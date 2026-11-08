"""统一 RAG 回答器模块 - 整合确定性回答与 LLM 增强回答

本模块合并了原 answerer.py（确定性回答）和 llm_answerer.py（LLM 增强回答）的功能，提供：
1. 统一的回答接口（RAGAnswerer）
2. 确定性回答引擎（DomainOntology + 规则匹配）
3. LLM 增强回答引擎（RAG + LLM）
4. 自动路由：根据问题类型和配置选择最佳回答模式
5. 多跳推理：基于本体关系图的路径搜索，发现实体间深层关联

架构目标：
- G4: 统一知识消费层
- G5: 统一质量闭环

[Phase 3 - RAG 2026-06-26]: Merged from answerer.py and llm_answerer.py
[Phase 3 - Multi-hop Reasoning]: Added ontology-based path search
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Deque
from collections import deque

import yaml

from src.domain.errors import DataContractError

from .formatting import (
    format_dify_structured_answer,
    format_document_result,
    format_neo4j_result,
    generate_ah_framework_report,
    synthesize_natural_answer,
)
from .models import (
    LLMAnswerResult,
    RetrievedChunk,
    EDGE_TYPES_SYSTEM_TO_CAP,
)
from .chunk_indexing import load_chunks_jsonl, load_idf, load_inverted_index
from .unified_query import parse as parse_query, get_rewriter, RewrittenQuery
from .retriever import retrieve_keyword
from .hybrid_rag import HybridIndexManager, HybridSearchResult  # RAG-P0-1 2026-08-04: true BM25+Vector fusion
from .core.unified_rag_service import UnifiedRAGService
from .evidence_gate import (
    EvidenceGate,
    EvidenceGateAction,
    EvidenceItem,
    get_evidence_gate,
)  # P1-1 (2026-09-01): RETRIEVE_MORE auto re-retrieval

try:
    from .neo4j_retriever import Neo4jRetriever
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

try:
    from ontology_compiler.rigor.llm_provider import UnifiedLLMProvider, LLMRequest
    LLM_PROVIDER_AVAILABLE = True
except ImportError:
    UnifiedLLMProvider = None  # type: ignore[assignment,misc]
    LLMRequest = None  # type: ignore[assignment,misc]
    LLM_PROVIDER_AVAILABLE = False

# Import UnifiedLLMClient from ontology_compiler for Pipeline-style LLM calls
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from ontology_compiler.rigor.llm_backend import UnifiedLLMClient
    UNIFIED_LLM_CLIENT_AVAILABLE = True
except ImportError:
    UnifiedLLMClient = None
    UNIFIED_LLM_CLIENT_AVAILABLE = False

# Import Dify client (preferred when LLM_PROVIDER=dify, env.env already configures it)
try:
    from .infrastructure.dify_client import (
        load_dify_config_from_env,
        ask_dify_kb_qa,
        get_dify_answer_text,
        DifyConfig as _DifyConfigT,
    )
    DIFY_CLIENT_AVAILABLE = True
except ImportError:
    DIFY_CLIENT_AVAILABLE = False
    load_dify_config_from_env = None  # type: ignore[assignment]
    ask_dify_kb_qa = None  # type: ignore[assignment]
    get_dify_answer_text = None  # type: ignore[assignment]
    _DifyConfigT = None  # type: ignore[assignment,misc]


# =============================================================================
# RAG-07 (§9.7, 2026-09-03): deterministic engine + LLM-call mixin extracted
# into sibling modules.  Re-exported here for backward compatibility (public
# __all__ + test patches on rag_answerer.* names).
# =============================================================================
from .deterministic_answerer import (
    QuestionType,
    AnswerResult,
    DomainOntology,
    answer_deterministic,
)
from .llm_calls import LLMCallMixin
from .retrieval_mixin import RetrievalMixin


class DifyCallLLMBackend:
    """Dify adapter exposing ``call_llm(prompt, max_tokens, temperature)``.

    P1-2 (2026-09-01): lets ``QueryRewriter._llm_expand`` Route 1
    (``hasattr(backend, "call_llm")``) run against a Dify workflow/chat app when
    no UnifiedLLMProvider / UnifiedLLMClient is configured.  Dify controls
    temperature/max_tokens inside the workflow/chat app, so both args are
    accepted for interface compatibility but not forwarded.
    """

    def __init__(self, config: Any) -> None:
        self._config = config

    def call_llm(self, prompt: str, max_tokens: int = 50, temperature: float = 0.0) -> str:
        raw = ask_dify_kb_qa(
            query=prompt,
            inputs={
                "system_prompt": "你是本体检索关键词提取助手，只输出英文关键词。",
                "user_question": prompt,
                "full_prompt": prompt,
            },
            config=self._config,
        )
        return get_dify_answer_text(raw) or ""


# =============================================================================
# Unified RAG Answerer (combines Deterministic + LLM)
# =============================================================================

class RAGAnswerer(LLMCallMixin, RetrievalMixin):
    """统一 RAG 解答器 - 支持确定性回答和 LLM 增强回答"""

    _MASTER_NODE_TYPES: Set[str] = {
        "master_system", "master_asset", "master_process",
        "master_actor", "master_data", "master_capability",
    }

    _L2_SCENARIO_TYPES: Set[str] = {
        "l2_business_scenario", "l2_business_logic",
        "master_business_scenario",
    }

    _prompts_config: Dict[str, Any] = {}
    _QUESTION_PATTERNS: Dict[str, List[str]] = {}
    _SYSTEM_PROMPTS: Dict[str, str] = {}

    def __init__(
        self,
        rag_dir: Path = Path("outputs/ontology/rag"),
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        # =====================================================================
        # MAINTENANCE NOTE (2026-08-04): Recommended default topk window
        # =====================================================================
        # Default retrieval topk should stay in the 4..6 range so that the
        # total ontology context length fits within the DIFY_MAX_CONTEXT_CHARS
        # threshold (default 12 000 chars). For typical 1 000 char/chunk the
        # relationship is roughly:
        #     topk 4 -> ~ 4 000..5 000 chars context  (well under 12 k)
        #     topk 5 -> ~ 5 000..6 500 chars context  (RECOMMENDED default)
        #     topk 6 -> ~ 6 000..8 000 chars context  (still under 12 k)
        #     topk >= 10 -> can exceed 12 k -> raise DIFY_MAX_CONTEXT_CHARS env
        #                 e.g. set DIFY_MAX_CONTEXT_CHARS=20000 in config/env.env
        # =====================================================================
        max_context_chunks: int = 5,
        max_chunk_length: int = 1000,
        use_neo4j: bool = False,
        neo4j_weight: float = 0.3,
        use_llm: bool = True,
        use_multi_hop: bool = True,
        max_hop_depth: int = 3,
        packet_path: Optional[Path] = None,
        packet: Optional[Dict[str, Any]] = None,
        rag_service: Optional[UnifiedRAGService] = None,
    ):
        self.rag_dir = rag_dir
        self.model = model
        self.max_context_chunks = max_context_chunks
        self.max_chunk_length = max_chunk_length
        self.use_neo4j = use_neo4j and NEO4J_AVAILABLE
        self.neo4j_weight = neo4j_weight
        self.use_llm = use_llm
        self.use_multi_hop = use_multi_hop
        self.max_hop_depth = max_hop_depth
        self.packet_path = packet_path
        self.packet = packet
        # --- governed packet hard constraint (2026-08-29) ------------------
        # Resolve the authoritative rag_dir here so callers which omit the arg
        # (the common case) automatically point at the governed packet.  Legacy
        # directories raise DataContractError unless RAG_ALLOW_LEGACY_DIR=1.
        from .governed_packets import resolve_governed_rag_dir

        self.rag_dir = resolve_governed_rag_dir(rag_dir)
        # -------------------------------------------------------------------
        self.rag_service = rag_service or (
            UnifiedRAGService.from_packet(packet_path=str(packet_path) if packet_path else None, packet=packet)
            if packet_path is not None or packet is not None
            else None
        )

        if self.rag_service is None:
            self.chunks = load_chunks_jsonl(self.rag_dir / "chunks.jsonl")
            self.index = load_inverted_index(self.rag_dir / "inverted_index.json")
            self.idf = load_idf(self.rag_dir / "idf.json")
        else:
            self.chunks = []
            self.index = {}
            self.idf = {}

        # P1-1 (2026-09-01): Evidence-gate drives RETRIEVE_MORE auto
        # re-retrieval. Defaults to the process-wide singleton so production
        # answers on sufficient evidence and re-retrieves (expanded top_k) when
        # the gate asks for more. Tests that bypass ``__init__`` leave this
        # attribute unset -> the gate loop is skipped (legacy behaviour).
        self.evidence_gate: Optional[EvidenceGate] = (
            evidence_gate if evidence_gate is not None else get_evidence_gate()
        )

        # RAG-P0-1 (2026-08-04): Hybrid (BM25 + Vector) retrieval through the
        # canonical HybridIndexManager. The old retrieve_keyword() call path
        # was pure BM25, ignoring the vector index on disk entirely.
        # Lazy-init: only construct the manager if someone actually calls a
        # retrieval helper, so quick operations (e.g. deterministic answerer)
        # don't pay the vector-load cost.
        self._hybrid_manager: Optional[HybridIndexManager] = None
        self._hybrid_manager_tried: bool = False  # avoid re-raising ImportError repeatedly

        self.neo4j_retriever = None
        if self.use_neo4j:
            try:
                self.neo4j_retriever = Neo4jRetriever()
            except Exception as e:
                self.use_neo4j = False

        self._load_env_file()

        self.dify_config = None  # Dify LLM (preferred when LLM_PROVIDER=dify)
        self.llm_provider: Optional[Any] = None
        self.llm_client: Optional[UnifiedLLMClient] = None
        if self.use_llm:
            try:
                # Priority 1: Dify LLM (explicit LLM_PROVIDER=dify or env keys present & available)
                force_dify = (os.getenv("LLM_PROVIDER") or "").strip().lower() == "dify"
                has_dify_keys = bool(
                    (os.getenv("DIFY_API_KEY") or "").strip()
                    and (os.getenv("DIFY_BASE_URL") or "").strip()
                )
                if DIFY_CLIENT_AVAILABLE and (force_dify or has_dify_keys):
                    try:
                        self.dify_config = load_dify_config_from_env()  # type: ignore[misc]
                        print(
                            f"✅ RAG using Dify LLM "
                            f"(base_url={self.dify_config.base_url}, app_type={self.dify_config.app_type})"
                        )
                    except Exception as e:
                        print(f"⚠️ RAG Dify init skipped (fallback to other LLMs): {e}")
                        self.dify_config = None

                # Priority 2: UnifiedLLMClient (Pipeline-style) if available
                if not self.dify_config and UNIFIED_LLM_CLIENT_AVAILABLE:
                    from src.application.bootstrap.llm_bootstrap import get_llm_client
                    self.llm_client = get_llm_client()
                    print(f"✅ RAG using UnifiedLLMClient (provider={self.llm_client.provider})")

                # Priority 3: UnifiedLLMProvider
                if not self.dify_config and not self.llm_client and LLM_PROVIDER_AVAILABLE:
                    self.llm_provider = UnifiedLLMProvider()
                    print(f"✅ RAG using UnifiedLLMProvider")
            except Exception as e:
                print(f"⚠️ RAG LLM init error: {e}")
                self.llm_provider = None

        if not RAGAnswerer._prompts_config:
            RAGAnswerer._prompts_config = self._load_prompts_config()
            RAGAnswerer._QUESTION_PATTERNS = RAGAnswerer._prompts_config.get("question_patterns", {})
            RAGAnswerer._SYSTEM_PROMPTS = RAGAnswerer._prompts_config.get("system_prompts", {})

        self._domain_ontology: Optional[DomainOntology] = None
        if self.rag_service is None:
            self._load_domain_ontology()

    def _load_env_file(self):
        env_paths = [
            Path("config/env.env"),
            Path("../config/env.env"),
            Path("../../config/env.env"),
        ]
        for env_path in env_paths:
            if env_path.exists():
                with open(env_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        if '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            if key and value and os.getenv(key) is None:
                                os.environ[key] = value
                break

    def _load_domain_ontology(self):
        """加载领域本体用于多跳推理。"""
        ontology_paths = [
            self.rag_dir.parent / "layer1_domain_ontology.json",
            Path("outputs/ontology/layer1_domain_ontology.json"),
            Path("../outputs/ontology/layer1_domain_ontology.json"),
        ]
        for path in ontology_paths:
            if path.exists():
                try:
                    self._domain_ontology = DomainOntology(path)
                    break
                except Exception as e:
                    pass

    def _is_multi_hop_question(self, question: str) -> bool:
        """判断是否为多跳推理问题。"""
        multi_hop_patterns = [
            "如何通过", "什么路径", "如何关联", "之间的关系",
            "从.*到.*", "连接", "通路", "链路", "路由",
            "中间经过", "涉及哪些", "关联哪些",
        ]
        for pattern in multi_hop_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                return True
        return False

    def _extract_entity_ids_from_context(self, context: str) -> List[str]:
        """从上下文中提取实体ID。"""
        patterns = [
            r"(CAP_L[12]_\d+_\w+)",
            r"(SYS_\d+_\w+)",
            r"(DATA_\d+_\w+)",
            r"(PROC_\d+_\w+)",
            r"(ASSET_\d+_\w+)",
        ]
        entity_ids = []
        for pattern in patterns:
            matches = re.findall(pattern, context)
            entity_ids.extend(matches)
        return list(set(entity_ids))

    def _build_multi_hop_context(self, question: str, entity_ids: List[str]) -> str:
        """
        构建多跳推理上下文。
        
        Args:
            question: 用户问题
            entity_ids: 已识别的实体ID列表
            
        Returns:
            多跳推理上下文字符串
        """
        if not self._domain_ontology or not entity_ids:
            return ""
        
        parts = ["=== 多跳推理路径 ==="]
        
        for i, start_id in enumerate(entity_ids):
            reachable = self._domain_ontology.find_reachable_nodes(
                start_id, max_hops=self.max_hop_depth
            )
            if reachable:
                parts.append(f"\n从 [{start_id}] {self._domain_ontology.node_name(start_id)} 出发:")
                for node in reachable[:10]:
                    path_str = " → ".join([f"{p[0]}({p[1]})" for p in node["path"]])
                    if path_str:
                        path_str += f" → {node['node_id']}"
                    parts.append(f"  • {node['hops']}跳: {node['name']} ({node['meta_class']})")
                    if node["path"]:
                        parts.append(f"    路径: {path_str}")
        
        if len(entity_ids) >= 2:
            common = self._domain_ontology.find_common_neighbors(entity_ids, max_hops=2)
            if common:
                parts.append(f"\n{len(entity_ids)}个实体的公共邻居:")
                for node in common[:5]:
                    parts.append(f"  • {node['name']} ({node['meta_class']})")
        
        return "\n".join(parts)

    def _load_prompts_config(self) -> Dict[str, Any]:
        config_paths = [
            Path("config/prompts.yaml"),
            Path("../config/prompts.yaml"),
            Path("../../config/prompts.yaml"),
        ]
        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        return yaml.safe_load(f)
                except Exception as e:
                    break
        return {
            "question_patterns": {
                "DEEP_ANALYSIS": [
                    "是什么", "分析", "详解", "介绍", "阐述", "深度", "业务",
                    "what is", "analyze", "explain", "detail", "business",
                    "定义", "作用", "功能", "用途", "影响", "重要性",
                    "definition", "function", "usage", "impact",
                ],
                "SYSTEM_RELATION": ["关系", "交互", "集成", "接口", "连接", "通信", "调用",
                                   "上下游", "依赖", "输入输出", "对接", "关联系统",
                                   "relation", "interface", "integration", "interact"],
                "BUSINESS_SCENARIO": ["流程", "场景", "如何", "怎么", "步骤", "操作",
                                      "触发", "条件", "输出", "结果", "工作流",
                                      "process", "scenario", "workflow", "how to"],
                "DATA_FLOW": ["数据", "数据流", "流转", "传输", "结构", "字段",
                              "来源", "去向", "存储", "格式", "BOM", "变更单",
                              "data", "flow", "field", "transform"],
                "ROLE_RESPONSIBILITY": ["职责", "负责", "谁", "角色", "人员", "部门",
                                         "owner", "responsibility", "role", "who"],
                "CONCEPT_DEFINITION": ["是什么", "定义", "含义", "解释", "什么是",
                                        "what is", "define", "meaning"],
            },
            "system_prompts": {
                "DEEP_ANALYSIS": (
                    "你是一个企业知识分析师，擅长进行数据资产的深度业务分析。\n"
                    "请基于提供的本体知识库上下文，按照以下A-H框架结构进行深度分析：\n\n"
                    "A. 数据对象定义：说明这个数据对象是什么，内部叫法、数据分类、核心特征\n"
                    "B. 涉及流程与系统：它在哪些流程里用，涉及哪些系统，关键步骤\n"
                    "C. 角色与决策：谁创建、谁修改、谁审批，谁使用，用来做什么决策\n"
                    "D. 常见问题与标准：目前最常见的问题，如果作为数据产品希望有什么标准\n"
                    "E. 典型业务场景：描述一个具体的业务流程场景\n"
                    "F. 数据来源与权威源：数据从哪里来，哪个系统是权威源\n\n"
                    "要求：\n"
                    "1. 用中文回答\n"
                    "2. 结合业务术语和具体系统名称\n"
                    "3. 如果上下文中有系统名称（如CMP, OHF, BIS, PLUS等），请提及\n"
                    "4. 如果上下文中有流程步骤，请详细描述\n"
                    "5. 如果信息不足，请基于已有信息进行合理推断\n"
                    "6. 使用清晰的小标题分段"
                ),
                "SYSTEM_RELATION": (
                    "你是一个企业系统集成分析师。\n"
                    "请基于提供的本体知识库，分析实体之间的关系。\n"
                    "要求：说明数据流向、依赖关系、接口交互等。"
                ),
                "BUSINESS_SCENARIO": (
                    "你是一个企业业务分析师。\n"
                    "请基于提供的本体知识库，描述业务流程和场景。\n"
                    "要求：按步骤说明、涉及系统、关键触发条件。"
                ),
                "GENERAL": (
                    "你是一个企业知识分析师。\n"
                    "请基于提供的本体知识库，用清晰、准确的中文回答问题。"
                ),
            }
        }

    def _rewrite_query(self, question: str) -> str:
        """[H7 FIX] 直接传递 provider/client 对象，不再访问私有属性 ``_client``。

        QueryRewriter 内部路由：
        - UnifiedLLMProvider → 公开 ``call_llm()``
        - UnifiedLLMClient   → 公开 async ``call()``
        - OpenAI SDK client  → 公开 ``chat.completions.create()`` (legacy)

        P1-2 (2026-09-01): 当未配置 UnifiedLLMProvider/Client 但已配置 Dify 时，
        注入 ``DifyCallLLMBackend``，恢复 Dify 分支的 LLM 扩展（此前永不执行）。
        """
        llm_provider = self.llm_provider
        llm_client = self.llm_client
        if llm_provider is None and llm_client is None and self.dify_config is not None:
            llm_provider = DifyCallLLMBackend(self.dify_config)
        rewriter = get_rewriter(
            self.rag_dir,
            llm_provider=llm_provider,  # Priority 3 provider (goes via call_llm)
            llm_client=llm_client,      # Priority 2 UnifiedLLMClient (goes via call)
            model=self.model,
        )
        rewrite_info: RewrittenQuery = rewriter.rewrite(question)
        return rewrite_info.rewritten

    def _search_neo4j(self, retrieval_query: str) -> list[dict[str, Any]]:
        if not self.use_neo4j or not self.neo4j_retriever:
            return []
        try:
            return self.neo4j_retriever.keyword_search(retrieval_query, topk=3)
        except Exception:
            return []

    def _build_evidence(self, results: list[Any], neo4j_results: list[dict[str, Any]], effective_topk: int) -> list[dict[str, Any]]:
        # G5c (2026-08-17): citation built via the shared policy builder — the
        # emitted dict shape is unchanged (locked by answer-path characterization).
        from .policy.fallback import build_citation

        evidence = []
        for r in results[:effective_topk]:
            chunk = r.chunk
            evidence.append(build_citation(
                chunk_id=getattr(chunk, "chunk_id", "unknown"),
                chunk_type=getattr(chunk, "chunk_type", "unknown"),
                source=chunk.meta.get("source", "unknown"),
                score=r.score,
                content_preview=chunk.content[:200],
            ))

        for r in neo4j_results:
            evidence.append(build_citation(
                chunk_id=f"neo4j:{r.get('id', 'unknown')}",
                chunk_type=f"Neo4j:{r.get('node_type', 'unknown')}",
                source="neo4j",
                score=1.0,
                content_preview=r.get('description', r.get('name', ''))[:200],
            ))
        return evidence

    def _compute_confidence(self, results: list[Any], neo4j_results: list[dict[str, Any]]) -> float:
        base_conf = min(0.9, 0.5 + len(results) * 0.08)
        if neo4j_results:
            base_conf = min(0.95, base_conf + 0.1)
        return base_conf

    async def _search_packet(self, query: str, top_k: int) -> list[Any]:
        if self.rag_service is None:
            return []
        return await self.rag_service.search(query, top_k=top_k)

    async def _search_packet_with_gate(
        self,
        retrieval_query: str,
        topk: int,
        max_retrieval_rounds: int = 3,
    ) -> list[Any]:
        """P1-1 (2026-09-01): Evidence-gate driven retrieval loop (packet path).

        Round 1 runs the packet search unchanged.  When the gate decides
        RETRIEVE_MORE, later rounds re-run the packet search with a 50%-expanded
        top_k and merge the new ``RAGSearchResult`` items (dedup by chunk_id).
        Stops on ANSWER / ESCALATE.  Last decision / rounds are recorded on
        ``self._last_gate_decision`` / ``self._last_gate_rounds_used``.
        """
        gate = getattr(self, "evidence_gate", None)
        if gate is None:
            return await self._search_packet(retrieval_query, topk)

        accumulated: Dict[str, Any] = {}
        for r in await self._search_packet(retrieval_query, topk):
            accumulated[getattr(r, "chunk_id", None) or id(r)] = r

        current_topk = topk
        round_num = 1
        decision = None
        while round_num <= max_retrieval_rounds:
            evidence_items = [
                EvidenceItem(
                    text=getattr(r, "content", "") or "",
                    score=max(0.0, min(1.0, float(getattr(r, "score", 0.0) or 0.0))),
                    source_id=getattr(r, "source", None) or getattr(r, "chunk_id", "packet"),
                )
                for r in accumulated.values()
            ]
            # 提取查询术语用于证据门控评估
            try:
                pq = parse_query(retrieval_query)
                query_terms = getattr(pq, "query_terms", retrieval_query.split())
            except Exception:
                query_terms = retrieval_query.split()
            decision = gate.evaluate(
                evidence_items,
                query_terms=query_terms,
                has_conflict=False,
                retrieval_round=round_num,
                max_retrieval_rounds=max_retrieval_rounds,
            )
            if decision.action in (EvidenceGateAction.ANSWER, EvidenceGateAction.ESCALATE):
                break
            current_topk = max(current_topk, int(current_topk * 1.5) + 1)
            for r in await self._search_packet(retrieval_query, current_topk):
                key = getattr(r, "chunk_id", None) or id(r)
                if key not in accumulated:
                    accumulated[key] = r
            round_num += 1

        self._last_gate_decision = decision
        self._last_gate_rounds_used = round_num
        return list(accumulated.values())

    async def _answer_async_impl(self, question: str, topk: int = 5, mode: str = "auto") -> LLMAnswerResult:
        """Async implementation for packet-first and legacy answer paths."""
        if self.rag_service is not None:
            retrieval_query = self._rewrite_query(question)
            packet_results = await self._search_packet_with_gate(retrieval_query, topk)
            if not packet_results:
                return LLMAnswerResult(
                    answer="未能找到相关信息来回答这个问题。",
                    evidence_chunks=[],
                    confidence=0.0,
                    model=self.model,
                )
            packet_evidence = self._packet_results_to_evidence(packet_results)
            context = "\n---\n".join(
                f"[{idx}] 来源: {item['source']}\n{item['content_preview']}"
                for idx, item in enumerate(packet_evidence, 1)
            )
            if self.use_llm and (self.dify_config or self.llm_client or self.llm_provider):
                try:
                    answer = self._call_llm(question, context)
                    confidence = min(0.95, 0.5 + len(packet_results) * 0.08)
                except Exception as exc:
                    answer = f"LLM 生成答案时出错: {exc}\n\n{context}"
                    confidence = 0.3
            else:
                answer = context
                confidence = min(0.9, 0.5 + len(packet_results) * 0.08)
            return LLMAnswerResult(
                answer=answer,
                evidence_chunks=packet_evidence,
                confidence=confidence,
                model="retrieval_asset_packet",
            )

        retrieval_query = self._rewrite_query(question)
        pq = parse_query(retrieval_query, base_topk=topk)
        effective_topk = pq.recommended_topk
        results = self._retrieve_with_gate(pq, topk=topk)
        neo4j_results = self._search_neo4j(retrieval_query)

        if not results and not neo4j_results:
            return LLMAnswerResult(
                answer="未能找到相关信息来回答这个问题。",
                evidence_chunks=[],
                confidence=0.0,
                model=self.model
            )

        context = self._build_context(results, neo4j_results, max_chunks=effective_topk)

        if self.use_multi_hop and self._is_multi_hop_question(question):
            entity_ids = self._extract_entity_ids_from_context(context)
            if entity_ids:
                multi_hop_context = self._build_multi_hop_context(question, entity_ids)
                if multi_hop_context:
                    context += "\n\n" + multi_hop_context

        if self.use_llm and (self.dify_config or self.llm_client or self.llm_provider):
            try:
                answer = self._call_llm(question, context)
                confidence = self._compute_confidence(results, neo4j_results)
            except Exception as e:
                import traceback as _tb
                _tb.print_exc()
                answer = f"❌ LLM 生成答案时出错: {e}\n\n以下是检索到的相关信息：\n\n{context}"
                confidence = 0.3
        else:
            answer = self._format_retrieved_results(question, results)
            confidence = 0.5

        actual_model = self.model
        if self.dify_config:
            actual_model = "dify"
        elif self.llm_client:
            actual_model = str(getattr(self.llm_client, "provider", "unknown"))
        elif self.llm_provider:
            actual_model = str(getattr(self.llm_provider, "backend", "unknown"))

        return LLMAnswerResult(
            answer=answer,
            evidence_chunks=self._build_evidence(results, neo4j_results, effective_topk),
            confidence=confidence,
            model=actual_model,
        )

    async def answer_async(self, question: str, topk: int = 5, mode: str = "auto") -> LLMAnswerResult:
        """Async answer API with no asyncio.run() usage."""
        return await self._answer_async_impl(question, topk=topk, mode=mode)

    def answer(self, question: str, topk: int = 5, mode: str = "auto") -> LLMAnswerResult:
        """Synchronous compatibility wrapper for legacy callers."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._answer_async_impl(question, topk=topk, mode=mode))
        raise RuntimeError(
            "RAGAnswerer.answer() cannot be called from a running event loop. "
            "Use await RAGAnswerer.answer_async(...) instead."
        )

    def _packet_results_to_evidence(self, results: list[Any]) -> list[dict[str, Any]]:
        # G5c (2026-08-17): shared citation builder keeps the packet shape
        # (extra ``metadata`` field) while sharing the citation output contract.
        from .policy.fallback import build_citation

        evidence = []
        for result in results:
            metadata = getattr(result, "metadata", {}) or {}
            evidence.append(build_citation(
                chunk_id=getattr(result, "chunk_id", "unknown"),
                chunk_type=metadata.get("entity_type") or metadata.get("relation_type") or "packet",
                source=getattr(result, "source", "retrieval_asset_packet"),
                score=float(getattr(result, "score", 0.0) or 0.0),
                content_preview=str(getattr(result, "content", ""))[:200],
                extra={"metadata": metadata},
            ))
        return evidence

    async def answer_detailed_async(self, question: str, topk: int = 5, mode: str = "auto") -> Dict[str, Any]:
        """生成详细答案（包含完整调试信息）的异步版本。"""
        result = await self.answer_async(question, topk=topk, mode=mode)
        return {
            "question": question,
            "answer": result.answer,
            "confidence": result.confidence,
            "model": result.model,
            "evidence_count": len(result.evidence_chunks),
            "evidence": result.evidence_chunks,
        }

    def answer_detailed(self, question: str, topk: int = 5) -> Dict[str, Any]:
        """LEGACY sync wrapper for detailed answers.

        Prefer `await answer_detailed_async(...)` in async code.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.answer_detailed_async(question, topk=topk))
        raise RuntimeError(
            "RAGAnswerer.answer_detailed() cannot be called from a running event loop. "
            "Use await RAGAnswerer.answer_detailed_async(...) instead."
        )

    def close(self):
        """关闭连接（Neo4j）"""
        if self.neo4j_retriever:
            self.neo4j_retriever.close()

    def fallback_policy(self):
        """Access the unified fallback policy (RAG-06 / B-6).

        Lazy import — the policy modules are additive; old paths that never call
        this stay untouched.
        """
        from .policy.fallback import FallbackPolicy

        return FallbackPolicy.default()


# =============================================================================
# LLMAnswerer Backward Compatibility Wrapper
# =============================================================================

class LLMAnswerer(RAGAnswerer):
    """LLMAnswerer 向后兼容包装 - 已整合到 RAGAnswerer"""
    
    def __init__(
        self,
        rag_dir: Optional[Path] = None,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_context_chunks: int = 5,
        max_chunk_length: int = 1000,
        use_neo4j: bool = False,
        neo4j_weight: float = 0.3,
    ):
        super().__init__(
            rag_dir=rag_dir,
            model=model,
            max_context_chunks=max_context_chunks,
            max_chunk_length=max_chunk_length,
            use_neo4j=use_neo4j,
            neo4j_weight=neo4j_weight,
            use_llm=True,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

async def answer_question_async(
    question: str,
    rag_dir: Path = Path("outputs/ontology/rag"),
    model: str = "deepseek-chat",
    packet_path: Optional[Path] = None,
    packet: Optional[Dict[str, Any]] = None,
    rag_service: Optional[UnifiedRAGService] = None,
) -> str:
    """便捷函数：异步回答问题。"""
    answerer = RAGAnswerer(
        rag_dir=rag_dir,
        model=model,
        packet_path=packet_path,
        packet=packet,
        rag_service=rag_service,
    )
    result = await answerer.answer_async(question)
    return result.answer


async def answer_detailed_async(
    question: str,
    rag_dir: Path = Path("outputs/ontology/rag"),
    model: str = "deepseek-chat",
    packet_path: Optional[Path] = None,
    packet: Optional[Dict[str, Any]] = None,
    rag_service: Optional[UnifiedRAGService] = None,
) -> Dict[str, Any]:
    """便捷函数：异步生成详细答案。"""
    answerer = RAGAnswerer(
        rag_dir=rag_dir,
        model=model,
        packet_path=packet_path,
        packet=packet,
        rag_service=rag_service,
    )
    return await answerer.answer_detailed_async(question)


def answer_question(
    question: str,
    rag_dir: Path = Path("outputs/ontology/rag"),
    model: str = "deepseek-chat",
    packet_path: Optional[Path] = None,
    packet: Optional[Dict[str, Any]] = None,
    rag_service: Optional[UnifiedRAGService] = None,
) -> str:
    """LEGACY sync convenience wrapper.

    Prefer `await answer_question_async(...)` in async code.
    """
    answerer = RAGAnswerer(
        rag_dir=rag_dir,
        model=model,
        packet_path=packet_path,
        packet=packet,
        rag_service=rag_service,
    )
    result = answerer.answer(question)
    return result.answer


__all__ = [
    # Core classes
    "RAGAnswerer",
    "LLMAnswerer",
    "DomainOntology",
    "QuestionType",
    # Result classes
    "AnswerResult",
    "LLMAnswerResult",
    # Functions
    "answer_deterministic",
    "answer_detailed_async",
    "answer_question_async",
    "answer_question",
]
