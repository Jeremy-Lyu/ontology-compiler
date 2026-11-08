"""统一查询处理模块 - 整合解析、重写、分类功能

本模块整合了原 query_parser.py、query_rewriter.py、query_classifier.py 的功能，提供：
1. 查询文本处理：NFC 正规化、token 化、实体提取
2. 查询重写：术语映射、LLM 扩展
3. 查询分类：类型、意图、复杂度识别
4. 统一接口：UnifiedQueryProcessor

架构目标：
- G4: 统一知识消费层
- G5: 统一质量闭环

[Phase 3 - RAG 2026-06-26]: Merged from query_parser.py, query_rewriter.py, query_classifier.py
"""

from __future__ import annotations

import asyncio
import json
import re
import unicodedata
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field

# LLM provider protocol shape — used by QueryRewriter to call the public
# ``call_llm`` / ``call`` methods without importing the concrete classes
# (avoids circular imports between rag <-> llm_provider/rigor).
_LLMProviderLike = Any  # UnifiedLLMProvider | UnifiedLLMClient | "OpenAI"


# =============================================================================
# Constants and Patterns
# =============================================================================

# 与 indexer._TOKEN_RE 保持一致（字母、数字、下划线）
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

# 实体名称模式：2 个以上大写字母/数字组成的词（如 OHF、CMP、BIS、MBOM）
_ENTITY_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z][A-Z0-9]{1,})(?![A-Za-z0-9])")

# 每个实体在多实体查询中额外补充的 chunk 数
_ENTITY_EXTRA_CHUNKS = 3

# 本体术语映射表（从本体数据中提取）
_DOMAIN_TERM_MAP = {
    # 生产计划相关
    "生产计划": ["Production Planning", "Order Scheduling", "Fixed Planning", "Preliminary Planning"],
    "排程": ["Order Scheduling", "Production Scheduling"],
    "固定计划": ["Fixed Planning"],
    "初步计划": ["Preliminary Planning"],
    "产能订单": ["Capacity Orders"],
    "主生产计划": ["Master Production Schedule", "MPS"],
    "订单复制": ["Order Replication"],
    "生产日历": ["Production Calendar"],
    # BOM 相关
    "物料清单": ["BOM", "Bill of Materials"],
    "工程BOM": ["EBOM", "Engineering BOM"],
    "制造BOM": ["MBOM", "Manufacturing BOM"],
    "变更通知": ["C-note", "Change Note"],
    # 系统相关
    "订单处理": ["Order Handling", "OHF"],
    "工厂BOM": ["CMP", "Factory BOM"],
    "业务介绍": ["BIS", "Business Introduction"],
    # 流程相关
    "锁定": ["Lock Request", "Order Lock"],
    "一致性检查": ["Consistency Check"],
    "故障报告": ["Fault Report"],
}

# LLM 提示词：提取搜索关键词
_LLM_EXPAND_PROMPT = """\
你是制造工程数字化平台的知识助手。
用户提出了一个查询，你需要提取出最适合检索本体知识库的英文关键词。

本体知识库覆盖：订单管理（OHF/ARO/BOP）、BOM管理（CMP/EBOM/MBOM）、
物料追溯（BIS/ATACQ）、生产计划（Production Planning/Order Scheduling）、
工程变更（C-note/Change Management）等制造领域。

规则：
- 只输出英文关键词，用空格分隔
- 最多输出 8 个关键词
- 不要输出任何解释或标点
- 输出示例：Production Planning Order Scheduling Fixed Planning ARO OHF

用户查询：{query}
关键词："""


# =============================================================================
# Query Type Enums
# =============================================================================

class QueryType(str, Enum):
    """查询类型"""
    ENTITY_LOOKUP = "entity_lookup"  # 实体查询（如：什么是 OHF？）
    RELATION_QUERY = "relation_query"  # 关系查询（如：OHF 和 ARO 的关系？）
    ATTRIBUTE_QUERY = "attribute_query"  # 属性查询（如：OHF 的字段有哪些？）
    SEMANTIC_SEARCH = "semantic_search"  # 语义搜索（如：生产计划流程）
    HYBRID_SEARCH = "hybrid_search"  # 混合查询


class QueryIntent(str, Enum):
    """查询意图"""
    DEFINITION = "definition"  # 定义（什么是 X？）
    RELATIONSHIP = "relationship"  # 关系（X 和 Y 的关系？）
    PROCESS = "process"  # 流程（X 的流程？）
    COMPARISON = "comparison"  # 对比（X 和 Y 的区别？）
    CAUSE_EFFECT = "cause_effect"  # 因果（为什么 X？）
    LIST = "list"  # 列表（列出所有 X？）
    VALIDATION = "validation"  # 验证（X 是否包含 Y？）


class QueryComplexity(str, Enum):
    """查询复杂度"""
    SIMPLE = "simple"  # 简单（单实体/单关系）
    MEDIUM = "medium"  # 中等（多实体/多关系）
    COMPLEX = "complex"  # 复杂（跨层级/多跳/聚合）


# =============================================================================
# Query Data Classes
# =============================================================================

@dataclass(frozen=True)
class ParsedQuery:
    """查询解析结果。

    Attributes:
        original: 原始查询字符串
        normalized: NFC 正规化后的查询（用于 LLM prompt）
        tokens: 小写 token 列表（与索引 tokenize() 输出一致，用于 BM25）
        entities: 提取到的大写实体名称列表（如 ['OHF', 'CMP', 'BIS']）
        is_multi_entity: 是否为多实体查询
        recommended_topk: 推荐的检索窗口大小（随实体数动态扩展）
    """

    original: str
    normalized: str
    tokens: List[str]
    entities: List[str]
    is_multi_entity: bool
    recommended_topk: int

    @property
    def entity_set(self) -> Set[str]:
        """实体名称集合（大写）。"""
        return set(self.entities)


@dataclass(frozen=True)
class RewrittenQuery:
    """重写后的查询结果。"""
    original: str
    rewritten: str
    method: str  # "term_mapping" | "llm_rewrite" | "llm_expand" | "none"
    confidence: float  # 0-1，表示重写可信度


class QueryClassification(BaseModel):
    """查询分类结果"""
    query: str = Field(..., description="原始查询")
    query_type: QueryType = Field(..., description="查询类型")
    intent: QueryIntent = Field(..., description="查询意图")
    complexity: QueryComplexity = Field(..., description="查询复杂度")
    entities: List[str] = Field(default_factory=list, description="识别的实体")
    relations: List[str] = Field(default_factory=list, description="识别的关系")
    attributes: List[str] = Field(default_factory=list, description="识别的属性")
    keywords: List[str] = Field(default_factory=list, description="关键词")
    confidence: float = Field(1.0, description="分类置信度")
    suggestions: List[str] = Field(default_factory=list, description="优化建议")
    created_at: str = Field("", description="创建时间")


# =============================================================================
# Query Parsing Functions
# =============================================================================

def normalize_text(text: str) -> str:
    """NFC Unicode 正规化，消除编码差异（如全角/半角、组合字符）。

    不做大小写转换，保留原始大小写（token 化时再做 lower）。
    """
    return unicodedata.normalize("NFC", text or "")


def tokenize_query(text: str) -> List[str]:
    """与 indexer.tokenize() 完全一致的 token 化，保证查询与索引对齐。

    流程：NFC 正规化 → 提取字母数字下划线序列 → 小写。
    """
    normalized = normalize_text(text)
    return [t.lower() for t in _TOKEN_RE.findall(normalized) if t]


def extract_entities(text: str) -> List[str]:
    """从文本中提取大写实体名称（系统名、缩写等）。

    使用 lookaround 替代 \\b，兼容中英文混合边界。
    去重并保留首次出现顺序。

    Examples:
        >>> extract_entities("讲下OHF, CMP, BIS的关系")
        ['OHF', 'CMP', 'BIS']
        >>> extract_entities("OHF system manufacturing")
        ['OHF']
        >>> extract_entities("what is MBOM and how does EBOM relate")
        ['MBOM', 'EBOM']
    """
    seen: Set[str] = set()
    result: List[str] = []
    for m in _ENTITY_RE.finditer(normalize_text(text)):
        entity = m.group(1)
        if entity not in seen:
            seen.add(entity)
            result.append(entity)
    return result


def recommended_topk(base_topk: int, entity_count: int) -> int:
    """根据实体数量动态计算推荐检索窗口大小。

    单实体查询不扩展；多实体查询每个实体额外增加 _ENTITY_EXTRA_CHUNKS 个 chunk。

    Args:
        base_topk: 用户指定的基础 topk
        entity_count: 查询中的实体数量

    Returns:
        推荐的 topk，不低于 base_topk
    """
    if entity_count <= 1:
        return base_topk
    return base_topk + entity_count * _ENTITY_EXTRA_CHUNKS


def parse(query: str, base_topk: int = 5) -> ParsedQuery:
    """完整解析查询，返回 ParsedQuery。

    这是本模块的主入口，调用方只需传入原始查询字符串。

    Args:
        query: 用户原始查询
        base_topk: 基础检索数量（默认 5）

    Returns:
        ParsedQuery 包含 tokens、entities、recommended_topk 等

    Examples:
        >>> pq = parse("讲下OHF, CMP, BIS的关系")
        >>> pq.entities
        ['OHF', 'CMP', 'BIS']
        >>> pq.is_multi_entity
        True
        >>> pq.recommended_topk
        14
    """
    normalized = normalize_text(query)
    tokens = tokenize_query(query)
    entities = extract_entities(query)
    n_entities = len(entities)

    return ParsedQuery(
        original=query,
        normalized=normalized,
        tokens=tokens,
        entities=entities,
        is_multi_entity=n_entities > 1,
        recommended_topk=recommended_topk(base_topk, n_entities),
    )


# =============================================================================
# Query Rewriter
# =============================================================================

def _load_term_map_from_ontology(ontology_dir: Path) -> dict[str, List[str]]:
    """从本体数据中提取术语映射。

    [Phase 6 - IMPLEMENTED 2026-06-22]: 自动从 L2 scenarios 和 layer 文件中提取中英映射。

    策略：
    1. 扫描 L2 scenarios 中的 category、domain 字段
    2. 从 description 中提取中文关键词及其英文对应词
    3. 结合领域特定术语库构建完整映射
    """
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')

    term_map: dict[str, List[str]] = {}

    # 1. 扫描 L2 scenarios 文件
    l2_files = [
        ontology_dir / "layers" / "l2_scenarios.json",
        ontology_dir / "layers" / "l2_definitions.json",
        ontology_dir / "layers" / "l2_logic.json",
    ]

    for l2_file in l2_files:
        if not l2_file.exists():
            continue
        try:
            with open(l2_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 提取 scenarios 或 definitions 列表
            items = data.get("scenarios", []) or data.get("definitions", []) or []

            for item in items:
                # 从 category 提取映射（如 BOM_MANAGEMENT -> BOM 管理）
                category = item.get("category", "")
                domain = item.get("domain", "")
                desc = item.get("description", "")

                # 2. 从 description 中提中英对照词
                # 匹配 "XXX (XXX)" 或 "XXX - XXX" 模式
                mixed_pattern = re.compile(r'([\u4e00-\u9fff]+(?:\s*[\u4e00-\u9fff]+)*)\s*[\(/]\s*([A-Za-z0-9\s\-]+?)(?:[\)/,]|$)')
                for match in mixed_pattern.finditer(desc):
                    cn_term = match.group(1).strip()
                    en_term = match.group(2).strip()
                    if cn_term and en_term and len(cn_term) > 1:
                        if cn_term not in term_map:
                            term_map[cn_term] = []
                        if en_term not in term_map[cn_term]:
                            term_map[cn_term].append(en_term)

                # 3. 从 domain 和 category 提取标准化术语
                domain_terms = {
                    "MANUFACTURING": ["Manufacturing", "Production"],
                    "SUPPLY_CHAIN": ["Supply Chain", "Logistics"],
                    "LOGISTICS": ["Logistics", "Transportation"],
                    "BOM_MANAGEMENT": ["BOM Management", "Bill of Materials"],
                    "ORDER_MANAGEMENT": ["Order Management", "Order Handling"],
                    "PRODUCTION_PLANNING": ["Production Planning", "Planning"],
                    "CHANGE_MANAGEMENT": ["Change Management", "Change Note"],
                }
                if domain in domain_terms:
                    cn_domain = {"MANUFACTURING": "制造", "SUPPLY_CHAIN": "供应链",
                                 "LOGISTICS": "物流", "BOM_MANAGEMENT": "BOM管理",
                                 "ORDER_MANAGEMENT": "订单管理", "PRODUCTION_PLANNING": "生产计划",
                                 "CHANGE_MANAGEMENT": "变更管理"}.get(domain, domain)
                    if cn_domain not in term_map:
                        term_map[cn_domain] = domain_terms[domain]

        except (json.JSONDecodeError, IOError) as e:
            continue

    # 4. 如果提取的映射太少，补充关键术语
    min_terms = 10
    if len(term_map) < min_terms:
        # 回退到硬编码映射
        term_map.update(_DOMAIN_TERM_MAP)

    return term_map


class QueryRewriter:
    """查询重写器，支持术语映射和 LLM 辅助重写。

    [H7 FIX] 不再依赖调用方传入的 ``_client`` 私有属性（OpenAI SDK 形状）。
    C-1: 统一 gateway 优先，raw OpenAI SDK 只作末端兼容。路由顺序：

    1. ``UnifiedLLMProvider`` → 调 ``provider.call_llm(...)``
    2. ``UnifiedLLMClient``（rigor/llm_backend.py）→ 调 ``asyncio.run(client.call(...))``
    3. ``LLMGateway``（统一入口 / budget / pricing / monitor）
    4. 原始 ``OpenAI`` SDK 客户端（legacy 兜底，不再作为主路径）
    """

    def __init__(
        self,
        ontology_dir: Optional[Path] = None,
        llm_client: Optional[_LLMProviderLike] = None,
        llm_provider: Optional[_LLMProviderLike] = None,
        model: str = "deepseek-chat",
    ):
        self.term_map = _load_term_map_from_ontology(ontology_dir) if ontology_dir else _DOMAIN_TERM_MAP.copy()
        # [H7] Accept both names; prefer explicit llm_provider
        self._llm_backend = llm_provider or llm_client
        self.model = model

    @property
    def llm_client(self) -> Optional[_LLMProviderLike]:
        """Backwards-compatible alias for ``_llm_backend``."""
        return self._llm_backend

    @llm_client.setter
    def llm_client(self, value: Optional[_LLMProviderLike]) -> None:
        self._llm_backend = value

    def rewrite(self, query: str) -> RewrittenQuery:
        """前置重写查询：术语映射 + LLM 扩展（如果有客户端）。

        **始终**在检索前调用，不是降级兜底。

        策略：
        1. 术语映射：从词典中匹配中文词汇 → 追加对应英文术语
        2. LLM 扩展：调用 LLM 提取额外搜索关键词（对术语映射的补充）
        3. 合并去重后返回扩展查询

        Args:
            query: 原始查询（中文/英文均可）

        Returns:
            RewrittenQuery，包含扩展后的查询和方法
        """
        extra_terms: List[str] = []
        method_parts: List[str] = []

        # 1. 术语映射
        mapped_terms = self._map_terms(query)
        if mapped_terms:
            for _, en_terms in mapped_terms:
                extra_terms.extend(en_terms)
            method_parts.append("term_mapping")

        # 2. LLM 扩展（总是尝试，与术语映射互补）
        if self._llm_backend:
            llm_keywords = self._llm_expand(query)
            if llm_keywords:
                extra_terms.extend(llm_keywords)
                method_parts.append("llm_expand")

        # 去重，保留顺序
        extra_unique = list(dict.fromkeys(extra_terms))

        if extra_unique:
            rewritten = f"{query} {' '.join(extra_unique)}"
            return RewrittenQuery(
                original=query,
                rewritten=rewritten,
                method="+".join(method_parts) if method_parts else "none",
                confidence=0.9 if self._llm_backend else 0.8,
            )

        return RewrittenQuery(
            original=query,
            rewritten=query,
            method="none",
            confidence=1.0,
        )
    
    def _map_terms(self, query: str) -> List[tuple[str, List[str]]]:
        """提取查询中可映射的术语。"""
        mapped = []
        for zh_term, en_terms in self.term_map.items():
            if zh_term in query:
                mapped.append((zh_term, en_terms))
        return mapped
    
    def _llm_expand(self, query: str) -> List[str]:
        """调用 LLM 提取检索关键词（快速轻量调用，max_tokens=50）。

        C-1: 优先通过统一 gateway / public provider 路由。
        保留 legacy async 客户端兼容，但不再把 raw OpenAI SDK 作为主路径。
        """
        backend = self._llm_backend
        if backend is None:
            return []
        try:
            prompt = _LLM_EXPAND_PROMPT.format(query=query)
            raw: str = ""

            # Route 1: UnifiedLLMProvider (llm_provider.py) — public call_llm
            if hasattr(backend, "call_llm") and callable(backend.call_llm):
                raw = backend.call_llm(
                    prompt,
                    max_tokens=50,
                    temperature=0.0,
                )

            # Route 2: UnifiedLLMClient / async LLM port — public async call
            elif hasattr(backend, "call") and callable(backend.call) and asyncio.iscoroutinefunction(backend.call):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop and loop.is_running():
                    # Cannot block inside running event loop; skip gracefully
                    print("[QueryRewriter] async loop running; skip LLM expand (synchronous rewriter)")
                    return []
                raw = asyncio.run(backend.call(
                    prompt,
                    temperature=0.0,
                    max_tokens=50,
                ))

            # Route 3: Legacy OpenAI SDK shape — kept only as compatibility shim.
            elif hasattr(backend, "chat") and hasattr(backend.chat, "completions"):
                try:
                    from src.application.bootstrap.llm_gateway import get_llm_gateway
                    gateway = get_llm_gateway()
                    # C-5: cost-aware routing for keyword extraction / query
                    # rewriting. This is a CLASSIFICATION task with tiny output
                    # (max 50 tokens) → LIGHT tier; temperature=0 for
                    # deterministic keyword tagging.
                    outcome = gateway.execute_routed(
                        prompt,
                        task_kind="classification",
                        entity_count_hint=0,
                        temperature_override=0.0,
                        max_tokens_override=50,
                        stage="rag.query_rewrite",
                    )
                    raw = outcome.text
                except Exception:
                    resp = backend.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=50,
                        temperature=0.0,
                    )
                    raw = resp.choices[0].message.content.strip()

            else:
                # Unknown backend shape — do not crash, just skip expansion.
                print(f"[QueryRewriter] Unknown llm backend type={type(backend).__name__}; skip LLM expand")
                return []

            # Parse: only pure ASCII words/phrases (filter Chinese & punctuation)
            keywords = [w.strip() for w in raw.replace(",", " ").split() if w.strip() and w.strip().isascii()]
            return keywords[:8]
        except Exception as e:
            print(f"[QueryRewriter] LLM 扩展失败（跳过）: {e}")
            return []


# 全局单例（由 LLMAnswerer 初始化时注入 llm_client / llm_provider）
_default_rewriter: Optional[QueryRewriter] = None


def get_rewriter(
    ontology_dir: Optional[Path] = None,
    llm_client: Optional[_LLMProviderLike] = None,
    llm_provider: Optional[_LLMProviderLike] = None,
    model: str = "deepseek-chat",
) -> QueryRewriter:
    """获取全局 QueryRewriter 单例。首次调用时初始化（含 LLM 后端注入）。

    [H7 FIX] 新增 ``llm_provider`` 参数，优先使用公开的 provider 对象，
    不再强制传递 ``_client`` 私有属性。
    """
    global _default_rewriter
    effective_backend = llm_provider or llm_client
    needs_init = (
        _default_rewriter is None
        or (effective_backend is not None and _default_rewriter._llm_backend is None)
    )
    if needs_init:
        _default_rewriter = QueryRewriter(
            ontology_dir,
            llm_client=llm_client,
            llm_provider=llm_provider,
            model=model,
        )
    return _default_rewriter


# =============================================================================
# Query Classifier
# =============================================================================

class QueryClassifier:
    """查询分类器
    
    基于规则和模式识别进行查询分类。
    """
    
    def __init__(self) -> None:
        self._definition_patterns = [
            "什么是", "what is", "define", "定义", "解释", "explain",
            "含义", "meaning", "作用", "function", "功能",
        ]
        self._relation_patterns = [
            "关系", "relationship", "关联", "connect", "link",
            "依赖", "depend", "交互", "interact", "联系",
        ]
        self._process_patterns = [
            "流程", "process", "步骤", "step", "顺序", "order",
            "如何", "how", "怎么", "way", "方式",
        ]
        self._comparison_patterns = [
            "区别", "difference", "对比", "compare", "不同",
            "差异", "vs", "versus", "异同",
        ]
        self._cause_patterns = [
            "为什么", "why", "原因", "reason", "导致", "cause",
            "影响", "impact", "结果", "result",
        ]
        self._list_patterns = [
            "列出", "list", "所有", "all", "有哪些", "what are",
            "包含", "include", "有哪些", "包含哪些",
        ]
        self._validation_patterns = [
            "是否", "whether", "有没有", "have", "包含", "contain",
            "存在", "exist", "是否包含",
        ]
    
    def classify(self, query: str) -> QueryClassification:
        """分类查询
        
        [Phase 7-ML-1] Enhanced with ensemble scoring:
        - Pattern-based classification (existing)
        - Statistical feature scoring (new)
        - Confidence calibration (new)
        """
        query_lower = query.lower()
        
        # 识别意图
        intent = self._classify_intent(query_lower)
        intent_scores = self._score_intent_patterns(query_lower)
        
        # 识别类型
        query_type = self._classify_type(query_lower, intent)
        
        # 识别复杂度
        complexity = self._classify_complexity(query_lower)
        complexity_score = self._score_complexity_features(query_lower)
        
        # 提取实体、关系、属性
        entities, relations, attributes = self._extract_entities_relations_attributes(query)
        
        # 提取关键词
        keywords = self._extract_keywords(query_lower)
        
        # 生成建议
        suggestions = self._generate_suggestions(query_type, intent, complexity)
        
        # [Phase 7-ML-1] Calibrate confidence based on pattern match strength
        confidence = self._calibrate_confidence(intent_scores, complexity_score)
        
        return QueryClassification(
            query=query,
            query_type=query_type,
            intent=intent,
            complexity=complexity,
            entities=entities,
            relations=relations,
            attributes=attributes,
            keywords=keywords,
            confidence=confidence,
            suggestions=suggestions,
            created_at=datetime.now().isoformat(),
        )
    
    def _score_intent_patterns(self, query_lower: str) -> Dict[str, float]:
        """Score each intent based on pattern matches.
        
        [Phase 7-ML-1] Returns normalized scores for all intents
        to support confidence calibration.
        """
        scores: Dict[str, float] = {}
        
        patterns = {
            "definition": self._definition_patterns,
            "relationship": self._relation_patterns,
            "process": self._process_patterns,
            "comparison": self._comparison_patterns,
            "cause_effect": self._cause_patterns,
            "list": self._list_patterns,
            "validation": self._validation_patterns,
        }
        
        for intent_name, pattern_list in patterns.items():
            match_count = sum(1 for p in pattern_list if p in query_lower)
            scores[intent_name] = min(match_count / 2.0, 1.0)  # Normalize to [0, 1]
        
        return scores
    
    def _score_complexity_features(self, query_lower: str) -> float:
        """Score query complexity based on statistical features.
        
        [Phase 7-ML-1] Extracts features and returns complexity score [0, 1].
        """
        features = {
            "length": min(len(query_lower) / 100.0, 1.0),
            "entity_count": (query_lower.count("和") + query_lower.count("与") + 
                           query_lower.count("and")) / 3.0,
            "multi_hop": 1.0 if any(kw in query_lower for kw in ["通过", "via", "间接", "indirect"]) else 0.0,
            "aggregation": 1.0 if any(kw in query_lower for kw in ["统计", "汇总", "平均"]) else 0.0,
            "question_words": sum(1 for w in ["什么", "怎么", "为什么", "哪些"] if w in query_lower) / 4.0,
        }
        
        # Weighted sum (simple linear model)
        weights = {"length": 0.2, "entity_count": 0.3, "multi_hop": 0.2, 
                  "aggregation": 0.2, "question_words": 0.1}
        score = sum(features[k] * weights[k] for k in features)
        return min(score, 1.0)
    
    def _calibrate_confidence(self, intent_scores: Dict[str, float], 
                             complexity_score: float) -> float:
        """Calibrate classification confidence.
        
        [Phase 7-ML-1] Higher confidence when:
        - One intent clearly dominates
        - Complexity features are consistent
        """
        if not intent_scores:
            return 0.5
        
        max_score = max(intent_scores.values())
        second_max = sorted(intent_scores.values(), reverse=True)[1] if len(intent_scores) > 1 else 0
        
        # Confidence based on margin between top two intents
        intent_confidence = min(max_score * (1 + (max_score - second_max)), 1.0)
        
        # Blend with complexity confidence (higher for clear signals)
        complexity_confidence = 0.7 + 0.3 * (1 - abs(complexity_score - 0.5) * 2)
        
        return round((intent_confidence * 0.7 + complexity_confidence * 0.3), 2)
    
    def _classify_intent(self, query_lower: str) -> QueryIntent:
        """分类意图"""
        for pattern in self._definition_patterns:
            if pattern in query_lower:
                return QueryIntent.DEFINITION
        
        for pattern in self._relation_patterns:
            if pattern in query_lower:
                return QueryIntent.RELATIONSHIP
        
        for pattern in self._process_patterns:
            if pattern in query_lower:
                return QueryIntent.PROCESS
        
        for pattern in self._comparison_patterns:
            if pattern in query_lower:
                return QueryIntent.COMPARISON
        
        for pattern in self._cause_patterns:
            if pattern in query_lower:
                return QueryIntent.CAUSE_EFFECT
        
        for pattern in self._list_patterns:
            if pattern in query_lower:
                return QueryIntent.LIST
        
        for pattern in self._validation_patterns:
            if pattern in query_lower:
                return QueryIntent.VALIDATION
        
        return QueryIntent.DEFINITION  # 默认
    
    def _classify_type(self, query_lower: str, intent: QueryIntent) -> QueryType:
        """分类类型"""
        if intent == QueryIntent.RELATIONSHIP:
            return QueryType.RELATION_QUERY
        
        if intent == QueryIntent.DEFINITION:
            return QueryType.ENTITY_LOOKUP
        
        if intent == QueryIntent.PROCESS:
            return QueryType.SEMANTIC_SEARCH
        
        if intent == QueryIntent.COMPARISON:
            return QueryType.HYBRID_SEARCH
        
        if intent == QueryIntent.CAUSE_EFFECT:
            return QueryType.SEMANTIC_SEARCH
        
        if intent == QueryIntent.LIST:
            return QueryType.ENTITY_LOOKUP
        
        if intent == QueryIntent.VALIDATION:
            return QueryType.RELATION_QUERY
        
        return QueryType.SEMANTIC_SEARCH
    
    def _classify_complexity(self, query_lower: str) -> QueryComplexity:
        """分类复杂度"""
        # 检查是否包含多个实体
        entity_count = query_lower.count("和") + query_lower.count("与") + query_lower.count("and")
        
        # 检查是否包含多跳关系
        multi_hop = any(kw in query_lower for kw in ["通过", "via", "间接", "indirect"])
        
        # 检查是否包含聚合
        aggregation = any(kw in query_lower for kw in ["统计", "statistic", "汇总", "summary", "平均", "average"])
        
        if entity_count >= 2 or multi_hop or aggregation:
            return QueryComplexity.COMPLEX
        
        if entity_count == 1 or len(query_lower.split()) > 10:
            return QueryComplexity.MEDIUM
        
        return QueryComplexity.SIMPLE
    
    def _extract_entities_relations_attributes(self, query: str) -> Tuple[List[str], List[str], List[str]]:
        """提取实体、关系、属性"""
        # 简单实现：基于已知术语提取
        # 实际应该从本体中提取
        
        entities = []
        relations = []
        attributes = []
        
        # 已知实体
        known_entities = [
            "OHF", "ARO", "BOP", "CMP", "BIS", "ATACQ",
            "MPP", "PLUS", "HERCULES", "TOPAS", "CIP",
            "EBOM", "MBOM", "BOM", "C-note", "Change Note",
        ]
        
        # 已知关系
        known_relations = [
            "依赖", "关联", "包含", "调用", "同步",
            "depend", "relate", "include", "call", "sync",
        ]
        
        # 已知属性
        known_attributes = [
            "字段", "属性", "配置", "参数", "设置",
            "field", "attribute", "config", "parameter", "setting",
        ]
        
        for entity in known_entities:
            if entity in query:
                entities.append(entity)
        
        for relation in known_relations:
            if relation in query:
                relations.append(relation)
        
        for attribute in known_attributes:
            if attribute in query:
                attributes.append(attribute)
        
        return entities, relations, attributes
    
    def _extract_keywords(self, query_lower: str) -> List[str]:
        """提取关键词"""
        # 简单实现：过滤停用词
        stop_words = {
            "的", "是", "在", "和", "与", "或", "但", "如果", "那么",
            "the", "is", "in", "and", "or", "but", "if", "then",
            "什么", "怎么", "如何", "为什么", "where", "when", "how", "why",
        }
        
        words = query_lower.split()
        keywords = [w for w in words if w not in stop_words and len(w) > 1]
        
        return keywords[:10]  # 最多 10 个关键词
    
    def _generate_suggestions(
        self,
        query_type: QueryType,
        intent: QueryIntent,
        complexity: QueryComplexity,
    ) -> List[str]:
        """生成优化建议"""
        suggestions = []
        
        if complexity == QueryComplexity.COMPLEX:
            suggestions.append("建议拆分为多个简单查询")
            suggestions.append("考虑使用多跳检索")
        
        if query_type == QueryType.ENTITY_LOOKUP:
            suggestions.append("建议优先使用结构化检索")
        
        if query_type == QueryType.SEMANTIC_SEARCH:
            suggestions.append("建议使用向量检索")
        
        if intent == QueryIntent.RELATIONSHIP:
            suggestions.append("建议同时检索源实体和目标实体")
        
        if intent == QueryIntent.PROCESS:
            suggestions.append("建议检索相关流程文档")
        
        return suggestions


# =============================================================================
# Unified Query Processor
# =============================================================================

class UnifiedQueryProcessor:
    """统一查询处理器 - 整合解析、重写、分类
    
    提供一站式查询处理入口：
    1. 解析：提取 tokens、entities
    2. 重写：术语映射 + LLM 扩展
    3. 分类：识别类型、意图、复杂度
    4. 可选：绑定 packet 路径，便于上层查询直接消费管道产物
    """
    
    def __init__(
        self,
        ontology_dir: Optional[Path] = None,
        llm_client: Optional[_LLMProviderLike] = None,
        llm_provider: Optional[_LLMProviderLike] = None,
        model: str = "deepseek-chat",
        packet_path: Optional[Path] = None,
        packet: Optional[Dict[str, Any]] = None,
        rag_service: Optional[Any] = None,
    ):
        self.rewriter = QueryRewriter(
            ontology_dir,
            llm_client=llm_client,
            llm_provider=llm_provider,
            model=model,
        )
        self.classifier = QueryClassifier()
        self.packet_path = packet_path
        self.packet = packet
        self.rag_service = rag_service
    
    def process(self, query: str, base_topk: int = 5) -> Dict[str, Any]:
        """完整处理查询
        
        Args:
            query: 原始查询
            base_topk: 基础检索数量
            
        Returns:
            dict 包含解析、重写、分类结果
        """
        # 1. 解析
        parsed = parse(query, base_topk)
        
        # 2. 重写
        rewritten = self.rewriter.rewrite(query)
        
        # 3. 分类
        classification = self.classifier.classify(query)
        
        payload = {
            "parsed": parsed,
            "rewritten": rewritten,
            "classification": classification,
            "search_query": rewritten.rewritten,  # 用于检索的最终查询
            "recommended_topk": parsed.recommended_topk,
        }
        if self.packet_path is not None:
            payload["packet_path"] = str(self.packet_path)
        if self.packet is not None:
            payload["packet"] = self.packet
        if self.rag_service is not None:
            payload["rag_service"] = self.rag_service
        return payload

    @classmethod
    def from_packet(
        cls,
        packet_path: Optional[Path] = None,
        packet: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "UnifiedQueryProcessor":
        """Create a query processor bound to the canonical packet service."""
        from .core.unified_rag_service import UnifiedRAGService

        service = UnifiedRAGService.from_packet(packet_path=str(packet_path) if packet_path else None, packet=packet)
        return cls(packet_path=packet_path, packet=packet, rag_service=service, **kwargs)

    async def search(self, query: str, top_k: Optional[int] = None, **kwargs: Any) -> list[Any]:
        """Process and execute a query through the packet-first RAG service."""
        if self.rag_service is None:
            if self.packet_path is not None or self.packet is not None:
                self.rag_service = self.from_packet(
                    packet_path=self.packet_path,
                    packet=self.packet,
                    ontology_dir=None,
                ).rag_service
            else:
                raise RuntimeError("No RAG service or RetrievalAssetPacket configured")
        processed = self.process(query, base_topk=top_k or 5)
        effective_top_k = top_k or processed["recommended_topk"]
        return await self.rag_service.search(processed["search_query"], top_k=effective_top_k, **kwargs)
    
    def get_search_query(self, query: str) -> str:
        """获取用于检索的优化查询（简化接口）"""
        rewritten = self.rewriter.rewrite(query)
        return rewritten.rewritten


__all__ = [
    # Enums
    "QueryType",
    "QueryIntent",
    "QueryComplexity",
    # Data classes
    "ParsedQuery",
    "RewrittenQuery",
    "QueryClassification",
    # Functions
    "parse",
    "normalize_text",
    "tokenize_query",
    "extract_entities",
    "recommended_topk",
    "get_rewriter",
    # Classes
    "QueryRewriter",
    "QueryClassifier",
    "UnifiedQueryProcessor",
]