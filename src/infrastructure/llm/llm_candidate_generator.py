"""LLM Candidate Generator — LLM候选生成器模式

根据 PIPELINE_REFACTOR_EXECUTION_CHECKLIST.md 和 SKILLS_STANDARDIZATION_AND_PIPELINE_RAG_ALIGNMENT.md 的要求：
- LLM 只做语义理解和候选生成，不做最终判定
- 将大任务拆成小任务，分步骤输出
- 建立 parse → schema → repair/retry → fallback 的链路
- 把规则外置到 config / schema / skill metadata

设计原则：
1. LLM 输出候选结果（candidates），而非最终结果（final result）
2. 每个候选结果包含置信度和证据
3. 由确定性系统（schema validator + rule engine）做最终筛选
4. 支持 JSON 错误自动修复和重试机制
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)

# C-1 (2026-08-18): candidate generation uses a configurable token policy
# instead of a hardcoded ceiling. Defaults remain environment-driven so the
# behavior stays stable until C-2 wires a richer prompt budget policy.
_DEFAULT_MAX_TOKENS = int(os.getenv("LLM_CANDIDATE_MAX_TOKENS", "16000"))
_MAX_MAX_TOKENS_CAP = int(os.getenv("LLM_CANDIDATE_MAX_TOKENS_CAP", "24000"))

# C-5: maps the internal `task` string passed through _generate_with_retry /
# _call_llm to the routing TaskKind so the gateway can pick a cost-aware tier.
# Imported lazily inside _call_llm to avoid a hard dependency at module import.
_TASK_KIND_MAP = {
    "scene_classification": "classification",
    "entity_candidates": "entity_extraction",
    "relation_candidates": "relation_extraction",
    "text_summary": "short_summary",
    "quality_report": "generic",
}


class LLMOutputType(Enum):
    ENTITY_CANDIDATES = "entity_candidates"
    RELATION_CANDIDATES = "relation_candidates"
    SCENE_CLASSIFICATION = "scene_classification"
    QUALITY_REPORT = "quality_report"
    TEXT_SUMMARY = "text_summary"


class GenerationStatus(Enum):
    SUCCESS = "success"
    PARSE_ERROR = "parse_error"
    SCHEMA_ERROR = "schema_error"
    RETRY = "retry"
    FALLBACK = "fallback"


@dataclass
class LLMCandidate:
    """LLM 候选结果"""
    type: str
    name: str
    confidence: float = 0.0
    fields: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    source: str = ""


@dataclass
class LLMGenerationResult:
    """LLM 生成结果"""
    status: GenerationStatus
    candidates: List[LLMCandidate] = field(default_factory=list)
    raw_output: str = ""
    error_message: str = ""
    retry_count: int = 0
    confidence_threshold: float = 0.0


class LLMCandidateGenerator:
    """LLM 候选生成器
    
    将 LLM 从"大而全输出"改成"候选生成器"模式：
    1. 场景分类 → 2. 实体候选 → 3. 关系候选 → 4. schema校验 → 5. 规则过滤
    """
    
    def __init__(
        self,
        llm_client=None,
        max_retries: int = 3,
        default_max_tokens: Optional[int] = None,
        max_tokens_cap: Optional[int] = None,
        budget_policy=None,
        gateway=None,
    ):
        self._llm_client = llm_client
        self._max_retries = max_retries
        self._default_max_tokens = self._coerce_max_tokens(
            default_max_tokens if default_max_tokens is not None else _DEFAULT_MAX_TOKENS
        )
        self._max_tokens_cap = self._coerce_max_tokens(
            max_tokens_cap if max_tokens_cap is not None else _MAX_MAX_TOKENS_CAP
        )
        # C-2: optional prompt budget policy for task-aware token allocation.
        self._budget_policy = budget_policy
        # C-5: optional LLM gateway for cost-aware model routing. When set,
        # _call_llm consults gateway.route() to pick max_tokens/temperature
        # per task kind, and attaches the RouteDecision to the result for
        # telemetry. Execution still flows through the existing async client
        # so sync/async semantics are unchanged.
        self._gateway = gateway
        
    async def generate_scene_classification(self, content: str, domain_hint: str = "", max_tokens: Optional[int] = None, document_type: Optional[str] = None) -> LLMGenerationResult:
        """步骤1：场景分类"""
        prompt = f"""请分析以下文档，识别其业务场景类型。

领域提示：{domain_hint}

文档内容：
{content[:4000]}

请返回JSON格式：
{{"scene_type": "场景类型", "confidence": 0.0-1.0, "keywords": ["关键词1", "关键词2"], "description": "场景描述"}}
"""
        return await self._generate_with_retry(
            prompt, LLMOutputType.SCENE_CLASSIFICATION,
            max_tokens=max_tokens, task="scene_classification", document_type=document_type,
        )

    async def generate_entity_candidates(self, content: str, entity_types: List[str], domain_hint: str = "", max_tokens: Optional[int] = None, document_type: Optional[str] = None) -> LLMGenerationResult:
        """步骤2：实体候选生成
        
        注意：这只是候选结果，最终结果由 schema validator 和 rule engine 决定。
        """
        entity_types_str = ", ".join(entity_types)

        prompt = f"""请从以下文档中提取实体候选。

领域提示：{domain_hint}

目标实体类型：{entity_types_str}

文档内容：
{content[:6000]}

请返回JSON格式，每个实体包含置信度：
{{
  "entities": [
    {{
      "type": "实体类型",
      "name": "实体名称",
      "confidence": 0.0-1.0,
      "fields": {{"字段名": "值"}},
      "evidence": ["证据片段"]
    }}
  ]
}}

注意：
1. 只返回候选，不需要做最终判定
2. 置信度反映你对识别准确性的把握
3. 证据片段要能在原文中找到对应内容
4. 最多返回 25 个实体候选，按重要性排序
"""
        return await self._generate_with_retry(
            prompt, LLMOutputType.ENTITY_CANDIDATES,
            max_tokens=max_tokens, task="entity_candidates", document_type=document_type,
        )

    async def generate_relation_candidates(self, content: str, entities: List[Dict[str, Any]], domain_hint: str = "", max_tokens: Optional[int] = None, document_type: Optional[str] = None) -> LLMGenerationResult:
        """步骤3：关系候选生成"""
        entities_str = json.dumps([{"type": e.get("type"), "name": e.get("name")} for e in entities], ensure_ascii=False, indent=2)

        prompt = f"""请分析以下实体之间的关系。

领域提示：{domain_hint}

实体列表：
{entities_str}

文档内容：
{content[:4000]}

请返回JSON格式：
{{
  "relationships": [
    {{
      "source": "源实体名称",
      "target": "目标实体名称",
      "type": "关系类型",
      "confidence": 0.0-1.0,
      "evidence": ["证据片段"]
    }}
  ]
}}

注意：
1. 只返回候选关系，不需要做最终判定
2. 最多返回 40 条关系候选，按重要性排序
3. 优先抽取 has_part / depends_on / uses / implements / derived_from 等具体语义类型
4. BOM 派生语义必须使用 derived_from：当文档说明 EBOM（工程BOM）或 MBOM（制造BOM）由通用 BOM 派生、生成或转化时，输出 source=EBOM/MBOM, target=BOM, type=derived_from，不要降级为 related_to
5. 库存结构语义优先使用 has_part 或 part_of：当文档说明 Cycle Stock、Safety Stock 是 Inventory 的组成部分时，输出 source=Cycle Stock/Safety Stock, target=Inventory, type=has_part
6. 如果上下文中同时出现 EBOM/MBOM/BOM 的对比、生成、转换、来源、派生、结构定义，请优先产生 derived_from、has_part、part_of，而不是 related_to
7. 如果上下文中同时出现 Inventory、Cycle Stock、Safety Stock、Production、Benchmark 相关描述，请优先保留独立实体名称，不要把复合短语无条件拆成多个不相干实体
"""
        return await self._generate_with_retry(
            prompt, LLMOutputType.RELATION_CANDIDATES,
            max_tokens=max_tokens, task="relation_candidates", document_type=document_type,
        )
    
    async def _generate_with_retry(self, prompt: str, output_type: LLMOutputType, max_tokens: Optional[int] = None, *, task: Optional[str] = None, document_type: Optional[str] = None) -> LLMGenerationResult:
        """带重试机制的生成，包含截断检测和自动升级 max_tokens"""
        max_tokens = self._resolve_budget(task=task, document_type=document_type, max_tokens=max_tokens)
        truncation_retry_max = self._resolve_truncation_retry_max(task=task)

        for attempt in range(self._max_retries + 1):
            try:
                raw_output = await self._call_llm(prompt, max_tokens=max_tokens, task=task)

                # Detect truncation before parsing
                if self._detect_truncation(raw_output):
                    logger.warning(f"LLM output truncation detected (attempt {attempt+1}), "
                                  f"output_len={len(raw_output)}, max_tokens={max_tokens}")
                    if attempt < truncation_retry_max and max_tokens < self._max_tokens_cap:
                        # Retry with higher max_tokens, but never beyond the
                        # configured escalation cap so spend stays bounded.
                        max_tokens = min(int(max_tokens * 1.5), self._max_tokens_cap)
                        logger.info(f"Retrying with increased max_tokens={max_tokens}")
                        continue
                    # If retries exhausted or cap reached, attempt repair
                    logger.warning("Truncation retries exhausted, attempting JSON repair")

                parsed = self._parse_json(raw_output)

                if not parsed:
                    logger.warning(f"LLM output parse failed (attempt {attempt+1})")
                    if attempt < self._max_retries:
                        continue
                    return LLMGenerationResult(
                        status=GenerationStatus.PARSE_ERROR,
                        raw_output=raw_output,
                        error_message="JSON parse failed after all retries and repair attempts",
                        retry_count=attempt,
                    )
                
                candidates = self._convert_to_candidates(parsed, output_type)

                return LLMGenerationResult(
                    status=GenerationStatus.SUCCESS,
                    candidates=candidates,
                    raw_output=raw_output,
                    retry_count=attempt,
                )

            except Exception as e:
                logger.warning(f"LLM generation failed (attempt {attempt+1}): {e}")
                if attempt < self._max_retries:
                    continue
                return LLMGenerationResult(
                    status=GenerationStatus.FALLBACK,
                    raw_output="",
                    error_message=str(e),
                    retry_count=attempt,
                )
    
    def _coerce_max_tokens(self, value: Optional[int]) -> int:
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 1

    def _resolve_max_tokens(self, override: Optional[int]) -> int:
        base = self._default_max_tokens if override is None else self._coerce_max_tokens(override)
        return min(base, self._max_tokens_cap)

    def _resolve_budget(
        self,
        *,
        task: Optional[str] = None,
        document_type: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> int:
        """C-2: if a budget policy is attached, use its profile-aware cap."""
        if self._budget_policy is not None:
            budget = self._budget_policy.resolve(
                task=task,
                document_type=document_type,
                completion_tokens=max_tokens,
            )
            return self._resolve_max_tokens(budget.max_completion_tokens)
        return self._resolve_max_tokens(max_tokens)

    def _resolve_truncation_retry_max(self, *, task: Optional[str] = None) -> int:
        """C-2: pull truncation retry max from budget policy when available."""
        if self._budget_policy is not None and task is not None:
            budget = self._budget_policy.resolve(task=task)
            return max(0, int(budget.truncation_retry_max))
        return 2

    def _resolve_temperature(self, *, task: Optional[str] = None, default: float = 0.2) -> float:
        """C-2: pull temperature from budget policy when available."""
        if self._budget_policy is not None and task is not None:
            budget = self._budget_policy.resolve(task=task)
            return float(budget.temperature)
        return default

    async def _call_llm(self, prompt: str, max_tokens: Optional[int] = None, *, task: Optional[str] = None) -> str:
        """调用 LLM

        C-5: when a gateway is attached, consult gateway.route() to derive
        cost-aware max_tokens / temperature for this task. The routed values
        override the budget-policy defaults but are still capped by
        ``_max_tokens_cap``. Execution flows through the existing async
        ``self._llm_client`` so monitored-client semantics are preserved.
        """
        if not self._llm_client:
            return ""

        routed_max_tokens: Optional[int] = None
        routed_temperature: Optional[float] = None
        if self._gateway is not None:
            try:
                from src.application.services.model_router import TaskKind

                task_kind = _TASK_KIND_MAP.get(task or "", TaskKind.GENERIC)
                decision = self._gateway.route(prompt, task_kind=task_kind)
                routed_max_tokens = decision.model_tier.max_tokens
                routed_temperature = decision.model_tier.temperature
                logger.debug(
                    "C-5 routed task=%s -> tier=%s model=%s max_tokens=%s",
                    task, decision.tier.value, decision.model_tier.model, routed_max_tokens,
                )
            except Exception as exc:  # pragma: no cover - routing is advisory
                logger.debug("C-5 routing unavailable for task=%s: %s", task, exc)

        # Resolution order: explicit caller override > routed > budget policy > default.
        effective_max = max_tokens if max_tokens is not None else routed_max_tokens
        resolved_max = self._resolve_max_tokens(effective_max)
        if routed_temperature is not None:
            temp = float(routed_temperature)
        else:
            temp = self._resolve_temperature(task=task)

        resp = await self._llm_client.call(
            prompt,
            max_tokens=resolved_max,
            temperature=temp,
        )
        return resp
    
    def _parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """解析 JSON，支持多种格式和自动修复链路

        修复链路：直接解析 → 去代码块 → 正则提取 → 截断修复 → 放弃
        """
        if not text or not text.strip():
            return None

        text = text.strip()

        # Step 1: Strip markdown code fences
        if text.startswith("```"):
            text = re.sub(r'^```(?:json)?\s*\n?', '', text)
            text = re.sub(r'\s*```\s*$', '', text)

        # Step 2: Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Step 3: Try regex extraction of first JSON object
        try:
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                return json.loads(match.group())
        except json.JSONDecodeError:
            pass

        # Step 4: Try truncation repair (close incomplete JSON)
        repaired = self._repair_truncated_json(text)
        if repaired is not None:
            logger.warning("JSON repaired from truncated output")
            return repaired

        return None

    @staticmethod
    def _repair_truncated_json(text: str) -> Optional[Dict[str, Any]]:
        """Attempt to repair truncated JSON by closing open structures.

        Handles:
        - Unterminated strings (add closing quote)
        - Open arrays (add closing bracket)
        - Open objects (add closing brace)
        - Trailing commas before closing
        """
        if not text or not text.strip():
            return None

        s = text.strip()

        # Remove any trailing partial key-value (e.g. "description": "some tex)
        # Find last complete entry ending with , or } or ]
        last_complete = max(s.rfind('},'), s.rfind('],'), s.rfind('"}'), s.rfind('"}'))
        if last_complete > 0:
            # Cut at last complete entry and close
            truncated = s[:last_complete + 1]
            # Count open vs close braces and brackets
            open_braces = truncated.count('{') - truncated.count('}')
            open_brackets = truncated.count('[') - truncated.count(']')

            # Remove trailing comma if present
            if truncated.rstrip().endswith(','):
                truncated = truncated.rstrip()[:-1]

            # Close open structures
            truncated += ']' * open_brackets + '}' * open_braces

            try:
                return json.loads(truncated)
            except json.JSONDecodeError:
                pass

        # Fallback: try to close with minimal repair
        open_braces = s.count('{') - s.count('}')
        open_brackets = s.count('[') - s.count(']')

        if open_braces > 0 or open_brackets > 0:
            # Remove trailing comma
            if s.rstrip().endswith(','):
                s = s.rstrip()[:-1]
            # Remove trailing incomplete string (unmatched quote)
            if s.count('"') % 2 != 0:
                s += '"'
            s += ']' * max(open_brackets, 0) + '}' * max(open_braces, 0)
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                pass

        return None

    def _detect_truncation(self, text: str) -> bool:
        """Detect if LLM output was likely truncated.

        Signals:
        - Brace/bracket imbalance
        - Missing closing quotes
        - Output ends mid-word (no newline, no punctuation)
        - Very short output for a complex task
        """
        if not text or not text.strip():
            return False

        s = text.strip()

        # Check brace balance
        if s.count('{') != s.count('}'):
            return True
        if s.count('[') != s.count(']'):
            return True

        # Check quote balance (simple heuristic)
        if s.count('"') % 2 != 0:
            return True

        return False
    
    def _convert_to_candidates(self, parsed: Dict[str, Any], output_type: LLMOutputType) -> List[LLMCandidate]:
        """将解析结果转换为候选列表"""
        candidates = []
        
        if output_type == LLMOutputType.ENTITY_CANDIDATES:
            for entity in parsed.get("entities", []):
                candidates.append(LLMCandidate(
                    type=entity.get("type", ""),
                    name=entity.get("name", ""),
                    confidence=entity.get("confidence", 0.0),
                    fields=entity.get("fields", {}),
                    evidence=entity.get("evidence", []),
                ))
        
        elif output_type == LLMOutputType.RELATION_CANDIDATES:
            for rel in parsed.get("relationships", []):
                candidates.append(LLMCandidate(
                    type=rel.get("type", ""),
                    name=f"{rel.get('source')} -> {rel.get('target')}",
                    confidence=rel.get("confidence", 0.0),
                    fields={
                        "source": rel.get("source"),
                        "target": rel.get("target"),
                    },
                    evidence=rel.get("evidence", []),
                ))
        
        elif output_type == LLMOutputType.SCENE_CLASSIFICATION:
            candidates.append(LLMCandidate(
                type="scene",
                name=parsed.get("scene_type", ""),
                confidence=parsed.get("confidence", 0.0),
                fields={
                    "keywords": parsed.get("keywords", []),
                    "description": parsed.get("description", ""),
                },
            ))
        
        return candidates


class CandidateFilter:
    """候选筛选器 - 确定性系统做最终判定"""
    
    def __init__(self, schema_validator=None, rule_engine=None):
        self._schema_validator = schema_validator
        self._rule_engine = rule_engine
    
    def filter(self, candidates: List[LLMCandidate], min_confidence: float = 0.5) -> List[LLMCandidate]:
        """筛选候选结果"""
        filtered = []
        
        for candidate in candidates:
            if candidate.confidence < min_confidence:
                logger.debug(f"Skipping candidate {candidate.name} due to low confidence: {candidate.confidence}")
                continue
            
            if self._schema_validator and not self._validate_schema(candidate):
                logger.debug(f"Skipping candidate {candidate.name} due to schema validation failure")
                continue
            
            if self._rule_engine and not self._validate_rules(candidate):
                logger.debug(f"Skipping candidate {candidate.name} due to rule validation failure")
                continue
            
            filtered.append(candidate)
        
        return filtered
    
    def _validate_schema(self, candidate: LLMCandidate) -> bool:
        """验证 schema"""
        return True
    
    def _validate_rules(self, candidate: LLMCandidate) -> bool:
        """验证规则"""
        return True
