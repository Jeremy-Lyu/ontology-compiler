"""Cost-aware model routing policy (C-5).

Decides which provider + model a call should use based on prompt features and
task kind, so the gateway can pick a cheaper model for trivial work and a
stronger model for low-confidence / conflict-heavy work without callers
hard-coding model names.

Design principles (C-5 plan):

- Classification / JSON repair / short summary  -> LIGHT model tier
- Candidate generation / entity / relation extraction -> STANDARD tier
- Low-confidence review / conflict resolution / high-value synthesis -> HEAVY tier

The router never executes the call; it only returns a :class:`RouteDecision`
that the gateway (or any caller) can feed into ``execute``. Default model
table is cost-aware but overridable through :class:`ModelRoutingConfig`.

The router is deterministic, side-effect free, and unit-testable without
network or LLM access. It is intentionally *advisory*: callers may still pass
an explicit ``model=`` to :meth:`LLMGateway.execute` and bypass routing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TaskKind(str, Enum):
    """Coarse task labels callers can pass to bias routing.

    Using ``str`` mixin so values are JSON-serializable and usable as dict
    keys directly.
    """

    CLASSIFICATION = "classification"
    JSON_REPAIR = "json_repair"
    SHORT_SUMMARY = "short_summary"
    CANDIDATE_GENERATION = "candidate_generation"
    ENTITY_EXTRACTION = "entity_extraction"
    RELATION_EXTRACTION = "relation_extraction"
    LOW_CONFIDENCE_REVIEW = "low_confidence_review"
    CONFLICT_RESOLUTION = "conflict_resolution"
    RAG_ANSWER = "rag_answer"
    GENERIC = "generic"


class TaskComplexity(str, Enum):
    """Three-tier complexity bucket used to pick a model tier."""

    LIGHT = "light"
    STANDARD = "standard"
    HEAVY = "heavy"


@dataclass(frozen=True)
class ModelTier:
    """A concrete (provider, model) pair plus suggested call parameters.

    ``max_tokens`` / ``temperature`` are *suggestions*: callers may override
    them, but routing makes a cost-aware default so trivial tasks do not
    inherit a 16k-token completion allowance.
    """

    provider: str
    model: str
    max_tokens: int = 2048
    temperature: float = 0.0


@dataclass(frozen=True)
class RouteDecision:
    """Result of a routing call.

    Attributes:
        tier: complexity bucket that drove the decision.
        model_tier: concrete (provider, model) recommendation.
        task_kind: the task kind used (explicit or inferred).
        reason: short human-readable explanation for audit / trace.
        signals: feature snapshot used by the classifier (for telemetry).
    """

    tier: TaskComplexity
    model_tier: ModelTier
    task_kind: TaskKind
    reason: str
    signals: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelRoutingConfig:
    """Tier -> ModelTier table + classification thresholds.

    Defaults are cost-aware: ``deepseek-chat`` (cheap) for LIGHT/STANDARD,
    ``deepseek-reasoner`` (stronger, pricier) for HEAVY. Override via
    :func:`get_model_routing_policy` injection in tests.
    """

    light: ModelTier = ModelTier(provider="deepseek", model="deepseek-chat", max_tokens=1024, temperature=0.0)
    standard: ModelTier = ModelTier(provider="deepseek", model="deepseek-chat", max_tokens=4096, temperature=0.2)
    heavy: ModelTier = ModelTier(provider="deepseek", model="deepseek-reasoner", max_tokens=8192, temperature=0.3)

    # Complexity classification thresholds. Tuned conservatively so most
    # extraction / RAG calls fall in STANDARD, and only genuinely long or
    # conflict-marked prompts escalate to HEAVY.
    light_max_prompt_chars: int = 1500
    standard_max_prompt_chars: int = 8000
    light_max_entity_hint: int = 8
    standard_max_entity_hint: int = 40
    low_confidence_threshold: float = 0.55  # below this => HEAVY


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

_JSON_REQUIREMENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bjson\s+schema\b", re.IGNORECASE),
    re.compile(r"\brespond\s+in\s+json\b", re.IGNORECASE),
    re.compile(r"\breturn\s+only\s+(?:valid\s+)?json\b", re.IGNORECASE),
    re.compile(r"```json", re.IGNORECASE),
    re.compile(r"\boutput_format\s*=\s*[\"']json[\"']", re.IGNORECASE),
)

_CONFLICT_MARKERS: tuple[str, ...] = (
    "conflict",
    "contradict",
    "disagree",
    "inconsistent",
    "mutually exclusive",
    "two candidates",
    "ambiguous",
)

_LOW_CONFIDENCE_MARKERS: tuple[str, ...] = (
    "low confidence",
    "low-confidence",
    "uncertain",
    "ambiguous",
    "unresolved",
)


def _approx_token_count(prompt: str) -> int:
    """Rough token estimate (4 chars / token). Good enough for tiering."""
    return max(1, len(prompt or "") // 4)


def _requires_json(prompt: str) -> bool:
    return any(p.search(prompt) for p in _JSON_REQUIREMENT_PATTERNS)


def _has_conflict_marker(prompt: str) -> bool:
    lowered = (prompt or "").lower()
    return any(marker in lowered for marker in _CONFLICT_MARKERS)


def _has_low_confidence_marker(prompt: str) -> bool:
    lowered = (prompt or "").lower()
    return any(marker in lowered for marker in _LOW_CONFIDENCE_MARKERS)


def extract_routing_signals(
    prompt: str,
    *,
    task_kind: Optional[TaskKind] = None,
    entity_count_hint: int = 0,
    confidence_hint: Optional[float] = None,
    has_conflict: Optional[bool] = None,
) -> dict[str, Any]:
    """Build the feature dict the classifier consumes.

    Callers may pass any combination of hints; missing ones are inferred from
    the prompt text so simple ``gateway.execute(prompt)`` calls still get a
    sensible routing without needing to label every call site.
    """
    text = prompt or ""
    inferred_conflict = has_conflict if has_conflict is not None else _has_conflict_marker(text)
    inferred_low_conf = (
        confidence_hint is not None and confidence_hint < 0.55
    ) or _has_low_confidence_marker(text)
    return {
        "prompt_chars": len(text),
        "approx_tokens": _approx_token_count(text),
        "requires_json": _requires_json(text),
        "entity_count_hint": max(0, int(entity_count_hint or 0)),
        "confidence_hint": float(confidence_hint) if confidence_hint is not None else None,
        "has_conflict": bool(inferred_conflict),
        "low_confidence_inferred": bool(inferred_low_conf),
        "task_kind": (task_kind or TaskKind.GENERIC).value,
    }


# ---------------------------------------------------------------------------
# Task kind inference (only used when caller does not pass task_kind)
# ---------------------------------------------------------------------------


def infer_task_kind(prompt: str) -> TaskKind:
    """Best-effort task kind inference from prompt content.

    Conservative: defaults to :attr:`TaskKind.GENERIC` rather than guessing
    high-stakes categories. Used only when the caller does not pass an
    explicit ``task_kind``.
    """
    lowered = (prompt or "").lower()
    if _requires_json(lowered) and ("repair" in lowered or "fix" in lowered or "valid" in lowered):
        return TaskKind.JSON_REPAIR
    if "classify" in lowered or "classification" in lowered or "scene" in lowered and "type" in lowered:
        return TaskKind.CLASSIFICATION
    if "summarize" in lowered or "summary" in lowered or "compress" in lowered:
        if len(lowered) < 1500:
            return TaskKind.SHORT_SUMMARY
    if "entity" in lowered and ("extract" in lowered or "candidate" in lowered):
        return TaskKind.ENTITY_EXTRACTION
    if "relation" in lowered and ("extract" in lowered or "candidate" in lowered):
        return TaskKind.RELATION_EXTRACTION
    if "conflict" in lowered or "contradict" in lowered:
        return TaskKind.CONFLICT_RESOLUTION
    if "answer" in lowered and ("question" in lowered or "query" in lowered):
        return TaskKind.RAG_ANSWER
    return TaskKind.GENERIC


# ---------------------------------------------------------------------------
# Routing policy
# ---------------------------------------------------------------------------

# Task-kind -> default complexity when signals are ambiguous.
_TASK_KIND_DEFAULT_COMPLEXITY: dict[TaskKind, TaskComplexity] = {
    TaskKind.CLASSIFICATION: TaskComplexity.LIGHT,
    TaskKind.JSON_REPAIR: TaskComplexity.LIGHT,
    TaskKind.SHORT_SUMMARY: TaskComplexity.LIGHT,
    TaskKind.CANDIDATE_GENERATION: TaskComplexity.STANDARD,
    TaskKind.ENTITY_EXTRACTION: TaskComplexity.STANDARD,
    TaskKind.RELATION_EXTRACTION: TaskComplexity.STANDARD,
    TaskKind.RAG_ANSWER: TaskComplexity.STANDARD,
    TaskKind.GENERIC: TaskComplexity.STANDARD,
    TaskKind.LOW_CONFIDENCE_REVIEW: TaskComplexity.HEAVY,
    TaskKind.CONFLICT_RESOLUTION: TaskComplexity.HEAVY,
}


class ModelRoutingPolicy:
    """Cost-aware model routing policy.

    Stateless aside from the :class:`ModelRoutingConfig` it holds; safe to
    share process-wide. Use :func:`get_model_routing_policy` for the
    process-wide singleton.
    """

    def __init__(self, config: Optional[ModelRoutingConfig] = None) -> None:
        self._config = config or ModelRoutingConfig()

    @property
    def config(self) -> ModelRoutingConfig:
        return self._config

    def classify_complexity(
        self,
        prompt: str,
        *,
        task_kind: Optional[TaskKind] = None,
        entity_count_hint: int = 0,
        confidence_hint: Optional[float] = None,
        has_conflict: Optional[bool] = None,
    ) -> tuple[TaskComplexity, dict[str, Any]]:
        """Classify a call into LIGHT / STANDARD / HEAVY.

        Returns the complexity bucket and the signals used so the caller can
        attach them to trace / telemetry.
        """
        signals = extract_routing_signals(
            prompt,
            task_kind=task_kind,
            entity_count_hint=entity_count_hint,
            confidence_hint=confidence_hint,
            has_conflict=has_conflict,
        )

        kind = task_kind or infer_task_kind(prompt)
        signals["task_kind"] = kind.value

        # Hard escalators: conflict or explicit low-confidence always HEAVY.
        if signals["has_conflict"] or signals["low_confidence_inferred"]:
            return TaskComplexity.HEAVY, signals

        # Task-kind driven baseline.
        base = _TASK_KIND_DEFAULT_COMPLEXITY.get(kind, TaskComplexity.STANDARD)

        # Down-grade prompt-size driven escalation when task kind is light.
        prompt_chars = signals["prompt_chars"]
        entity_hint = signals["entity_count_hint"]

        if base == TaskComplexity.LIGHT:
            # Light tasks stay light unless prompt is unusually long.
            if prompt_chars > self._config.standard_max_prompt_chars:
                return TaskComplexity.STANDARD, signals
            return TaskComplexity.LIGHT, signals

        # base is STANDARD or HEAVY.
        if base == TaskComplexity.HEAVY:
            return TaskComplexity.HEAVY, signals

        # base STANDARD: stay light if everything is small.
        if (
            prompt_chars <= self._config.light_max_prompt_chars
            and entity_hint <= self._config.light_max_entity_hint
            and not signals["requires_json"]
        ):
            return TaskComplexity.LIGHT, signals

        # Escalate to HEAVY when prompt is very long or entity hint very high.
        if (
            prompt_chars > self._config.standard_max_prompt_chars
            or entity_hint > self._config.standard_max_entity_hint
        ):
            return TaskComplexity.HEAVY, signals

        return TaskComplexity.STANDARD, signals

    def resolve_model_tier(self, complexity: TaskComplexity) -> ModelTier:
        if complexity == TaskComplexity.LIGHT:
            return self._config.light
        if complexity == TaskComplexity.HEAVY:
            return self._config.heavy
        return self._config.standard

    def route(
        self,
        prompt: str,
        *,
        task_kind: Optional[TaskKind] = None,
        entity_count_hint: int = 0,
        confidence_hint: Optional[float] = None,
        has_conflict: Optional[bool] = None,
    ) -> RouteDecision:
        """Produce a :class:`RouteDecision` for a single call.

        The decision is purely advisory; the gateway applies it only when the
        caller does not pass an explicit ``model=``.
        """
        complexity, signals = self.classify_complexity(
            prompt,
            task_kind=task_kind,
            entity_count_hint=entity_count_hint,
            confidence_hint=confidence_hint,
            has_conflict=has_conflict,
        )
        tier = self.resolve_model_tier(complexity)
        kind = TaskKind(signals["task_kind"])
        reasons: list[str] = [f"task={kind.value}", f"complexity={complexity.value}"]
        if signals["has_conflict"]:
            reasons.append("conflict-detected")
        if signals["low_confidence_inferred"]:
            reasons.append("low-confidence")
        if signals["requires_json"]:
            reasons.append("requires-json")
        reasons.append(f"prompt_chars={signals['prompt_chars']}")
        return RouteDecision(
            tier=complexity,
            model_tier=tier,
            task_kind=kind,
            reason=";".join(reasons),
            signals=signals,
        )


# ---------------------------------------------------------------------------
# Process-wide singleton (overridable for tests)
# ---------------------------------------------------------------------------

_policy: Optional[ModelRoutingPolicy] = None


def get_model_routing_policy() -> ModelRoutingPolicy:
    global _policy
    if _policy is None:
        _policy = ModelRoutingPolicy()
    return _policy


def set_model_routing_policy(policy: Optional[ModelRoutingPolicy]) -> None:
    """Inject a policy (test-only). Pass ``None`` to reset to default."""
    global _policy
    _policy = policy
