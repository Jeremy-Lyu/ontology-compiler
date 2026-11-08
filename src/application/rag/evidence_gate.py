"""RAG evidence gate (C-5).

Decides whether a RAG answer path should:

- ``ANSWER``: evidence is sufficient, proceed to synthesize the answer.
- ``RETRIEVE_MORE``: evidence is thin on coverage / count, fetch more chunks.
- ``ESCALATE``: evidence is contradictory / extremely low-confidence; escalate
  to a stronger model or human review rather than answer directly.

Signals evaluated (per C-5 plan):

- ``coverage``: fraction of the query's key terms covered by retrieved chunks.
- ``diversity``: number of distinct source documents / sections represented.
- ``score_margin``: gap between top-1 and top-2 retrieval scores; a large
  margin means a single dominant answer, a tiny margin signals ambiguity.
- ``evidence_count``: raw number of retrieved items.
- ``top_score``: absolute score of the best evidence.
- ``has_conflict``: explicit conflict marker (caller-supplied).

The gate is deterministic and side-effect free; it produces an
:class:`EvidenceGateDecision` that the RAG answer path consumes. It is
intentionally conservative: when signals are missing or zero, it returns
``RETRIEVE_MORE`` rather than answering on weak evidence.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, Iterable, Optional

logger = logging.getLogger(__name__)


class EvidenceGateAction(str, Enum):
    """What the RAG answer path should do next."""

    ANSWER = "answer"
    RETRIEVE_MORE = "retrieve_more"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class EvidenceItem:
    """A single retrieved evidence item used by the gate.

    ``source_id`` should identify the document / section so diversity can be
    measured; ``score`` is the retrieval score in [0, 1]; ``text`` is the
    raw chunk text used for coverage scoring.
    """

    text: str
    score: float = 0.0
    source_id: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceSignals:
    """Computed signals from a batch of evidence items.

    All numeric fields are normalized to [0, 1] so thresholds are stable
    across retrieval backends with different raw score distributions.
    """

    evidence_count: int
    top_score: float
    score_margin: float
    coverage: float
    diversity: float
    has_conflict: bool


@dataclass(frozen=True)
class EvidenceGateConfig:
    """Thresholds for the evidence gate.

    Defaults are intentionally conservative: prefer ``RETRIEVE_MORE`` over
    answering on weak evidence, and escalate on conflict or extreme margin.
    """

    min_evidence_count: int = 3
    min_top_score: float = 0.45
    min_score_margin: float = 0.10  # below this => ambiguous => retrieve more
    conflict_margin_threshold: float = 0.03  # margin this tight + conflict => escalate
    min_coverage: float = 0.60
    min_diversity: float = 0.20


@dataclass(frozen=True)
class EvidenceGateDecision:
    """Outcome of an evidence gate evaluation.

    Attributes:
        action: what to do next.
        signals: the computed signals (for trace / telemetry).
        reason: short human-readable explanation.
    """

    action: EvidenceGateAction
    signals: EvidenceSignals
    reason: str


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------


def _normalize_score(score: float) -> float:
    """Clamp a raw retrieval score to [0, 1]."""
    if score is None:
        return 0.0
    try:
        s = float(score)
    except (TypeError, ValueError):
        return 0.0
    if s < 0:
        return 0.0
    if s > 1:
        return 1.0
    return s


def _compute_coverage(query_terms: Iterable[str], evidence: list[EvidenceItem]) -> float:
    """Fraction of query terms appearing in at least one evidence item."""
    terms = [t.lower().strip() for t in query_terms if t and t.strip()]
    if not terms:
        # No query terms supplied: coverage unknown => treat as 1.0 so the
        # gate does not artificially demand more retrieval.
        return 1.0
    corpus = " ".join((e.text or "").lower() for e in evidence)
    if not corpus:
        return 0.0
    covered = sum(1 for t in terms if t in corpus)
    return covered / len(terms)


def _compute_diversity(evidence: list[EvidenceItem]) -> float:
    """Diversity = distinct source_ids / total evidence count.

    Returns 0 when no source_ids are present (degrades gracefully; the gate
    will then rely on count / score / coverage instead).
    """
    if not evidence:
        return 0.0
    sources = {e.source_id for e in evidence if e.source_id}
    if not sources:
        return 0.0
    return min(1.0, len(sources) / len(evidence))


def _score_margin(evidence: list[EvidenceItem]) -> tuple[float, float]:
    """Return ``(top_score, margin)`` from the sorted evidence list."""
    if not evidence:
        return 0.0, 0.0
    scores = sorted((_normalize_score(e.score) for e in evidence), reverse=True)
    top = scores[0]
    runner_up = scores[1] if len(scores) > 1 else 0.0
    return top, top - runner_up


def compute_signals(
    evidence: Iterable[EvidenceItem],
    *,
    query_terms: Iterable[str] = (),
    has_conflict: bool = False,
) -> EvidenceSignals:
    """Build :class:`EvidenceSignals` from raw evidence."""
    items = list(evidence)
    top, margin = _score_margin(items)
    return EvidenceSignals(
        evidence_count=len(items),
        top_score=top,
        score_margin=margin,
        coverage=_compute_coverage(query_terms, items),
        diversity=_compute_diversity(items),
        has_conflict=bool(has_conflict),
    )


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


class EvidenceGate:
    """Deterministic RAG evidence gate.

    Stateless aside from its :class:`EvidenceGateConfig`; safe to share
    process-wide via :func:`get_evidence_gate`.
    """

    def __init__(self, config: Optional[EvidenceGateConfig] = None) -> None:
        self._config = config or EvidenceGateConfig()
        # P1-3 (2026-09-01): decision telemetry so the regression runner can
        # record ``evidence_signals`` in the manifest / baseline. The gate
        # itself stays side-effect free w.r.t. routing decisions; these
        # counters only observe what ``evaluate`` produced.
        self._decision_counts: Dict[str, int] = {}
        self._last_decision: Optional[Dict[str, object]] = None

    @property
    def config(self) -> EvidenceGateConfig:
        return self._config

    def evaluate(
        self,
        evidence: Iterable[EvidenceItem],
        *,
        query_terms: Iterable[str] = (),
        has_conflict: bool = False,
        retrieval_round: int = 1,
        max_retrieval_rounds: int = 3,
    ) -> EvidenceGateDecision:
        """Decide the next action for the RAG answer path.

        Args:
            evidence: retrieved evidence items.
            query_terms: key terms extracted from the query (for coverage).
            has_conflict: explicit conflict flag (e.g. from contradiction
                detector or entity-resolution disagreement).
            retrieval_round: current retrieval iteration (1-indexed).
            max_retrieval_rounds: cap after which the gate escalates instead
                of requesting more retrieval.
        """
        decision = self._decide(
            evidence,
            query_terms=query_terms,
            has_conflict=has_conflict,
            retrieval_round=retrieval_round,
            max_retrieval_rounds=max_retrieval_rounds,
        )
        # P1-3 (2026-09-01): observe the decision for baseline telemetry.
        # Never raises so the answer path is untouched by telemetry failures.
        try:
            action = decision.action.value
            self._decision_counts[action] = self._decision_counts.get(action, 0) + 1
            self._last_decision = {
                "action": action,
                "reason": decision.reason,
                "signals": asdict(decision.signals),
            }
        except Exception:  # pragma: no cover - defensive
            pass
        return decision

    def _decide(
        self,
        evidence: Iterable[EvidenceItem],
        *,
        query_terms: Iterable[str] = (),
        has_conflict: bool = False,
        retrieval_round: int = 1,
        max_retrieval_rounds: int = 3,
    ) -> EvidenceGateDecision:
        """Pure decision core (no telemetry); see :meth:`evaluate`."""
        signals = compute_signals(evidence, query_terms=query_terms, has_conflict=has_conflict)
        cfg = self._config

        # Escalate: explicit conflict + very tight margin means the top-2
        # answers are nearly tied and contradictory.
        if signals.has_conflict and signals.score_margin < cfg.conflict_margin_threshold:
            return EvidenceGateDecision(
                action=EvidenceGateAction.ESCALATE,
                signals=signals,
                reason="conflict+tight-margin",
            )

        # Escalate: zero evidence after exhausting retrieval rounds.
        if signals.evidence_count == 0 and retrieval_round >= max_retrieval_rounds:
            return EvidenceGateDecision(
                action=EvidenceGateAction.ESCALATE,
                signals=signals,
                reason="no-evidence-after-max-rounds",
            )

        # Retrieve more: any single signal below threshold and we still have
        # retrieval budget left.
        weak_signals: list[str] = []
        if signals.evidence_count < cfg.min_evidence_count:
            weak_signals.append(f"count={signals.evidence_count}<{cfg.min_evidence_count}")
        if signals.top_score < cfg.min_top_score:
            weak_signals.append(f"top_score={signals.top_score:.2f}<{cfg.min_top_score}")
        if signals.score_margin < cfg.min_score_margin and signals.evidence_count >= 2:
            weak_signals.append(f"margin={signals.score_margin:.2f}<{cfg.min_score_margin}")
        if signals.coverage < cfg.min_coverage:
            weak_signals.append(f"coverage={signals.coverage:.2f}<{cfg.min_coverage}")
        if signals.diversity < cfg.min_diversity and signals.evidence_count > 0:
            weak_signals.append(f"diversity={signals.diversity:.2f}<{cfg.min_diversity}")

        if weak_signals:
            if retrieval_round >= max_retrieval_rounds:
                return EvidenceGateDecision(
                    action=EvidenceGateAction.ESCALATE,
                    signals=signals,
                    reason="weak-signals-after-max-rounds:" + "|".join(weak_signals),
                )
            return EvidenceGateDecision(
                action=EvidenceGateAction.RETRIEVE_MORE,
                signals=signals,
                reason="weak-signals:" + "|".join(weak_signals),
            )

        return EvidenceGateDecision(
            action=EvidenceGateAction.ANSWER,
            signals=signals,
            reason="sufficient-evidence",
        )

    # ------------------------------------------------------------------
    # P1-3 (2026-09-01): decision telemetry accessors
    # ------------------------------------------------------------------

    def get_evidence_signals(self) -> Dict[str, object]:
        """Snapshot of observed gate decisions for baseline ``evidence_signals``.

        Returns a JSON-serializable dict::

            {
                "decision_counts": {"answer": 2, "retrieve_more": 1},
                "last": {"action": "...", "reason": "...", "signals": {...}},
            }
        """
        return {
            "decision_counts": dict(self._decision_counts),
            "last": dict(self._last_decision) if self._last_decision else None,
        }

    def reset_decision_counts(self) -> None:
        """Clear telemetry (per-sample reset for the regression runner)."""
        self._decision_counts.clear()
        self._last_decision = None


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_gate: Optional[EvidenceGate] = None


def get_evidence_gate() -> EvidenceGate:
    global _gate
    if _gate is None:
        _gate = EvidenceGate()
    return _gate


def set_evidence_gate(gate: Optional[EvidenceGate]) -> None:
    """Inject a gate (test-only). Pass ``None`` to reset to default."""
    global _gate
    _gate = gate
