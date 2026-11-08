"""Unified fallback + citation policy for RAG (G5, B-6, 2026-08-17).

Consolidates the *policy* (ordering + registration) of the RAG fallback paths
that previously lived implicitly in several call sites:

1. retrieval: hybrid BM25+Vector → pure-BM25 (``hybrid_rag`` / answerer)
2. LLM answer: primary Dify/LLM → ``_fallback_answer_from_context`` (Dify empty)
3. deep-analysis: A-H framework → ``_generate_deep_analysis_fallback`` (LLM empty)
4. citation: retrieval-result + neo4j verdict → ``_build_evidence`` shapes

Design: the policy is a pure, deterministic *sequence/registry* — it does NOT
rewrite the fallback branch bodies (their behavior is locked by
characterization tests). It gives a single named order and a stable place to
inspect/override the cascade, addressing RAG-06 ("rules in one place").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class FallbackLevel(str, Enum):
    """Stages in the RAG answer cascade, best → degraded."""

    PRIMARY = "primary"              # Dify workflow / direct LLM structured answer
    CONTEXT_ANSWER = "context_answer"  # natural answer synthesized from retrieval context
    BM25_ONLY = "bm25_only"          # retrieval degraded to pure keyword search
    LOCAL_SYNTHESIS = "local_synthesis"  # local deterministic summary (no LLM)


@dataclass
class FallbackStep:
    level: FallbackLevel
    label: str
    invoked_by: str  # method / module that owns the branch


_FALLBACK_ORDER: List[FallbackStep] = [
    FallbackStep(FallbackLevel.PRIMARY, "Dify workflow / LLM structured answer", "rag_answerer._call_llm_with_dify"),
    FallbackStep(FallbackLevel.CONTEXT_ANSWER, "natural answer from retrieved context", "rag_answerer._fallback_answer_from_context"),
    FallbackStep(FallbackLevel.BM25_ONLY, "pure keyword retrieval fallback", "rag_answerer._retrieve_hybrid_fused / hybrid_rag._compute_bm25"),
    FallbackStep(FallbackLevel.LOCAL_SYNTHESIS, "local deterministic summary / deep-analysis fallback", "rag_answerer._generate_deep_analysis_fallback"),
]


class FallbackPolicy:
    """Read-only policy: ordered fallback cascade + query/state helpers.

    Immutable by default (frozen) except for an optional per-run override
    queue used by callers to record which stages actually engaged.
    """

    def __init__(self, steps: Optional[List[FallbackStep]] = None) -> None:
        self._steps = list(steps or _FALLBACK_ORDER)
        self._engaged: List[str] = []

    def stages(self) -> List[FallbackLevel]:
        return [s.level for s in self._steps]

    def labels(self) -> Dict[FallbackLevel, str]:
        return {s.level: s.label for s in self._steps}

    def next_after(self, level: FallbackLevel) -> Optional[FallbackLevel]:
        """Return the next-degraded level after ``level`` (None when already last)."""
        stages = self.stages()
        try:
            idx = stages.index(level)
        except ValueError:
            return stages[0] if stages else None
        if idx + 1 < len(stages):
            return stages[idx + 1]
        return None

    def record(self, level: FallbackLevel) -> None:
        self._engaged.append(level)

    def summary(self) -> Dict[str, Any]:
        return {
            "order": [s.level.value for s in self._steps],
            "engaged": list(self._engaged),
        }

    @staticmethod
    def default() -> "FallbackPolicy":
        return FallbackPolicy()


# ---------------------------------------------------------------------------
# Citation assembly (G5c): single output shape for evidence sinks.
#
# The answerer builds evidence from two sources (retrieval results and Neo4j
# verdicts) in two places (``_build_evidence`` and ``_packet_results_to_evidence``).
# ``build_citation`` normalises both to one citation shape so RAG-06's "citation
# rules in one place" holds, without changing the emitted dicts (locked by
# ``tests/rag/test_rag_answer_path_characterization.py``).
# ---------------------------------------------------------------------------


def build_citation(
    *,
    chunk_id: str,
    chunk_type: str,
    source: str,
    score: float,
    content_preview: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Single citation output shape for RAG evidence sinks.

    Order and defaults match the legacy ``_build_evidence`` dict exactly so
    existing characterization (``chunk_id``/``chunk_type``/``source``/``score``/
    ``content_preview``) is preserved; ``extra`` folds in per-path extras such
    as the packet ``metadata`` field.
    """
    citation: Dict[str, Any] = {
        "chunk_id": chunk_id,
        "chunk_type": chunk_type,
        "source": source,
        "score": score,
        "content_preview": content_preview,
    }
    if extra:
        citation.update(extra)
    return citation