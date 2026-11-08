"""Context compaction for long prompts / RAG context (C-5).

Goal: keep the most *recent* + most *relevant* + all *conflict* items when a
context grows beyond a token budget, instead of naive truncation that drops
the tail (which is often where the latest evidence lives).

Inputs are generic "context items" — typically retrieved chunks, extracted
entities, candidate relations, or evidence snippets. Each item carries:

- ``text``: the actual content.
- ``score``: optional relevance score in [0, 1] (e.g. retrieval score,
  confidence, normalized recency).
- ``kind``: optional category label (chunk / entity / relation / evidence).
- ``is_conflict``: when True the item is *pinned* (always kept) because
  conflicts must survive compaction for downstream resolution.
- ``index``: original insertion order, used as recency tiebreaker.

The compactor is deterministic and side-effect free. It does not call any
LLM; compaction is rule-based so the cost-aware model routing stays the
single point where tokens get spent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass
class ContextItem:
    """A single compactable context entry.

    ``metadata`` is preserved through compaction so callers can attach
    provenance (chunk_id, source, run_id) without the compactor needing
    to understand it.
    """

    text: str
    score: float = 0.0
    kind: str = "generic"
    is_conflict: bool = False
    index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompactionConfig:
    """Compaction thresholds.

    Defaults keep a generous budget so compaction only kicks in when prompts
    genuinely blow past typical LLM context windows. The character budget is
    the hard cap; ``recent_keep`` and ``relevant_keep`` are soft quotas used
    to decide what survives when the budget is exceeded.
    """

    max_chars: int = 24000
    recent_keep: int = 6
    relevant_keep: int = 12
    min_items: int = 4  # never compact below this many items


@dataclass(frozen=True)
class CompactionResult:
    """Outcome of a compaction pass.

    Attributes:
        kept: items retained (in original order).
        dropped: items removed (in original order).
        kept_chars: total characters in kept items.
        budget_exceeded: True if compaction actually dropped something.
        reason: short explanation for trace / telemetry.
    """

    kept: list[ContextItem]
    dropped: list[ContextItem]
    kept_chars: int
    budget_exceeded: bool
    reason: str


def _approx_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


def _score_weighted_rank(item: ContextItem, *, total_items: int) -> float:
    """Combine score + recency into a single rank value.

    Recency is normalized to [0, 1] using ``index`` against ``total_items``;
    conflict items get +infinity so they always survive.
    """
    if item.is_conflict:
        return float("inf")
    recency = item.index / max(1, total_items - 1) if total_items > 1 else 1.0
    # 60% relevance score, 40% recency — keeps most-relevant first while
    # still respecting insertion order for tie-breaking.
    return 0.6 * float(item.score or 0.0) + 0.4 * recency


class ContextCompactor:
    """Rule-based context compactor (no LLM calls).

    Strategy when budget exceeded:

    1. Pin all conflict items (always kept, regardless of score).
    2. Keep the most recent ``recent_keep`` items (recency = highest index).
    3. Keep the top ``relevant_keep`` items by score.
    4. Fill remaining budget by score-weighted rank until ``max_chars`` hit.
    5. Never drop below ``min_items`` (prefer dropping longest non-pinned
       items first).
    """

    def __init__(self, config: Optional[CompactionConfig] = None) -> None:
        self._config = config or CompactionConfig()

    @property
    def config(self) -> CompactionConfig:
        return self._config

    def compact(self, items: Iterable[ContextItem]) -> CompactionResult:
        """Compact ``items`` to fit within the configured character budget."""
        item_list = list(items)
        if not item_list:
            return CompactionResult(
                kept=[], dropped=[], kept_chars=0, budget_exceeded=False, reason="empty-input"
            )

        total_chars = sum(len(i.text) for i in item_list)
        if total_chars <= self._config.max_chars:
            return CompactionResult(
                kept=item_list,
                dropped=[],
                kept_chars=total_chars,
                budget_exceeded=False,
                reason="within-budget",
            )

        n = len(item_list)
        # 1. Pin conflicts.
        pinned = [i for i in item_list if i.is_conflict]
        pinned_ids = {id(i) for i in pinned}

        # 2. Recent items (highest index).
        recent = sorted(
            [i for i in item_list if id(i) not in pinned_ids],
            key=lambda i: i.index,
            reverse=True,
        )[: self._config.recent_keep]
        recent_ids = {id(i) for i in recent}

        # 3. Top relevant by score.
        relevant = sorted(
            [i for i in item_list if id(i) not in pinned_ids and id(i) not in recent_ids],
            key=lambda i: i.score,
            reverse=True,
        )[: self._config.relevant_keep]
        relevant_ids = {id(i) for i in relevant}

        # 4. Remaining pool ranked by score-weighted rank.
        selected_ids = pinned_ids | recent_ids | relevant_ids
        pool = [i for i in item_list if id(i) not in selected_ids]
        pool.sort(key=lambda i: _score_weighted_rank(i, total_items=n), reverse=True)

        kept: list[ContextItem] = []
        kept_chars = 0
        kept_ids: set[int] = set()

        def _try_add(item: ContextItem, *, force: bool = False) -> bool:
            nonlocal kept_chars
            if id(item) in kept_ids:
                return False
            if not force and kept_chars + len(item.text) > self._config.max_chars:
                return False
            kept.append(item)
            kept_ids.add(id(item))
            kept_chars += len(item.text)
            return True

        # Add pinned first (always), then recent, then relevant, then pool.
        for item in pinned:
            _try_add(item, force=True)
        for item in recent:
            _try_add(item)
        for item in relevant:
            _try_add(item)
        # Pool: force-add until min_items floor is reached, then respect budget.
        for item in pool:
            if len(kept) < self._config.min_items:
                _try_add(item, force=True)
                continue
            _try_add(item)

        kept_set = {id(i) for i in kept}
        dropped = [i for i in item_list if id(i) not in kept_set]

        # Re-order kept to original insertion order for downstream consumers.
        kept.sort(key=lambda i: i.index)

        return CompactionResult(
            kept=kept,
            dropped=dropped,
            kept_chars=kept_chars,
            budget_exceeded=True,
            reason=f"compacted-{len(item_list)}-to-{len(kept)}",
        )

    def compact_texts(self, texts: Iterable[str], *, scores: Optional[Iterable[float]] = None) -> list[str]:
        """Convenience: compact a flat list of strings.

        Returns the kept texts in original order. ``scores`` (optional) maps
        1:1 to ``texts``; missing scores default to 0.
        """
        text_list = list(texts)
        score_list = list(scores) if scores is not None else [0.0] * len(text_list)
        items = [
            ContextItem(text=t, score=s, kind="text", index=i)
            for i, (t, s) in enumerate(zip(text_list, score_list))
        ]
        result = self.compact(items)
        return [i.text for i in result.kept]


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_compactor: Optional[ContextCompactor] = None


def get_context_compactor() -> ContextCompactor:
    global _compactor
    if _compactor is None:
        _compactor = ContextCompactor()
    return _compactor


def set_context_compactor(compactor: Optional[ContextCompactor]) -> None:
    """Inject a compactor (test-only). Pass ``None`` to reset to default."""
    global _compactor
    _compactor = compactor
