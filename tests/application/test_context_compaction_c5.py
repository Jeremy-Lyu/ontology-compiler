from __future__ import annotations

"""C-5 characterization tests: context compaction.

Covers:
- within-budget: no compaction, all items kept in original order
- over-budget: conflict items always pinned
- over-budget: recent items kept (highest index)
- over-budget: top-score items kept
- min_items floor prevents compacting below threshold
- compact_texts convenience wrapper
"""

import unittest

from src.application.services.context_compaction import (
    CompactionConfig,
    ContextCompactor,
    ContextItem,
    get_context_compactor,
    set_context_compactor,
)


def _item(text: str, *, score: float = 0.0, kind: str = "generic", index: int = 0, is_conflict: bool = False) -> ContextItem:
    return ContextItem(text=text, score=score, kind=kind, index=index, is_conflict=is_conflict)


class WithinBudgetTest(unittest.TestCase):
    def test_no_compaction_when_under_budget(self):
        compactor = ContextCompactor(CompactionConfig(max_chars=1000))
        items = [_item("a", index=0), _item("b", index=1)]
        result = compactor.compact(items)
        self.assertFalse(result.budget_exceeded)
        self.assertEqual(len(result.kept), 2)
        self.assertEqual(len(result.dropped), 0)
        self.assertEqual(result.reason, "within-budget")

    def test_empty_input(self):
        result = ContextCompactor().compact([])
        self.assertEqual(result.kept, [])
        self.assertEqual(result.reason, "empty-input")


class ConflictPinningTest(unittest.TestCase):
    def test_conflict_items_always_kept(self):
        # Make budget so tight only one item can survive.
        compactor = ContextCompactor(CompactionConfig(max_chars=5, min_items=1, recent_keep=0, relevant_keep=0))
        items = [
            _item("long text that exceeds budget", score=0.9, index=0),
            _item("x", score=0.1, index=1, is_conflict=True),
        ]
        result = compactor.compact(items)
        self.assertTrue(any(i.is_conflict for i in result.kept))


class RecencyAndRelevanceTest(unittest.TestCase):
    def test_recent_items_kept(self):
        compactor = ContextCompactor(CompactionConfig(max_chars=20, recent_keep=2, relevant_keep=0, min_items=1))
        items = [
            _item("first item here", score=0.9, index=0),
            _item("second item", score=0.1, index=1),
            _item("third item", score=0.1, index=2),
        ]
        result = compactor.compact(items)
        kept_indices = [i.index for i in result.kept]
        # Most recent (highest index) should be in kept.
        self.assertIn(2, kept_indices)

    def test_top_score_items_kept(self):
        compactor = ContextCompactor(CompactionConfig(max_chars=20, recent_keep=0, relevant_keep=1, min_items=1))
        items = [
            _item("low score text", score=0.1, index=0),
            _item("high score", score=0.95, index=1),
            _item("mid score", score=0.5, index=2),
        ]
        result = compactor.compact(items)
        kept_scores = [i.score for i in result.kept]
        self.assertIn(0.95, kept_scores)


class MinItemsFloorTest(unittest.TestCase):
    def test_min_items_floor_respected(self):
        # Budget too small for min_items, but we still keep min_items.
        compactor = ContextCompactor(CompactionConfig(max_chars=1, min_items=3, recent_keep=0, relevant_keep=0))
        items = [_item(f"item-{i}", score=0.5, index=i) for i in range(5)]
        result = compactor.compact(items)
        self.assertGreaterEqual(len(result.kept), 3)


class OrderPreservationTest(unittest.TestCase):
    def test_kept_items_in_original_order(self):
        compactor = ContextCompactor(CompactionConfig(max_chars=10, min_items=2, recent_keep=1, relevant_keep=1))
        items = [
            _item("a", score=0.9, index=0),
            _item("b", score=0.5, index=1),
            _item("c", score=0.3, index=2),
        ]
        result = compactor.compact(items)
        kept_indices = [i.index for i in result.kept]
        self.assertEqual(kept_indices, sorted(kept_indices))


class CompactTextsTest(unittest.TestCase):
    def test_compact_texts_returns_kept_strings(self):
        compactor = ContextCompactor(CompactionConfig(max_chars=20, min_items=1, recent_keep=1, relevant_keep=1))
        texts = ["first text", "second text", "third text"]
        kept = compactor.compact_texts(texts, scores=[0.9, 0.5, 0.1])
        self.assertIsInstance(kept, list)
        self.assertGreaterEqual(len(kept), 1)
        for t in kept:
            self.assertIsInstance(t, str)


class SingletonTest(unittest.TestCase):
    def tearDown(self):
        set_context_compactor(None)

    def test_default_singleton(self):
        c = get_context_compactor()
        self.assertIsInstance(c, ContextCompactor)

    def test_inject_custom(self):
        custom = ContextCompactor(CompactionConfig(max_chars=42))
        set_context_compactor(custom)
        self.assertIs(get_context_compactor(), custom)
        self.assertEqual(get_context_compactor().config.max_chars, 42)


if __name__ == "__main__":
    unittest.main()
