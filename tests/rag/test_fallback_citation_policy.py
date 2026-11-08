"""Fallback + citation policy tests (G5, B-6, 2026-08-17).

Locks the unified fallback cascade (RAG-06) and the shared citation shape:
- FallbackPolicy order (primary → context_answer → bm25_only → local_synthesis)
- next_after / record / summary semantics
- build_citation output contract (matches the legacy _build_evidence dict)
Pure unit tests.
"""
from __future__ import annotations

import unittest

from src.application.rag.policy.fallback import (
    FallbackLevel,
    FallbackPolicy,
    build_citation,
)


class FallbackPolicyTest(unittest.TestCase):
    def test_default_order_is_best_to_degraded(self):
        stages = FallbackPolicy.default().stages()
        self.assertEqual(
            stages,
            [
                FallbackLevel.PRIMARY,
                FallbackLevel.CONTEXT_ANSWER,
                FallbackLevel.BM25_ONLY,
                FallbackLevel.LOCAL_SYNTHESIS,
            ],
        )

    def test_next_after_moves_down_cascade(self):
        policy = FallbackPolicy.default()
        self.assertEqual(policy.next_after(FallbackLevel.PRIMARY), FallbackLevel.CONTEXT_ANSWER)
        self.assertEqual(policy.next_after(FallbackLevel.CONTEXT_ANSWER), FallbackLevel.BM25_ONLY)
        self.assertEqual(policy.next_after(FallbackLevel.BM25_ONLY), FallbackLevel.LOCAL_SYNTHESIS)
        self.assertIsNone(policy.next_after(FallbackLevel.LOCAL_SYNTHESIS))

    def test_next_after_unknown_level_returns_head(self):
        policy = FallbackPolicy.default()
        self.assertEqual(policy.next_after("not-a-level"), FallbackLevel.PRIMARY)

    def test_record_and_summary(self):
        policy = FallbackPolicy.default()
        policy.record(FallbackLevel.PRIMARY)
        policy.record(FallbackLevel.BM25_ONLY)
        summary = policy.summary()
        self.assertEqual(summary["engaged"], ["primary", "bm25_only"])
        self.assertEqual(len(summary["order"]), 4)

    def test_labels_cover_all_stages(self):
        labels = FallbackPolicy.default().labels()
        self.assertEqual(len(labels), 4)
        self.assertIn("Dify workflow", labels[FallbackLevel.PRIMARY])


class BuildCitationTest(unittest.TestCase):
    def test_output_shape_matches_legacy_evidence(self):
        citation = build_citation(
            chunk_id="c1",
            chunk_type="capability_card",
            source="doc.md",
            score=0.9,
            content_preview="content",
        )
        self.assertEqual(
            citation,
            {
                "chunk_id": "c1",
                "chunk_type": "capability_card",
                "source": "doc.md",
                "score": 0.9,
                "content_preview": "content",
            },
        )

    def test_extra_metadata_folded_in(self):
        citation = build_citation(
            chunk_id="c1",
            chunk_type="packet",
            source="retrieval_asset_packet",
            score=0.5,
            content_preview="x",
            extra={"metadata": {"entity_type": "System"}},
        )
        self.assertEqual(citation["metadata"], {"entity_type": "System"})
        self.assertEqual(citation["chunk_id"], "c1")

    def test_citation_keys_stable(self):
        citation = build_citation(
            chunk_id="n1", chunk_type="Neo4j:System", source="neo4j",
            score=1.0, content_preview="desc",
        )
        self.assertEqual(
            list(citation.keys()),
            ["chunk_id", "chunk_type", "source", "score", "content_preview"],
        )


if __name__ == "__main__":
    unittest.main()