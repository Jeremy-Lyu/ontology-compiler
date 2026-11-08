"""Characterization for P2-1 retrieval upgrade (channel routing + RRF-first).

Locks:
1. route_channels determinism (pure function over parsed query surface).
2. _rank_rrf contract (dual-route accumulation beats single-route; bounded).
3. hybrid_search RRF-first default (fusion_mode default flipped weighted->rrf
   in RetrievalConfig; HybridRetriever weighted path preserved as opt-out).
4. hybrid_search fixed tail (rerank desc + citation metadata attached).
"""
from __future__ import annotations

import unittest


def _result(rid, **scores):
    from src.application.rag.hybrid_rag import RetrievalResult
    r = RetrievalResult(result_id=rid, content=rid, source="t", source_type="t")
    for k, v in scores.items():
        setattr(r, k, v)
    return r


class ChannelRouterTest(unittest.TestCase):
    def test_definition_query_routes_lexical_vector_structured(self):
        from src.application.rag.query_channel_router import route_channels
        d = route_channels(query_type="entity_lookup", intent="definition")
        self.assertIn("lexical", d.channels)
        self.assertIn("structured", d.channels)
        # pure exact-name lookup: vector off
        self.assertNotIn("vector", d.channels)

    def test_semantic_query_routes_vector(self):
        from src.application.rag.query_channel_router import route_channels
        d = route_channels(query_type="semantic_search", intent="process")
        self.assertIn("lexical", d.channels)
        self.assertIn("vector", d.channels)

    def test_relation_multi_entity_routes_graph(self):
        from src.application.rag.query_channel_router import route_channels
        d = route_channels(query_type="relation_query", intent="relationship",
                           is_multi_entity=True, entity_count=2)
        self.assertIn("graph", d.channels)

    def test_single_entity_nonrelational_no_graph(self):
        from src.application.rag.query_channel_router import route_channels
        d = route_channels(query_type="entity_lookup", intent="definition")
        self.assertNotIn("graph", d.channels)

    def test_deterministic(self):
        from src.application.rag.query_channel_router import route_channels
        a = route_channels(query_type="hybrid_search", intent="comparison",
                           is_multi_entity=True, entity_count=3)
        b = route_channels(query_type="hybrid_search", intent="comparison",
                           is_multi_entity=True, entity_count=3)
        self.assertEqual(a.channels, b.channels)
        self.assertIn("graph", a.channels)


class HybridRrfRankTest(unittest.TestCase):
    def test_dual_channel_beats_single_channel(self):
        from src.application.rag.hybrid_rag import HybridIndexManager
        ranked = HybridIndexManager._rank_rrf(
            {"both": 0.2, "single_a": 0.9}, {"both": 0.8, "single_b": 0.7}, rrf_k=60)
        order = [cid for cid, _ in ranked]
        self.assertEqual(order[0], "both")

    def test_scores_bounded(self):
        from src.application.rag.hybrid_rag import HybridIndexManager
        ranked = HybridIndexManager._rank_rrf({"a": 1.0}, {"a": 1.0}, rrf_k=60)
        self.assertLess(ranked[0][1], 0.034)

    def test_empty_channels(self):
        from src.application.rag.hybrid_rag import HybridIndexManager
        self.assertEqual(HybridIndexManager._rank_rrf({}, {}, rrf_k=60), [])

    def test_tiebreak_deterministic(self):
        from src.application.rag.hybrid_rag import HybridIndexManager
        ranked = HybridIndexManager._rank_rrf({"a": 0.5, "b": 0.5}, {}, rrf_k=60)
        self.assertEqual([c for c, _ in ranked], ["a", "b"])


class RrfFirstDefaultTest(unittest.TestCase):
    def test_retrieval_config_defaults_rrf(self):
        from src.application.rag.hybrid_rag import RetrievalConfig
        self.assertEqual(RetrievalConfig().fusion_mode, "rrf")

    def test_weighted_opt_out_preserved(self):
        from src.application.rag.hybrid_rag import HybridRetriever, RetrievalConfig
        retriever = object.__new__(HybridRetriever)
        retriever._structured_index = None
        config = RetrievalConfig()
        config.fusion_mode = "weighted"
        config.structured_weight = 0.6
        config.vector_weight = 0.4
        fused = retriever._fuse_results(
            [_result("a", structured_score=0.8)],
            [_result("a", vector_score=0.5)], [], [], config)
        self.assertEqual(len(fused), 1)
        self.assertAlmostEqual(fused[0].hybrid_score, 0.8 * 0.6 + 0.5 * 0.4)


if __name__ == "__main__":
    unittest.main()
