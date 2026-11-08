"""RAG answer-path + hybrid-fusion characterization (G9, 2026-08-17).

Before RRF fusion (G6/B-3) and MainPipeline decision extraction (G7), lock the
current behavior of:

- `_answer_async_impl` main path (packet-first + legacy) via mocks — field/
  fallback/confidence behavior without real LLM or network.
- `_build_evidence` / `_compute_confidence` — evidence assembly + confidence
  scaling semantics.
- `HybridRetriever._fuse_results` / `_rerank_results` — three-way fusion and
  ranking behavior (the thing G6 will upgrade).

Pure unit tests: no LLM, no network, no filesystem reads.
"""
from __future__ import annotations

import asyncio
import unittest
from unittest import mock

from src.application.rag.rag_answerer import RAGAnswerer
from src.application.rag.models import ChunkRecord, RetrievedChunk


def _answerer(**attrs):
    a = object.__new__(RAGAnswerer)
    a.model = "test-model"
    a.max_context_chunks = 5
    a.max_chunk_length = 1000
    a.use_neo4j = False
    a.neo4j_weight = 0.3
    a.use_llm = False  # deterministic by default; tests that need LLM enable it
    a.use_multi_hop = False
    a.max_hop_depth = 3
    a.rag_service = None
    a.chunks = []
    a.index = {}
    a.idf = {}
    a._hybrid_manager = None
    a._hybrid_manager_tried = False
    a.neo4j_retriever = None
    a.dify_config = None
    a.llm_provider = None
    a.llm_client = None
    for k, v in attrs.items():
        setattr(a, k, v)
    return a


def _retrieved(content: str, *, chunk_id: str = "cap_1", chunk_type: str = "capability_card", score: float = 0.9, meta=None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=ChunkRecord(
            chunk_id=chunk_id,
            chunk_type=chunk_type,
            content=content,
            node_ids=[],
            linked_node_ids=[],
            meta=meta or {"source": "doc.md"},
        ),
        score=score,
    )


class AnswerAsyncImplTest(unittest.TestCase):
    """Lock answer_async main-path behavior with mock retrieval/no LLM."""

    def test_legacy_no_results_returns_empty(self):
        a = _answerer()
        with mock.patch.object(a, "_rewrite_query", return_value="q"), \
             mock.patch.object(a, "_retrieve_with_entity_expansion", return_value=[]), \
             mock.patch.object(a, "_search_neo4j", return_value=[]), \
             mock.patch("src.application.rag.rag_answerer.parse_query") as pq:
            pq.return_value.recommended_topk = 5
            result = asyncio.run(a._answer_async_impl("question"))
        self.assertEqual(result.answer, "未能找到相关信息来回答这个问题。")
        self.assertEqual(result.evidence_chunks, [])
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.model, "test-model")

    def test_packet_path_empty_returns_empty(self):
        a = _answerer()

        class _EmptyService:
            async def search(self, query, top_k=5):
                return []

        a.rag_service = _EmptyService()
        with mock.patch.object(a, "_rewrite_query", return_value="q"):
            result = asyncio.run(a._answer_async_impl("question"))
        self.assertEqual(result.answer, "未能找到相关信息来回答这个问题。")
        self.assertEqual(result.confidence, 0.0)

    def test_legacy_deterministic_answer_builds_evidence(self):
        a = _answerer()
        res = [_retrieved("content A", chunk_id="c1", score=0.8), _retrieved("content B", chunk_id="c2", score=0.6)]
        with mock.patch.object(a, "_rewrite_query", return_value="q"), \
             mock.patch.object(a, "_retrieve_with_entity_expansion", return_value=res), \
             mock.patch.object(a, "_search_neo4j", return_value=[]), \
             mock.patch.object(a, "_build_context", return_value="ctx"), \
             mock.patch.object(a, "_format_retrieved_results", return_value="formatted"), \
             mock.patch("src.application.rag.rag_answerer.parse_query") as pq:
            pq.return_value.recommended_topk = 5
            result = asyncio.run(a._answer_async_impl("question"))
        self.assertIn("formatted", result.answer)
        self.assertEqual(result.confidence, 0.5)  # no-LLM deterministic branch
        self.assertEqual(len(result.evidence_chunks), 2)
        self.assertEqual(result.model, "test-model")

    def test_legacy_llm_branch_uses_compute_confidence(self):
        a = _answerer(use_llm=True)
        a.llm_client = mock.Mock()
        a.llm_client.provider = "deepseek"
        res = [_retrieved("content A", chunk_id="c1")]
        with mock.patch.object(a, "_rewrite_query", return_value="q"), \
             mock.patch.object(a, "_retrieve_with_entity_expansion", return_value=res), \
             mock.patch.object(a, "_search_neo4j", return_value=[]), \
             mock.patch.object(a, "_build_context", return_value="ctx"), \
             mock.patch.object(a, "_call_llm", return_value="llm answer"), \
             mock.patch.object(a, "_compute_confidence", return_value=0.9), \
             mock.patch("src.application.rag.rag_answerer.parse_query") as pq:
            pq.return_value.recommended_topk = 5
            result = asyncio.run(a._answer_async_impl("question"))
        self.assertEqual(result.answer, "llm answer")
        self.assertEqual(result.confidence, 0.9)
        self.assertEqual(result.model, "deepseek")


class BuildEvidenceAndConfidenceTest(unittest.TestCase):
    def test_evidence_assembly_from_results_and_neo4j(self):
        a = _answerer()
        res = [_retrieved("content A", chunk_id="c1")]
        neo4j = [{"id": "n1", "node_type": "System", "description": "desc"}]
        evidence = a._build_evidence(res, neo4j, effective_topk=5)
        self.assertEqual(len(evidence), 2)
        self.assertEqual(evidence[0]["chunk_id"], "c1")
        self.assertEqual(evidence[0]["source"], "doc.md")
        self.assertEqual(evidence[0]["score"], 0.9)
        self.assertEqual(evidence[1]["chunk_id"], "neo4j:n1")
        self.assertEqual(evidence[1]["chunk_type"], "Neo4j:System")
        self.assertEqual(evidence[1]["score"], 1.0)

    def test_evidence_topk_slices_document_results(self):
        a = _answerer()
        res = [_retrieved("x", chunk_id=f"c{i}") for i in range(5)]
        evidence = a._build_evidence(res, [], effective_topk=2)
        self.assertEqual([e["chunk_id"] for e in evidence], ["c0", "c1"])

    def test_confidence_scales_with_results(self):
        a = _answerer()
        zero = a._compute_confidence([], [])
        one = a._compute_confidence([_retrieved("x")], [])
        five = a._compute_confidence([_retrieved("x") for _ in range(5)], [])
        with_neo4j = a._compute_confidence([_retrieved("x") for _ in range(5)], [{"id": "n1"}])
        self.assertEqual(zero, 0.5)
        self.assertGreater(one, zero)
        self.assertEqual(five, 0.9)  # capped at 0.9 without neo4j
        self.assertEqual(with_neo4j, 0.95)  # capped at 0.95 with neo4j


def _retrieval_result(result_id: str, structured: float | None = None, vector: float | None = None, rule: float | None = None):
    from src.application.rag.hybrid_rag import RetrievalResult
    r = RetrievalResult(
        result_id=result_id,
        content="content",
        source="src",
        source_type="doc",
        metadata={},
    )
    if structured is not None:
        r.structured_score = structured
    if vector is not None:
        r.vector_score = vector
    if rule is not None:
        r.rule_score = rule
    return r


class HybridFusionCharacterizationTest(unittest.TestCase):
    """Lock _fuse_results weighted-sum fusion (P2-1: weighted is the opt-out
    fallback; RRF is the default — tests pin fusion_mode explicitly)."""

    def _retriever(self):
        from src.application.rag.hybrid_rag import HybridRetriever, RetrievalConfig
        retriever = object.__new__(HybridRetriever)
        retriever._structured_index = None
        config = RetrievalConfig()
        config.fusion_mode = "weighted"
        return retriever, config

    def test_merge_structured_and_vector_weights_sum(self):
        retriever, config = self._retriever()
        config.structured_weight = 0.6
        config.vector_weight = 0.4
        structured = [_retrieval_result("a", structured=0.8)]
        vector = [_retrieval_result("a", vector=0.5)]
        fused = retriever._fuse_results(structured, vector, [], [], config)
        self.assertEqual(len(fused), 1)
        self.assertAlmostEqual(fused[0].hybrid_score, 0.8 * 0.6 + 0.5 * 0.4)

    def test_three_route_merge_reads_all_weights(self):
        retriever, config = self._retriever()
        config.structured_weight = 0.5
        config.vector_weight = 0.3
        config.rule_weight = 0.2
        structured = [_retrieval_result("a", structured=1.0)]
        vector = [_retrieval_result("a", vector=0.5)]
        rule = [_retrieval_result("a", rule=0.25)]
        fused = retriever._fuse_results(structured, vector, rule, [], config)
        self.assertEqual(len(fused), 1)
        self.assertAlmostEqual(fused[0].hybrid_score, 1.0 * 0.5 + 0.5 * 0.3 + 0.25 * 0.2)

    def test_vector_only_result_uses_vector_weight(self):
        retriever, config = self._retriever()
        config.vector_weight = 0.4
        vector = [_retrieval_result("b", vector=0.7)]
        fused = retriever._fuse_results([], vector, [], [], config)
        self.assertEqual(len(fused), 1)
        self.assertAlmostEqual(fused[0].hybrid_score, 0.7 * 0.4)

    def test_rerank_sorts_by_hybrid_score(self):
        retriever, config = self._retriever()
        fused = [
            _retrieval_result("low", structured=0.2),
            _retrieval_result("high", structured=0.9),
            _retrieval_result("mid", structured=0.5),
        ]
        for r, s in zip(fused, [0.2, 0.9, 0.5]):
            r.hybrid_score = s
        ranked = retriever._rerank_results(fused, config)
        self.assertEqual([r.result_id for r in ranked], ["high", "mid", "low"])


class RrfFusionTest(unittest.TestCase):
    """G6 (2026-08-17): RRF fusion — cross-route hits rank above single-route,
    and weighted mode stays the default (backward compatible)."""

    def _retriever(self):
        from src.application.rag.hybrid_rag import HybridRetriever, RetrievalConfig
        retriever = object.__new__(HybridRetriever)
        retriever._structured_index = None
        return retriever, RetrievalConfig()

    def test_fusion_mode_defaults_rrf(self):
        # P2-1 (2026-09-09): RRF-first flip — weighted kept as opt-out.
        _, config = self._retriever()
        self.assertEqual(config.fusion_mode, "rrf")

    def test_rrf_bumps_double_route_above_single_route(self):
        retriever, config = self._retriever()
        config.fusion_mode = "rrf"
        config.rrf_k = 60
        # Route A ranks: [both, single_a]  (both is #1 -> high 1/61)
        # Route B ranks: [single_b, both]  (both is #2 -> adds 1/62)
        both = _retrieval_result("both", structured=0.2)   # mediocre single-score
        single_a = _retrieval_result("single_a", structured=0.9)  # high single-score
        single_b = _retrieval_result("single_b", structured=0.8)
        structured = [both, single_a]
        vector = [single_b, both]
        fused = retriever._fuse_results(structured, vector, [], [], config)
        ranked = sorted(fused, key=lambda r: r.hybrid_score, reverse=True)
        # both appears in 2 routes -> 1/61 + 1/62 > single_a's 1/61
        self.assertEqual(ranked[0].result_id, "both")
        self.assertGreater(ranked[0].hybrid_score, ranked[1].hybrid_score)
        self.assertLess(ranked[0].hybrid_score, 0.0335)  # ~1/60+1/61, bounded

    def test_rrf_confidence_equals_rrf_score(self):
        retriever, config = self._retriever()
        config.fusion_mode = "rrf"
        fused = retriever._fuse_results([_retrieval_result("a", structured=0.9)], [], [], [], config)
        self.assertAlmostEqual(fused[0].confidence, fused[0].hybrid_score)

    def test_rrf_single_route_equals_weighted_ignored(self):
        retriever, config = self._retriever()
        config.fusion_mode = "rrf"
        a = _retrieval_result("x", structured=0.7)
        b = _retrieval_result("y", structured=0.3)
        fused = retriever._fuse_results([a, b], [], [], [], config)
        ranked = sorted(fused, key=lambda r: r.hybrid_score, reverse=True)
        # order determined by rank order (a before b), not raw score
        self.assertEqual(ranked[0].result_id, "x")
        self.assertEqual(ranked[1].result_id, "y")
        # both have the same RRF contribution (rank 0 vs 1 differ by 1/k term)
        self.assertGreater(ranked[0].hybrid_score, ranked[1].hybrid_score)


if __name__ == "__main__":
    unittest.main()