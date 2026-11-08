"""P1-1 (2026-09-01): Evidence-gate integration characterization.

Locks the RETRIEVE_MORE auto re-retrieval behaviour introduced when the C-5
evidence gate is wired into the production retrieval loops of RAGAnswerer:

- legacy path: ``_retrieve_with_gate`` (round 1 = entity-expansion funnel,
  later rounds = raw fused funnel with expanded top_k).
- packet path: ``_search_packet_with_gate`` (round 1 = packet search, later
  rounds re-run the packet search with expanded top_k).
- ``_answer_async_impl`` drives both through the gate when ``evidence_gate``
  is set; tests that bypass ``__init__`` leave it unset -> gate skipped
  (legacy behaviour, locked by test_rag_answer_path_characterization).

Pure unit tests: no LLM, no network, no filesystem reads.
"""
from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

from src.application.rag.evidence_gate import (
    EvidenceGate,
    EvidenceGateAction,
)
from src.application.rag.rag_answerer import RAGAnswerer
from src.application.rag.models import ChunkRecord, RetrievedChunk


def _answerer(**attrs):
    a = object.__new__(RAGAnswerer)
    a.model = "test-model"
    a.max_context_chunks = 5
    a.max_chunk_length = 1000
    a.use_neo4j = False
    a.neo4j_weight = 0.3
    a.use_llm = False
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


def _retrieved(content: str, *, chunk_id: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=ChunkRecord(
            chunk_id=chunk_id,
            chunk_type="capability_card",
            content=content,
            node_ids=[],
            linked_node_ids=[],
            meta={"source": "doc.md"},
        ),
        score=score,
    )


def _pq(normalized: str = "q", entities=None, recommended_topk: int = 5) -> SimpleNamespace:
    return SimpleNamespace(
        normalized=normalized,
        entities=list(entities or []),
        is_multi_entity=len(entities or []) > 1,
        recommended_topk=recommended_topk,
    )


class LegacyGateRetrievalTest(unittest.TestCase):
    """Lock ``_retrieve_with_gate`` loop semantics (legacy path)."""

    def test_gate_disabled_delegates_unchanged(self):
        a = _answerer()  # evidence_gate unset -> gate skipped
        sentinel = [_retrieved("x", chunk_id="c1")]
        with mock.patch.object(a, "_retrieve_with_entity_expansion", return_value=sentinel) as funnel:
            out = a._retrieve_with_gate(_pq(), topk=5)
        funnel.assert_called_once_with(_pq(), topk=5)
        self.assertIs(out, sentinel)
        self.assertFalse(hasattr(a, "_last_gate_decision"))

    def test_sufficient_evidence_answers_in_one_round(self):
        a = _answerer(evidence_gate=EvidenceGate())
        # Scores must clear the default gate thresholds with a clean margin
        # (0.9-0.8 == 0.0999... < 0.10 would trigger RETRIEVE_MORE via float).
        strong = [
            _retrieved("A", chunk_id="c1", score=0.9),
            _retrieved("B", chunk_id="c2", score=0.7),
            _retrieved("C", chunk_id="c3", score=0.5),
        ]
        with mock.patch.object(a, "_retrieve_with_entity_expansion", return_value=strong), \
             mock.patch.object(a, "_retrieve_hybrid_fused", side_effect=AssertionError("must not re-retrieve")):
            out = a._retrieve_with_gate(_pq(), topk=5)
        self.assertEqual([r.chunk.chunk_id for r in out], ["c1", "c2", "c3"])
        self.assertEqual(a._last_gate_rounds_used, 1)
        self.assertEqual(a._last_gate_decision.action, EvidenceGateAction.ANSWER)

    def test_weak_evidence_triggers_retrieve_more_then_answer(self):
        a = _answerer(evidence_gate=EvidenceGate())
        weak = [_retrieved("weak A", chunk_id="w1", score=0.3)]
        strong = [
            _retrieved("strong B", chunk_id="s1", score=0.9),
            _retrieved("strong C", chunk_id="s2", score=0.7),
        ]
        with mock.patch.object(a, "_retrieve_with_entity_expansion", return_value=weak), \
             mock.patch.object(a, "_retrieve_hybrid_fused", return_value=strong) as fused:
            out = a._retrieve_with_gate(_pq(), topk=5)
        # Round 2 must expand top_k by 50% (5 -> max(5, int(5*1.5)+1) = 8).
        fused.assert_called_once_with("q", topk=8)
        self.assertEqual([r.chunk.chunk_id for r in out], ["w1", "s1", "s2"])
        self.assertEqual(a._last_gate_rounds_used, 2)
        self.assertEqual(a._last_gate_decision.action, EvidenceGateAction.ANSWER)

    def test_weak_evidence_after_max_rounds_escalates(self):
        a = _answerer(evidence_gate=EvidenceGate())
        weak = [_retrieved("weak A", chunk_id="w1", score=0.3)]
        with mock.patch.object(a, "_retrieve_with_entity_expansion", return_value=weak), \
             mock.patch.object(a, "_retrieve_hybrid_fused", return_value=weak):
            out = a._retrieve_with_gate(_pq(), topk=5, max_retrieval_rounds=2)
        self.assertEqual([r.chunk.chunk_id for r in out], ["w1"])
        self.assertEqual(a._last_gate_rounds_used, 2)
        self.assertEqual(a._last_gate_decision.action, EvidenceGateAction.ESCALATE)

    def test_no_evidence_after_max_rounds_escalates(self):
        a = _answerer(evidence_gate=EvidenceGate())
        with mock.patch.object(a, "_retrieve_with_entity_expansion", return_value=[]), \
             mock.patch.object(a, "_retrieve_hybrid_fused", return_value=[]):
            out = a._retrieve_with_gate(_pq(), topk=5, max_retrieval_rounds=2)
        self.assertEqual(out, [])
        self.assertEqual(a._last_gate_rounds_used, 2)
        self.assertEqual(a._last_gate_decision.action, EvidenceGateAction.ESCALATE)


class _FakePacketResult:
    def __init__(self, chunk_id: str, content: str, score: float, source: str):
        self.chunk_id = chunk_id
        self.content = content
        self.score = score
        self.source = source
        self.metadata = {}


class PacketGateRetrievalTest(unittest.TestCase):
    """Lock ``_search_packet_with_gate`` loop semantics (packet path)."""

    def test_gate_disabled_delegates_unchanged(self):
        a = _answerer()
        a.rag_service = mock.Mock()

        async def _search(query, top_k=5):
            return [_FakePacketResult("p1", "x", 0.9, "packet:a")]

        a.rag_service.search = _search
        with mock.patch.object(a, "_search_packet", wraps=a._search_packet) as sp:
            out = asyncio.run(a._search_packet_with_gate("q", 5))
        sp.assert_called_once_with("q", 5)
        self.assertEqual(len(out), 1)
        self.assertFalse(hasattr(a, "_last_gate_decision"))

    def test_sufficient_evidence_answers_in_one_round(self):
        a = _answerer(evidence_gate=EvidenceGate())

        async def _search(query, top_k=5):
            # 2026-10-16: gate 增加 coverage/diversity 信号后，证据需覆盖查询词
            # 且来源多样，才能一轮 ANSWER（原 "A"/"B"/"C" 覆盖=0 恒弱）。
            return [
                _FakePacketResult("p1", "q answer one", 0.9, "packet:a"),
                _FakePacketResult("p2", "q answer two", 0.7, "packet:b"),
                _FakePacketResult("p3", "q answer three", 0.5, "packet:c"),
            ]

        a.rag_service = mock.Mock()
        a.rag_service.search = _search
        out = asyncio.run(a._search_packet_with_gate("q", 5))
        self.assertEqual([r.chunk_id for r in out], ["p1", "p2", "p3"])
        self.assertEqual(a._last_gate_rounds_used, 1)
        self.assertEqual(a._last_gate_decision.action, EvidenceGateAction.ANSWER)

    def test_weak_evidence_triggers_retrieve_more_then_answer(self):
        a = _answerer(evidence_gate=EvidenceGate())
        calls = []

        async def _search(query, top_k=5):
            calls.append(top_k)
            if top_k == 5:
                return [_FakePacketResult("w1", "weak", 0.3, "packet:a")]
            return [
                _FakePacketResult("s1", "q strong one", 0.9, "packet:b"),
                _FakePacketResult("s2", "q strong two", 0.7, "packet:c"),
            ]

        a.rag_service = mock.Mock()
        a.rag_service.search = _search
        out = asyncio.run(a._search_packet_with_gate("q", 5))
        self.assertEqual(calls, [5, 8])
        self.assertEqual([r.chunk_id for r in out], ["w1", "s1", "s2"])
        self.assertEqual(a._last_gate_rounds_used, 2)
        self.assertEqual(a._last_gate_decision.action, EvidenceGateAction.ANSWER)


class AnswerAsyncImplGatedTest(unittest.TestCase):
    """Lock that ``_answer_async_impl`` drives the gate on both paths."""

    def test_legacy_answer_retrieves_more_when_evidence_weak(self):
        a = _answerer(evidence_gate=EvidenceGate())
        weak = [_retrieved("weak A", chunk_id="w1", score=0.3)]
        strong = [
            _retrieved("strong B", chunk_id="s1", score=0.9),
            _retrieved("strong C", chunk_id="s2", score=0.7),
        ]
        with mock.patch.object(a, "_rewrite_query", return_value="q"), \
             mock.patch.object(a, "_retrieve_with_entity_expansion", return_value=weak), \
             mock.patch.object(a, "_retrieve_hybrid_fused", return_value=strong), \
             mock.patch.object(a, "_search_neo4j", return_value=[]), \
             mock.patch.object(a, "_build_context", return_value="ctx"), \
             mock.patch.object(a, "_format_retrieved_results", return_value="formatted"), \
             mock.patch("src.application.rag.rag_answerer.parse_query") as pq:
            pq.return_value.recommended_topk = 5
            pq.return_value.normalized = "q"
            pq.return_value.entities = []
            result = asyncio.run(a._answer_async_impl("question"))
        self.assertIn("formatted", result.answer)
        # merged w1+s1+s2 all surface as evidence (effective_topk=5 slices to 3)
        self.assertEqual(len(result.evidence_chunks), 3)
        self.assertEqual(a._last_gate_rounds_used, 2)
        self.assertEqual(a._last_gate_decision.action, EvidenceGateAction.ANSWER)

    def test_packet_answer_retrieves_more_when_evidence_weak(self):
        a = _answerer(evidence_gate=EvidenceGate())

        class _Svc:
            def __init__(self):
                self.calls = []

            async def search(self, query, top_k=5):
                self.calls.append(top_k)
                if top_k == 5:
                    return [_FakePacketResult("w1", "weak", 0.3, "packet:a")]
                return [
                    _FakePacketResult("s1", "q strong1", 0.9, "packet:b"),
                    _FakePacketResult("s2", "q strong2", 0.7, "packet:c"),
                ]

        a.rag_service = _Svc()
        with mock.patch.object(a, "_rewrite_query", return_value="q"):
            result = asyncio.run(a._answer_async_impl("question"))
        self.assertEqual(a.rag_service.calls, [5, 8])
        self.assertEqual(a._last_gate_rounds_used, 2)
        self.assertEqual(a._last_gate_decision.action, EvidenceGateAction.ANSWER)
        self.assertEqual(len(result.evidence_chunks), 3)


if __name__ == "__main__":
    unittest.main()
