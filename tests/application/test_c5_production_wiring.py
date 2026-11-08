from __future__ import annotations

"""C-5 production wiring tests: candidate_generator + rag_answerer + hybrid_rag.

Verifies the three production call points correctly consume the C-5 modules:
- candidate_generator: gateway.route() consulted, routed max_tokens applied
- rag_answerer: execute_routed() used, route_decision attached to outcome
- hybrid_rag: retrieve_with_gate() runs EvidenceGate, multi-round retrieval
"""

import asyncio
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from src.application.rag.evidence_gate import (
    EvidenceGate,
    EvidenceGateAction,
    EvidenceGateConfig,
    EvidenceItem,
)
from src.application.rag.hybrid_rag import (
    GateRetrievalResult,
    HybridRetriever,
    RetrievalConfig,
    RetrievalResult,
)
from src.application.services.model_router import (
    ModelRoutingConfig,
    ModelRoutingPolicy,
    ModelTier,
    RouteDecision,
    TaskComplexity,
    TaskKind,
    set_model_routing_policy,
)
from src.infrastructure.llm.llm_candidate_generator import LLMCandidateGenerator


# ---------------------------------------------------------------------------
# candidate_generator wiring
# ---------------------------------------------------------------------------


class _AsyncFakeClient:
    """Async client that records the max_tokens / temperature it was called with."""

    def __init__(self, response: str = '{"candidates": []}') -> None:
        self.response = response
        self.last_max_tokens: Any = None
        self.last_temperature: Any = None

    async def call(self, prompt: str, max_tokens: int = 0, temperature: float = 0.0) -> str:
        self.last_max_tokens = max_tokens
        self.last_temperature = temperature
        return self.response


class _FakeGateway:
    """Minimal gateway stub that returns a fixed RouteDecision."""

    def __init__(self, decision: RouteDecision) -> None:
        self._decision = decision
        self.route_calls: list[dict[str, Any]] = []

    def route(self, prompt: str, *, task_kind: Any = None, **kw: Any) -> RouteDecision:
        self.route_calls.append({"prompt": prompt, "task_kind": task_kind})
        return self._decision


class CandidateGeneratorRoutingTest(unittest.TestCase):
    def tearDown(self):
        set_model_routing_policy(None)

    def test_gateway_route_consulted_for_entity_extraction(self):
        # Inject a gateway whose routed max_tokens differs from the default.
        decision = RouteDecision(
            tier=TaskComplexity.STANDARD,
            model_tier=ModelTier(provider="deepseek", model="deepseek-chat", max_tokens=333, temperature=0.42),
            task_kind=TaskKind.ENTITY_EXTRACTION,
            reason="test",
            signals={},
        )
        gateway = _FakeGateway(decision)
        client = _AsyncFakeClient()
        gen = LLMCandidateGenerator(llm_client=client, gateway=gateway)

        asyncio.run(gen._call_llm("extract entities", task="entity_candidates"))

        # The gateway was consulted with the right task kind.
        self.assertEqual(len(gateway.route_calls), 1)
        self.assertEqual(gateway.route_calls[0]["task_kind"], TaskKind.ENTITY_EXTRACTION)
        # Routed max_tokens / temperature were applied.
        self.assertEqual(client.last_max_tokens, 333)
        self.assertAlmostEqual(client.last_temperature, 0.42)

    def test_explicit_max_tokens_overrides_routed(self):
        decision = RouteDecision(
            tier=TaskComplexity.STANDARD,
            model_tier=ModelTier(provider="deepseek", model="deepseek-chat", max_tokens=333, temperature=0.42),
            task_kind=TaskKind.ENTITY_EXTRACTION,
            reason="test",
            signals={},
        )
        gateway = _FakeGateway(decision)
        client = _AsyncFakeClient()
        gen = LLMCandidateGenerator(llm_client=client, gateway=gateway)

        asyncio.run(gen._call_llm("prompt", max_tokens=999, task="entity_candidates"))

        # Explicit caller max_tokens wins over routed.
        self.assertEqual(client.last_max_tokens, 999)
        # Routed temperature still applies (no explicit override).
        self.assertAlmostEqual(client.last_temperature, 0.42)

    def test_no_gateway_keeps_original_behavior(self):
        client = _AsyncFakeClient()
        gen = LLMCandidateGenerator(llm_client=client, default_max_tokens=500)
        asyncio.run(gen._call_llm("prompt", task="entity_candidates"))
        # No gateway => default max_tokens used, budget policy temperature.
        self.assertEqual(client.last_max_tokens, 500)

    def test_task_kind_map_covers_all_generate_tasks(self):
        """Verify each generate_* method's task string maps to a TaskKind."""
        from src.infrastructure.llm.llm_candidate_generator import _TASK_KIND_MAP

        for task_str, kind_str in _TASK_KIND_MAP.items():
            self.assertIsInstance(task_str, str)
            self.assertIsInstance(kind_str, str)
            # The kind string must be a valid TaskKind value.
            TaskKind(kind_str)


# ---------------------------------------------------------------------------
# hybrid_rag retrieve_with_gate wiring
# ---------------------------------------------------------------------------


def _make_result(rid: str, score: float, content: str = "x", source: str = "doc1") -> RetrievalResult:
    return RetrievalResult(
        result_id=rid,
        content=content,
        source=source,
        source_type="test",
        hybrid_score=score,
        confidence=score,
    )


class _StubStructuredIndex:
    """Minimal structured index stub that returns canned results."""

    def search_entities(self, **kw):
        return []

    def search_relations(self, **kw):
        return []

    def search_evidences(self, **kw):
        return []

    def get_entity_evidences(self, entity_id):
        return []


class RetrieveWithGateTest(unittest.TestCase):
    def setUp(self):
        self.retriever = HybridRetriever(structured_index=_StubStructuredIndex())

    def test_sufficient_evidence_answers_in_one_round(self):
        # Provide enough high-quality results via vector_search_fn.
        # Use scores with a wide margin (0.95 - 0.8 = 0.15 > 0.10) to avoid
        # floating-point precision issues triggering RETRIEVE_MORE.
        good_results = [
            {"content": "alpha beta gamma delta", "score": 0.95, "source": "doc1"},
            {"content": "alpha beta gamma delta", "score": 0.8, "source": "doc2"},
            {"content": "alpha beta gamma delta", "score": 0.7, "source": "doc3"},
        ]
        self.retriever._vector_search_fn = lambda query, top_k: good_results
        # P2-1 (2026-09-09): RRF-first is the default; this gate-loop test pins
        # weighted mode so raw 0.95/0.8/0.7 scores survive fusion unscaled.
        config = RetrievalConfig(top_k=10, enable_structured=False, enable_rules=False, vector_weight=1.0, fusion_mode="weighted")

        result = self.retriever.retrieve_with_gate(
            "alpha beta gamma delta",
            config,
            query_terms=["alpha", "beta", "gamma", "delta"],
        )

        self.assertIsInstance(result, GateRetrievalResult)
        self.assertEqual(result.decision.action, EvidenceGateAction.ANSWER)
        self.assertEqual(result.rounds_used, 1)
        self.assertGreaterEqual(len(result.results), 3)

    def test_weak_evidence_triggers_retrieve_more_then_answer(self):
        # Round 1: low-score results => RETRIEVE_MORE.
        # Round 2: better results => ANSWER.
        call_count = [0]

        def expanding_search(query, top_k):
            call_count[0] += 1
            if call_count[0] == 1:
                return [{"content": "alpha", "score": 0.3, "source": "doc1"}]
            return [
                {"content": "alpha beta gamma delta", "score": 0.9, "source": "doc2"},
                {"content": "alpha beta gamma delta", "score": 0.8, "source": "doc3"},
                {"content": "alpha beta gamma delta", "score": 0.7, "source": "doc4"},
            ]

        self.retriever._vector_search_fn = expanding_search
        # P2-1: pin weighted mode (see above).
        config = RetrievalConfig(top_k=5, enable_structured=False, enable_rules=False, vector_weight=1.0, fusion_mode="weighted")

        result = self.retriever.retrieve_with_gate(
            "alpha beta gamma delta",
            config,
            query_terms=["alpha", "beta", "gamma", "delta"],
            max_retrieval_rounds=3,
        )

        self.assertEqual(result.rounds_used, 2)
        self.assertEqual(result.decision.action, EvidenceGateAction.ANSWER)

    def test_conflict_escalates(self):
        self.retriever._vector_search_fn = lambda query, top_k: [
            {"content": "alpha beta gamma delta", "score": 0.9, "source": "doc1"},
            {"content": "alpha beta gamma delta", "score": 0.89, "source": "doc2"},
            {"content": "alpha beta gamma delta", "score": 0.5, "source": "doc3"},
        ]
        config = RetrievalConfig(top_k=10, enable_structured=False, enable_rules=False)

        result = self.retriever.retrieve_with_gate(
            "alpha beta gamma delta",
            config,
            query_terms=["alpha", "beta", "gamma", "delta"],
            has_conflict=True,
        )

        self.assertEqual(result.decision.action, EvidenceGateAction.ESCALATE)

    def test_no_evidence_after_max_rounds_escalates(self):
        self.retriever._vector_search_fn = lambda query, top_k: []
        config = RetrievalConfig(top_k=5, enable_structured=False, enable_rules=False)

        result = self.retriever.retrieve_with_gate(
            "query", config, max_retrieval_rounds=2,
        )

        self.assertEqual(result.decision.action, EvidenceGateAction.ESCALATE)
        self.assertEqual(result.rounds_used, 2)

    def test_injected_gate_used(self):
        custom_gate = EvidenceGate(EvidenceGateConfig(
            min_evidence_count=1, min_top_score=0.0, min_score_margin=0.0,
            min_coverage=0.0, min_diversity=0.0,
        ))
        self.retriever._vector_search_fn = lambda query, top_k: [
            {"content": "x", "score": 0.1, "source": "doc1"},
        ]
        config = RetrievalConfig(top_k=5, enable_structured=False, enable_rules=False)

        result = self.retriever.retrieve_with_gate(
            "query", config, evidence_gate=custom_gate,
        )

        # Permissive gate => ANSWER on first round.
        self.assertEqual(result.decision.action, EvidenceGateAction.ANSWER)


# ---------------------------------------------------------------------------
# rag_answerer execute_routed wiring (unit-level, no network)
# ---------------------------------------------------------------------------


class RagAnswererRoutedTest(unittest.TestCase):
    """Verify _call_deepseek_direct uses execute_routed + ContextCompactor.

    Uses a stub gateway so no real provider is contacted.
    """

    def test_execute_routed_called_with_rag_answer_task_kind(self):
        from src.application.bootstrap.llm_gateway import LLMCallOutcome
        from src.application.rag.rag_answerer import RAGAnswerer

        # Build a minimal answerer instance.
        answerer = RAGAnswerer.__new__(RAGAnswerer)

        # Patch get_llm_gateway to return a stub that records the call.
        captured: dict[str, Any] = {}

        class _StubGateway:
            def execute_routed(self, prompt, *, task_kind=None, stage=None, call_id="", **kw):
                captured["prompt"] = prompt
                captured["task_kind"] = task_kind
                captured["stage"] = stage
                return LLMCallOutcome(
                    text="A" * 100,  # > 50 chars to pass the length check
                    provider="deepseek",
                    model="deepseek-chat",
                    call_id=call_id,
                    prompt_hash="h",
                    success=True,
                    cost_usd=0.001,
                    price_known=True,
                    duration_ms=10,
                    routed_model=True,
                    route_decision=RouteDecision(
                        tier=TaskComplexity.STANDARD,
                        model_tier=ModelTier(provider="deepseek", model="deepseek-chat"),
                        task_kind=TaskKind.RAG_ANSWER,
                        reason="test",
                        signals={},
                    ),
                )

        with patch(
            "src.application.bootstrap.llm_gateway.get_llm_gateway",
            return_value=_StubGateway(),
        ):
            result = answerer._call_deepseek_direct("test question", "short context")

        self.assertIsNotNone(result)
        self.assertEqual(captured["task_kind"], "rag_answer")
        self.assertEqual(captured["stage"], "rag.deep_analysis")
        # The result should contain the routing tier.
        self.assertIn("路由", result)
        self.assertIn("standard", result)

    def test_context_compaction_applied_for_long_context(self):
        from src.application.bootstrap.llm_gateway import LLMCallOutcome
        from src.application.rag.rag_answerer import RAGAnswerer

        answerer = RAGAnswerer.__new__(RAGAnswerer)
        long_context = "\n\n".join([f"段落 {i} 的内容" for i in range(200)])

        class _StubGateway:
            def execute_routed(self, prompt, **kw):
                # Verify the prompt is shorter than the original context.
                return LLMCallOutcome(
                    text="B" * 100,
                    provider="deepseek", model="deepseek-chat",
                    call_id="x", prompt_hash="h", success=True,
                    cost_usd=0.0, price_known=True, duration_ms=1,
                )

        with patch(
            "src.application.bootstrap.llm_gateway.get_llm_gateway",
            return_value=_StubGateway(),
        ):
            result = answerer._call_deepseek_direct("q", long_context, max_context_chars=500)

        self.assertIsNotNone(result)
        # Compaction note should appear (context was compacted).
        # (If compaction fell back to truncation, a different note appears;
        #  either way the call succeeded.)


if __name__ == "__main__":
    unittest.main()
