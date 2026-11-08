"""P1-3 (2026-09-01) characterization tests: C-5 telemetry in the baseline.

Locks the behavior that:
1. ``LLMMonitor.record_route`` aggregates task_kind -> tier counts and
   ``get_summary`` exposes them as ``route_decisions``.
2. ``LLMGateway.execute_routed`` records one routing decision per call.
3. ``EvidenceGate.evaluate`` records ANSWER / RETRIEVE_MORE / ESCALATE
   decision counts plus a last-decision snapshot.
4. The regression runner merges per-sample telemetry into the baseline
   fields ``route_decisions`` / ``evidence_signals``.

All tests are hermetic: no network, no real LLM calls, no file I/O.
"""
from __future__ import annotations

import unittest
from typing import Any

from src.application.rag.evidence_gate import (
    EvidenceGate,
    EvidenceGateAction,
    EvidenceItem,
    get_evidence_gate,
)
from src.application.services.llm_monitor import get_llm_monitor


# ---------------------------------------------------------------------------
# 1. LLMMonitor route telemetry
# ---------------------------------------------------------------------------


class MonitorRouteDecisionsTest(unittest.TestCase):
    def test_record_route_aggregates_counts(self):
        from src.application.services.llm_monitor import LLMMonitor

        monitor = LLMMonitor()
        monitor.record_route(task_kind="classification", tier="light")
        monitor.record_route(task_kind="classification", tier="light")
        monitor.record_route(task_kind="rag_answer", tier="standard")
        self.assertEqual(
            monitor.get_summary()["route_decisions"],
            {"classification": {"light": 2}, "rag_answer": {"standard": 1}},
        )

    def test_record_route_tolerates_none_as_unknown(self):
        from src.application.services.llm_monitor import LLMMonitor

        monitor = LLMMonitor()
        monitor.record_route(task_kind=None, tier=None)  # type: ignore[arg-type]
        self.assertEqual(
            monitor.get_summary()["route_decisions"],
            {"unknown": {"unknown": 1}},
        )

    def test_reset_clears_route_decisions(self):
        from src.application.services.llm_monitor import LLMMonitor

        monitor = LLMMonitor()
        monitor.record_route(task_kind="classification", tier="light")
        monitor.reset()
        self.assertEqual(monitor.get_summary()["route_decisions"], {})


# ---------------------------------------------------------------------------
# 2. Gateway wiring: execute_routed records the route
# ---------------------------------------------------------------------------


class _FakeClient:
    def call(self, prompt: str, temperature: Any = None, max_tokens: Any = None) -> str:
        return "ok"


class _FakeBootstrap:
    def __init__(self, client: _FakeClient) -> None:
        self._client = client
        from src.application.bootstrap.llm_bootstrap import LLMProviderType

        self.config = type("Cfg", (), {"primary_provider": LLMProviderType.DEEPSEEK})()

    def get_client(self, provider: Any) -> Any:
        return self._client


class GatewayRecordsRouteTest(unittest.TestCase):
    def setUp(self):
        from src.application.bootstrap.llm_gateway import LLMGateway

        get_llm_monitor().reset()
        self.gateway = LLMGateway(bootstrap=_FakeBootstrap(_FakeClient()))  # type: ignore[arg-type]

    def tearDown(self):
        get_llm_monitor().reset()

    def test_execute_routed_records_route(self):
        self.gateway.execute_routed("Classify scene type", task_kind="classification")
        decisions = get_llm_monitor().get_summary()["route_decisions"]
        self.assertEqual(decisions.get("classification", {}).get("light"), 1)

    def test_execute_routed_records_conflict_escalation(self):
        from src.application.services.model_router import TaskKind

        self.gateway.execute_routed(
            "two candidates conflict",
            task_kind=TaskKind.ENTITY_EXTRACTION,
            has_conflict=True,
        )
        decisions = get_llm_monitor().get_summary()["route_decisions"]
        self.assertEqual(decisions.get("entity_extraction", {}).get("heavy"), 1)


# ---------------------------------------------------------------------------
# 3. EvidenceGate decision telemetry
# ---------------------------------------------------------------------------


class EvidenceGateTelemetryTest(unittest.TestCase):
    def setUp(self):
        self.gate = EvidenceGate()
        self.gate.reset_decision_counts()

    def _answer_evidence(self) -> list[EvidenceItem]:
        # Satisfies every default threshold: count=3, top=0.9, margin=0.5,
        # coverage=1.0 (term "alpha" present), diversity=3/3.
        return [
            EvidenceItem(text="alpha content", score=0.9, source_id="doc1"),
            EvidenceItem(text="beta content", score=0.4, source_id="doc2"),
            EvidenceItem(text="gamma content", score=0.3, source_id="doc3"),
        ]

    def test_evaluate_records_answer_count(self):
        decision = self.gate.evaluate(self._answer_evidence(), query_terms=("alpha",))
        self.assertEqual(decision.action, EvidenceGateAction.ANSWER)
        signals = self.gate.get_evidence_signals()
        self.assertEqual(signals["decision_counts"], {"answer": 1})

    def test_evaluate_records_retrieve_more_count(self):
        decision = self.gate.evaluate([EvidenceItem(text="thin", score=0.1, source_id="d1")])
        self.assertEqual(decision.action, EvidenceGateAction.RETRIEVE_MORE)
        signals = self.gate.get_evidence_signals()
        self.assertEqual(signals["decision_counts"], {"retrieve_more": 1})

    def test_last_decision_snapshot_is_json_ready(self):
        self.gate.evaluate(self._answer_evidence(), query_terms=("alpha",))
        last = self.gate.get_evidence_signals()["last"]
        self.assertIsNotNone(last)
        self.assertEqual(last["action"], "answer")
        self.assertEqual(last["reason"], "sufficient-evidence")
        self.assertIsInstance(last["signals"], dict)

    def test_reset_decision_counts(self):
        self.gate.evaluate(self._answer_evidence(), query_terms=("alpha",))
        self.gate.reset_decision_counts()
        self.assertEqual(self.gate.get_evidence_signals(), {"decision_counts": {}, "last": None})

    def test_singleton_gate_shares_telemetry(self):
        gate = get_evidence_gate()
        try:
            gate.reset_decision_counts()
            gate.evaluate(self._answer_evidence(), query_terms=("alpha",))
            self.assertEqual(gate.get_evidence_signals()["decision_counts"], {"answer": 1})
        finally:
            gate.reset_decision_counts()


# ---------------------------------------------------------------------------
# 4. Runner: per-sample collection + batch merge + baseline payload
# ---------------------------------------------------------------------------


class RunnerTelemetryAggregationTest(unittest.TestCase):
    def test_merge_route_decisions(self):
        from scripts.run_regression_samples import _merge_route_decisions

        results = [
            {"llm_cost": {"route_decisions": {"classification": {"light": 2}}}},
            {"llm_cost": {"route_decisions": {"classification": {"light": 1, "heavy": 1}}}},
            {"llm_cost": {}},
        ]
        self.assertEqual(
            _merge_route_decisions(results),
            {"classification": {"light": 3, "heavy": 1}},
        )

    def test_merge_route_decisions_empty(self):
        from scripts.run_regression_samples import _merge_route_decisions

        self.assertEqual(_merge_route_decisions([]), {})

    def test_merge_evidence_signals(self):
        from scripts.run_regression_samples import _merge_evidence_signals

        results = [
            {"llm_cost": {"evidence_signals": {"decision_counts": {"answer": 1}}}},
            {
                "llm_cost": {
                    "evidence_signals": {
                        "decision_counts": {"answer": 2, "retrieve_more": 1},
                        "last": {"action": "answer"},
                    }
                }
            },
        ]
        # Batch merge keeps counts only; per-sample "last" is not merged.
        self.assertEqual(
            _merge_evidence_signals(results),
            {"decision_counts": {"answer": 3, "retrieve_more": 1}},
        )

    def test_baseline_payload_includes_telemetry(self):
        from scripts.run_regression_samples import _baseline_payload

        summary = {
            "manifest": "test.json",
            "executed_at": "2026-09-01T00:00:00",
            "count": 1,
            "passed_count": 1,
            "failed_count": 0,
            "overall_status": "PASS",
            "thresholds": {},
            "results": [
                {
                    "llm_cost": {
                        "total_calls": 1,
                        "total_tokens": 100,
                        "total_cost_estimate": 0.001,
                        "total_duration_ms": 50,
                        "unknown_price_calls": 0,
                        "route_decisions": {"classification": {"light": 1}},
                        "evidence_signals": {"decision_counts": {"answer": 1}},
                    },
                },
            ],
        }
        payload = _baseline_payload(summary)
        self.assertIn("route_decisions", payload)
        self.assertIn("evidence_signals", payload)
        self.assertEqual(payload["route_decisions"], {"classification": {"light": 1}})
        self.assertEqual(payload["evidence_signals"], {"decision_counts": {"answer": 1}})

    def test_collect_llm_cost_summary_includes_telemetry(self):
        from scripts.run_regression_samples import _collect_llm_cost_summary

        monitor = get_llm_monitor()
        gate = get_evidence_gate()
        monitor.reset()
        gate.reset_decision_counts()
        try:
            monitor.record_route(task_kind="classification", tier="light")
            gate.evaluate(
                [EvidenceItem(text="thin", score=0.1, source_id="d1")],
            )
            result = _collect_llm_cost_summary()
            self.assertIn("route_decisions", result)
            self.assertEqual(result["route_decisions"].get("classification", {}).get("light"), 1)
            self.assertIn("evidence_signals", result)
            self.assertEqual(
                result["evidence_signals"].get("decision_counts", {}).get("retrieve_more"), 1
            )
        finally:
            monitor.reset()
            gate.reset_decision_counts()

    def test_collect_llm_cost_summary_tolerates_empty_monitor(self):
        from scripts.run_regression_samples import _collect_llm_cost_summary

        get_llm_monitor().reset()
        get_evidence_gate().reset_decision_counts()
        try:
            result = _collect_llm_cost_summary()
            self.assertEqual(result["route_decisions"], {})
            self.assertEqual(result["evidence_signals"], {"decision_counts": {}, "last": None})
        finally:
            get_llm_monitor().reset()


if __name__ == "__main__":
    unittest.main()
