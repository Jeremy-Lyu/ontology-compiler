from __future__ import annotations

"""C-5 characterization tests: RAG evidence gate.

Covers:
- signal extraction: count, top_score, score_margin, coverage, diversity
- ANSWER when evidence is sufficient
- RETRIEVE_MORE on weak signals (low count / top_score / coverage / margin)
- ESCALATE on conflict + tight margin
- ESCALATE after max retrieval rounds with no evidence
- ESCALATE after max retrieval rounds with weak signals
- coverage = 1.0 when no query terms supplied (no artificial demand)
- diversity degrades gracefully when source_ids missing
"""

import unittest

from src.application.rag.evidence_gate import (
    EvidenceGate,
    EvidenceGateAction,
    EvidenceGateConfig,
    EvidenceItem,
    compute_signals,
    get_evidence_gate,
    set_evidence_gate,
)


def _item(text: str, score: float = 0.5, source_id: str = "") -> EvidenceItem:
    return EvidenceItem(text=text, score=score, source_id=source_id)


class SignalExtractionTest(unittest.TestCase):
    def test_count_and_top_score(self):
        sig = compute_signals([_item("a", 0.7), _item("b", 0.4)])
        self.assertEqual(sig.evidence_count, 2)
        self.assertAlmostEqual(sig.top_score, 0.7)

    def test_score_margin_between_top_two(self):
        sig = compute_signals([_item("a", 0.8), _item("b", 0.5)])
        self.assertAlmostEqual(sig.score_margin, 0.3)

    def test_score_margin_single_item(self):
        sig = compute_signals([_item("a", 0.8)])
        self.assertEqual(sig.score_margin, 0.8)  # top - 0

    def test_coverage_with_query_terms(self):
        sig = compute_signals(
            [_item("the inventory cycle stock")],
            query_terms=["inventory", "cycle", "missing"],
        )
        # 2 of 3 terms covered
        self.assertAlmostEqual(sig.coverage, 2 / 3)

    def test_coverage_one_when_no_query_terms(self):
        sig = compute_signals([_item("anything")])
        self.assertEqual(sig.coverage, 1.0)

    def test_diversity_distinct_sources(self):
        sig = compute_signals(
            [_item("a", source_id="doc1"), _item("b", source_id="doc1"), _item("c", source_id="doc2")]
        )
        self.assertAlmostEqual(sig.diversity, 2 / 3)

    def test_diversity_zero_when_no_sources(self):
        sig = compute_signals([_item("a"), _item("b")])
        self.assertEqual(sig.diversity, 0.0)

    def test_score_clamped_to_unit_interval(self):
        sig = compute_signals([_item("a", score=2.5)])
        self.assertEqual(sig.top_score, 1.0)


class AnswerDecisionTest(unittest.TestCase):
    def test_sufficient_evidence_answers(self):
        gate = EvidenceGate()
        evidence = [
            _item("inventory cycle stock safety", score=0.9, source_id="doc1"),
            _item("cycle stock benchmark", score=0.7, source_id="doc2"),
            _item("safety stock vsm", score=0.6, source_id="doc3"),
        ]
        decision = gate.evaluate(evidence, query_terms=["inventory", "cycle", "safety"])
        self.assertEqual(decision.action, EvidenceGateAction.ANSWER)
        self.assertEqual(decision.reason, "sufficient-evidence")


class RetrieveMoreTest(unittest.TestCase):
    def test_low_count_triggers_retrieve_more(self):
        gate = EvidenceGate()
        decision = gate.evaluate([_item("a", score=0.9)], query_terms=["a"])
        self.assertEqual(decision.action, EvidenceGateAction.RETRIEVE_MORE)

    def test_low_top_score_triggers_retrieve_more(self):
        gate = EvidenceGate()
        evidence = [
            _item("a b c d e", score=0.2, source_id="doc1"),
            _item("a b c d e", score=0.15, source_id="doc2"),
            _item("a b c d e", score=0.1, source_id="doc3"),
        ]
        decision = gate.evaluate(evidence, query_terms=["a", "b", "c", "d", "e"])
        self.assertEqual(decision.action, EvidenceGateAction.RETRIEVE_MORE)
        self.assertIn("top_score", decision.reason)

    def test_low_coverage_triggers_retrieve_more(self):
        gate = EvidenceGate()
        evidence = [
            _item("alpha", score=0.9, source_id="doc1"),
            _item("alpha", score=0.8, source_id="doc2"),
            _item("alpha", score=0.7, source_id="doc3"),
        ]
        decision = gate.evaluate(evidence, query_terms=["alpha", "beta", "gamma"])
        # coverage = 1/3 < 0.60
        self.assertEqual(decision.action, EvidenceGateAction.RETRIEVE_MORE)
        self.assertIn("coverage", decision.reason)

    def test_tight_margin_triggers_retrieve_more(self):
        gate = EvidenceGate()
        evidence = [
            _item("a b c d e f", score=0.9, source_id="doc1"),
            _item("a b c d e f", score=0.89, source_id="doc2"),
            _item("a b c d e f", score=0.5, source_id="doc3"),
        ]
        decision = gate.evaluate(evidence, query_terms=["a", "b", "c", "d", "e", "f"])
        # margin 0.01 < 0.10
        self.assertEqual(decision.action, EvidenceGateAction.RETRIEVE_MORE)
        self.assertIn("margin", decision.reason)


class EscalateTest(unittest.TestCase):
    def test_conflict_with_tight_margin_escalates(self):
        gate = EvidenceGate()
        evidence = [
            _item("a b c d e", score=0.9, source_id="doc1"),
            _item("a b c d e", score=0.89, source_id="doc2"),
            _item("a b c d e", score=0.5, source_id="doc3"),
        ]
        decision = gate.evaluate(evidence, query_terms=["a", "b", "c", "d", "e"], has_conflict=True)
        self.assertEqual(decision.action, EvidenceGateAction.ESCALATE)
        self.assertIn("conflict", decision.reason)

    def test_no_evidence_after_max_rounds_escalates(self):
        gate = EvidenceGate()
        decision = gate.evaluate([], retrieval_round=3, max_retrieval_rounds=3)
        self.assertEqual(decision.action, EvidenceGateAction.ESCALATE)
        self.assertEqual(decision.reason, "no-evidence-after-max-rounds")

    def test_weak_signals_after_max_rounds_escalates(self):
        gate = EvidenceGate()
        evidence = [
            _item("a", score=0.2, source_id="doc1"),
            _item("a", score=0.1, source_id="doc2"),
        ]
        decision = gate.evaluate(evidence, retrieval_round=3, max_retrieval_rounds=3)
        self.assertEqual(decision.action, EvidenceGateAction.ESCALATE)
        self.assertIn("weak-signals-after-max-rounds", decision.reason)


class CustomConfigTest(unittest.TestCase):
    def test_custom_thresholds_change_decision(self):
        # Very permissive config: should ANSWER even on weak evidence.
        cfg = EvidenceGateConfig(
            min_evidence_count=1,
            min_top_score=0.0,
            min_score_margin=0.0,
            min_coverage=0.0,
            min_diversity=0.0,
        )
        gate = EvidenceGate(cfg)
        decision = gate.evaluate([_item("a", score=0.1)])
        self.assertEqual(decision.action, EvidenceGateAction.ANSWER)


class SingletonTest(unittest.TestCase):
    def tearDown(self):
        set_evidence_gate(None)

    def test_default_singleton(self):
        self.assertIsInstance(get_evidence_gate(), EvidenceGate)

    def test_inject_custom(self):
        custom = EvidenceGate(EvidenceGateConfig(min_evidence_count=42))
        set_evidence_gate(custom)
        self.assertIs(get_evidence_gate(), custom)


if __name__ == "__main__":
    unittest.main()
