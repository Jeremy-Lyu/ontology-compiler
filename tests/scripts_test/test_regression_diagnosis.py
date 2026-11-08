"""Tests for regression diagnosis helpers."""
from __future__ import annotations

import unittest

from scripts.regression_diagnosis import master_dominance_signal, summarize_regression_failures


class RegressionDiagnosisTest(unittest.TestCase):
    def _summary(self):
        return {"results": [
            {"sample_id": "a", "pass_fail": "FAIL", "evaluation": {"observed": {"entity_count": 405, "relation_count": 682}, "checks": [{"name": "golden_entity_recall", "passed": False}], "golden": {"entity_recall": 0.4, "relation_recall": 1.0, "unmatched_entities": [{"name": "X"}], "unmatched_relations": []}}},
            {"sample_id": "b", "pass_fail": "FAIL", "evaluation": {"observed": {"entity_count": 408, "relation_count": 685}, "checks": [{"name": "golden_relation_recall", "passed": False}], "golden": {"entity_recall": 1.0, "relation_recall": 0.0, "unmatched_entities": [], "unmatched_relations": [{"source": "A"}]}}},
            {"sample_id": "c", "pass_fail": "PASS", "evaluation": {"observed": {"entity_count": 405, "relation_count": 682}, "checks": [], "golden": {}}},
        ]}

    def test_failure_summary_is_compact(self):
        row = summarize_regression_failures(self._summary())[0]
        self.assertEqual(row["failed_checks"], ["golden_entity_recall"])
        self.assertEqual(row["unmatched_entities"], [{"name": "X"}])

    def test_stable_counts_emit_diagnostic_signal(self):
        report = master_dominance_signal(self._summary())
        self.assertTrue(report["suspiciously_stable"])
        self.assertEqual(report["entity_span"], 3)
        self.assertEqual(report["relation_span"], 3)


if __name__ == "__main__":
    unittest.main()
