"""Tests for detailed regression artifact diagnostics."""
from __future__ import annotations

import unittest

from scripts.regression_artifact_diagnosis import source_breakdown, trace_relation_type


class RegressionArtifactDiagnosisTest(unittest.TestCase):
    def test_source_breakdown_uses_provenance(self):
        stage = {"metadata": {"stage_result_data": {"nodes": [
            {"meta_id": "a", "provenance": {"stage": "extract"}},
            {"meta_id": "b", "attributes": [{"name": "auto_generated", "value": "true"}]},
        ], "relationships": [{"from": "a", "to": "b", "provenance": {"stage": "extract"}}]}}}
        result = source_breakdown(stage)
        self.assertEqual(result["entities_by_source"]["input"], 1)
        self.assertEqual(result["entities_by_source"]["injected"], 1)
        self.assertEqual(result["relations_by_source"]["input"], 1)

    def test_relation_type_absence_is_classified_at_extract(self):
        extract = {"metadata": {"stage_result_data": {"relationships": [{"type": "DEPENDS_ON"}]}}}
        governed = {"metadata": {"stage_result_data": {"violations": []}}}
        result = trace_relation_type(extract, governed, "DERIVED_FROM")
        self.assertEqual(result["extract_count"], 0)
        self.assertEqual(result["classification"], "missing_at_extract")

    def test_relation_type_present_is_not_mislabeled(self):
        extract = {"metadata": {"stage_result_data": {"relationships": [{"type": "DERIVED_FROM", "from_name": "EBOM", "to_name": "BOM"}]}}}
        governed = {"metadata": {"stage_result_data": {"violations": [{"relationship_type": "DERIVED_FROM"}]}}}
        result = trace_relation_type(extract, governed, "DERIVED_FROM")
        self.assertEqual(result["extract_count"], 1)
        self.assertEqual(result["governance_violation_count"], 1)
        self.assertEqual(result["classification"], "present_at_extract_review_governance_or_merge")


if __name__ == "__main__":
    unittest.main()
