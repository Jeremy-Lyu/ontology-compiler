"""Tests for regression summary artifact diagnostics integration."""
from __future__ import annotations

import unittest
from pathlib import Path

from scripts.regression_artifact_diagnosis import source_breakdown, trace_relation_type


class RegressionArtifactDiagnosisIntegrationTest(unittest.TestCase):
    def test_real_ebom_extract_has_no_derived_from(self):
        root = Path("outputs/runs/ontology/pipeline/REG-ebom-mbom-2023-20260827-170733/artifacts")
        extracts = list(root.glob("*-extract-bundle.json"))
        relations = list(root.glob("*-relationship-bundle.json"))
        if not extracts or not relations:
            self.skipTest("real regression artifacts are not present")
        import json
        extract = json.loads(extracts[0].read_text())
        relationship = json.loads(relations[0].read_text())
        result = trace_relation_type(extract, relationship, "DERIVED_FROM")
        self.assertEqual(result["classification"], "missing_at_extract")
        self.assertEqual(result["extract_count"], 0)

    def test_source_breakdown_handles_real_extract_shape(self):
        root = Path("outputs/runs/ontology/pipeline/REG-ebom-mbom-2023-20260827-170733/artifacts")
        extracts = list(root.glob("*-extract-bundle.json"))
        if not extracts:
            self.skipTest("real regression artifacts are not present")
        import json
        result = source_breakdown(json.loads(extracts[0].read_text()))
        self.assertGreater(result["entity_total"], 0)
        self.assertGreater(result["relation_total"], 0)


if __name__ == "__main__":
    unittest.main()
