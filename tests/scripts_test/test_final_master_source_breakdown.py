"""Tests for final master source breakdown."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.regression_artifact_diagnosis import (
    master_source_breakdown,
    source_breakdown,
    summarize_sources,
)


class FinalMasterSourceBreakdownTest(unittest.TestCase):
    def test_origin_kind_precedes_heuristics(self):
        stage = {"metadata": {"stage_result_data": {
            "nodes": [
                {"origin_kind": "master_reused", "source_file": "old.md"},
                {"origin_kind": "input", "source_file": "new.md"},
                {"origin_kind": "injected", "auto_generated": True},
            ],
            "instances": [],
            "relationships": [{"origin_kind": "derived"}],
        }}}
        result = source_breakdown(stage)
        self.assertEqual(result["entities_by_source"], {"master_reused": 1, "input": 1, "injected": 1})
        self.assertEqual(result["relations_by_source"], {"derived": 1})

    def test_injected_share_and_related_to(self):
        summary = summarize_sources(
            [{"origin_kind": "input"}, {"origin_kind": "injected"}],
            [
                {"origin_kind": "input", "relation_type": "DERIVED_FROM"},
                {"origin_kind": "injected", "relation_type": "RELATED_TO"},
                {"origin_kind": "input", "relation_type": "RELATED_TO"},
            ],
        )
        self.assertEqual(summary["injected_entity_share"], 0.5)
        self.assertEqual(summary["extracted_relation_share"], round(2 / 3, 4))
        self.assertEqual(summary["related_to_count"], 2)
        self.assertEqual(summary["related_to_share"], round(2 / 3, 4))

    def test_master_dir_breakdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nodes.json").write_text(
                '[{"name":"A","origin_kind":"input"},{"name":"B","origin_kind":"injected"}]',
                encoding="utf-8",
            )
            (root / "relationships.json").write_text(
                '[{"relation_type":"PART_OF","origin_kind":"injected"}]',
                encoding="utf-8",
            )
            summary = master_source_breakdown(root)
            self.assertEqual(summary["entity_total"], 2)
            self.assertEqual(summary["injected_relation_share"], 1.0)


if __name__ == "__main__":
    unittest.main()
