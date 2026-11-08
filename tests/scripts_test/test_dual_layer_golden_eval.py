"""Regression checks for the Phase B dual-layer golden eval (extract vs merge).

双层评估口径 (2026-08-29): golden entities/relations are evaluated twice —
Layer 1 against the raw extraction output (stage "extract"), Layer 2 against
the post-merge master file (``stage_results["merge"].data.master_path``) — plus
a retention derivative (merge recall / extract recall) isolating merge loss.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_regression_samples import (
    _aggregate_layered_golden,
    _evaluate_sample,
    _render_layered_golden_lines,
)

GOLDEN = {
    "entities": [
        {"name": "MBOM", "meta_class": "System", "layer": "L1", "aliases": ["制造BOM"]},
        {"name": "EBOM", "meta_class": "System", "layer": "L1", "aliases": []},
    ],
    "relations": [
        {"source": "MBOM", "target": "EBOM", "types": ["DEPENDS_ON"]},
    ],
}


def _mk_payload(*, extract_has_mbom: bool, merge_has_mbom: bool, master_file: Path) -> dict:
    extract_nodes = []
    if extract_has_mbom:
        extract_nodes.append({"id": "N1", "name": "MBOM", "meta_class": "System", "layer": "L1"})
    extract_nodes.append({"id": "N2", "name": "EBOM", "meta_class": "System", "layer": "L1"})
    extract_rels = [{"source": "N1", "target": "N2", "relation_type": "DEPENDS_ON"}] if extract_has_mbom else []
    master_nodes = []
    if merge_has_mbom:
        master_nodes.append({"id": "M1", "name": "MBOM", "meta_class": "System", "layer": "L1"})
    master_nodes.append({"id": "M2", "name": "EBOM", "meta_class": "System", "layer": "L1"})
    master_rels = [{"source": "M1", "target": "M2", "relation_type": "DEPENDS_ON"}] if merge_has_mbom else []
    master_file.write_text(
        json.dumps({"nodes": master_nodes, "instances": [], "relationships": master_rels}, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "success": True,
        "status": "SUCCESS",
        "stage_results": {
            "extract": {"success": True, "data": {"nodes": extract_nodes, "instances": [], "relationships": extract_rels}},
            "merge": {"success": True, "data": {"master_path": str(master_file)}},
        },
        "golden_semantic": GOLDEN,
    }


class DualLayerGoldenEvalTest(unittest.TestCase):
    def test_merge_loss_isolated_by_retention(self):
        with tempfile.TemporaryDirectory() as td:
            master = Path(td) / "master_ontology_unified.json"
            payload = _mk_payload(extract_has_mbom=True, merge_has_mbom=False, master_file=master)
            ev = _evaluate_sample(payload, {}, {})
            layered = ev["layered_golden"]
            self.assertEqual(layered["extract"]["entity_recall"], 1.0)
            self.assertEqual(layered["merge"]["entity_recall"], 0.5)  # EBOM survives in master
            self.assertEqual(layered["retention"]["entity_retention"], 0.5)
            # legacy mixed metrics still present and computed over ALL stages
            names = [c["name"] for c in ev["checks"]]
            self.assertIn("golden_entity_recall", names)
            self.assertEqual(ev["golden"]["entity_recall"], 1.0)

    def test_no_threshold_means_observational_only(self):
        with tempfile.TemporaryDirectory() as td:
            master = Path(td) / "master_ontology_unified.json"
            payload = _mk_payload(extract_has_mbom=False, merge_has_mbom=False, master_file=master)
            ev = _evaluate_sample(payload, {}, {})
            layered_names = [c["name"] for c in ev["checks"]]
            self.assertFalse(
                any(n.startswith(("extract_", "merge_", "entity_retention")) for n in layered_names),
                layered_names,
            )

    def test_threshold_gates_layered_check(self):
        with tempfile.TemporaryDirectory() as td:
            master = Path(td) / "master_ontology_unified.json"
            payload = _mk_payload(extract_has_mbom=False, merge_has_mbom=False, master_file=master)
            ev = _evaluate_sample(payload, {}, {"extract_entity_recall_min": 0.9})
            checks = {c["name"]: c for c in ev["checks"]}
            self.assertIn("extract_entity_recall", checks)
            self.assertFalse(checks["extract_entity_recall"]["passed"])

    def test_aggregation_and_rendering(self):
        with tempfile.TemporaryDirectory() as td:
            master = Path(td) / "master_ontology_unified.json"
            payload = _mk_payload(extract_has_mbom=True, merge_has_mbom=False, master_file=master)
            ev = _evaluate_sample(payload, {}, {})
            results = [
                {"evaluation": {"layered_golden": ev["layered_golden"]}},
                {"evaluation": {"layered_golden": ev["layered_golden"]}},
            ]
            agg = _aggregate_layered_golden(results)
            self.assertIn("extract_entity_recall", agg["metrics"])
            lines = _render_layered_golden_lines({"layered_golden_summary": agg})
            self.assertTrue(any("Golden Dual-Layer Eval" in ln for ln in lines))

    def test_missing_master_path_does_not_crash(self):
        with tempfile.TemporaryDirectory() as td:
            master = Path(td) / "master_ontology_unified.json"
            payload = _mk_payload(extract_has_mbom=True, merge_has_mbom=True, master_file=master)
            payload["stage_results"]["merge"]["data"].pop("master_path")
            ev = _evaluate_sample(payload, {}, {})
            self.assertIsNone(ev["layered_golden"]["merge"])
            self.assertIsNone(ev["layered_golden"]["retention"])
            # extract layer still measured
            self.assertEqual(ev["layered_golden"]["extract"]["entity_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
