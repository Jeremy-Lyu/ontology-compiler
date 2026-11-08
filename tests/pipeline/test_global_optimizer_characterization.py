from __future__ import annotations

"""Characterization tests for global_optimizer.optimize_before_master_write.

Locks the behavior of the 30-step post-merge optimizer: stats counters for each
step, master invariants (unique meta_ids, no self-loops/dangling rels, entities
bucket rebuilt, legacy buckets cleared), and in-place mutation semantics. This
is the regression baseline for splitting the 214-line orchestrator.
"""

import unittest
from copy import deepcopy

from src.pipeline.stages.global_optimizer import optimize_before_master_write


def _node(nid: str, name: str, layer: str, meta_class: str, **extra) -> dict:
    node = {
        "meta_id": nid,
        "name": name,
        "layer": layer,
        "meta_class": meta_class,
        "description": f"description of {name} with enough length to score " * 3,
        "attributes": [
            {"name": "attr1", "value": "v1"},
            {"name": "attr2", "value": "v2"},
            {"name": "attr3", "value": "v3"},
        ],
        "evidence": {"content": "evidence content for " + name, "line_range": [1, 5]},
        "provenance": {"source_file": "master.json", "run_id": "RUN-1", "stage": "merge"},
        "source_file": "master.json",
    }
    node.update(extra)
    return node


def _build_master() -> dict:
    return {
        "nodes": [
            _node("L0-001", "BusinessScenario", "L0", "BusinessScenario"),
            _node("L1-001", "ProductBOM", "L1", "Data"),
            # duplicate meta_id pair -> Step 0 fix must kick in
            _node("L2-001", "DuplicatedScenario", "L2", "BusinessScenario"),
            _node("L2-001", "DuplicatedScenario", "L2", "BusinessScenario", description=""),
            # same-layer duplicate (name+layer+meta_class) -> Step 1a
            _node("L2-002", "PlantBOM", "L2", "Data"),
            _node("L2-003", "PlantBOM", "L2", "Data", description=""),
            # cross-layer duplicate (same name, different layer) -> Step 1
            _node("L1-002", "MaterialFlow", "L1", "Process"),
            _node("L2-004", "MaterialFlow", "L2", "Process"),
            # low-quality node (missing everything) -> quality scoring flags it
            {"meta_id": "L2-005", "name": "MinimalNode", "layer": "L2", "meta_class": "Data"},
            # L3 instance stays an instance
            _node("L3-001", "CPT311", "L3", "System"),
        ],
        "instances": [
            {"meta_id": "L3-002", "name": "AGVSystem", "layer": "L3", "meta_class": "System"},
        ],
        "relationships": [
            # valid
            {"source": "L1-001", "target": "L2-004", "relation_type": "contains", "type": "contains"},
            # dangling (target does not exist)
            {"source": "L1-001", "target": "L9-999", "relation_type": "uses", "type": "uses"},
            # self-loop
            {"source": "L2-005", "target": "L2-005", "relation_type": "relates_to", "type": "relates_to"},
            # valid, to L3 instance
            {"source": "L1-001", "target": "L3-001", "relation_type": "realizes", "type": "realizes"},
        ],
    }


class OptimizeBeforeMasterWriteCharacterizationTest(unittest.TestCase):
    """Characterize optimize_before_master_write on a crafted master."""

    def test_optimize_preserves_master_invariants(self):
        master = _build_master()
        original = deepcopy(master)
        result_master, stats = optimize_before_master_write(master, run_id="RUN-CHAR")

        # In-place mutation: same object returned.
        self.assertIs(result_master, master)

        # --- master invariants ---
        nodes = master["nodes"]
        # Every node is a dict with a meta_id.
        self.assertTrue(all(isinstance(n, dict) and str(n.get("meta_id") or "") for n in nodes))
        # No duplicate meta_ids survive.
        node_ids = [str(n.get("meta_id")) for n in nodes]
        self.assertEqual(len(node_ids), len(set(node_ids)), f"duplicate ids: {node_ids}")
        # Relationships have no self-loops.
        for rel in master["relationships"]:
            self.assertNotEqual(str(rel.get("source")), str(rel.get("target")))
        # entities bucket mirrors final nodes+instances; legacy buckets rebuilt
        # from nodes by layer (L0/L1/L2/L2.5), not blindly cleared.
        self.assertIn("entities", master)
        entity_count = sum(len(v) for v in master.get("entities", {}).values() if isinstance(v, list))
        self.assertGreaterEqual(entity_count, len(nodes) + len(master.get("instances", [])))
        # Legacy layer buckets are rebuilt from the final nodes by layer.
        layer_to_legacy = {"L0": "layer0_metadata", "L1": "layer1_concepts",
                           "L2": "layer2_scenarios", "L2.5": "layer25_mapping_rules"}
        for layer, legacy in layer_to_legacy.items():
            self.assertIn(legacy, master)
            expected = [n for n in nodes + master.get("instances", [])
                        if isinstance(n, dict) and str(n.get("layer")) == layer]
            self.assertEqual(
                len(master[legacy]), len(expected),
                f"{legacy} should mirror {layer} nodes"
            )
            if expected:
                self.assertTrue(master[legacy])

        # --- stats schema ---
        for key in (
            "cross_layer_merged", "same_layer_merged", "mser_merged",
            "duplicate_meta_ids_detected", "duplicate_meta_ids_fixed",
            "l25_parent_ids_assigned", "provenance_filled", "evidence_enriched",
            "quality_scored", "low_quality_flagged", "dangling_rels_fixed",
            "relationships_redirected", "relationships_merged", "self_loops_removed",
            "entities_rebuilt", "legacy_buckets_cleared", "id_remap",
        ):
            self.assertIn(key, stats)

        # --- behavior locks: the crafted defects must have been handled ---
        self.assertGreaterEqual(stats["duplicate_meta_ids_fixed"], 1)
        self.assertGreaterEqual(stats["same_layer_merged"], 1)
        self.assertGreaterEqual(stats["cross_layer_merged"], 1)
        self.assertGreaterEqual(stats["provenance_filled"], 1)   # MinimalNode + L3-002
        self.assertGreaterEqual(stats["quality_scored"], 1)
        self.assertGreaterEqual(stats["low_quality_flagged"], 1)  # MinimalNode
        self.assertGreaterEqual(stats["dangling_rels_fixed"] + stats["dangling_rels_removed"], 1)
        self.assertGreaterEqual(stats["self_loops_removed"], 1)
        self.assertGreaterEqual(stats["entities_rebuilt"], 1)

    def test_optimize_empty_master_is_noop(self):
        master = {"nodes": [], "instances": [], "relationships": [], "entities": {}}
        result_master, stats = optimize_before_master_write(master, run_id="RUN-EMPTY")
        self.assertIs(result_master, master)
        self.assertEqual(stats["duplicate_meta_ids_fixed"], 0)
        self.assertEqual(stats["same_layer_merged"], 0)
        self.assertEqual(stats["cross_layer_merged"], 0)
        self.assertEqual(stats["self_loops_removed"], 0)
        self.assertEqual(master["nodes"], [])
        self.assertEqual(master["relationships"], [])
        # entities bucket still rebuilt (may contain empty layer lists).
        self.assertIn("entities", master)

    def test_optimize_returns_stats_breakdown_and_id_remap(self):
        master = _build_master()
        _, stats = optimize_before_master_write(master, run_id="RUN-CHAR")
        # id_remap maps removed ids -> kept ids.
        self.assertIsInstance(stats["id_remap"], dict)
        for old, new in stats["id_remap"].items():
            self.assertIsInstance(old, str)
            self.assertIsInstance(new, str)
        # instances also enriched (L3-002 has no provenance).
        self.assertGreaterEqual(stats["provenance_filled"], 1)


class BackfillMissingEvidenceRefTest(unittest.TestCase):
    """C-5: every relationship must end up with a non-empty evidence_ref.

    The quality gate's relationship_validity is binary for HIGH-severity
    evidence_ref issues, so gold requires 100% coverage.
    """

    def test_backfills_from_source_file(self):
        from src.pipeline.stages.global_optimizer import _backfill_missing_evidence_ref

        nodes = [
            {"meta_id": "L1-001", "source_file": "C:/docs/23w41.pptx"},
            {"meta_id": "L2-100", "source_file": "C:/docs/7.01.docx"},
        ]
        rels = [
            {"source": "L1-001", "target": "L2-100", "relation_type": "CONTAINS"},
        ]
        n = _backfill_missing_evidence_ref(rels, nodes, [])
        self.assertEqual(n, 1)
        self.assertEqual(rels[0]["evidence_ref"], "doc:23w41.pptx")

    def test_backfills_from_target_when_source_unknown(self):
        from src.pipeline.stages.global_optimizer import _backfill_missing_evidence_ref

        nodes = [{"meta_id": "L3-001", "source_file": "C:/docs/x.md"}]
        rels = [
            {"source": "GHOST", "target": "L3-001", "relation_type": "BELONGS_TO"},
        ]
        _backfill_missing_evidence_ref(rels, [], nodes)
        self.assertEqual(rels[0]["evidence_ref"], "doc:x.md")

    def test_synthetic_tag_when_no_entity_source(self):
        from src.pipeline.stages.global_optimizer import _backfill_missing_evidence_ref

        rels = [
            {"source": "L0-999", "target": "L0-998", "relation_type": "RELATED_TO"},
        ]
        _backfill_missing_evidence_ref(rels, [], [])
        self.assertEqual(rels[0]["evidence_ref"], "synthetic:related_to")

    def test_skips_relationships_with_existing_evidence_ref(self):
        from src.pipeline.stages.global_optimizer import _backfill_missing_evidence_ref

        rels = [
            {"source": "A", "target": "B", "relation_type": "USES", "evidence_ref": "doc:real"},
            {"source": "C", "target": "D", "relation_type": "USES"},
        ]
        n = _backfill_missing_evidence_ref(rels, [], [])
        self.assertEqual(n, 1)
        self.assertEqual(rels[0]["evidence_ref"], "doc:real")  # untouched
        self.assertTrue(rels[1]["evidence_ref"])

    def test_optimize_stats_include_evidence_ref_backfilled(self):
        # The orchestrator records the backfill counter in stats.
        master = _build_master()
        _, stats = optimize_before_master_write(master, run_id="RUN-EV")
        self.assertIn("evidence_ref_backfilled", stats)
        self.assertGreaterEqual(stats["evidence_ref_backfilled"], 0)
        # Every surviving relationship has a non-empty evidence_ref.
        for rel in master["relationships"]:
            self.assertTrue(
                str(rel.get("evidence_ref") or "").strip(),
                f"relationship {rel.get('source')}->{rel.get('target')} missing evidence_ref",
            )


if __name__ == "__main__":
    unittest.main()