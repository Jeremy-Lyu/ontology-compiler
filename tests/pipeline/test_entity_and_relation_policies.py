"""Entity-layer & relation-rewrite policy tests (G7, B-7, 2026-08-17).

Locks the extracted MainPipeline business decisions as pure functions:
- entity_layer_policy: layer inference, entity-layer index, profile summary,
  metadata projection, domain-context inference, layer-pair gating
- relation_rewrite_policy: RELATED_TO candidate collection / dedupe / finalize
- stage_args: Namespace assembly shape
"""
from __future__ import annotations

import unittest
from argparse import Namespace
from pathlib import Path

from src.pipeline import entity_layer_policy as elp
from src.pipeline import relation_rewrite_policy as rrp
from src.pipeline.stage_args import build_stage_args


class InferEntityLayerTest(unittest.TestCase):
    def test_keeps_explicit_layer(self):
        entity = {"name": "Something", "type": "System", "layer": "L1"}
        self.assertEqual(elp.infer_entity_layer(entity), entity)

    def test_infers_l3_for_instance_token(self):
        entity = {"name": "Plant A benchmark", "type": "System"}
        out = elp.infer_entity_layer(entity)
        self.assertEqual(out["layer"], "L3")

    def test_infers_l25_for_bridge_token(self):
        entity = {"name": "Reporting Interface", "type": "System"}
        out = elp.infer_entity_layer(entity)
        self.assertEqual(out["layer"], "L2.5")

    def test_infers_l2_for_plain_type(self):
        entity = {"name": "Something", "type": "System"}
        out = elp.infer_entity_layer(entity)
        self.assertEqual(out["layer"], "L2")

    def test_no_layer_when_no_type(self):
        entity = {"name": "Something"}
        out = elp.infer_entity_layer(entity)
        self.assertNotIn("layer", out)


class EntityLayerIndexTest(unittest.TestCase):
    def test_index_builds_id_to_layer(self):
        data = {
            "nodes": [
                {"id": "n1", "layer": "L1", "type": "X"},
                {"entity_id": "n2", "layer": "L2", "type": "Y"},
            ]
        }
        idx = elp.build_entity_layer_index(data)
        self.assertEqual(idx, {"n1": "L1", "n2": "L2"})


class ProfileSummaryTest(unittest.TestCase):
    def test_summary_counts_types_and_layers(self):
        entities = [
            {"name": "A", "type": "System", "layer": "L1", "aliases": ["a"]},
            {"name": "B", "type": "System", "layer": "L1", "evidence_span": "x", "confidence": 0.9},
        ]
        summary = elp.build_entity_profile_summary(entities)
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["types"], {"System": 2})
        self.assertEqual(summary["layers"], {"L1": 2})
        self.assertEqual(summary["with_aliases"], 1)
        self.assertEqual(summary["with_evidence"], 1)
        # Original semantics: completeness = avg populated-completeness-key
        # count per entity (entity A: aliases=1; entity B: evidence+confidence=2)
        self.assertAlmostEqual(summary["profile_completeness"], 3 / 2)


class MetadataProjectionTest(unittest.TestCase):
    def test_projects_known_keys_and_domain_context(self):
        metadata = {
            "effective_dates": ["2025"],
            "source_hints": ["inventory plant"],
            "rewrite_context": {"k": "v"},
        }
        out = elp.project_structured_metadata(metadata)
        self.assertEqual(out["effective_dates"], ["2025"])
        self.assertEqual(out["rewrite_context"], {"k": "v"})
        self.assertEqual(out["domain_context"]["operational_context"], "supply_chain_inventory")

    def test_infer_domain_context_bom(self):
        out = elp.infer_domain_metadata_context({"source_hints": ["mbom bom variant"]})
        self.assertEqual(out.get("manufacturing_context"), "bom_centric")


class LayerPairGateTest(unittest.TestCase):
    def test_blocked_pair_via_engine(self):
        class _Gov:
            def is_layer_pair_allowed(self, s, t):
                return s == "L1" and t == "L2"

        rel = {"source_layer": "L3", "target_layer": "L1"}
        self.assertTrue(elp.is_layer_pair_blocked(rel, governance_engine=_Gov()))

        rel_ok = {"source_layer": "L1", "target_layer": "L2"}
        self.assertFalse(elp.is_layer_pair_blocked(rel_ok, governance_engine=_Gov()))

    def test_missing_layers_not_blocked(self):
        self.assertFalse(elp.is_layer_pair_blocked({"type": "X"}, governance_engine=object()))


class RelationRewritePolicyTest(unittest.TestCase):
    def test_rewrite_depends_on(self):
        rel = {"type": "RELATED_TO", "text_blob": "system B depends on system A"}
        out = rrp.rewrite_related_to(rel)
        self.assertEqual(out["type"], "DEPENDS_ON")
        self.assertEqual(out["rewritten_from"], "RELATED_TO")
        self.assertIn("DEPENDS_ON", out["rewrite_candidates"])

    def test_rewrite_supports_via_source_types(self):
        rel = {"type": "RELATED_TO", "source_type": "actor", "target_type": "system", "text": "x supports y"}
        out = rrp.rewrite_related_to(rel)
        self.assertEqual(out["type"], "SUPPORTS")

    def test_finalize_preserves_non_related(self):
        ordered = []
        out = rrp.finalize_related_to_relation(
            rel={"type": "RELATED_TO"},
            ordered_candidates=ordered,
            source_type="",
            target_type="",
            source_name="",
            target_name="",
        )
        self.assertNotIn("rewritten_from", out)  # no candidate -> no rewrite

    def test_dedupe_keeps_order(self):
        self.assertEqual(rrp.dedupe_ordered_candidates(["A", "B", "A", "C"]), ["A", "B", "C"])


class StageArgsTest(unittest.TestCase):
    def test_build_stage_args_shape(self):
        class _Ctx:
            domain = "SUPPLY_CHAIN"
            source_path = "data/sample.pptx"
            output_dir = "/tmp/out"
            fast_mode = False
            enable_rigor_extract = False
            enable_llm_conflict_resolver = True
            regression_mode = False
            metadata = Namespace(strict_governance=True)

        class _Paths:
            run_root = Path("/tmp/run")
            stage_root = Path("/tmp/run/stage")
            file_root = Path("/tmp/run/files")
            artifact_root = Path("/tmp/run/artifacts")

        args = build_stage_args(
            context=_Ctx(),
            run_paths=_Paths(),
            run_id="R1",
            raw_context_output_dir=None,
            project_root=Path("/repo"),
        )
        self.assertEqual(args.stage, "full")
        self.assertEqual(args.domain, "SUPPLY_CHAIN")
        self.assertEqual(args.run_id, "R1")
        self.assertTrue(args.strict_governance)
        self.assertFalse(args.no_backup)
        self.assertEqual(args.fast_mode, False)
        self.assertEqual(args.enable_llm_resolver, True)


if __name__ == "__main__":
    unittest.main()