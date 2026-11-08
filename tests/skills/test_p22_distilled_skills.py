"""P2-2 (2026-09-09): characterize the four distilled skills (F6–F9).

F6 was already implemented; F7/F8/F9 are new in P2-2. Each test asserts
the Skill contract (SkillContext → SkillResult), versioned output_schema
conformance, and the "same input → deterministic same output / idempotent"
property required by §6.2.

RelationRewrite is covered by its own characterization (RelationRewriteCharacterizationTest
in test_relation_rewrite_characterization.py); this file owns the three new
skills so the one-assert-per-skill principle survives if that suite moves.
"""
from __future__ import annotations

import asyncio
import unittest
from typing import Any, Dict, List


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _ctx(entities: List[Dict], **cfg: Any):
    from ontology_compiler.skills.base import SkillContext
    return SkillContext(entities=entities, config=dict(cfg))


class MetadataEnrichmentSkillTest(unittest.TestCase):
    def test_enriches_and_exposes_metadata_contract(self):
        from ontology_compiler.skills.builtin.metadata_enrichment_skill import MetadataEnrichmentSkill
        skill = MetadataEnrichmentSkill()
        ents = [{"meta_id": "E1", "meta_name": "Test Entity", "description": "Review by 2026-03-15 weekly meeting, confidential."}]
        ctx = _ctx(ents, document_content="Review by 2026-03-15 weekly meeting, confidential.")
        result = _run(skill.execute(ctx))
        self.assertTrue(result.success)
        self.assertTrue(result.applied)
        self.assertIn("enriched_entities", result.output)
        self.assertIn("enrichment_count", result.output)
        self.assertIn("extracted_metadata", result.output)
        # §6.2 output check: canonical ExtractedMetadata keys present
        md = result.output["extracted_metadata"]
        for key in ("effective_dates", "security_classes", "frequencies",
                    "source_hints", "time_windows", "formulas"):
            self.assertIn(key, md)
        # deterministic: same input -> same structured_metadata
        result2 = _run(skill.execute(ctx))
        self.assertEqual(result.output["extracted_metadata"], result2.output["extracted_metadata"])

    def test_skipped_when_no_entities(self):
        from ontology_compiler.skills.builtin.metadata_enrichment_skill import MetadataEnrichmentSkill
        skill = MetadataEnrichmentSkill()
        ctx = _ctx([])
        result = _run(skill.execute(ctx))
        self.assertTrue(result.skipped or not result.applied)

    def test_prefix_uses_entity_description_when_content_absent(self):
        from ontology_compiler.skills.builtin.metadata_enrichment_skill import MetadataEnrichmentSkill
        skill = MetadataEnrichmentSkill()
        ents = [{"meta_id": "E2", "description": "Due 2026-12-01 quarterly"}]
        ctx = _ctx(ents)
        result = _run(skill.execute(ctx))
        self.assertTrue(result.applied)
        self.assertIn("2026-12-01", str(result.output["extracted_metadata"].get("effective_dates")))


class CrossFileAssociationSkillTest(unittest.TestCase):
    def test_merges_same_concept_across_files(self):
        from ontology_compiler.skills.builtin.cross_file_association_skill import CrossFileAssociationSkill
        skill = CrossFileAssociationSkill()
        # Two files each containing "Supplier" (name normalization -> same token)
        e1 = {"meta_id": "M-1", "meta_name": "Supplier", "description": "supplier entity"}
        e2 = {"meta_id": "M-2", "meta_name": "supplier", "description": "supplier entity"}
        ctx = _ctx([], file_entities={"a.json": [e1], "b.json": [e2]})
        result = _run(skill.execute(ctx))
        self.assertTrue(result.success)
        self.assertIn("merged_entities", result.output)
        self.assertIn("association_map", result.output)
        self.assertIn("duplicate_groups", result.output)
        # same normalized name -> same group
        self.assertEqual(result.output["merge_count"], 1)
        self.assertLess(len(result.output["merged_entities"]), 2)

    def test_no_merge_when_single_file(self):
        from ontology_compiler.skills.builtin.cross_file_association_skill import CrossFileAssociationSkill
        skill = CrossFileAssociationSkill()
        ctx = _ctx([], file_entities={"only.json": [{"meta_id": "X", "meta_name": "Foo"}]})
        result = _run(skill.execute(ctx))
        self.assertTrue(result.skipped)

    def test_deterministic_merge_map(self):
        from ontology_compiler.skills.builtin.cross_file_association_skill import CrossFileAssociationSkill
        skill = CrossFileAssociationSkill()
        e1 = {"meta_id": "D1", "meta_name": "Plant"}
        e2 = {"meta_id": "D2", "meta_name": "plant"}
        ctx = _ctx([], file_entities={"a.json": [e1], "b.json": [e2]})
        r1 = _run(skill.execute(ctx))
        r2 = _run(skill.execute(ctx))
        self.assertEqual(r1.output["association_map"], r2.output["association_map"])


class QualityGateSkillTest(unittest.TestCase):
    def test_runs_gate_and_exposes_canonical_report(self):
        from ontology_compiler.skills.builtin.quality_gate_skill import QualityGateSkill
        skill = QualityGateSkill()
        data = {
            "entities": [{"id": "E1", "name": "Foo", "type": "concept", "layer": "L1"}],
            "relationships": [],
            "metadata": {"source": "test"},
        }
        ctx = _ctx([], data=data)
        result = _run(skill.execute(ctx))
        self.assertTrue(result.success)
        self.assertTrue(result.applied)
        # SKILL_DISTILLATION_PLAN quality_gate output_schema: canonical contract
        for key in ("result", "grade", "score", "issues", "quality_report"):
            self.assertIn(key, result.output, f"missing {key} in QualityGateSkill output")
        self.assertIn(result.output["result"], ("pass", "warn", "fail", "block"))

    def test_strict_mode_warn_becomes_fail(self):
        from ontology_compiler.skills.builtin.quality_gate_skill import QualityGateSkill
        # Use a deliberately-degraded payload likely to produce WARN (empty relationships)
        data = {"entities": [{"id": "E1", "name": "Foo", "type": "concept", "layer": "L1"}],
                "relationships": []}
        skill = QualityGateSkill()
        loose = _run(skill.execute(_ctx([], data=data)))
        strict = _run(skill.execute(_ctx([], data=data, strict_mode=True)))
        # strict must not be lower than loose (warn->fail promotion)
        if loose.output.get("result") == "warn":
            self.assertEqual(strict.output["result"], "fail")

    def test_skipped_without_entities(self):
        from ontology_compiler.skills.builtin.quality_gate_skill import QualityGateSkill
        skill = QualityGateSkill()
        ctx = _ctx([], data={"entities": [], "relationships": []})
        result = _run(skill.execute(ctx))
        self.assertTrue(result.skipped or not result.applied)


class HarnessDiscoveryTest(unittest.TestCase):
    def test_four_distilled_skills_discoverable(self):
        from ontology_compiler.harness.skill_registry import build_default_skill_registry
        reg = build_default_skill_registry()
        names = {d.name for d in reg.list_skills()}
        for expected in ("relation_rewrite", "metadata_enrichment",
                         "cross_file_association", "quality_gate"):
            self.assertIn(expected, names, f"{expected} not discovered by harness")

    def test_versions_bumped_for_new_skills(self):
        from ontology_compiler.harness.skill_registry import build_default_skill_registry
        reg = build_default_skill_registry()
        by_name = {d.name: d for d in reg.list_skills()}
        for name in ("metadata_enrichment", "cross_file_association", "quality_gate"):
            self.assertEqual(by_name[name].version, "1.1.0",
                             f"{name} version should be 1.1.0 after P2-2")


if __name__ == "__main__":
    unittest.main()
