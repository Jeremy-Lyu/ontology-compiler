"""SkillRegistry fact-source characterization tests.

Locks the current SkillRegistry ownership model (Phase 4 fact-source
unification) so the reconciliation batch can proceed safely:

- canonical DISCOVERY: `ontology_compiler.harness.skill_registry`
  (`build_default_skill_registry`, reads `skills/**/.ontology-skill/skill.json`).
- runtime singleton `ontology_compiler.skills.registry.get_skill_registry`
  holds metadata only — its `get()` returns None for metadata-only skills and
  its `list_active()` currently claims metadata-only skills as active (this is
  the misleading behavior the bridge must not propagate).
- `SkillPipelineBridge` is the pipeline runtime entry point; its `list_skills()`
  must reflect the real, executable executors, not the metadata-only set.
"""
from __future__ import annotations

import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class HarnessDiscoveryCharacterization(unittest.TestCase):
    """Canonical discovery via build_default_skill_registry."""

    @classmethod
    def setUpClass(cls):
        from ontology_compiler.harness.skill_registry import build_default_skill_registry

        cls.registry = build_default_skill_registry(PROJECT_ROOT)
        cls.all_skills = cls.registry.list_skills(enabled_only=False)
        cls.enabled_skills = cls.registry.list_skills(enabled_only=True)

    def test_discovers_manifest_skills(self):
        # At least the core workflow/ontology skills must be discovered.
        names = {s.name for s in self.all_skills}
        for expected in (
            "ontology_merger",
            "ontology_validator",
            "consensus_resolver",
            "drift_detector",
            "entity_classification",
            "capsule_publisher",
        ):
            self.assertIn(expected, names)

    def test_disabled_skills_excluded_from_enabled_list(self):
        enabled_names = {s.name for s in self.enabled_skills}
        for s in self.all_skills:
            if not s.enabled:
                self.assertNotIn(s.name, enabled_names)

    def test_every_discovered_skill_has_description(self):
        for s in self.all_skills:
            self.assertTrue(s.description, f"skill {s.name} missing description")

    def test_get_returns_registered_definition(self):
        sample = self.enabled_skills[0] if self.enabled_skills else None
        if sample:
            self.assertIs(self.registry.get(sample.name), sample)


class RuntimeSingletonCharacterization(unittest.TestCase):
    """The runtime singleton holds metadata only; get() is None for those."""

    @classmethod
    def setUpClass(cls):
        from ontology_compiler.skills.registry import get_skill_registry, reset_skill_registry

        reset_skill_registry()
        cls.registry = get_skill_registry()
        cls.loaded = cls.registry.load_metadata_from_skill_json(str(PROJECT_ROOT / "skills"))

    def test_metadata_loaded_from_skill_json(self):
        self.assertGreater(self.loaded, 0)

    def test_metadata_only_skills_not_executable(self):
        # get() must return None because no runtime Skill instances are
        # registered in production — metadata presence is NOT executability.
        for name in self.registry.list_active()[:5]:
            self.assertIsNone(self.registry.get(name))

    def test_list_active_claims_metadata_only_as_active(self):
        # Lock the CURRENT misleading behavior so the reconciliation batch can
        # change it deliberately (metadata-only counted as active).
        self.assertEqual(
            len(self.registry.list_active()),
            self.loaded,
            "list_active currently counts all metadata-only skills as active",
        )


class PipelineBridgeCharacterization(unittest.TestCase):
    """SkillPipelineBridge must reflect real executors, not metadata-only ids."""

    def test_list_skills_matches_executors(self):
        # The invariant: what the bridge advertises as skills must actually be
        # executable. Currently list_skills() leaks metadata-only ids from the
        # runtime singleton, so this assertion FAILS until the reconciliation
        # batch fixes the bridge to report _skill_executors.
        from src.pipeline.skill_integration import get_skill_pipeline_bridge

        bridge = get_skill_pipeline_bridge()
        bridge.load_skills()
        advertised = set(bridge.list_skills())
        executable = set(bridge._skill_executors.keys())
        self.assertTrue(executable, "no executors loaded")
        self.assertLessEqual(
            len(advertised),
            len(executable),
            "bridge advertises more skills than it can execute",
        )

    def test_executors_populated_by_harness_discovery(self):
        from src.pipeline.skill_integration import get_skill_pipeline_bridge

        bridge = get_skill_pipeline_bridge()
        bridge.load_skills()
        self.assertGreaterEqual(len(bridge._skill_executors), 20)


if __name__ == "__main__":
    unittest.main()
