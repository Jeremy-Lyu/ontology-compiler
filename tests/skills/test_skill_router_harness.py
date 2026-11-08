"""SkillsRouter discovers via harness — no skills.skill_registry."""
from __future__ import annotations

import unittest


class SkillRouterHarnessTest(unittest.TestCase):
    def test_router_imports_without_legacy_registry(self):
        from skills.skill_router import SkillsRouter

        router = SkillsRouter()
        self.assertTrue(hasattr(router.skill_registry, "list_skills"))
        skills = router.route({"meta_class": "System", "domain": "QUALITY"})
        self.assertIn("ontology_validator", skills)

    def test_legacy_skill_registry_module_is_archived(self):
        with self.assertRaises(ImportError):
            import skills.skill_registry  # noqa: F401


if __name__ == "__main__":
    unittest.main()
