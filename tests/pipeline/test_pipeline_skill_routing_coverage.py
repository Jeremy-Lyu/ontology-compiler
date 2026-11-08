"""Route-config vs executor coverage test.

Locks the fact-source invariant behind `config/pipeline_skill_routing.json`:
every non-empty skill id routed by the pipeline must be resolvable by the
pipeline skill executor (`SkillPipelineBridge`). Without this, a stage route
can silently reference a skill that is not a pipeline-executable skill (e.g.
the L2.5 prompt-only skills in `skills/l25/*.yaml`), producing a stable
"Skill not found" execution error on every matching run.

History:
- 2026-08-13: routing config was routing `l25_mbom_extraction` /
  `l25_logistics_extraction` to the extract stage. Those ids are prompt-only
  skills owned by the extraction kernel (L25SkillLoader), not pipeline
  executor skills, so every MANUFACTURING/SUPPLY_CHAIN extract raised
  `Skill not found`. Config fixed (v1.1.0) and this test added to prevent
  regression.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROUTING_CONFIG = PROJECT_ROOT / "config" / "pipeline_skill_routing.json"


def _all_routed_skill_ids() -> set[str]:
    payload = json.loads(ROUTING_CONFIG.read_text(encoding="utf-8"))
    ids: set[str] = set()
    for rule in payload.get("routing_rules", []):
        ids.update(rule.get("skills") or [])
    for rule in payload.get("fallback_rules", []):
        ids.update(rule.get("skills") or [])
    return ids


class PipelineSkillRoutingCoverageTest(unittest.TestCase):
    def test_routing_config_is_valid_json(self):
        payload = json.loads(ROUTING_CONFIG.read_text(encoding="utf-8"))
        self.assertIn("routing_rules", payload)

    def test_non_empty_routed_skills_are_registered_executor_skills(self):
        routed = _all_routed_skill_ids()
        # Skip when no skills are routed (the current fixed state).
        if not routed:
            return
        from src.pipeline.skill_integration import get_skill_pipeline_bridge

        registered = set(get_skill_pipeline_bridge().list_skills())
        missing = sorted(routed - registered)
        self.assertEqual(
            missing,
            [],
            "routing config references pipeline skills that the executor cannot "
            f"resolve: {missing}. Prompt-only skills (e.g. skills/l25/*.yaml) "
            "are owned by the extraction kernel, not the pipeline skill executor.",
        )


if __name__ == "__main__":
    unittest.main()
