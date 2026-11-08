"""P2-3 (2026-10-16): skill manifest 完整性棘轮。

背景（本次整改的事实发现）：

- 仓库 discovery 出 **29 个 skill**，其中 **24 个 `script_path` 指向不存在的文件**
  （`scripts/pipeline/`、`scripts/diagnostics/`、`scripts/maintenance/` 三个目录在
  git 全历史中从未存在）。
- 这 24 个 **全部 `enabled=True`**。
- **更正（2026-10-16 复查）**：路由由 `config/pipeline_skill_routing.json` 驱动，
  且无任何代码按 `SkillDefinition.stages` 选择 skill（`get_by_stage` 为死代码）。
  因此这 24 个 phantom skill 实际 **inert**——注册为 executor 但永不被选中。
  真实风险降级为：① preflight 上报的 `skills_count` 虚高；② 若将来新增路由规则
  指向 phantom 才会炸。
- 长期存量（非 `6ecccdc` 引入），因此以**棘轮**方式锁定：phantom 只减不增，
  evaluable 只增不减。修好一个就从基线移除，防止债务回涨。
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from ontology_compiler.harness.skill_registry import build_default_skill_registry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE = PROJECT_ROOT / "config" / "quality" / "skill_manifest_integrity_baseline.json"


class SkillManifestIntegrityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        cls.reg = build_default_skill_registry()
        cls.skills = cls.reg.list_skills(enabled_only=False)

    def _phantom(self) -> set[str]:
        return {
            s.name
            for s in self.skills
            if not (s.script_path and os.path.isfile(s.script_path))
        }

    def test_phantom_skills_do_not_grow(self):
        allowed = set(self.baseline["phantom_skills"])
        current = self._phantom()
        new = sorted(current - allowed)
        self.assertEqual(
            new, [],
            f"新增 phantom skill（script_path 指向不存在的文件）：{new}。"
            "要么修正 script_path，要么明确加入基线（需说明理由）。",
        )

    def test_phantom_skills_actually_shrink_over_time(self):
        """若已修复若干项，基线必须同步收窄——防止棘轮失去意义。"""
        allowed = set(self.baseline["phantom_skills"])
        current = self._phantom()
        stale = sorted(allowed - current)
        self.assertEqual(
            stale, [],
            f"基线含已不再 phantom 的 skill：{stale}。"
            "请从 config/quality/skill_manifest_integrity_baseline.json 移除，锁定成果。",
        )

    def test_evaluable_count_does_not_regress(self):
        current = len(self.reg.evaluable_skill_names(enabled_only=False))
        self.assertGreaterEqual(
            current, self.baseline["evaluable_count"],
            "可评估 skill 数下降：某 skill.json 的 output_schema 被清空或 manifest 解析退化。",
        )

    def test_total_skill_count_does_not_regress(self):
        self.assertGreaterEqual(
            len(self.skills), self.baseline["total_skills"],
            "skill discovery 数量下降：可能是 manifest 校验失败被跳过（查 get_discovery_errors）。",
        )

    def test_every_evaluable_skill_reports_no_violation_for_empty_output(self):
        """口径锁：空输出对含 required 的 schema 必须报缺失，证明 evaluator 真的接上而不是恒空。"""
        checked = 0
        for name in self.reg.evaluable_skill_names(enabled_only=False):
            skill = self.reg.get(name)
            schema = skill.output_schema or {}
            if not schema.get("required"):
                continue
            self.assertTrue(
                self.reg.evaluate_output(name, {}),
                f"{name}: 声明了 required 但空输出未报违规，evaluator 可能未生效",
            )
            checked += 1
        self.assertGreater(checked, 0, "没有任何带 required 的可评估 skill，检查基线是否失效")

    def test_every_routed_skill_has_an_existing_script(self):
        """真正的风险面：被路由配置选中的 skill 必须能执行。"""
        routing = PROJECT_ROOT / "config" / "pipeline_skill_routing.json"
        if not routing.exists():
            self.skipTest("routing config absent")
        cfg = json.loads(routing.read_text(encoding="utf-8"))
        routed: set[str] = set()
        for rule in (cfg.get("routing_rules") or []) + (cfg.get("fallback_rules") or []):
            routed.update(rule.get("skills") or [])

        by_name = {s.name: s for s in self.skills}
        broken = []
        for name in sorted(routed):
            skill = by_name.get(name)
            if skill is None:
                broken.append(f"{name} (未 discovery)")
            elif not (skill.script_path and os.path.isfile(skill.script_path)):
                broken.append(f"{name} (script 缺失: {skill.script_path})")
        self.assertEqual(broken, [], f"路由配置引用了无法执行的 skill：{broken}")

    @staticmethod
    def _error_keys() -> set[str]:
        return {
            f"{e['path'].split('/ontology_20260702/', 1)[-1]} :: {e['error']}"
            for e in build_default_skill_registry().get_discovery_errors()
        }

    def test_discovery_errors_do_not_grow(self):
        """manifest 校验失败被静默跳过的 skill：只减不增。

        2026-10-16 存量 16 项：14 项 category 误用目录名
        （business_orchestration / perception / semantic_organization / cross_layer）、
        2 项缺 `command`。修复方式需领域决策（放宽白名单 vs 改语义 category），
        故以棘轮锁定而非擅自修改。
        """
        allowed = set(self.baseline["discovery_errors"])
        new = sorted(self._error_keys() - allowed)
        self.assertEqual(new, [], f"新增 discovery 错误：{new}")

    def test_discovery_errors_actually_shrink_over_time(self):
        allowed = set(self.baseline["discovery_errors"])
        stale = sorted(allowed - self._error_keys())
        self.assertEqual(
            stale, [],
            f"基线含已修复的 discovery 错误：{stale}。请从基线移除，锁定成果。",
        )


if __name__ == "__main__":
    unittest.main()
