"""P2-3c (2026-10-16): skill registry 健康报告。

契约锁：
- collect() 四类信息齐全且与棘轮基线一致
- check() 对"新增 phantom / 新增 discovery error / evaluable 回退 / 路由断裂"各报一条
- check() 基线一致时为空（--check 退出 0）
- 渲染不含 traceback、含四节标题
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import report_skill_registry_health as mod  # noqa: E402


class CollectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = mod.collect()

    def test_all_sections_present(self):
        for key in (
            "total_skills", "enabled_skills", "evaluable_skills", "evaluable_count",
            "phantom_skills", "discovery_errors", "routed_skills", "routing_broken",
        ):
            self.assertIn(key, self.report)

    def test_matches_ratchet_baseline(self):
        base = json.loads(mod.BASELINE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            sorted(self.report["phantom_skills"]), sorted(base["phantom_skills"]),
            "phantom 清单与棘轮基线不一致 → 基线已过期",
        )
        self.assertEqual(
            sorted(self.report["discovery_errors"]), sorted(base["discovery_errors"]),
            "discovery_errors 与棘轮基线不一致 → 基线已过期",
        )
        self.assertGreaterEqual(self.report["evaluable_count"], base["evaluable_count"])

    def test_known_counts(self):
        self.assertEqual(self.report["total_skills"], 29)
        self.assertGreaterEqual(self.report["evaluable_count"], 18)
        self.assertEqual(self.report["routing_broken"], [])


class CheckTest(unittest.TestCase):
    def _base(self):
        return {
            "phantom_skills": ["a", "b"],
            "discovery_errors": ["e1"],
            "evaluable_count": 5,
        }

    def test_no_new_debt_passes(self):
        report = {
            "phantom_skills": ["a", "b"],
            "discovery_errors": ["e1"],
            "evaluable_count": 5,
            "routing_broken": [],
        }
        self.assertEqual(mod.check(report, baseline=self._base()), [])

    def test_new_phantom_reported(self):
        report = dict(self._base(), phantom_skills=["a", "b", "new"], routing_broken=[])
        problems = mod.check(report, baseline=self._base())
        self.assertEqual(len(problems), 1)
        self.assertIn("phantom", problems[0])
        self.assertIn("new", problems[0])

    def test_new_discovery_error_reported(self):
        report = dict(self._base(), discovery_errors=["e1", "e2"], routing_broken=[])
        problems = mod.check(report, baseline=self._base())
        self.assertIn("discovery error", problems[0])

    def test_evaluable_regression_reported(self):
        report = dict(self._base(), evaluable_count=4, routing_broken=[])
        problems = mod.check(report, baseline=self._base())
        self.assertEqual(len(problems), 1)
        self.assertIn("回退", problems[0])

    def test_routing_broken_reported_even_when_baseline_clean(self):
        report = dict(self._base(), routing_broken=["stage_preflight (script 缺失)"])
        problems = mod.check(report, baseline=self._base())
        self.assertTrue(any("路由" in p for p in problems))


class RenderTest(unittest.TestCase):
    def test_render_has_sections_and_no_traceback(self):
        report = mod.collect()
        text = mod._render(report)
        for marker in ("[phantom]", "[discovery errors]", "[routed]", "[evaluable]"):
            self.assertIn(marker, text)
        self.assertNotIn("Traceback", text)


class CliTest(unittest.TestCase):
    def test_cli_json_and_check_exit_zero(self):
        import subprocess

        out = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "report_skill_registry_health.py"),
             "--json", "--check"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=120,
        )
        self.assertEqual(out.returncode, 0, out.stdout[-2000:] + out.stderr[-2000:])
        payload = json.loads(out.stdout)
        self.assertIn("total_skills", payload)


if __name__ == "__main__":
    unittest.main()
