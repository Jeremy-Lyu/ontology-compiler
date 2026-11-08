"""2026-10-16: compat usage 扫描口径锁。

背景：`check_compat_usage_gate` 反复因**新增测试**引用 canonical API 而假红
（本会话 3 次：33→35、35→37、37→38，均为 `tests/**` 新增引用）。

修正后的语义：`total_references` 只计**生产引用**（retirement-debt 的真实风险面）；
`tests/` 的引用计入 `test_references` / `test_files`，仅作可见性报告，
不影响门禁“只减不增”判定——因为测试无法“迁移离开”被测 API。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from compat_usage_stats import scan  # noqa: E402


class CompatScanSemanticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stats = scan()

    def test_production_files_exclude_tests(self):
        for name, st in self.stats.items():
            leaked = [f for f in st["files"] if f.startswith("tests/")]
            self.assertEqual(leaked, [], f"{name}: tests/ 引用泄漏进 total_references")

    def test_test_refs_are_reported_separately(self):
        for name, st in self.stats.items():
            self.assertIn("test_files", st)
            self.assertIn("test_references", st)
            for f in st["test_files"]:
                self.assertTrue(f.startswith("tests/"), f"{name}: test_files 含非测试文件 {f}")

    def test_skill_registry_harness_has_test_only_refs_isolated(self):
        """该目标正是被假红 3 次的：其测试引用必须落在 test_* 而非生产计数里。"""
        st = self.stats.get("skill_registry.harness")
        self.assertIsNotNone(st)
        self.assertGreater(st["test_references"], 0)
        self.assertFalse(any(f.startswith("tests/") for f in st["files"]))


if __name__ == "__main__":
    unittest.main()
