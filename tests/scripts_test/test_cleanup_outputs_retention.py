from __future__ import annotations

"""Tests for scripts/cleanup_outputs_retention.py (F-2, 2026-08-15).

Locks the retention policy: newest N runs kept, plus any run carrying a
``.keep`` marker (preserves approved/failed runs intentionally).
"""

import tempfile
import unittest
from pathlib import Path

from scripts.cleanup_outputs_retention import collect_runs, keep_set


class RetentionTest(unittest.TestCase):
    def _make_runs(self, names):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        for name in names:
            (root / name).mkdir()
        return root

    def test_newest_n_kept(self):
        root = self._make_runs(["MAIN-20260815-003000", "MAIN-20260815-002000", "MAIN-20260815-001000"])
        runs = collect_runs(root)
        self.assertEqual([r.name for r in runs], ["MAIN-20260815-003000", "MAIN-20260815-002000", "MAIN-20260815-001000"])
        kept = keep_set(runs, keep=2)
        kept_names = [r.name for r in runs if r in kept]
        self.assertEqual(kept_names, ["MAIN-20260815-003000", "MAIN-20260815-002000"])

    def test_keep_marker_preserves_old_run(self):
        root = self._make_runs(["MAIN-20260815-003000", "MAIN-20260815-001000", "MAIN-20260701-000000"])
        (root / "MAIN-20260701-000000" / ".keep").touch()
        runs = collect_runs(root)
        kept = keep_set(runs, keep=1)
        kept_names = {r.name for r in runs if r in kept}
        self.assertIn("MAIN-20260815-003000", kept_names)
        self.assertIn("MAIN-20260701-000000", kept_names)  # marked keep
        self.assertNotIn("MAIN-20260815-001000", kept_names)

    def test_empty_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.assertEqual(collect_runs(root), [])
            self.assertEqual(keep_set([], keep=5), set())


if __name__ == "__main__":
    unittest.main()

    def test_apply_deletes_victims_dry_run_deletes_nothing(self):
        """2026-10-16 P0-3：--apply 真删（空操作回归锁），dry-run 不删。"""
        import sys
        from scripts.cleanup_outputs_retention import main
        root = self._make_runs(["MAIN-20260815-003000", "MAIN-20260815-002000",
                                "MAIN-20260815-001000"])
        argv = sys.argv
        try:
            sys.argv = ["cleanup", f"--root={root}", "--keep=2"]
            self.assertEqual(main(), 0)
            self.assertTrue((root / "MAIN-20260815-001000").exists())
            sys.argv = ["cleanup", f"--root={root}", "--keep=2", "--apply"]
            self.assertEqual(main(), 0)
            self.assertFalse((root / "MAIN-20260815-001000").exists())
            self.assertTrue((root / "MAIN-20260815-003000").exists())
        finally:
            sys.argv = argv
            self._tmp.cleanup()
