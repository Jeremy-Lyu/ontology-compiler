from __future__ import annotations

"""Characterization tests for run_extract (execution plan Phase 1).

Locks the current extract-stage behavior for the core paths: single-file
success, folder with mixed success/failure, missing source, and empty folder.
The LLM entry point (compile_source_file), document cache, and L25 skill
loader are stubbed so the test characterizes aggregation/normalization
semantics only, without network or real extraction.
"""

import asyncio
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock


class _FakeCache:
    def get_cached_result(self, path, version_bundle=None):
        return None

    def cache_result(self, *args, **kwargs):
        return None


class _FakeSkillLoader:
    def get_stats(self):
        return {"skills_count": 0}

    def find_matching_skills(self, text):
        return []


def _success_result(nodes, relationships=None):
    return {
        "status": "SUCCESS",
        "data": {
            "nodes": nodes,
            "instances": [],
            "relationships": relationships or [],
        },
    }


def _failed_result(message="boom"):
    return {"status": "FAILED", "error": message}


def _run(args):
    return asyncio.run(run_extract(args))


from src.pipeline.stages.extract import run_extract


class RunExtractCharacterizationTest(unittest.TestCase):
    """Characterize run_extract aggregation and status logic."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        os.environ["EXTRACT_ENABLE_CANDIDATE_GEN"] = "0"
        self._patches = [
            mock.patch("src.pipeline.stages.extract.get_l25_skill_loader", return_value=_FakeSkillLoader()),
            mock.patch("src.pipeline.stages.extract.DocumentCache", return_value=_FakeCache()),
        ]
        for p in self._patches:
            p.start()
        self.addCleanup(self._tear_down_patches)

    def _tear_down_patches(self):
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()

    def _args(self, source) -> Namespace:
        return Namespace(
            file=str(source) if source.is_file() else None,
            folder=str(source) if source.is_dir() else None,
            domain="SUPPLY_CHAIN",
            run_id="EXTRACT-CHAR",
            output_dir=str(self.root / "output"),
        )

    def _args_file(self, file_path: Path) -> Namespace:
        # Explicit file path even when it does not exist (exercises the
        # "source not found" branch instead of the missing-arg branch).
        return Namespace(
            file=str(file_path),
            folder=None,
            domain="SUPPLY_CHAIN",
            run_id="EXTRACT-CHAR",
            output_dir=str(self.root / "output"),
        )

    @staticmethod
    def _node_by_id(nodes, meta_id):
        for n in nodes:
            if n.get("meta_id") == meta_id:
                return n
        return None

    def test_single_file_success_aggregates_and_writes_fragment(self):
        source = self.root / "a.md"
        source.write_text("some markdown content", encoding="utf-8")
        node = {"meta_id": "L1-001", "name": "MBOM", "layer": "L1", "meta_class": "Data"}

        with mock.patch(
            "ontology_compiler.compile_source_file",
            new=mock.AsyncMock(return_value=_success_result([node], [])),
        ):
            result = _run(self._args(source))

        self.assertTrue(result.success, result.message)
        self.assertEqual(result.stage, "extract")
        self.assertEqual(result.data["run_id"], "EXTRACT-CHAR")
        self.assertEqual(result.data["domain"], "SUPPLY_CHAIN")
        # Behavior lock: single MBOM node triggers deterministic BOM backfill
        # and MBOM -> BOM DERIVED_FROM safety-net relation.
        self.assertEqual(len(result.data["relationships"]), 1)
        self.assertEqual(result.data["relationships"][0]["type"], "DERIVED_FROM")
        self.assertEqual(result.data["relationships"][0]["to_name"], "BOM")
        # Behavior lock: run_extract prepends the combined fragment
        # (fragment_<run_id>.json) to fragment_paths in addition to the
        # per-file fragment, so the list carries 2 entries (artifact_paths
        # aliases the same list).
        self.assertEqual(len(result.data["fragment_paths"]), 2)
        self.assertEqual(len(result.artifact_paths), 2)
        self.assertTrue(Path(result.artifact_paths[0]).exists())
        self.assertIn("processed 1/1 files", result.message)
        # Contract: data["nodes"] is a FLAT list of dicts (not nested by layer).
        nodes = result.data["nodes"]
        self.assertTrue(all(isinstance(n, dict) for n in nodes), "nodes must be flat dicts")
        # Behavior lock: 11 L0 base classes + 1 input node + 1 backfilled BOM.
        self.assertEqual(len(nodes), 13)
        self.assertEqual(sum(1 for n in nodes if n.get("layer") == "L0"), 11)
        l1 = self._node_by_id(nodes, "L1-001")
        self.assertIsNotNone(l1)
        # Behavior lock: alias rewrite "mbom" -> "MBOM" via node optimizations.
        self.assertEqual(l1.get("name"), "MBOM")

    def test_folder_mixed_success_and_failure(self):
        (self.root / "b.md").write_text("b content", encoding="utf-8")
        (self.root / "c.md").write_text("c content", encoding="utf-8")
        ok_node = {"meta_id": "L1-010", "name": "SAP", "layer": "L1", "meta_class": "System"}

        async def fake_compile(path, domain=None, document_metadata=None):
            if str(path).endswith("b.md"):
                return _success_result([ok_node], [])
            return _failed_result("extraction failed")

        with mock.patch("ontology_compiler.compile_source_file", new=fake_compile):
            result = _run(self._args(self.root))

        self.assertFalse(result.success)
        # Only the successful file's node survives aggregation (flat nodes).
        nodes = result.data["nodes"]
        self.assertTrue(all(isinstance(n, dict) for n in nodes))
        self.assertIsNotNone(self._node_by_id(nodes, "L1-010"))
        # Behavior lock: combined fragment is prepended alongside the per-file
        # fragment (see test_single_file_success_aggregates_and_writes_fragment).
        self.assertEqual(len(result.data["fragment_paths"]), 2)
        self.assertEqual(len(result.artifact_paths), 2)
        self.assertIn("processed 1/2 files", result.message)
        self.assertIn("failed 1", result.message)

    def test_missing_source_returns_failure(self):
        missing = self.root / "missing.md"
        result = _run(self._args_file(missing))
        self.assertFalse(result.success)
        self.assertIn("source not found", result.message)

    def test_empty_folder_returns_failure(self):
        empty = self.root / "empty"
        empty.mkdir()
        result = _run(self._args(empty))
        self.assertFalse(result.success)
        self.assertIn("no supported files found", result.message)


if __name__ == "__main__":
    unittest.main()