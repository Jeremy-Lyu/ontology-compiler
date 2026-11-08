from __future__ import annotations

"""Characterization tests for run_merge (execution plan Phase 1).

Locks the current merge behavior for the four core paths: normal merge,
duplicate suppression, master-ID collision remap with relationship reconnect,
and empty input. The global optimizer and master store are dependency-stubbed
so the test characterizes merge semantics only; integration with the optimizer
and the store is covered by the regression / E2E suite.
"""

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from src.pipeline.stages.merge import run_merge


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class RunMergeCharacterizationTest(unittest.TestCase):
    """Characterize run_merge success/failure and data fields."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self._optimizer_patch = mock.patch(
            "src.pipeline.stages.global_optimizer.optimize_before_master_write",
            side_effect=lambda master, run_id: (master, {}),
        )
        self._optimizer_patch.start()
        self.addCleanup(self._optimizer_patch.stop)
        # Hermeticity (P0-4): writing a single-file master derives a temp
        # ``master_ontology/`` dir, and MasterOntologyStore.write() would fire
        # the governed-packet auto-rebuild hook with that *temp* master —
        # spawning a subprocess that writes a tiny packet into the production
        # governed dir and repoints latest_governed_packet.json away from the
        # real 1367-chunk packet (observed 2026-09-01). Stub the hook so merge
        # characterization never touches the production RAG assets.
        self._rebuild_patch = mock.patch(
            "src.pipeline.storage.master_store.MasterOntologyStore._post_write_rebuild_governed_packet"
        )
        self._rebuild_patch.start()
        self.addCleanup(self._rebuild_patch.stop)

    def _args(self, fragment_path: Path, master_path: Path) -> Namespace:
        return Namespace(
            output_dir=str(self.root / "output"),
            master_path=str(master_path),
            fragment=str(fragment_path),
            run_dir=str(self.root / "run_dir"),
            run_id="MERGE-CHAR",
        )

    def _write_fragment(self, nodes, relationships, name="fragment_char.json") -> Path:
        path = self.root / name
        _write_json(
            path,
            {"nodes": nodes, "instances": [], "relationships": relationships},
        )
        return path

    def test_normal_merge_success_counts(self):
        fragment = self._write_fragment(
            nodes=[
                {"meta_id": "L0-001", "name": "System", "layer": "L0", "meta_class": "Capability"},
                {"meta_id": "L1-001", "name": "MBOM", "layer": "L1", "meta_class": "Data", "source_file": "a.md"},
                {"meta_id": "L2-001", "name": "ChangeManagement", "layer": "L2", "meta_class": "Process", "source_file": "a.md"},
            ],
            relationships=[{"source": "L1-001", "target": "L2-001", "relation_type": "contains"}],
        )
        master = self.root / "master.json"
        _write_json(master, {"nodes": [], "instances": [], "relationships": []})

        result = run_merge(self._args(fragment, master))

        self.assertTrue(result.success, result.message)
        self.assertEqual(result.stage, "merge")
        self.assertEqual(result.data["merged_nodes"], 3)
        self.assertEqual(result.data["new_nodes"], 3)
        self.assertEqual(result.data["duplicate_nodes"], 0)
        self.assertEqual(result.data["merged_relationships"], 1)
        self.assertIn(str(fragment), result.artifact_paths)
        self.assertIn(str(master), result.artifact_paths)

    def test_duplicate_entities_collapsed(self):
        fragment = self._write_fragment(
            nodes=[
                {"meta_id": "L1-001", "name": "MBOM", "layer": "L1", "meta_class": "Data", "source_file": "a.md"},
            ],
            relationships=[],
        )
        master = self.root / "master.json"
        _write_json(
            master,
            {
                "nodes": [
                    {"meta_id": "L1-001", "name": "MBOM", "layer": "L1", "meta_class": "Data", "source_file": "a.md"},
                ],
                "instances": [],
                "relationships": [],
            },
        )

        result = run_merge(self._args(fragment, master))

        self.assertTrue(result.success, result.message)
        # Same signature (layer/name/meta_class/source) collapses; nothing new added.
        self.assertEqual(result.data["duplicate_nodes"], 1)
        self.assertEqual(result.data["new_nodes"], 0)
        self.assertEqual(result.data["merged_nodes"], 1)

    def test_id_collision_remaps_and_reconnects_relationship(self):
        master_node = {"meta_id": "L1-001", "name": "MBOM", "layer": "L1", "meta_class": "Data", "source_file": "m.md"}
        master = self.root / "master.json"
        _write_json(
            master,
            {"nodes": [master_node], "instances": [], "relationships": []},
        )

        fragment = self._write_fragment(
            nodes=[
                {"meta_id": "L1-001", "name": "SAP", "layer": "L1", "meta_class": "System", "source_file": "f.md"},
                {"meta_id": "L2-001", "name": "TARR", "layer": "L2", "meta_class": "Process", "source_file": "f.md"},
            ],
            relationships=[{"source": "L1-001", "target": "L2-001", "relation_type": "uses"}],
        )

        result = run_merge(self._args(fragment, master))

        self.assertTrue(result.success, result.message)
        merged_master = json.loads(master.read_text(encoding="utf-8"))
        fragment_node_ids = [n.get("meta_id") for n in merged_master["nodes"] if n.get("name") == "SAP"]
        self.assertEqual(len(fragment_node_ids), 1)
        new_id = fragment_node_ids[0]
        self.assertNotEqual(new_id, "L1-001")
        # The fragment relationship must point at the remapped id, not the master's.
        # Note: run_merge normalizes relation_type to uppercase.
        fragment_rels = [r for r in merged_master["relationships"] if r.get("relation_type") == "USES"]
        self.assertEqual(len(fragment_rels), 1)
        self.assertEqual(fragment_rels[0]["source"], new_id)
        self.assertEqual(fragment_rels[0]["target"], "L2-001")

    def test_empty_input_blocked_and_master_not_written(self):
        """P0-3: an empty extraction must never be merged into master.

        Previously an empty fragment merged successfully and could wipe master
        when the LLM provider silently returned nothing (regression 2026-09-01).
        The empty-extraction guard now blocks merge before master is written.
        """
        fragment = self._write_fragment(nodes=[], relationships=[])
        master = self.root / "master.json"
        master_before = {"nodes": [{"meta_id": "L0-001", "name": "System", "layer": "L0"}], "instances": [], "relationships": []}
        _write_json(master, master_before)

        result = run_merge(self._args(fragment, master))

        self.assertFalse(result.success)
        self.assertIn("blocked", result.message)
        # Master is untouched (still the pre-merge content).
        self.assertEqual(json.loads(master.read_text(encoding="utf-8")), master_before)

    def test_under_threshold_input_blocked(self):
        """P0-3: fragment below MERGE_MIN_FRAGMENT_ENTITIES is blocked."""
        fragment = self._write_fragment(
            nodes=[{"meta_id": "L1-001", "name": "MBOM", "layer": "L1", "meta_class": "Data", "source_file": "a.md"}],
            relationships=[],
        )
        master = self.root / "master.json"
        _write_json(master, {"nodes": [], "instances": [], "relationships": []})

        with mock.patch.dict("os.environ", {"MERGE_MIN_FRAGMENT_ENTITIES": "5"}):
            result = run_merge(self._args(fragment, master))

        self.assertFalse(result.success)
        self.assertIn("under-threshold", result.message)

    def test_non_empty_fragment_passes_guard(self):
        """P0-3: a fragment carrying entities still merges normally."""
        fragment = self._write_fragment(
            nodes=[{"meta_id": "L1-001", "name": "MBOM", "layer": "L1", "meta_class": "Data", "source_file": "a.md"}],
            relationships=[],
        )
        master = self.root / "master.json"
        _write_json(master, {"nodes": [], "instances": [], "relationships": []})

        result = run_merge(self._args(fragment, master))

        self.assertTrue(result.success, result.message)
        self.assertEqual(result.data["new_nodes"], 1)

    def test_missing_fragment_returns_failure(self):
        master = self.root / "master.json"
        _write_json(master, {"nodes": [], "instances": [], "relationships": []})
        missing = self.root / "no_such_fragment.json"

        result = run_merge(self._args(missing, master))

        self.assertFalse(result.success)
        self.assertIn("no fragment artifact", result.message)


if __name__ == "__main__":
    unittest.main()