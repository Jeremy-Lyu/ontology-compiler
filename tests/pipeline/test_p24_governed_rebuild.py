"""P2-4 (2026-09-09): debounce + failure counter + governed publish gate."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def _conventional_layout(tmp: Path) -> tuple[Path, Path]:
    """Production-conventional layout: unified sibling to master_ontology dir."""
    unified = tmp / "master_ontology_unified.json"
    store_dir = tmp / "master_ontology"
    store_dir.mkdir(parents=True, exist_ok=True)
    (store_dir / "nodes.json").write_text('[{"id":"n1"}]', encoding="utf-8")
    (store_dir / "relationships.json").write_text('[]', encoding="utf-8")
    return unified, store_dir


class DebounceTest(unittest.TestCase):
    def test_same_hash_second_write_does_not_rebuild(self):
        from src.pipeline.storage.master_store import MasterOntologyStore

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            unified, _ = _conventional_layout(tmp_path)
            store = MasterOntologyStore(str(unified))
            fake_governed = tmp_path / "fake_governed"
            fake_governed.mkdir()
            fake_pointer = tmp_path / "fake_latest.json"
            fake_pointer.write_text(
                '{"packet_id":"GOV-OLD","master_hashes":{"nodes.json":"other","relationships.json":"other"}}',
                encoding="utf-8",
            )
            with mock.patch("src.application.rag.governed_packets.ASSET_PACKETS_ROOT", tmp_path / "fake_governed"):
                with mock.patch("src.application.rag.governed_packets.LATEST_POINTER", fake_pointer):
                    with mock.patch(
                        "src.pipeline.storage.master_store._sha256_file",
                        side_effect=lambda p: "hash-nodes" if "nodes" in str(p) else "hash-rels",
                    ):
                        with mock.patch("src.pipeline.storage.master_store.subprocess.run") as run:
                            run.return_value = mock.MagicMock(returncode=0, stdout="", stderr="")
                            with mock.patch.object(MasterOntologyStore, "_rebuild_status_path",
                                                   return_value=tmp_path / "st.json"):
                                store._post_write_rebuild_governed_packet()
                                self.assertEqual(run.call_count, 1)
                                run.reset_mock()
                                store._post_write_rebuild_governed_packet()
                                self.assertEqual(run.call_count, 0)

    def test_different_hash_rebuilds_again(self):
        from src.pipeline.storage.master_store import MasterOntologyStore

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            unified, _ = _conventional_layout(tmp_path)
            store = MasterOntologyStore(str(unified))
            hashes = ["h1-nodes", "h1-rels", "h2-nodes", "h2-rels"]
            it = iter(hashes * 2)
            fake_governed = tmp_path / "fake_governed2"
            fake_governed.mkdir()
            fake_pointer = tmp_path / "fake_latest2.json"
            fake_pointer.write_text('{"packet_id":"GOV-OLD2"}', encoding="utf-8")
            with mock.patch("src.application.rag.governed_packets.ASSET_PACKETS_ROOT", tmp_path / "fake_governed2"):
                with mock.patch("src.application.rag.governed_packets.LATEST_POINTER", fake_pointer):
                    with mock.patch(
                        "src.pipeline.storage.master_store._sha256_file",
                        side_effect=lambda p: next(it),
                    ):
                        with mock.patch("src.pipeline.storage.master_store.subprocess.run") as run:
                            run.return_value = mock.MagicMock(returncode=0, stdout="", stderr="")
                            with mock.patch.object(MasterOntologyStore, "_rebuild_status_path",
                                                   return_value=tmp_path / "st2.json"):
                                store._post_write_rebuild_governed_packet()
                                self.assertEqual(run.call_count, 1)
                                run.reset_mock()
                                store._post_write_rebuild_governed_packet()
                                self.assertEqual(run.call_count, 1)


class FailureCounterTest(unittest.TestCase):
    def test_failure_increments_and_alert_threshold(self):
        from src.pipeline.storage.master_store import MasterOntologyStore

        with tempfile.TemporaryDirectory() as tmp:
            store = MasterOntologyStore(str(Path(tmp) / "master.json"))
            status_path = Path(tmp) / "governed_rebuild_status.json"
            with mock.patch.object(MasterOntologyStore, "_rebuild_status_path", return_value=status_path):
                with mock.patch.dict(os.environ, {"RAG_REBUILD_FAIL_ALERT_THRESHOLD": "2"}):
                    store._record_rebuild_outcome(False, "boom 1")
                    self.assertEqual(store._rebuild_failure_count, 1)
                    self.assertFalse(json.loads(status_path.read_text())["alert"])
                    store._record_rebuild_outcome(False, "boom 2")
                    self.assertEqual(store._rebuild_failure_count, 2)
                    self.assertTrue(json.loads(status_path.read_text())["alert"])
                    store._record_rebuild_outcome(True)
                    self.assertEqual(store._rebuild_failure_count, 0)
                    self.assertFalse(json.loads(status_path.read_text())["alert"])
                    self.assertEqual(json.loads(status_path.read_text())["last_outcome"], "ok")

    def test_manifest_hit_resets_counter(self):
        from src.pipeline.storage.master_store import MasterOntologyStore

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            unified, _ = _conventional_layout(tmp_path)
            store = MasterOntologyStore(str(unified))
            store._rebuild_failure_count = 2
            status_path = tmp_path / "status.json"
            with mock.patch.object(MasterOntologyStore, "_rebuild_status_path", return_value=status_path):
                with mock.patch("src.pipeline.storage.master_store._sha256_file", return_value="same"):
                    governed_root = tmp_path / "asset_packets" / "governed" / "GOV-1"
                    governed_root.mkdir(parents=True)
                    (governed_root / "manifest.json").write_text(
                        json.dumps({"master_hashes": {"nodes.json": "same", "relationships.json": "same"}}),
                        encoding="utf-8",
                    )
                    pointer = tmp_path / "latest.json"
                    pointer.write_text(json.dumps({"packet_id": "GOV-1"}), encoding="utf-8")
                    with mock.patch("src.application.rag.governed_packets.LATEST_POINTER", pointer):
                        with mock.patch("src.application.rag.governed_packets.ASSET_PACKETS_ROOT",
                                        tmp_path / "asset_packets"):
                            store._post_write_rebuild_governed_packet()
                    self.assertEqual(store._rebuild_failure_count, 0)


class GovernedPublishGateTest(unittest.TestCase):
    def test_freshness_mismatch_fails(self):
        import scripts.check_governed_publish as gate

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            master_dir = tmp_path / "master_ontology"
            master_dir.mkdir()
            (master_dir / "nodes.json").write_text("[]", encoding="utf-8")
            (master_dir / "relationships.json").write_text("[]", encoding="utf-8")
            governed = tmp_path / "asset_packets" / "governed" / "GOV-OLD"
            governed.mkdir(parents=True)
            (governed / "manifest.json").write_text(
                json.dumps({"packet_id": "GOV-OLD",
                            "master_hashes": {"nodes.json": "stale", "relationships.json": "stale"}}),
                encoding="utf-8",
            )
            pointer = tmp_path / "asset_packets" / "latest_governed_packet.json"
            pointer.parent.mkdir(parents=True, exist_ok=True)
            pointer.write_text(json.dumps({"packet_id": "GOV-OLD"}), encoding="utf-8")

            with mock.patch.object(gate, "DEFAULT_MASTER_DIR", master_dir):
                with mock.patch.object(gate, "LATEST_POINTER", pointer):
                    with mock.patch.object(gate, "ROOT", tmp_path):
                        with mock.patch.object(gate, "STATUS_PATH", tmp_path / "no_status.json"):
                            import sys
                            orig_argv = sys.argv[:]
                            sys.argv = ["check_governed_publish.py", "--master-dir", str(master_dir)]
                            try:
                                ret = gate.main()
                            finally:
                                sys.argv = orig_argv
                            self.assertEqual(ret, 1)

    def test_status_alert_fails(self):
        import scripts.check_governed_publish as gate
        import scripts.build_governed_packet as bgp

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            master_dir = tmp_path / "master_ontology"
            master_dir.mkdir()
            (master_dir / "nodes.json").write_text("[]", encoding="utf-8")
            (master_dir / "relationships.json").write_text("[]", encoding="utf-8")
            hashes = bgp._master_file_hashes(master_dir)
            governed = tmp_path / "asset_packets" / "governed" / "GOV-NOW"
            governed.mkdir(parents=True)
            (governed / "manifest.json").write_text(
                json.dumps({"packet_id": "GOV-NOW", "master_hashes": hashes}),
                encoding="utf-8",
            )
            pointer = tmp_path / "asset_packets" / "latest_governed_packet.json"
            pointer.parent.mkdir(parents=True, exist_ok=True)
            pointer.write_text(json.dumps({"packet_id": "GOV-NOW"}), encoding="utf-8")
            status = tmp_path / "governed_rebuild_status.json"
            status.write_text(json.dumps({"alert": True, "consecutive_failures": 3,
                                          "last_outcome": "failed", "last_error": "boom"}),
                              encoding="utf-8")
            with mock.patch.object(gate, "DEFAULT_MASTER_DIR", master_dir):
                with mock.patch.object(gate, "LATEST_POINTER", pointer):
                    with mock.patch.object(gate, "STATUS_PATH", status):
                        with mock.patch.object(gate, "ROOT", tmp_path):
                            import sys
                            orig_argv = sys.argv[:]
                            sys.argv = ["check_governed_publish.py", "--master-dir", str(master_dir)]
                            try:
                                ret = gate.main()
                            finally:
                                sys.argv = orig_argv
                            self.assertEqual(ret, 1)


if __name__ == "__main__":
    unittest.main()
