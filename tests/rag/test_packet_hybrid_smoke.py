"""Offline smoke tests for packet → hybrid index and retrieval.

Passes LocalEmbedder explicitly. HybridIndexManager's default embedder is
SentenceTransformers, which downloads a model; these tests stay offline.

Governed-only guard (2026-09-01, P0-4): ``HybridIndexManager(rag_dir=<tmp>)``
resolves a non-governed path to the *real* latest governed packet, so an
unpatched build here would silently overwrite the production packet's index
files (chunks.jsonl / inverted / vector_index) with 2-chunk test data — the
same corruption class that emptied GOV-20260901-180956.  These tests therefore
patch ``GOVERNED_DIR`` to the temp dir's parent and turn the temp dir into a
*valid governed packet* (manifest + checksum), so every write stays inside the
temp dir while still exercising the real manager reload/retrieval path.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.application.rag.governed_packets import build_file_manifest, file_sha256
from src.application.rag.infrastructure.embedding import LocalEmbedder
from src.application.rag.packet_index_bridge import build_hybrid_index_from_packet
from src.application.rag.hybrid_rag import HybridIndexManager


def _finalize_governed_packet(packet_dir: Path, *, chunk_count: int) -> None:
    """Write a valid manifest + checksum for ``packet_dir``.

    Must run *after* the index files exist so the recorded hashes match.
    """
    files = build_file_manifest(packet_dir, [
        "chunks.jsonl",
        "inverted_index.json",
        "idf.json",
        "vector_index/metadata.json",
        "packet.json",
    ])
    manifest = {
        "schema_version": "1.0",
        "packet_id": packet_dir.name,
        "built_at": "2026-09-01T00:00:00",
        "master_dir": str(packet_dir.parent),
        "master_hashes": {"nodes": "x", "relationships": "x"},
        "source_code_version": {},
        "files": [{"path": f.path, "size": f.size, "sha256": f.sha256} for f in files],
        "chunk_count": chunk_count,
        "entity_count": 0,
        "bilingual_alias_groups": 0,
        "vector_dimension": 0,
    }
    (packet_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (packet_dir / "manifest.sha256").write_text(
        file_sha256(packet_dir / "manifest.json"), encoding="utf-8"
    )


def _init_empty_governed_packet(packet_dir: Path) -> None:
    """Turn ``packet_dir`` into a valid governed packet with an empty index.

    `resolve_governed_rag_dir` only honors an explicit path once a valid
    manifest exists, but the index build overwrites those files afterwards —
    so callers re-finalize the manifest after building real content.
    """
    (packet_dir / "vector_index").mkdir(parents=True, exist_ok=True)
    (packet_dir / "chunks.jsonl").write_text("", encoding="utf-8")
    (packet_dir / "inverted_index.json").write_text("{}", encoding="utf-8")
    (packet_dir / "idf.json").write_text("{}", encoding="utf-8")
    (packet_dir / "vector_index" / "metadata.json").write_text("{}", encoding="utf-8")
    (packet_dir / "packet.json").write_text("{}", encoding="utf-8")
    _finalize_governed_packet(packet_dir, chunk_count=0)


class PacketHybridSmokeTest(unittest.TestCase):
    @staticmethod
    def _packet():
        return {
            "document": {"file_name": "mbom_smoke.md", "domain": "MANUFACTURING"},
            "entities": {
                "L0": [],
                "L1": [
                    {"id": "P1", "name": "HV Battery Module", "layer": "L1",
                     "type": "Asset", "description": "High voltage battery module",
                     "aliases": ["battery module"], "tags": ["HV", "battery"],
                     "evidence_ref": "doc:§1"},
                    {"id": "M1", "name": "MY23 MBOM Shanghai", "layer": "L1",
                     "type": "Asset", "description": "Shanghai manufacturing BOM",
                     "evidence_ref": "doc:§2"},
                ],
                "L2": [], "L2.5": [], "L3": [],
            },
            "relations": [],
        }

    def test_real_local_hybrid_index_builds_and_reloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _init_empty_governed_packet(tmp_path)
            # Patch GOVERNED_DIR so the temp dir is the governed root and the
            # build/resolution stay hermetic (see module docstring).
            with mock.patch(
                "src.application.rag.governed_packets.GOVERNED_DIR", tmp_path.parent
            ):
                embedder = LocalEmbedder()
                result = build_hybrid_index_from_packet(
                    self._packet(), tmp_path, embedder=embedder, force_rebuild=True
                )
                self.assertEqual(result.chunk_count, 2)
                self.assertGreaterEqual(result.token_count, 1)
                self.assertTrue(Path(result.inverted_index_path).exists())
                self.assertTrue(Path(result.vector_index_path).exists())

                # Re-finalize the manifest against the just-built index files so
                # the reload path can `require_governed_packet` it.
                _finalize_governed_packet(tmp_path, chunk_count=result.chunk_count)

                # Reload through the actual manager and retrieve, not just
                # inspect files.
                manager = HybridIndexManager(rag_dir=tmp_path, embedder=embedder)
                hits = manager.hybrid_search("battery module", topk=2)
                self.assertTrue(hits)
                self.assertEqual(hits[0].chunk.chunk_id, "ragc:P1")

    def test_empty_packet_builds_valid_empty_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _init_empty_governed_packet(tmp_path)
            with mock.patch(
                "src.application.rag.governed_packets.GOVERNED_DIR", tmp_path.parent
            ):
                result = build_hybrid_index_from_packet(
                    {"document": {"file_name": "empty"}, "entities": {}, "relations": []},
                    tmp_path,
                    embedder=LocalEmbedder(),
                    force_rebuild=True,
                )
                self.assertEqual(result.chunk_count, 0)
                self.assertTrue(Path(result.inverted_index_path).exists())


if __name__ == "__main__":
    unittest.main()
