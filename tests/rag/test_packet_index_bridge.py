"""Tests for explicit packet→RAG ChunkRecord index bridge."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.application.rag.packet_index_bridge import (
    rag_chunks_from_packet,
    write_packet_chunks,
)
from src.application.rag.chunk_indexing import load_chunks_jsonl


class PacketIndexBridgeTest(unittest.TestCase):
    def _packet(self):
        return {
            "document": {"file_name": "ecn.md"},
            "entities": {
                "L0": [],
                "L1": [{
                    "id": "P1", "name": "Part A", "layer": "L1", "type": "Asset",
                    "description": "Battery module", "aliases": ["电池A"],
                    "tags": ["battery"], "evidence_ref": "doc:§1",
                    "valid_from": "2026-01-01",
                }],
                "L2": [], "L2.5": [], "L3": [],
            },
            "relations": [],
        }

    def test_packet_chunks_are_existing_chunk_records(self):
        chunks = rag_chunks_from_packet(self._packet())
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.chunk_id, "ragc:P1")
        self.assertEqual(chunk.node_ids, ["P1"])
        self.assertIn("Part A", chunk.content)
        self.assertIn("电池A", chunk.content)
        self.assertEqual(chunk.meta["valid_from"], "2026-01-01")
        self.assertEqual(chunk.meta["evidence_ref"], "doc:§1")

    def test_jsonl_round_trip_uses_canonical_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chunks.jsonl"
            chunks = write_packet_chunks(self._packet(), path)
            loaded = load_chunks_jsonl(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].chunk_id, chunks[0].chunk_id)
            self.assertEqual(loaded[0].content, chunks[0].content)
            record = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertNotIn("evidence", record)  # absent, not null noise

    def test_empty_packet_has_no_chunks(self):
        packet = {"document": {"file_name": "empty"}, "entities": {}, "relations": []}
        self.assertEqual(rag_chunks_from_packet(packet), [])


if __name__ == "__main__":
    unittest.main()
