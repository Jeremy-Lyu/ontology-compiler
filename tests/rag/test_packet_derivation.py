"""Characterization tests for T4.1 RAG consumption-object derivation.

Locks: canonical/graph projection, retrievable chunk assembly (keywords,
cycle-safe parent paths), evidence-bundle grouping, and whole-packet
derivation — all pure functions over the packet shape produced by
extraction_packet_adapter.
"""
from __future__ import annotations

import unittest

from src.application.rag.packet_derivation import (
    derive_canonical_entity,
    derive_evidence_bundles,
    derive_rag_chunk_entity,
    derive_rag_objects,
)


def _packet() -> dict:
    return {
        "document": {"file_name": "ecn_case.pptx", "file_path": "docs/ecn_case.pptx",
                     "domain": "MANUFACTURING"},
        "layer_scope": "spec",
        "entities": {
            "L0": [],
            "L1": [
                {"id": "L1-PART", "name": "HV_Battery_Module_001", "layer": "L1",
                 "type": "Asset", "status": "active", "version": "2",
                 "description": "高压电池模组 V1", "aliases": ["电池模组V1"],
                 "evidence_ref": "doc:§2"},
                {"id": "L1-MBOM", "name": "MY23_MBOM_SH", "layer": "L1",
                 "type": "Asset",
                 "evidence_ref": "doc:§2"},
            ],
            "L2": [],
            "L2.5": [],
            "L3": [],
        },
        "relations": [
            {"source_id": "L1-PART", "target_id": "L1-MBOM", "relation_type": "USED_IN",
             "evidence_ref": "doc:§2", "source_cq": "CQ-MBOM-001"},
        ],
        "warnings": [],
    }


class CanonicalEntityTest(unittest.TestCase):
    def test_graph_projection_fields(self):
        ent = _packet()["entities"]["L1"][0]
        canon = derive_canonical_entity(ent)
        self.assertEqual(canon["id"], "L1-PART")
        self.assertEqual(canon["type"], "Asset")
        self.assertEqual(canon["status"], "active")
        self.assertEqual(canon["attributes"], {})
        # no extraction-task fields leak into the storage shape
        self.assertNotIn("source_span", canon)
        self.assertNotIn("relations_summary", canon)

    def test_malformed_attributes_degrade_to_empty(self):
        canon = derive_canonical_entity({"id": "X", "attributes": ["bad"]})
        self.assertEqual(canon["attributes"], {})


class RagChunkEntityTest(unittest.TestCase):
    def test_keywords_from_aliases_then_tags_deduped(self):
        by_id = {}
        chunk = derive_rag_chunk_entity(
            {"id": "E1", "name": "N", "aliases": ["a1", "a2"], "tags": ["a1", "t1"]},
            by_id,
        )
        self.assertEqual(chunk["keywords"], ["a1", "a2", "t1"])

    def test_parent_path_walks_chain_root_first(self):
        ents = {
            "child": {"id": "child", "name": "C", "parent_id": "mid"},
            "mid": {"id": "mid", "name": "M", "parent_id": "root"},
            "root": {"id": "root", "name": "R", "parent_id": ""},
        }
        chunk = derive_rag_chunk_entity(ents["child"], ents)
        self.assertEqual(chunk["parent_path"], ["R", "M", "C"])

    def test_parent_cycle_terminates(self):
        ents = {
            "a": {"id": "a", "name": "A", "parent_id": "b"},
            "b": {"id": "b", "name": "B", "parent_id": "a"},
        }
        chunk = derive_rag_chunk_entity(ents["a"], ents)
        self.assertLessEqual(len(chunk["parent_path"]), 17)  # depth guard hit, no hang


class EvidenceBundleTest(unittest.TestCase):
    def test_shared_ref_groups_entities_and_relations(self):
        bundles = derive_evidence_bundles(_packet())
        by_ref = {b["evidence_ref"]: b for b in bundles}
        self.assertIn("doc:§2", by_ref)
        b = by_ref["doc:§2"]
        self.assertEqual(b["document"], "ecn_case.pptx")
        self.assertIn("L1-PART", b["entity_ids"])
        self.assertIn("L1-MBOM", b["entity_ids"])
        self.assertTrue(any("USED_IN" in r for r in b["relation_ids"]))

    def test_missing_ref_skipped(self):
        packet = {"document": {"file_name": "d"}, "entities": {"L1": [
            {"id": "E", "name": "n"}  # no evidence_ref
        ]}, "relations": []}
        self.assertEqual(derive_evidence_bundles(packet), [])


class DeriveAllTest(unittest.TestCase):
    def test_whole_packet_derivation_counts_align(self):
        out = derive_rag_objects(_packet())
        self.assertEqual(len(out["canonical_entities"]), 2)
        self.assertEqual(len(out["rag_chunks"]), 2)
        self.assertTrue(out["evidence_bundles"])
        # chunk ↔ canonical stay aligned by entity id
        self.assertEqual(
            {c["entity_id"] for c in out["rag_chunks"]},
            {c["id"] for c in out["canonical_entities"]},
        )

    def test_empty_packet_safe(self):
        out = derive_rag_objects({"entities": {}, "relations": []})
        self.assertEqual(out, {"canonical_entities": [], "rag_chunks": [], "evidence_bundles": []})


if __name__ == "__main__":
    unittest.main()
