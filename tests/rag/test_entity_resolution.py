"""Characterization tests for T4.2 reversible entity canonicalization.

Locks: blocking/matching/clustering determinism, alias-based duplicate
discovery, type-mismatch protection, relation endpoint rewriting, collapse
dedup, and the Palantir-style reversibility guarantee (undo restores the
input deep-equal).
"""
from __future__ import annotations

import copy
import unittest

from src.application.rag.entity_resolution import (
    apply_canonicalization,
    block_entities,
    cluster_entities,
    normalize_entity_name,
    undo_canonicalization,
)


def _packet_with_duplicates() -> dict:
    return {
        "document": {"file_name": "d.pptx"},
        "entities": {
            "L1": [
                {"id": "A1", "name": "HV_Battery_Module_001", "layer": "L1",
                 "type": "Asset", "confidence": 0.6, "evidence_ref": "doc:§1",
                 "aliases": ["电池模组V1"]},
                {"id": "B2", "name": "hv battery module 001", "layer": "L1",
                 "type": "Asset", "confidence": 0.95, "evidence_ref": "doc:§2"},
                # different type: same normalized name must NOT merge
                {"id": "C3", "name": "HV_Battery_Module_001", "layer": "L2",
                 "type": "BusinessScenario", "evidence_ref": "doc:§3"},
                # unrelated singleton
                {"id": "D4", "name": "上海仓库", "layer": "L1", "type": "Actor",
                 "evidence_ref": "doc:§4"},
            ],
        },
        "relations": [
            {"source_id": "A1", "target_id": "C3", "relation_type": "USED_IN",
             "evidence_ref": "doc:§9", "source_cq": "CQ-1"},
            {"source_id": "D4", "target_id": "B2", "relation_type": "STORES",
             "evidence_ref": "doc:§8", "source_cq": "CQ-2"},
        ],
    }


class BlockingMatchingTest(unittest.TestCase):
    def test_normalize_key_semantics(self):
        self.assertEqual(normalize_entity_name("HV_Battery-Module 001"),
                         normalize_entity_name("hv battery module 001"))
        self.assertEqual(normalize_entity_name(None), "")

    def test_blocks_bound_pairwise_space(self):
        ents = _packet_with_duplicates()["entities"]["L1"]
        blocks = block_entities(ents)
        # A1 and B2 share type+first-char; C3/D4 land elsewhere
        a_block = [ents[i]["id"] for i in blocks.get(("Asset", "h"), [])]
        self.assertIn("A1", a_block)
        self.assertIn("B2", a_block)
        self.assertNotIn("D4", a_block)

    def test_alias_overlap_matches_despite_different_names(self):
        a = {"id": "x", "name": "Part X", "layer": "L1", "type": "Asset",
             "aliases": ["电池模组V1"]}
        b = {"id": "y", "name": "电池模组V1", "layer": "L1", "type": "Asset"}
        clusters = cluster_entities([a, b])
        self.assertEqual(len(clusters), 1)

    def test_type_or_layer_mismatch_protects(self):
        a = {"id": "x", "name": "Same", "layer": "L1", "type": "Asset"}
        b = {"id": "y", "name": "same", "layer": "L1", "type": "Process"}
        c = {"id": "z", "name": "Same", "layer": "L2", "type": "Asset"}
        self.assertEqual(len(cluster_entities([a, b])), 2)
        self.assertEqual(len(cluster_entities([a, c])), 2)


class ApplyCanonicalizationTest(unittest.TestCase):
    def test_merge_picks_highest_confidence_and_rewrites_relations(self):
        new_packet, record = apply_canonicalization(_packet_with_duplicates())
        flat_ids = [e["id"] for bucket in new_packet["entities"].values() for e in bucket]
        # B2 wins (0.95 > 0.6); C3/D4 untouched
        self.assertNotIn("A1", flat_ids)
        self.assertIn("B2", flat_ids)
        self.assertIn("C3", flat_ids)
        rels = {(r["source_id"], r["target_id"]) for r in new_packet["relations"]}
        self.assertIn(("B2", "C3"), rels)   # rewritten from A1
        self.assertIn(("D4", "B2"), rels)   # already canonical

    def test_record_carries_decisions_and_rewrites(self):
        _, record = apply_canonicalization(_packet_with_duplicates())
        self.assertEqual(record["decisions"][0]["canonical_id"], "B2")
        self.assertIn("A1", record["decisions"][0]["member_ids"])
        self.assertTrue(any(
            r.get("from") == "A1" and r.get("to") == "B2"
            for r in record["endpoint_rewrites"]
        ))
        self.assertIn("A1", record["id_map"])

    def test_input_packet_never_mutated(self):
        packet = _packet_with_duplicates()
        snapshot = copy.deepcopy(packet)
        apply_canonicalization(packet)
        self.assertEqual(packet, snapshot)

    def test_no_candidates_yields_identity(self):
        packet = _packet_with_duplicates()
        for e in packet["entities"]["L1"]:
            e["name"] = f"unique-{e['id']}"
            e.pop("aliases", None)
        new_packet, record = apply_canonicalization(packet)
        self.assertEqual(record["decisions"], [])
        flat_before = sorted(e["id"] for e in packet["entities"]["L1"])
        flat_after = sorted(e["id"] for b in new_packet["entities"].values() for e in b)
        self.assertEqual(flat_before, flat_after)

    def test_determinism_same_input_same_output(self):
        p1, r1 = apply_canonicalization(_packet_with_duplicates())
        p2, r2 = apply_canonicalization(_packet_with_duplicates())
        self.assertEqual(p1, p2)
        self.assertEqual(r1["decisions"], r2["decisions"])
        self.assertEqual(r1["id_map"], r2["id_map"])


class UndoCanonicalizationTest(unittest.TestCase):
    def test_undo_restores_deep_equal_original(self):
        original = _packet_with_duplicates()
        merged, record = apply_canonicalization(original)
        restored = undo_canonicalization(merged, record)
        self.assertEqual(restored, original)
        self.assertIsNot(restored, original)

    def test_undo_after_downstream_mutation_still_restores(self):
        original = _packet_with_duplicates()
        merged, record = apply_canonicalization(original)
        merged["relations"].append({"source_id": "X", "target_id": "Y"})  # drift after merge
        restored = undo_canonicalization(merged, record)
        self.assertEqual(restored["relations"], original["relations"])


if __name__ == "__main__":
    unittest.main()
