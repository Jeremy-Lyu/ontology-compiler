"""Tests for packet → Canonical models → StructuredIndex bridge."""
from __future__ import annotations

import unittest

from src.application.rag.structured_index import StructuredIndex
from src.application.rag.structured_packet_bridge import (
    canonical_models_from_packet,
    index_packet,
)


def packet():
    return {
        "document": {"file_name": "mbom.md", "domain": "MANUFACTURING"},
        "entities": {
            "L0": [],
            "L1": [
                {"id": "P1", "name": "Part A", "layer": "L1", "type": "Asset",
                 "description": "Battery module", "aliases": ["电池A"],
                 "attributes": {"part_number": "PN-1"}, "confidence": 0.95},
                {"id": "M1", "name": "MBOM X", "layer": "L1", "type": "Asset"},
            ],
            "L2": [], "L2.5": [], "L3": [],
        },
        "relations": [{
            "source_id": "P1", "target_id": "M1", "relation_type": "USED_IN",
            "source_cq": "CQ-MBOM-001", "evidence_ref": "doc:§2",
        }],
    }


class StructuredPacketBridgeTest(unittest.TestCase):
    def test_models_preserve_identity_and_relation_metadata(self):
        models = canonical_models_from_packet(packet())
        self.assertEqual({e.entity_id for e in models["entities"]}, {"P1", "M1"})
        part = next(e for e in models["entities"] if e.entity_id == "P1")
        self.assertEqual(part.properties["part_number"], "PN-1")
        self.assertEqual(part.confidence, 0.95)
        relation = models["relations"][0]
        self.assertEqual(relation.relation_type, "USED_IN")
        self.assertEqual(relation.metadata["source_cq"], "CQ-MBOM-001")
        self.assertEqual(relation.metadata["evidence_ref"], "doc:§2")

    def test_index_packet_supports_entity_and_relation_queries(self):
        index = index_packet(packet())
        self.assertIsInstance(index, StructuredIndex)
        found = index.search_entities(query="电池A")
        self.assertEqual([e.entity_id for e in found], ["P1"])
        rels = index.search_relations(from_entity="P1", relation_type="USED_IN")
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0][1:], ("Part A", "MBOM X"))
        self.assertEqual(index.get_stats().total_entities, 2)
        self.assertEqual(index.get_stats().total_relations, 1)

    def test_supplied_index_is_incrementally_reused(self):
        index = StructuredIndex()
        self.assertIs(index_packet(packet(), index), index)
        self.assertEqual(index.get_stats().total_entities, 2)
        # second indexing replaces records, no count inflation
        self.assertIs(index_packet(packet(), index), index)
        self.assertEqual(index.get_stats().total_entities, 2)


if __name__ == "__main__":
    unittest.main()
