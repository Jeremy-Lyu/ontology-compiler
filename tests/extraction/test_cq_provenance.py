"""Characterization tests for T2.3 CQ provenance registry (ROADMAP).

Locks: packet relation schema declares ``source_cq``; the adapter passes it
through from legacy dicts (EN/CN aliases); ``cq_provenance_stats`` computes
dead-relation metrics used as the Phase-2 exit signal.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from ontology_compiler.services.extraction_packet_adapter import (
    adapt_extraction_json_to_packet,
)
from scripts.golden_semantic_eval import cq_provenance_stats


class SourceCqSchemaTest(unittest.TestCase):
    def test_relation_schema_declares_source_cq(self):
        schema = json.loads(
            Path("schemas/ontology_packet.schema.json").read_text(encoding="utf-8")
        )
        props = schema["definitions"]["relation"]["properties"]
        self.assertIn("source_cq", props)
        self.assertEqual(props["source_cq"]["type"], "string")
        # optional field — must not be required
        self.assertNotIn(
            "source_cq", schema["definitions"]["relation"].get("required", [])
        )


class AdapterSourceCqTest(unittest.TestCase):
    def test_source_cq_passthrough_en_alias(self):
        legacy = {
            "L1_entities": [{"meta_id": "P1", "name": "Part A"}],
            "L2_entities": [{"meta_id": "M1", "name": "MBOM X"}],
            "relationships": [
                {"from": "Part A", "to": "MBOM X", "type": "USED_IN",
                 "evidence_ref": "doc:§1", "source_cq": "CQ-MBOM-001"}
            ],
        }
        rel = adapt_extraction_json_to_packet(legacy)["relations"][0]
        self.assertEqual(rel["source_cq"], "CQ-MBOM-001")

    def test_source_cq_cn_alias(self):
        legacy = {
            "L1_entities": [{"meta_id": "P1", "name": "零件A"}],
            "L2_entities": [{"meta_id": "M1", "name": "MBOM-X"}],
            "relationships": [
                {"from": "零件A", "to": "MBOM-X", "type": "USED_IN", "来源CQ": "CQ-MBOM-003"}
            ],
        }
        rel = adapt_extraction_json_to_packet(legacy)["relations"][0]
        self.assertEqual(rel["source_cq"], "CQ-MBOM-003")

    def test_missing_source_cq_is_empty_string(self):
        legacy = {
            "L1_entities": [{"meta_id": "P1", "name": "A"}],
            "relationships": [
                {"from": "A", "to": "A", "type": "SELF_LOOP"}
            ],
        }
        rel = adapt_extraction_json_to_packet(legacy)["relations"][0]
        self.assertNotIn("source_cq", rel)  # schema-optional: omitted when absent
        stats = cq_provenance_stats([rel])
        self.assertEqual(stats["dead_relations"], 1)


class CqProvenanceStatsTest(unittest.TestCase):
    def test_counts_dead_and_annotated_relations(self):
        relations = [
            {"relation_type": "USED_IN", "source_cq": "CQ-MBOM-001"},
            {"relation_type": "PART_OF", "source_cq": "CQ-MBOM-002"},
            {"relation_type": "RELATED_TO"},  # dead, no justification
            {"relation_type": "SUPPLIED_BY", "source_cq": ""},
        ]
        stats = cq_provenance_stats(relations)
        self.assertEqual(stats["total_relations"], 4)
        self.assertEqual(stats["with_source_cq"], 2)
        self.assertEqual(stats["dead_relations"], 2)
        self.assertEqual(stats["dead_ratio"], 0.5)
        self.assertEqual(stats["dead_by_type"], {"RELATED_TO": 1, "SUPPLIED_BY": 1})

    def test_empty_input(self):
        stats = cq_provenance_stats([])
        self.assertEqual(stats["total_relations"], 0)
        self.assertEqual(stats["dead_ratio"], 0.0)
        self.assertEqual(stats["dead_by_type"], {})

    def test_all_annotated(self):
        relations = [
            {"relation_type": t, "source_cq": f"CQ-{i}"}
            for i, t in enumerate(["A", "B"])
        ]
        stats = cq_provenance_stats(relations)
        self.assertEqual(stats["dead_relations"], 0)
        self.assertEqual(stats["dead_ratio"], 0.0)
        self.assertEqual(stats["dead_by_type"], {})


if __name__ == "__main__":
    unittest.main()
