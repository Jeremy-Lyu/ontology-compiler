"""Meaning-first surface: search / get / answerable + publish gate."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.application.rag.publish_gate import (
    evaluate_publish_gate,
    read_publish_receipt,
    write_publish_receipt,
)
from src.application.rag.semantic_surface import (
    check_answerable,
    flatten_entities,
    get_entity,
    search_entities,
)


def _sample_packet() -> dict:
    return {
        "document": {"id": "doc-1"},
        "entities": {
            "L0": [
                {
                    "id": "L0-001",
                    "name": "Asset",
                    "layer": "L0",
                    "type": "Asset",
                    "aliases": [],
                }
            ],
            "L1": [
                {
                    "id": "L1-EBOM",
                    "name": "EBOM",
                    "layer": "L1",
                    "type": "Artifact",
                    "aliases": ["Engineering BOM", "工程BOM"],
                    "description": "Engineering bill of materials",
                },
                {
                    "id": "L1-MBOM",
                    "name": "MBOM",
                    "layer": "L1",
                    "type": "Artifact",
                    "aliases": ["Manufacturing BOM"],
                },
            ],
            "L2": [
                {
                    "id": "L2-001",
                    "name": "Chassis Module",
                    "layer": "L2",
                    "type": "Module",
                }
            ],
            "L3": [],
        },
        "relations": [
            {
                "source_id": "L1-MBOM",
                "target_id": "L1-EBOM",
                "relation_type": "DERIVED_FROM",
                "provenance": "extracted",
            }
        ],
    }


class SemanticSurfaceTest(unittest.TestCase):
    def test_flatten_and_search(self):
        pkt = _sample_packet()
        self.assertEqual(len(flatten_entities(pkt)), 4)
        hits = search_entities("MBOM", packet=pkt)
        self.assertTrue(hits)
        self.assertEqual(hits[0].id, "L1-MBOM")

    def test_get_entity_includes_incident_relations(self):
        view = get_entity("L1-MBOM", packet=_sample_packet())
        self.assertIsNotNone(view)
        self.assertEqual(view["entity"]["name"], "MBOM")
        self.assertEqual(view["relation_count"], 1)

    def test_answerable_requires_derived_from_when_asked(self):
        ok = check_answerable("MBOM 如何由 EBOM 派生", packet=_sample_packet())
        self.assertTrue(ok.answerable)
        self.assertIn("DERIVED_FROM", ok.matched_relation_types)

        broken = _sample_packet()
        broken["relations"] = []
        bad = check_answerable("MBOM 如何由 EBOM 派生", packet=broken)
        self.assertFalse(bad.answerable)
        self.assertIn("DERIVED_FROM", bad.missing_relation_types)

    def test_answerable_fails_on_missing_concept(self):
        pkt = _sample_packet()
        pkt["entities"]["L1"] = [e for e in pkt["entities"]["L1"] if e["id"] != "L1-MBOM"]
        report = check_answerable("什么是 MBOM", packet=pkt)
        self.assertFalse(report.answerable)
        self.assertIn("mbom", report.missing_concepts)


class PublishGateTest(unittest.TestCase):
    def test_gate_passes_on_complete_packet(self):
        gate = evaluate_publish_gate(_sample_packet())
        self.assertTrue(gate.publishable)
        self.assertFalse(gate.certified)

    def test_gate_rejects_empty_required_layer(self):
        pkt = _sample_packet()
        pkt["entities"]["L2"] = []
        gate = evaluate_publish_gate(pkt)
        self.assertFalse(gate.publishable)
        self.assertTrue(any("L2" in e for e in gate.errors))

    def test_gate_rejects_instance_code_on_l0(self):
        pkt = _sample_packet()
        pkt["entities"]["L0"].append(
            {
                "id": "L0-BAD",
                "name": "PN-123456",
                "layer": "L0",
                "type": "Asset",
                "code": "PN-123456",
            }
        )
        gate = evaluate_publish_gate(pkt)
        self.assertFalse(gate.publishable)
        self.assertTrue(any("instance-like" in e for e in gate.errors))

    def test_receipt_roundtrip(self):
        pkt = _sample_packet()
        gate = evaluate_publish_gate(pkt, certified=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "packet.json").write_text(json.dumps(pkt), encoding="utf-8")
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            write_publish_receipt(root, gate, packet_id="GOV-TEST", promoted=True)
            receipt = read_publish_receipt(root)
            self.assertEqual(receipt["packet_id"], "GOV-TEST")
            self.assertTrue(receipt["publishable"])
            self.assertTrue(receipt["certified"])
            self.assertTrue(receipt["receipt_sha256"])


class LayerInstanceDisciplineRuleTest(unittest.TestCase):
    def test_rule_penalizes_pn_on_l1(self):
        import asyncio

        from ontology_compiler.services.quality_assurance.base import QualityContext
        from ontology_compiler.services.quality_assurance.quality_rules import (
            LayerInstanceDisciplineRule,
        )

        rule = LayerInstanceDisciplineRule()
        ctx = QualityContext(
            fragment={
                "nodes": [
                    {
                        "meta_id": "L1-1",
                        "meta_name": "PN-998877",
                        "layer": "L1",
                        "meta_class": "Asset",
                        "code": "PN-998877",
                    }
                ]
            }
        )
        score = asyncio.run(rule.check(ctx))
        self.assertLess(score, 1.0)
        self.assertTrue(ctx.issues)


if __name__ == "__main__":
    unittest.main()
