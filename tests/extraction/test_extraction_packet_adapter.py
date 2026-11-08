"""Characterization tests for the extraction→packet protocol bridge (T1.2).

Locks the legacy→OntologyPacket mapping (extraction_packet_adapter) and the
dual output-schema template switch in PromptService, per
docs/architectures/ENTITY_CLASSIFICATION_ROADMAP.md Phase 1 / T1.2.
"""
from __future__ import annotations

import os
import unittest
from unittest import mock


class ExtractionPacketAdapterTest(unittest.TestCase):
    def test_maps_layer_buckets_and_relations(self):
        from ontology_compiler.services.extraction_packet_adapter import (
            adapt_extraction_json_to_packet,
        )

        legacy = {
            "L0_entities": [
                {"meta_id": "L0-001", "meta_class": "Asset", "name": "Part", "evidence": "doc:§1"}
            ],
            "L1_entities": [
                {
                    "meta_id": "L1-001",
                    "meta_class": "Asset",
                    "parent_class": "Part",
                    "name": "Battery Module",
                    "attributes": [{"name": "kpi", "value": "energy density"}],
                }
            ],
            "L2_entities": [
                {"meta_id": "L2-001", "name": "MBOM 变更场景", "parent_id": "L1-001", "description": "ECN 影响"}
            ],
            "L2_5_entities": [],
            "L3_entities": [
                {"meta_id": "L3-001", "name": "HV_Battery_Module_001", "parent_id": "L1-001"}
            ],
            "relationships": [
                {"from": "HV_Battery_Module_001", "to": "MBOM 变更场景", "type": "USED_IN", "evidence_ref": "doc:slide-3"},
            ],
        }

        packet = adapt_extraction_json_to_packet(legacy, document_name="t.pptx", domain="MANUFACTURING")

        self.assertEqual(
            sorted(packet.keys()),
            sorted([
                "document", "layer_scope", "entities", "relations", "evidence",
                "provenance", "quality", "warnings", "assumptions",
            ]),
        )
        # entities are grouped by layer buckets (schema contract)
        self.assertEqual(
            set(packet["entities"].keys()), {"L0", "L1", "L2", "L2.5", "L3"}
        )
        from ontology_compiler.services.extraction_packet_adapter import iter_packet_entities
        flat = list(iter_packet_entities(packet))
        layers = {e["id"]: e["layer"] for e in flat}
        self.assertEqual(layers["L0-001"], "L0")
        self.assertEqual(layers["L1-001"], "L1")
        self.assertEqual(layers["L2-001"], "L2")
        self.assertEqual(layers["L3-001"], "L3")
        by_id = {e["id"]: e for e in flat}
        # attributes list-of-dicts normalized to dict
        self.assertEqual(by_id["L1-001"]["attributes"], {"kpi": "energy density"})
        # parent_class fallback fills parent_id
        self.assertEqual(by_id["L1-001"]["parent_id"], "Part")
        # layer_scope defaults to spec
        for ent in flat:
            self.assertEqual(ent["layer_scope"], "spec")
        # relation endpoint name → id resolution
        rel = packet["relations"][0]
        self.assertEqual(rel["source_id"], "L3-001")
        self.assertEqual(rel["target_id"], "L2-001")
        self.assertEqual(rel["relation_type"], "USED_IN")
        self.assertEqual(rel["provenance"], "llm")
        self.assertEqual(rel["evidence_ref"], "doc:slide-3")
        self.assertEqual(packet["warnings"], [])
        self.assertEqual(packet["document"]["file_name"], "t.pptx")

    def test_unresolved_endpoints_reported_as_warnings(self):
        from ontology_compiler.services.extraction_packet_adapter import (
            adapt_extraction_json_to_packet,
        )

        legacy = {
            "L1_entities": [{"meta_id": "L1-001", "name": "A"}],
            "relationships": [
                {"from": "A", "to": "Ghost", "type": "DEPENDS_ON"},
            ],
        }
        packet = adapt_extraction_json_to_packet(legacy)
        # Ghost is not an extracted entity: kept verbatim but warned
        self.assertEqual(len(packet["relations"]), 1)
        self.assertTrue(any(
            w.get("code") == "ENDPOINT_UNRESOLVED" for w in packet["warnings"]
        ))

    def test_pure_function_never_mutates_input(self):
        from ontology_compiler.services.extraction_packet_adapter import (
            adapt_extraction_json_to_packet,
        )

        legacy = {"L1_entities": [{"meta_id": "X", "name": "n"}], "relationships": []}
        snapshot = repr(legacy)
        adapt_extraction_json_to_packet(legacy)
        self.assertEqual(repr(legacy), snapshot)


class PacketSchemaTemplateSwitchTest(unittest.TestCase):
    """build_base_fragment/_select_output_schema live on PromptStrategy."""

    def test_default_uses_legacy_schema(self):
        from ontology_compiler.services.prompt_service import PromptStrategy

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EXTRACT_OUTPUT_PROTOCOL", None)
            schema = PromptStrategy._select_output_schema()
        self.assertIn("L0_entities", schema)
        self.assertNotIn("layer_scope", schema)

    def test_packet_env_switches_template(self):
        from ontology_compiler.services.prompt_service import PromptStrategy

        with mock.patch.dict(os.environ, {"EXTRACT_OUTPUT_PROTOCOL": "packet"}):
            schema = PromptStrategy._select_output_schema()
        self.assertIn("layer_scope", schema)
        self.assertIn('"relations"', schema)

    def test_base_fragment_honors_switch(self):
        from ontology_compiler.services.prompt_service import PromptStrategy

        svc = PromptStrategy()
        with mock.patch.dict(os.environ, {"EXTRACT_OUTPUT_PROTOCOL": "packet"}):
            prompt = svc.build_base_fragment(domain="MANUFACTURING")
        self.assertIn('"layer_scope"', prompt)
        with mock.patch.dict(os.environ, {}):
            os.environ.pop("EXTRACT_OUTPUT_PROTOCOL", None)
            prompt_legacy = svc.build_base_fragment(domain="MANUFACTURING")
        self.assertIn("L0_entities", prompt_legacy)


if __name__ == "__main__":
    unittest.main()
