"""Phase-2 exit-criteria E2E validation (ENTITY_CLASSIFICATION_ROADMAP).

Fixture models the reference ECN scenario (HV battery module replacement):
8 entities / 7 relations with CN temporal aliases and mixed CQ provenance.
Chain under test: legacy extraction JSON → packet adapter → JSON-schema
validation → P0 competency-question graph answerability → dead-relation stats,
plus the T2.1 dual-track attachment on a real UnifiedExtractionResult.

Pure and offline — no LLM calls. This validates the *mechanism* side of the
Phase-2 exit criteria ("P0 级 5 条 CQ 全部可答"); a live-data run remains a
follow-up backfill action.
"""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest import mock

from jsonschema import Draft7Validator

from ontology_compiler.services.extraction_packet_adapter import (
    adapt_extraction_json_to_packet,
    iter_packet_entities,
    maybe_attach_ontology_packet,
)
from ontology_compiler.services.true_unified_extractor import (
    UnifiedExtractionResult,
)
from scripts.golden_semantic_eval import cq_provenance_stats

SCHEMA = json.loads(
    Path("schemas/ontology_packet.schema.json").read_text(encoding="utf-8")
)


def _mbom_ecn_legacy_fixture() -> dict:
    """Legacy task-oriented extraction output for the ECN replacement case."""
    return {
        "L0_entities": [],
        "L1_entities": [
            {"meta_id": "L1-PART-001", "meta_class": "Asset", "parent_class": "Product_Definition",
             "name": "HV_Battery_Module_001", "evidence": "doc:§2",
             "attributes": [{"name": "part_number", "value": "HV_BM_001"}]},
            {"meta_id": "L1-MBOM-001", "meta_class": "Asset", "parent_class": "Product_Definition",
             "name": "MY23_MBOM_SH", "evidence": "doc:§2"},
            {"meta_id": "L1-VModel-001", "meta_class": "Concept", "parent_class": "Vehicle_Model",
             "name": "Model-Y(上海)", "evidence": "doc:§3"},
            {"meta_id": "L1-WS-001", "meta_class": "Asset", "parent_class": "Equipment_Infra",
             "name": "ST-015", "evidence": "doc:§4"},
            {"meta_id": "L1-Line-001", "meta_class": "Asset", "parent_class": "Equipment_Infra",
             "name": "L-Shanghai-01", "evidence": "doc:§4"},
            {"meta_id": "L1-INV-001", "meta_class": "Data", "parent_class": "Data_Record",
             "name": "库存_上海_1200件", "evidence": "doc:§5"},
            {"meta_id": "L1-WH-001", "meta_class": "Actor", "parent_class": "Equipment_Infra",
             "name": "上海仓库", "evidence": "doc:§5"},
            {"meta_id": "L1-ECN-0715", "meta_class": "Process", "parent_class": "Business_Process",
             "name": "ECN-2026-0715", "evidence": "doc:§1",
             "effective_date": "2026-09-01"},
            {"meta_id": "L1-SUP-001", "meta_class": "Actor", "name": "宁德时代", "evidence": "doc:§6"},
        ],
        "L2_entities": [],
        "L2_5_entities": [],
        "L3_entities": [],
        "relationships": [
            {"from": "HV_Battery_Module_001", "to": "MY23_MBOM_SH", "type": "USED_IN",
             "evidence_ref": "doc:§2", "source_cq": "CQ-MBOM-001"},
            {"from": "MY23_MBOM_SH", "to": "Model-Y(上海)", "type": "PART_OF",
             "evidence_ref": "doc:§3", "source_cq": "CQ-MBOM-002"},
            {"from": "HV_Battery_Module_001", "to": "ST-015", "type": "USED_AT",
             "evidence_ref": "doc:§4", "source_cq": "CQ-MBOM-003"},
            {"from": "ST-015", "to": "L-Shanghai-01", "type": "PART_OF",
             "evidence_ref": "doc:§4", "source_cq": "CQ-MBOM-003"},
            {"from": "HV_Battery_Module_001", "to": "库存_上海_1200件", "type": "HAS_INVENTORY",
             "evidence_ref": "doc:§5", "source_cq": "CQ-MBOM-005"},
            {"from": "库存_上海_1200件", "to": "上海仓库", "type": "STORED_AT",
             "evidence_ref": "doc:§5", "source_cq": "CQ-MBOM-005"},
            {"from": "ECN-2026-0715", "to": "HV_Battery_Module_001", "type": "AFFECTS",
             "evidence_ref": "doc:§1", "source_cq": "CQ-MBOM-011"},
            # deliberate noise edge: no business justification registered
            {"from": "宁德时代", "to": "上海仓库", "type": "RELATED_TO", "evidence_ref": "doc:§6"},
        ],
    }


# --- pure graph-traversal helpers over the packet shape ---------------------

def _all_entities(packet: dict) -> list:
    return list(iter_packet_entities(packet))


def _by_id(packet: dict, eid: str) -> dict:
    for ent in _all_entities(packet):
        if ent["id"] == eid:
            return ent
    raise AssertionError(f"entity {eid!r} not found")


def _targets(packet: dict, src_id: str, rel_type: str) -> list:
    return sorted(
        r["target_id"] for r in packet["relations"]
        if r["source_id"] == src_id and r["relation_type"] == rel_type
    )


class PacketSchemaConformanceTest(unittest.TestCase):
    def test_adapted_fixture_conforms_to_packet_schema(self):
        packet = adapt_extraction_json_to_packet(
            _mbom_ecn_legacy_fixture(), document_name="ecn_case.pptx"
        )
        errors = sorted(
            Draft7Validator(SCHEMA).iter_errors(packet),
            key=lambda e: list(e.path),
        )
        self.assertEqual(
            [], [f"{list(e.path)}: {e.message}" for e in errors]
        )


class P0CompetencyQuestionAnswerabilityTest(unittest.TestCase):
    """The five P0 CQs from the reference case must be graph-answerable."""

    def setUp(self):
        self.packet = adapt_extraction_json_to_packet(_mbom_ecn_legacy_fixture())
        self.part = "L1-PART-001"

    def test_cq001_part_used_in_mboms(self):
        mboms = _targets(self.packet, self.part, "USED_IN")
        self.assertIn("L1-MBOM-001", mboms)

    def test_cq002_mbom_maps_to_vehicle_model(self):
        for mbom in _targets(self.packet, self.part, "USED_IN"):
            models = _targets(self.packet, mbom, "PART_OF")
            self.assertIn("L1-VModel-001", models, f"{mbom} lacks vehicle model")

    def test_cq003_part_traces_to_workstation_and_line(self):
        stations = _targets(self.packet, self.part, "USED_AT")
        self.assertIn("L1-WS-001", stations)
        lines = _targets(self.packet, "L1-WS-001", "PART_OF")
        self.assertIn("L1-Line-001", lines)

    def test_cq005_inventory_chain_reaches_warehouse(self):
        inventories = _targets(self.packet, self.part, "HAS_INVENTORY")
        self.assertIn("L1-INV-001", inventories)
        warehouses = _targets(self.packet, "L1-INV-001", "STORED_AT")
        self.assertIn("L1-WH-001", warehouses)

    def test_cq011_ecn_effective_date_via_temporal_discipline(self):
        """T2.2 integration: 生效日期 alias lands in valid_from."""
        ecn = _by_id(self.packet, "L1-ECN-0715")
        self.assertEqual(ecn["valid_from"], "2026-09-01")

    def test_change_impact_chain_ecn_to_vehicle_model(self):
        """Two-hop impact walk: ECN -AFFECTS-> part -USED_IN-> mbom."""
        affected = _targets(self.packet, "L1-ECN-0715", "AFFECTS")
        self.assertIn(self.part, affected)
        self.assertTrue(_targets(self.packet, self.part, "USED_IN"))


class DeadRelationStatsOnRealisticFixtureTest(unittest.TestCase):
    def test_stats_flag_the_noise_edge_only(self):
        packet = adapt_extraction_json_to_packet(_mbom_ecn_legacy_fixture())
        stats = cq_provenance_stats(packet["relations"])
        self.assertEqual(stats["total_relations"], 8)
        self.assertEqual(stats["with_source_cq"], 7)
        self.assertEqual(stats["dead_relations"], 1)
        self.assertEqual(stats["dead_by_type"], {"RELATED_TO": 1})
        self.assertAlmostEqual(stats["dead_ratio"], 0.125)


class DualTrackE2EAttachmentTest(unittest.TestCase):
    def test_unified_result_carries_answerable_packet(self):
        legacy = _mbom_ecn_legacy_fixture()
        result = UnifiedExtractionResult(
            l0_entities=legacy["L0_entities"],
            l1_entities=legacy["L1_entities"],
            l2_entities=legacy["L2_entities"],
            l25_entities=legacy["L2_5_entities"],
            l3_entities=legacy["L3_entities"],
            relationships=legacy["relationships"],
            raw_response="<fixture>",
            processing_time_ms=1.0,
            token_usage={"estimated": 1},
        )
        setattr(result, "metadata", {"source_path": "ecn_case.pptx"})
        with mock.patch.dict(os.environ, {"EXTRACT_OUTPUT_PROTOCOL": "packet"}):
            maybe_attach_ontology_packet(result)

        packet = getattr(result, "ontology_packet")
        errors = list(Draft7Validator(SCHEMA).iter_errors(packet))
        self.assertEqual([], [str(e.message) for e in errors])
        # P0 CQ-001 answerable directly off the attached view
        self.assertIn("L1-MBOM-001", _targets(packet, "L1-PART-001", "USED_IN"))
        # legacy buckets untouched
        self.assertEqual(len(result.l1_entities), 9)


if __name__ == "__main__":
    unittest.main()
