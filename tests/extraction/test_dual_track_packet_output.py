"""Characterization tests for T2.1 dual-track packet output (ROADMAP).

Locks: EXTRACT_OUTPUT_PROTOCOL=packet attaches result.ontology_packet
alongside legacy layer buckets; default (flag off) attaches nothing; adapter
failure never breaks extraction.
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

from ontology_compiler.services.extraction_packet_adapter import (
    iter_packet_entities,
    maybe_attach_ontology_packet,
)
from ontology_compiler.services.true_unified_extractor import (
    TrueUnifiedExtractor,
    UnifiedExtractionResult,
)


def _make_result() -> UnifiedExtractionResult:
    r = UnifiedExtractionResult(
        l0_entities=[{"meta_id": "L0-001", "meta_class": "Asset", "name": "Part"}],
        l1_entities=[
            {"meta_id": "L1-001", "meta_class": "Asset", "parent_class": "Part",
             "name": "Battery Module"}
        ],
        l2_entities=[],
        l25_entities=[],
        l3_entities=[
            {"meta_id": "L3-001", "name": "HV_Battery_Module_001", "parent_id": "L1-001"}
        ],
        relationships=[
            {"from": "HV_Battery_Module_001", "to": "Battery Module",
             "type": "USED_IN", "evidence_ref": "doc:§2"}
        ],
        raw_response="{}",
        processing_time_ms=1.0,
        token_usage={"estimated": 1},
    )
    setattr(r, "metadata", {"source_path": "/tmp/doc.pptx"})
    return r


class DualTrackPacketAttachmentTest(unittest.TestCase):
    def test_flag_off_attaches_nothing(self):
        result = _make_result()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EXTRACT_OUTPUT_PROTOCOL", None)
            maybe_attach_ontology_packet(result)
        self.assertFalse(hasattr(result, "ontology_packet"))

    def test_flag_on_attaches_valid_packet(self):
        result = _make_result()
        with mock.patch.dict(os.environ, {"EXTRACT_OUTPUT_PROTOCOL": "packet"}):
            maybe_attach_ontology_packet(result)
        packet = getattr(result, "ontology_packet")
        ids = {e["id"] for e in iter_packet_entities(packet)}
        self.assertEqual(ids, {"L0-001", "L1-001", "L3-001"})
        layers = {e["id"]: e["layer"] for e in iter_packet_entities(packet)}
        self.assertEqual(layers["L3-001"], "L3")
        # relation endpoint name → id resolution across buckets
        rel = packet["relations"][0]
        self.assertEqual(rel["source_id"], "L3-001")
        self.assertEqual(rel["target_id"], "L1-001")
        # legacy attributes untouched (dual-track coexistence)
        self.assertEqual(result.l1_entities[0]["meta_id"], "L1-001")

    def test_adapter_failure_is_swallowed(self):
        result = _make_result()
        with mock.patch.dict(os.environ, {"EXTRACT_OUTPUT_PROTOCOL": "packet"}):
            with mock.patch(
                "ontology_compiler.services.extraction_packet_adapter"
                ".adapt_extraction_json_to_packet",
                side_effect=RuntimeError("boom"),
            ):
                # must not raise
                maybe_attach_ontology_packet(result)
        self.assertFalse(hasattr(result, "ontology_packet"))


if __name__ == "__main__":
    unittest.main()
