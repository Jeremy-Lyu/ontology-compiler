"""Characterization tests for T2.2 temporal discipline (ROADMAP).

Locks: adapter maps T1/T2/T3 + version-interval aliases (EN/CN) into the
canonical packet entity fields; three time axes stay independent; absent
fields are simply omitted.
"""
from __future__ import annotations

import unittest

from ontology_compiler.services.extraction_packet_adapter import (
    _extract_temporal_fields,
    adapt_extraction_json_to_packet,
    iter_packet_entities,
)



class TemporalExtractionTest(unittest.TestCase):
    def test_cn_aliases_mapped_to_canonical_fields(self):
        raw = {
            "meta_id": "ECN-001",
            "name": "C-201 国产化切换",
            "attributes": [
                {"name": "计划切换日期", "value": "2024-09-01"},
                {"name": "供货日期", "value": "2024-08-20"},
                {"name": "装车日期", "value": "2024-09-15"},
            ],
        }
        fields = _extract_temporal_fields(raw, {"计划切换日期": "2024-09-01"})
        packet = adapt_extraction_json_to_packet({"L1_entities": [raw]})
        ent = list(iter_packet_entities(packet))[0]
        self.assertEqual(ent["t1_planned_switch_date"], "2024-09-01")
        self.assertEqual(ent["t2_supplier_delivery_date"], "2024-08-20")
        self.assertEqual(ent["t3_actual_install_date"], "2024-09-15")

    def test_en_aliases_and_effective_interval(self):
        raw = {
            "meta_id": "V2",
            "name": "H-A version 2",
            "effective_date": "2024-06-01",
            "attributes": {"valid_to": "2024-09-01"},
        }
        fields = _extract_temporal_fields(raw, {"valid_to": "2024-09-01"})
        self.assertEqual(fields["valid_from"], "2024-06-01")
        self.assertEqual(fields["valid_to"], "2024-09-01")

    def test_three_axes_stay_independent(self):
        raw = {
            "meta_id": "E1",
            "name": "x",
            "t1": "2026-01-01",
            "attributes": {},
        }
        fields = _extract_temporal_fields(raw, {})
        self.assertEqual(set(fields), {"t1_planned_switch_date"})

    def test_absent_temporal_info_omitted(self):
        packet = adapt_extraction_json_to_packet(
            {"L1_entities": [{"meta_id": "N1", "name": "plain node"}]}
        )
        ent = list(iter_packet_entities(packet))[0]
        for key in (
            "t1_planned_switch_date",
            "t2_supplier_delivery_date",
            "t3_actual_install_date",
            "valid_from",
            "valid_to",
        ):
            self.assertNotIn(key, ent)

    def test_top_level_field_wins_over_attribute(self):
        raw = {
            "meta_id": "E2",
            "name": "y",
            "t1": "2026-01-01",
            "attributes": {"t1_planned_switch_date": "1999-01-01"},
        }
        fields = _extract_temporal_fields(raw, {"t1_planned_switch_date": "1999-01-01"})
        self.assertEqual(fields["t1_planned_switch_date"], "2026-01-01")


if __name__ == "__main__":
    unittest.main()
