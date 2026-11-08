"""Tests for real local hybrid smoke and entity drift reporting."""
from __future__ import annotations

import unittest

from src.application.rag.entity_drift import entity_drift_report


class EntityDriftReportTest(unittest.TestCase):
    def test_bucketed_snapshot_delta(self):
        before = {
            "entities": {"L1": [
                {"id": "A", "name": "Part A", "type": "Asset", "layer": "L1", "version": "1"},
                {"id": "B", "name": "Part B", "type": "Asset", "layer": "L1"},
            ]}
        }
        after = {
            "entities": {"L1": [
                {"id": "A", "name": "Part A updated", "type": "Asset", "layer": "L1", "version": "2"},
                {"id": "C", "name": "Part C", "type": "Asset", "layer": "L1"},
            ]}
        }
        report = entity_drift_report(before, after)
        self.assertEqual(report["added_ids"], ["C"])
        self.assertEqual(report["removed_ids"], ["B"])
        self.assertEqual(report["changed_ids"], ["A"])
        self.assertEqual(report["unchanged_ids"], [])
        self.assertEqual(report["changed_ratio"], 1.5)

    def test_same_snapshot_is_zero_drift(self):
        packet = {"entities": {"L1": [{"id": "A", "name": "a"}]}}
        report = entity_drift_report(packet, packet)
        self.assertEqual(report["changed_ratio"], 0.0)
        self.assertEqual(report["unchanged_ids"], ["A"])

    def test_flat_packet_and_empty_are_safe(self):
        report = entity_drift_report(
            {"entities": [{"id": "A", "name": "a"}]},
            {"entities": {}},
        )
        self.assertEqual(report["removed_ids"], ["A"])


if __name__ == "__main__":
    unittest.main()
