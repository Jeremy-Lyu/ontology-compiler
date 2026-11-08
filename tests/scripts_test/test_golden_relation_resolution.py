"""Regression checks for name-based golden relation endpoint resolution."""
from __future__ import annotations

import unittest

from scripts.golden_semantic_eval import golden_metrics, load_golden_set


class GoldenRelationResolutionTest(unittest.TestCase):
    def test_golden_names_resolve_to_observed_ids(self):
        golden = load_golden_set({
            "sample_id": "r",
            "entities": [
                {"name": "Cycle Stock", "meta_class": "Concept", "layer": "L2"},
                {"name": "Inventory", "meta_class": "Concept", "layer": "L2"},
            ],
            "relations": [{"source": "Cycle Stock", "target": "Inventory", "types": ["PART_OF"]}],
        })
        observed_entities = [
            {"id": "L2-1", "name": "Cycle Stock", "type": "Concept", "layer": "L2"},
            {"id": "L2-2", "name": "Inventory", "type": "Concept", "layer": "L2"},
        ]
        observed_relations = [{
            "source_id": "L2-1", "target_id": "L2-2", "relation_type": "PART_OF"
        }]
        result = golden_metrics(golden, observed_entities, observed_relations)
        self.assertEqual(result["relation_recall"], 1.0)
        self.assertEqual(result["unmatched_relations"], [])

    def test_alias_endpoint_is_accepted(self):
        golden = load_golden_set({
            "relations": [{"source": "MBOM", "target": "BOM", "types": ["DERIVED_FROM"]}],
        })
        entities = [
            {"id": "m", "name": "Manufacturing BOM", "aliases": ["MBOM"]},
            {"id": "b", "name": "Bill of Materials", "aliases": ["BOM"]},
        ]
        relations = [{"source_id": "m", "target_id": "b", "relation_type": "DERIVED_FROM"}]
        self.assertEqual(golden_metrics(golden, entities, relations)["relation_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
