from __future__ import annotations

"""Tests for the E (golden semantic regression) gate.

Locks recall/precision/F1 semantics of the golden-set matcher and the
manifest loader, without any LLM / network / filesystem dependency.
"""

import unittest

from scripts.golden_semantic_eval import (
    GoldenEntity,
    GoldenRelation,
    GoldenSample,
    golden_metrics,
    load_golden_set,
)


class GoldenMetricsTest(unittest.TestCase):
    def test_full_match_recall_100(self):
        sample = GoldenSample(
            sample_id="s1",
            entities=[
                GoldenEntity(name="MBOM", meta_class="System", layer="L1"),
            ],
            relations=[
                GoldenRelation(source_name="MBOM", target_name="BOM", relation_types=["DERIVED_FROM"]),
            ],
        )
        observed_entities = [
            {"meta_id": "L1-001", "name": "MBOM", "meta_class": "System", "layer": "L1"},
            {"meta_id": "L1-002", "name": "BOM", "meta_class": "System", "layer": "L1"},
        ]
        observed_relations = [
            {"source": "L1-001", "target": "L1-002", "relation_type": "DERIVED_FROM"},
        ]
        metrics = golden_metrics(sample, observed_entities, observed_relations)
        self.assertEqual(metrics["entity_recall"], 1.0)
        self.assertEqual(metrics["relation_recall"], 1.0)
        self.assertTrue(metrics["ok"])
        self.assertEqual(metrics["matched_entities"], 1)
        self.assertEqual(metrics["matched_relations"], 1)

    def test_missing_entity_lowers_recall(self):
        sample = GoldenSample(sample_id="s1", entities=[GoldenEntity(name="Missing", meta_class="System")])
        metrics = golden_metrics(sample, [], [])
        self.assertEqual(metrics["entity_recall"], 0.0)
        self.assertFalse(metrics["ok"])
        self.assertEqual(metrics["unmatched_entities"][0]["name"], "Missing")

    def test_layer_requirement_blocks_match(self):
        sample = GoldenSample(sample_id="s1", entities=[GoldenEntity(name="MBOM", meta_class="System", layer="L2")])
        observed_entities = [{"name": "MBOM", "meta_class": "System", "layer": "L1"}]
        metrics = golden_metrics(sample, observed_entities, [])
        # name+class match but layer differs -> not matched
        self.assertEqual(metrics["entity_recall"], 0.0)

    def test_rejected_entity_is_false_positive(self):
        sample = GoldenSample(
            sample_id="s1",
            rejected_entities=[GoldenEntity(name="GHOST", meta_class="Concept")],
        )
        observed_entities = [{"name": "GHOST", "meta_class": "Concept", "layer": "L1"}]
        metrics = golden_metrics(sample, observed_entities, [])
        self.assertEqual(len(metrics["false_positives"]), 1)
        self.assertFalse(metrics["ok"])

    def test_rejected_relation_type_flagged(self):
        sample = GoldenSample(
            sample_id="s1",
            reject_relation_types=["RELATED_TO"],
        )
        observed_relations = [{"source": "A", "target": "B", "relation_type": "RELATED_TO"}]
        metrics = golden_metrics(sample, [], observed_relations)
        self.assertEqual(len(metrics["rejected_relation_observations"]), 1)
        self.assertFalse(metrics["ok"])

    def test_relation_match_by_name_resolution(self):
        sample = GoldenSample(
            sample_id="s1",
            relations=[GoldenRelation(source_name="MBOM", target_name="BOM", relation_types=["DERIVED_FROM"])],
        )
        observed_entities = [
            {"meta_id": "L1-001", "name": "MBOM", "meta_class": "System"},
            {"meta_id": "L1-002", "name": "BOM", "meta_class": "System"},
        ]
        observed_relations = [{"source": "L1-001", "target": "L1-002", "relation_type": "DERIVED_FROM"}]
        metrics = golden_metrics(sample, observed_entities, observed_relations)
        # endpooints resolved by name -> ids
        self.assertEqual(metrics["matched_relations"], 1)

    def test_load_golden_set_from_manifest_block(self):
        data = {
            "sample_id": "mbom-du-v2",
            "entities": [{"name": "MBOM", "meta_class": "System", "layer": "L1"}],
            "relations": [{"source": "MBOM", "target": "BOM", "types": ["DERIVED_FROM"]}],
            "rejected_entities": [{"name": "GHOST", "meta_class": "Concept"}],
            "reject_relation_types": ["RELATED_TO"],
        }
        sample = load_golden_set(data)
        self.assertEqual(sample.sample_id, "mbom-du-v2")
        self.assertEqual(len(sample.entities), 1)
        self.assertEqual(sample.entities[0].name, "MBOM")
        self.assertEqual(len(sample.relations), 1)
        self.assertEqual(sample.relations[0].relation_types, ["DERIVED_FROM"])
        self.assertEqual(sample.reject_relation_types, ["RELATED_TO"])

    def test_empty_golden_is_ok(self):
        sample = GoldenSample(sample_id="s1")
        metrics = golden_metrics(sample, [], [])
        self.assertTrue(metrics["ok"])
        self.assertEqual(metrics["entity_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()