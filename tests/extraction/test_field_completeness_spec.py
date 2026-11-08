"""Field completeness follows ENTITY_CLASSIFICATION_SPEC.md §3 field names."""
from __future__ import annotations

import unittest

from ontology_compiler.services.quality_assurance.base import QualityContext
from ontology_compiler.services.quality_assurance.quality_rules import FieldCompletenessRule


class FieldCompletenessSpecTest(unittest.TestCase):
    def setUp(self):
        self.rule = FieldCompletenessRule()
        self.ctx = QualityContext(fragment={})

    def test_packet_names_and_legacy_names_score_the_same(self):
        packet = {"id": "E1", "name": "Part A", "layer": "L1", "type": "Asset"}
        legacy = {"meta_id": "E1", "meta_name": "Part A", "layer": "L1", "meta_class": "Asset"}
        self.assertEqual(
            self.rule._score_entity_fields(packet, self.ctx),
            self.rule._score_entity_fields(legacy, self.ctx),
        )
        self.assertEqual(self.ctx.issues, [])

    def test_missing_spec_field_is_reported_by_canonical_name(self):
        self.rule._check_required_fields(
            {"meta_id": "E1", "meta_name": "Part A", "meta_class": "Asset"},
            self.ctx,
        )
        self.assertTrue(any("layer" in issue for issue in self.ctx.issues))
        self.assertFalse(any("meta_id" in issue for issue in self.ctx.issues))


if __name__ == "__main__":
    unittest.main()
