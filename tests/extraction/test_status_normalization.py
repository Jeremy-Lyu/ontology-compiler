"""Characterization tests for entity ``status`` normalization.

Locks the behavior introduced to fix the R2 fact-source split between
``schemas/ontology_packet.schema.json`` (canonical enum) and the runtime
valid-status sets that used to live inline in ``entity_normalizer.py`` and
``merge.py``.

Key guarantee under test: LLM-extraction-emitted ``planned`` is coerced to
``proposed`` (a legal schema enum value) before it can reach schema validation.
"""

import json
import unittest
from pathlib import Path

from ontology_compiler.services.entity_normalizer import (
    _VALID_STATUSES,
    normalize_entities,
    normalize_status,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = PROJECT_ROOT / "schemas" / "ontology_packet.schema.json"


def _schema_status_enum():
    """Read the canonical ``status`` enum from the packet schema."""
    with SCHEMA_PATH.open("r", encoding="utf-8") as fh:
        schema = json.load(fh)
    # Entity definition lives under definitions/entity (entities per-layer
    # arrays reference it via $ref). The status enum is on that definition.
    entity_props = schema.get("definitions", {}).get("entity", {}).get("properties", {})
    return set(entity_props.get("status", {}).get("enum", []))


class TestNormalizeStatusAliases(unittest.TestCase):
    """Alias mapping: non-canonical values produced by LLM/history → canonical."""

    def test_planned_maps_to_proposed(self):
        # The primary fix: schema rejects "planned", allows "proposed".
        self.assertEqual(normalize_status("planned"), "proposed")

    def test_inactive_maps_to_deprecated(self):
        self.assertEqual(normalize_status("inactive"), "deprecated")

    def test_unknown_maps_to_draft(self):
        self.assertEqual(normalize_status("unknown"), "draft")

    def test_alias_mapping_is_case_insensitive(self):
        self.assertEqual(normalize_status("PLANNED"), "proposed")
        self.assertEqual(normalize_status("Planned"), "proposed")
        self.assertEqual(normalize_status("INACTIVE"), "deprecated")


class TestNormalizeStatusCanonical(unittest.TestCase):
    """Canonical legal values pass through unchanged."""

    def test_canonical_values_pass_through(self):
        for value in ["active", "deprecated", "proposed", "draft", "ghost", "archived"]:
            self.assertEqual(normalize_status(value), value)

    def test_canonical_values_case_insensitive(self):
        self.assertEqual(normalize_status("Active"), "active")
        self.assertEqual(normalize_status("PROPOSED"), "proposed")
        self.assertEqual(normalize_status("Draft"), "draft")


class TestNormalizeStatusDefaults(unittest.TestCase):
    """Empty / None / illegal values fall back to ``active``."""

    def test_none_returns_active(self):
        self.assertEqual(normalize_status(None), "active")

    def test_empty_string_returns_active(self):
        self.assertEqual(normalize_status(""), "active")
        self.assertEqual(normalize_status("   "), "active")

    def test_literal_none_string_returns_active(self):
        # merge.py historically treated the literal "None" string as missing.
        self.assertEqual(normalize_status("None"), "active")

    def test_non_string_returns_active(self):
        self.assertEqual(normalize_status(123), "active")
        self.assertEqual(normalize_status([]), "active")

    def test_illegal_value_returns_active(self):
        self.assertEqual(normalize_status("foobar"), "active")
        self.assertEqual(normalize_status("retired"), "active")


class TestNormalizeStatusSchemaAlignment(unittest.TestCase):
    """R2 guard: runtime valid set MUST equal the schema enum (no re-split)."""

    def test_valid_statuses_match_schema_enum(self):
        if not SCHEMA_PATH.exists():
            self.skipTest(f"schema not found at {SCHEMA_PATH}")
        schema_enum = _schema_status_enum()
        self.assertTrue(schema_enum, "could not read status enum from schema")
        self.assertEqual(_VALID_STATUSES, schema_enum)

    def test_alias_targets_are_all_canonical(self):
        # Every alias must map to a value that is itself legal — otherwise the
        # alias would just move the problem instead of fixing it.
        from ontology_compiler.services.entity_normalizer import _STATUS_ALIASES

        for src, dst in _STATUS_ALIASES.items():
            self.assertIn(
                dst, _VALID_STATUSES,
                f"alias {src!r} → {dst!r} but {dst!r} is not a canonical status",
            )


class TestNormalizeEntitiesIntegration(unittest.TestCase):
    """End-to-end: normalize_entities coerces status on real entity dicts."""

    def test_planned_entity_is_coerced_to_proposed(self):
        l1 = [{"meta_id": "L1-001", "name": "Planned System", "status": "planned"}]
        result = normalize_entities([], l1, [], [], [])
        self.assertEqual(result[1][0]["status"], "proposed")

    def test_missing_status_defaults_to_active(self):
        l2 = [{"meta_id": "L2-001", "name": "No Status System"}]
        result = normalize_entities([], [], l2, [], [])
        self.assertEqual(result[2][0]["status"], "active")

    def test_canonical_status_preserved_through_normalize_entities(self):
        l0 = [{"meta_id": "L0-001", "name": "Concept", "status": "draft"}]
        result = normalize_entities(l0, [], [], [], [])
        self.assertEqual(result[0][0]["status"], "draft")

    def test_illegal_status_reset_to_active(self):
        l3 = [{"meta_id": "L3-001", "name": "Bad Status Part", "status": "retired"}]
        result = normalize_entities([], [], [], [], l3)
        self.assertEqual(result[4][0]["status"], "active")


if __name__ == "__main__":
    unittest.main()
