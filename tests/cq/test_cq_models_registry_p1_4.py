"""P1-4 (2026-09-01) characterization tests: CQ M0 contract + M1 registry.

Locks the behavior that:
1. ``CQ`` requires cq_id/cq_text/entities/relations/query_path and a known
   status; list fields are stripped; transitions are linear to deprecated.
2. ``CQCatalog`` round-trips JSONL and filters by status/frozen.
3. ``CQRegistry.load`` is tolerant: missing file → empty registry + warning;
   malformed lines are skipped (never raised).

All tests are hermetic: no network, no real LLM calls, tmp_path file I/O only.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from src.application.cq.cq_models import (
    CQ,
    CQCatalog,
    CQStatus,
    CQValidationResult,
    allowed_transitions,
)
from src.application.cq.cq_registry import CQRegistry, default_catalog_path


def _valid_payload(**overrides) -> dict:
    payload = {
        "cq_id": "CQ-9001",
        "cq_text": "哪些车型使用了该线束？",
        "entities": ["Part", "Model"],
        "relations": ["part_of"],
        "query_path": "Part -> part_of -> Model",
        "status": "frozen",
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# 1. CQ model contract (M0)
# ---------------------------------------------------------------------------


class CQModelContractTest(unittest.TestCase):
    def test_minimal_required_fields_roundtrip(self):
        cq = CQ.model_validate(_valid_payload())
        self.assertEqual(cq.cq_id, "CQ-9001")
        self.assertEqual(cq.status, CQStatus.FROZEN.value)
        self.assertEqual(cq.version, "v1")
        self.assertEqual(cq.attributes, [])

    def test_missing_required_field_raises(self):
        for field in ("cq_id", "cq_text", "entities", "relations", "query_path"):
            payload = _valid_payload()
            payload.pop(field)
            with self.assertRaises(ValidationError):
                CQ.model_validate(payload)

    def test_unknown_status_rejected(self):
        with self.assertRaises(ValidationError):
            CQ.model_validate(_valid_payload(status="published"))

    def test_string_lists_are_stripped(self):
        cq = CQ.model_validate(
            _valid_payload(entities=["  Part ", "", "Model"], relations=[" part_of "])
        )
        self.assertEqual(cq.entities, ["Part", "Model"])
        self.assertEqual(cq.relations, ["part_of"])

    def test_transition_matrix_is_linear(self):
        self.assertEqual(
            allowed_transitions(CQStatus.DRAFT), frozenset({"reviewed", "deprecated"})
        )
        self.assertEqual(
            allowed_transitions(CQStatus.REVIEWED), frozenset({"frozen", "deprecated"})
        )
        self.assertEqual(allowed_transitions(CQStatus.FROZEN), frozenset({"deprecated"}))
        self.assertEqual(allowed_transitions(CQStatus.DEPRECATED), frozenset())

    def test_can_transition_to(self):
        cq = CQ.model_validate(_valid_payload(status="reviewed"))
        self.assertTrue(cq.can_transition_to("frozen"))
        self.assertTrue(cq.can_transition_to(CQStatus.DEPRECATED))
        self.assertFalse(cq.can_transition_to("draft"))


class CQValidationResultContractTest(unittest.TestCase):
    def test_defaults(self):
        result = CQValidationResult(cq_id="CQ-9001", executable=False)
        self.assertEqual(result.missing_entities, [])
        self.assertEqual(result.missing_relations, [])
        self.assertEqual(result.notes, "")


# ---------------------------------------------------------------------------
# 2. CQCatalog (M0)
# ---------------------------------------------------------------------------


class CQCatalogTest(unittest.TestCase):
    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
            encoding="utf-8",
        )

    def test_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "catalog.jsonl"
            self._write_jsonl(path, [_valid_payload(), _valid_payload(cq_id="CQ-9002", status="draft")])
            catalog = CQCatalog.from_jsonl(path)
            self.assertEqual(len(catalog), 2)
            self.assertEqual(catalog.get("CQ-9002").status, "draft")
            out = Path(tmp) / "roundtrip.jsonl"
            catalog.to_jsonl(out)
            self.assertEqual(len(CQCatalog.from_jsonl(out)), 2)

    def test_by_status_and_frozen(self):
        catalog = CQCatalog(
            items=[
                CQ.model_validate(_valid_payload()),
                CQ.model_validate(_valid_payload(cq_id="CQ-9003", status="draft")),
            ]
        )
        self.assertEqual(len(catalog.frozen()), 1)
        self.assertEqual(len(catalog.by_status("draft")), 1)
        self.assertIsNone(catalog.get("CQ-NOPE"))


# ---------------------------------------------------------------------------
# 3. CQRegistry tolerant loading (M1)
# ---------------------------------------------------------------------------


class CQRegistryTest(unittest.TestCase):
    def test_missing_file_yields_empty_registry_with_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = CQRegistry.load(Path(tmp) / "nope.jsonl")
            self.assertEqual(len(registry), 0)
            self.assertEqual(len(registry.warnings), 1)
            self.assertIn("not found", registry.warnings[0])

    def test_malformed_lines_skipped_with_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "catalog.jsonl"
            good = json.dumps(_valid_payload(), ensure_ascii=False)
            path.write_text(
                "\n".join(["{not json", good, "", '{"cq_id": "broken"}']) + "\n",
                encoding="utf-8",
            )
            registry = CQRegistry.load(path)
            self.assertEqual(len(registry), 1)
            self.assertEqual(len(registry.warnings), 2)
            self.assertIn("line 1", registry.warnings[0])
            self.assertIn("line 4", registry.warnings[1])

    def test_default_catalog_path_points_to_seed(self):
        path = default_catalog_path()
        self.assertEqual(path.name, "seed_cq.jsonl")
        self.assertEqual(path.parent.name, "cq")


if __name__ == "__main__":
    unittest.main()
