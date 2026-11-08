"""M4 (2026-09-02) characterization tests: CQ 抽取约束 (focus vocabulary + coverage).

Locks the behavior that (CQ_PIPELINE_IMPLEMENTATION_PLAN.md §9-M4):

1. ``CQRegistry.focus_vocabulary`` aggregates non-deprecated CQ terms into a
   de-duplicated 关注词表 (entities/relations/attributes) with a source map
   back to the originating cq_ids.
2. ``extract_cq_coverage`` computes read-only coverage statistics over an
   extract fragment — ``cq_entity_coverage`` / ``cq_relation_coverage``,
   present/missing focus terms, per-CQ detail, and 漏抽提示 (missing term →
   originating CQ ids). It never mutates fragment data (CQ 只做约束/提示).
3. Zero-regression: no catalog / only deprecated CQs → ``None``, extract
   stays byte-identical to the pre-CQ baseline.

All tests are hermetic: tmp_path file I/O only, no LLM calls.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.application.cq.cq_registry import CQRegistry
from src.pipeline.cq_integration import (
    extract_cq_coverage,
    extract_cq_coverage_artifact,
)


def _write_catalog(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    return path


def _cq(cq_id: str, status: str = "frozen", **overrides) -> dict:
    record = {
        "cq_id": cq_id,
        "cq_text": "业务问题",
        "entities": ["Part", "Model"],
        "relations": ["part_of"],
        "query_path": "Part -> part_of -> Model",
        "status": status,
    }
    record.update(overrides)
    return record


class FocusVocabularyTest(unittest.TestCase):
    def test_aggregates_and_dedups_across_active_cqs(self):
        registry = CQRegistry.load(_write_catalog(Path(tempfile.mkdtemp()) / "c.jsonl", [
            _cq("CQ-0001"),
            _cq("CQ-0002", entities=["Part", "Vehicle"], relations=["affects"]),
        ]))
        vocab = registry.focus_vocabulary()
        self.assertEqual(vocab["entities"], ["Part", "Model", "Vehicle"])
        self.assertEqual(vocab["relations"], ["part_of", "affects"])
        # source traceability: shared term lists both CQ ids
        self.assertEqual(vocab["sources"]["entity:part"], ["CQ-0001", "CQ-0002"])
        self.assertEqual(vocab["sources"]["relation:affects"], ["CQ-0002"])

    def test_deprecated_cqs_excluded(self):
        tmp = Path(tempfile.mkdtemp()) / "c.jsonl"
        CQRegistry.load(_write_catalog(tmp, [
            _cq("CQ-0001", status="deprecated"),
            _cq("CQ-0002", status="frozen", entities=["Vehicle"], relations=["drives"]),
        ]))
        registry = CQRegistry.load(tmp)
        vocab = registry.focus_vocabulary()
        self.assertEqual(vocab["entities"], ["Vehicle"])
        self.assertNotIn("entity:part", vocab["sources"])

    def test_attributes_aggregated(self):
        tmp = Path(tempfile.mkdtemp()) / "c.jsonl"
        CQRegistry.load(_write_catalog(tmp, [
            _cq("CQ-0001", attributes=["effective_time", "status"]),
        ]))
        registry = CQRegistry.load(tmp)
        self.assertEqual(registry.focus_vocabulary()["attributes"], ["effective_time", "status"])


class ExtractCoverageTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _catalog(self, records: list[dict]) -> Path:
        return _write_catalog(self.root / "cq" / "catalog.jsonl", records)

    def _nodes(self, *names: str) -> list[dict]:
        return [{"meta_name": n} for n in names]

    def _relations(self, *types: str) -> list[dict]:
        return [{"type": t} for t in types]

    def test_full_coverage_when_all_focus_terms_present(self):
        catalog = self._catalog([_cq("CQ-0001")])
        coverage = extract_cq_coverage(
            self._nodes("Harness Part", "Vehicle Model"),
            [],
            self._relations("PART_OF"),
            catalog_path=catalog,
        )
        self.assertEqual(coverage["cq_entity_coverage"], 1.0)
        self.assertEqual(coverage["cq_relation_coverage"], 1.0)
        self.assertEqual(coverage["missing_focus_entities"], [])
        self.assertEqual(coverage["missing_focus_relations"], [])
        self.assertEqual(coverage["missing_hints"], [])
        self.assertEqual(coverage["focus_vocabulary"]["entity_count"], 2)
        per_cq = coverage["per_cq"][0]
        self.assertEqual(per_cq["cq_id"], "CQ-0001")
        self.assertEqual(per_cq["entity_coverage"], 1.0)

    def test_partial_coverage_emits_missing_terms_and_hints(self):
        catalog = self._catalog([
            _cq("CQ-0001", entities=["Part", "Vehicle"], relations=["part_of", "affects"]),
        ])
        coverage = extract_cq_coverage(
            self._nodes("Harness Part"),
            [],
            self._relations("part_of"),
            catalog_path=catalog,
        )
        self.assertEqual(coverage["cq_entity_coverage"], 0.5)
        self.assertEqual(coverage["cq_relation_coverage"], 0.5)
        self.assertEqual(coverage["missing_focus_entities"], ["Vehicle"])
        self.assertEqual(coverage["missing_focus_relations"], ["affects"])
        # 漏抽提示 carries the originating CQ id
        hints = {(h["kind"], h["term"]): h["cq_ids"] for h in coverage["missing_hints"]}
        self.assertEqual(hints[("entity", "Vehicle")], ["CQ-0001"])
        self.assertEqual(hints[("relation", "affects")], ["CQ-0001"])

    def test_attribute_gap_hint(self):
        catalog = self._catalog([
            _cq("CQ-0001", attributes=["effective_time"]),
        ])
        coverage = extract_cq_coverage(
            self._nodes("Harness Part", "Vehicle Model"),
            [],
            self._relations("part_of"),
            catalog_path=catalog,
        )
        self.assertEqual(coverage["cq_attribute_coverage"], 0.0)
        self.assertEqual(coverage["missing_focus_attributes"], ["effective_time"])
        attr_hints = [h for h in coverage["missing_hints"] if h["kind"] == "attribute"]
        self.assertEqual(attr_hints[0]["cq_ids"], ["CQ-0001"])

    def test_does_not_mutate_fragment(self):
        catalog = self._catalog([_cq("CQ-0001")])
        nodes = self._nodes("Harness Part")
        relationships = self._relations("part_of")
        before = json.dumps(nodes, ensure_ascii=False)
        rel_before = json.dumps(relationships, ensure_ascii=False)
        extract_cq_coverage(nodes, [], relationships, catalog_path=catalog)
        self.assertEqual(json.dumps(nodes, ensure_ascii=False), before)
        self.assertEqual(json.dumps(relationships, ensure_ascii=False), rel_before)

    def test_missing_catalog_returns_none(self):
        self.assertIsNone(
            extract_cq_coverage(self._nodes("Part"), [], self._relations("part_of"),
                                catalog_path=self.root / "nope.jsonl")
        )

    def test_only_deprecated_cqs_returns_none(self):
        catalog = self._catalog([_cq("CQ-0001", status="deprecated")])
        self.assertIsNone(
            extract_cq_coverage(self._nodes("Part"), [], self._relations("part_of"),
                                catalog_path=catalog)
        )


class ExtractCoverageArtifactTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_artifact_written_under_cq_dir(self):
        catalog = _write_catalog(self.root / "c.jsonl", [_cq("CQ-0001")])
        coverage = extract_cq_coverage(
            [{"meta_name": "Harness Part"}, {"meta_name": "Vehicle Model"}],
            [],
            [{"type": "part_of"}],
            catalog_path=catalog,
        )
        path = extract_cq_coverage_artifact(coverage, self.root)
        self.assertIsNotNone(path)
        self.assertEqual(Path(path).name, "cq_extract_coverage.json")
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.assertEqual(payload["cq_entity_coverage"], 1.0)
        self.assertIn("per_cq", payload)

    def test_none_inputs_skip(self):
        self.assertIsNone(extract_cq_coverage_artifact(None, self.root))
        self.assertIsNone(extract_cq_coverage_artifact({"cq_entity_coverage": 1.0}, None))


if __name__ == "__main__":
    unittest.main()
