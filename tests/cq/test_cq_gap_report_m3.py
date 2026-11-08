"""M3 (2026-09-01) characterization tests: CQ 验证闭环 (gap report + mapping).

Locks the behavior that (CQ_PIPELINE_IMPLEMENTATION_PLAN.md §9-M3):

1. ``audit_cq_validation`` walks the ``query_path`` chain and localizes
   breaks (``position N: ...``), and checks CQ attributes against the master
   attribute vocabulary; ``executable`` now covers all four gap kinds.
2. ``audit_cq_gap_report_artifact`` writes ``cq_gap_report.md`` whose detail
   section names the missing entities/relations/attributes and chain breaks.
3. ``load_cq_mapping`` is tolerant (missing/malformed → ``{}``) and the
   shipped ``config/cq/cq_mapping.json`` binds CQ-0004/0005 and marks
   CQ-0001/0002/0003 unbound.
4. Zero-regression: no frozen CQ → no gap report, audit behaviour unchanged.

Acceptance scenario (计划 §9-M3 验收): a "必然缺东西" master (bi-temporal
口径缺失) must make the gap report point out the concrete missing items.

All tests are hermetic: tmp_path file I/O only, no LLM calls.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.application.cq.cq_models import CQ, CQCatalog, CQValidationResult
from src.application.cq.cq_registry import load_cq_mapping
from src.pipeline.cq_integration import (
    audit_cq_gap_report_artifact,
    audit_cq_validation,
    parse_query_path,
)
from src.application.cq.cq_artifacts import write_cq_gap_report


def _write_catalog(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# 1. CQValidationResult M3 extension
# ---------------------------------------------------------------------------


class CQValidationResultExtensionTest(unittest.TestCase):
    def test_m3_fields_default_empty(self):
        result = CQValidationResult(cq_id="CQ-9001", executable=False)
        self.assertEqual(result.missing_attributes, [])
        self.assertEqual(result.chain_breaks, [])

    def test_has_gap_reflects_all_four_kinds(self):
        base = {"cq_id": "CQ-9001", "executable": False}
        self.assertFalse(CQValidationResult(**base).has_gap)
        for field, value in [
            ("missing_entities", ["Part"]),
            ("missing_relations", ["part_of"]),
            ("missing_attributes", ["effective_time"]),
            ("chain_breaks", ["position 1: relation 'affects' not found in master"]),
        ]:
            self.assertTrue(CQValidationResult(**base, **{field: value}).has_gap)


# ---------------------------------------------------------------------------
# 2. query_path parsing
# ---------------------------------------------------------------------------


class ParseQueryPathTest(unittest.TestCase):
    def test_alternating_entity_relation_segments(self):
        parsed = parse_query_path("A -> rel1 -> B -> rel2 -> C")
        self.assertEqual(
            parsed,
            [("entity", "A"), ("relation", "rel1"), ("entity", "B"), ("relation", "rel2"), ("entity", "C")],
        )

    def test_filter_suffix_dropped(self):
        parsed = parse_query_path("A -> rel -> B | filter: x < y")
        self.assertEqual(parsed, [("entity", "A"), ("relation", "rel"), ("entity", "B")])

    def test_empty_path(self):
        self.assertEqual(parse_query_path(""), [])


# ---------------------------------------------------------------------------
# 3. audit_cq_validation M3 semantics
# ---------------------------------------------------------------------------


class AuditValidationM3Test(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _catalog(self, records: list[dict]) -> Path:
        return _write_catalog(self.root / "cq" / "catalog.jsonl", records)

    def test_chain_break_localizes_missing_relation(self):
        catalog = self._catalog(
            [
                {
                    "cq_id": "CQ-9101",
                    "cq_text": "q",
                    "entities": ["Part"],
                    "relations": ["part_of"],
                    "query_path": "Part -> part_of -> Model -> affects -> Vehicle",
                    "status": "frozen",
                }
            ]
        )
        master = {
            "nodes": [{"meta_name": "Harness Part"}, {"meta_name": "Vehicle Model"}],
            "relationships": [{"type": "part_of"}],
        }
        validation = audit_cq_validation(master, catalog)
        result = validation["results"][0]
        self.assertFalse(result["executable"])
        self.assertEqual(result["missing_entities"], [])  # entities list itself is fine
        self.assertEqual(result["missing_relations"], [])
        self.assertEqual(
            result["chain_breaks"],
            ["position 3: relation 'affects' not found in master"],
        )

    def test_attribute_gap_reported(self):
        catalog = self._catalog(
            [
                {
                    "cq_id": "CQ-9102",
                    "cq_text": "q",
                    "entities": ["Part"],
                    "relations": ["part_of"],
                    "query_path": "Part -> part_of -> Model",
                    "attributes": ["effective_time", "status"],
                    "status": "frozen",
                }
            ]
        )
        master = {
            "nodes": [
                {"meta_name": "Harness Part", "attributes": [{"name": "status", "value": "x"}]},
            ],
            "relationships": [{"type": "part_of"}],
        }
        validation = audit_cq_validation(master, catalog)
        result = validation["results"][0]
        self.assertFalse(result["executable"])
        self.assertEqual(result["missing_attributes"], ["effective_time"])

    def test_fully_covered_cq_executable(self):
        catalog = self._catalog(
            [
                {
                    "cq_id": "CQ-9103",
                    "cq_text": "q",
                    "entities": ["Part", "Model"],
                    "relations": ["part_of"],
                    "query_path": "Part -> part_of -> Model",
                    "attributes": ["status"],
                    "status": "frozen",
                }
            ]
        )
        master = {
            "nodes": [
                {"meta_name": "Harness Part", "attributes": [{"name": "status", "value": "x"}]},
                {"meta_name": "Vehicle Model"},
            ],
            "relationships": [{"type": "part_of"}],
        }
        validation = audit_cq_validation(master, catalog)
        result = validation["results"][0]
        self.assertTrue(result["executable"])
        self.assertEqual(result["chain_breaks"], [])

    def test_no_frozen_cqs_returns_none(self):
        catalog = self._catalog(
            [{"cq_id": "CQ-9104", "cq_text": "q", "entities": ["A"], "relations": ["r"], "query_path": "A -> r", "status": "draft"}]
        )
        self.assertIsNone(audit_cq_validation({"nodes": []}, catalog))


# ---------------------------------------------------------------------------
# 4. gap report artifact (M3 验收场景: 必然缺东西)
# ---------------------------------------------------------------------------


class GapReportArtifactTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_acceptance_bi_temporal_gap_is_reported(self):
        """计划 §9-M3 验收：bi-temporal 口径缺失 → gap report 指出缺项。"""
        catalog = self._write_seed_like_catalog()
        # Master 覆盖实体与基础关系，但缺双层时间属性（C-Notice 场景必然缺）
        master = {
            "nodes": [
                {"meta_name": "C-Notice"},
                {"meta_name": "Part"},
                {"meta_name": "Vehicle"},
                {"meta_name": "ProductionOrder", "attributes": [{"name": "status", "value": "scheduled"}]},
            ],
            "instances": [],
            "relationships": [
                {"type": "affects"},
                {"type": "part_of"},
                {"type": "scheduled_for"},
            ],
        }
        validation = audit_cq_validation(master, catalog)
        path = audit_cq_gap_report_artifact(validation, self.root, catalog_path=catalog)
        self.assertIsNotNone(path)
        text = Path(path).read_text(encoding="utf-8")
        # 汇总与明细都要点名缺项
        self.assertIn("CQ-9201", text)
        self.assertIn("缺属性 effective_time", text)
        self.assertIn("缺属性 installation_time", text)
        self.assertIn("❌", text)
        self.assertIn("待确认口径", text)

    def _write_seed_like_catalog(self) -> Path:
        return _write_catalog(
            self.root / "cq" / "catalog.jsonl",
            [
                {
                    "cq_id": "CQ-9201",
                    "cq_text": "这些车型中，哪些 VIN 已排产、还没上线，且上线时间早于变更生效？",
                    "entities": ["Vehicle", "ProductionOrder", "C-Notice"],
                    "relations": ["scheduled_for", "affects"],
                    "attributes": ["effective_time", "installation_time"],
                    "query_path": "C-Notice -> affects -> Part -> part_of -> Model -> scheduled_for -> Vehicle | filter: installation_time < effective_time",
                    "status": "frozen",
                    "open_questions": ["'还没装' = 实际上线时间 < 生效时间，需要两层时间（bi-temporal），口径待业务确认"],
                }
            ],
        )

    def test_executable_cq_excluded_from_detail_section(self):
        catalog = _write_catalog(
            self.root / "cq" / "catalog.jsonl",
            [
                {
                    "cq_id": "CQ-9202",
                    "cq_text": "q",
                    "entities": ["Part"],
                    "relations": ["part_of"],
                    "query_path": "Part -> part_of",
                    "status": "frozen",
                }
            ],
        )
        master = {"nodes": [{"meta_name": "Part"}], "relationships": [{"type": "part_of"}]}
        validation = audit_cq_validation(master, catalog)
        path = audit_cq_gap_report_artifact(validation, self.root, catalog_path=catalog)
        text = Path(path).read_text(encoding="utf-8")
        self.assertIn("✅", text)
        self.assertNotIn("## 缺口明细\n\n### CQ-9202", text)

    def test_write_cq_gap_report_direct_includes_mapping_column(self):
        catalog = CQCatalog(
            items=[
                CQ.model_validate(
                    {
                        "cq_id": "CQ-9203",
                        "cq_text": "q",
                        "entities": ["A"],
                        "relations": ["r"],
                        "query_path": "A -> r",
                        "status": "frozen",
                    }
                )
            ]
        )
        results = [
            CQValidationResult(cq_id="CQ-9203", executable=False, missing_entities=["A"])
        ]
        path = write_cq_gap_report(
            results,
            catalog,
            self.root,
            mapping={"CQ-9203": {"sample_id": "some-sample"}},
        )
        text = path.read_text(encoding="utf-8")
        self.assertIn("| CQ-9203 | some-sample | ❌ | 缺实体 A |", text)

    def test_no_validation_returns_none(self):
        self.assertIsNone(audit_cq_gap_report_artifact(None, self.root))
        self.assertIsNone(audit_cq_gap_report_artifact({"results": []}, None))


# ---------------------------------------------------------------------------
# 5. CQ ↔ regression sample mapping
# ---------------------------------------------------------------------------


class CQMappingTest(unittest.TestCase):
    def test_shipped_mapping_binds_two_of_five(self):
        from src.application.cq.cq_registry import default_mapping_path

        path = default_mapping_path()
        self.assertTrue(path.exists())
        mapping = load_cq_mapping(path)
        self.assertEqual(len(mapping), 5)
        self.assertEqual(mapping["CQ-0004"]["sample_id"], "mbom-du-v2")
        self.assertEqual(mapping["CQ-0005"]["sample_id"], "production-inventory-md")
        for cq_id in ("CQ-0001", "CQ-0002", "CQ-0003"):
            self.assertEqual(mapping[cq_id]["status"], "unbound")

    def test_missing_file_returns_empty_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_cq_mapping(Path(tmp) / "nope.json"), {})

    def test_malformed_file_returns_empty_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{not json", encoding="utf-8")
            self.assertEqual(load_cq_mapping(path), {})


if __name__ == "__main__":
    unittest.main()
