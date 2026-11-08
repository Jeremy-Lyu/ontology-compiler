"""P1-4 (2026-09-01) characterization tests: CQ ↔ pipeline integration (M2).

Locks the zero-regression contract (CQ_PIPELINE_IMPLEMENTATION_PLAN.md §9-M2):

1. Every integration helper returns ``None`` when there is no CQ input —
   missing catalog, empty catalog, or no artifact_root — so stages stay
   byte-identical to the pre-CQ baseline.
2. ``preflight`` summary / catalog snapshot appear only when a non-empty
   catalog exists.
3. ``audit`` validates frozen CQs read-only against the master vocabulary
   (lenient entity containment, exact case-insensitive relations).
4. ``summarize`` writes the review sheet + ArtifactBundle and reports
   executable / unanswered / pending counts.
5. Stage-level: ``run_summarize`` adds ``summary["cq_review"]`` only when a
   CQ catalog exists.

All tests are hermetic: tmp_path file I/O only, no LLM calls.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from src.pipeline.cq_integration import (
    audit_cq_report_artifact,
    audit_cq_validation,
    preflight_cq_catalog_artifact,
    preflight_cq_summary,
    summarize_cq_review_artifact,
)


def _write_catalog(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    return path


def _cq_record(cq_id: str = "CQ-9001", status: str = "frozen", **overrides) -> dict:
    record = {
        "cq_id": cq_id,
        "cq_text": "哪些车型使用了该线束？",
        "entities": ["Part", "Model"],
        "relations": ["part_of"],
        "query_path": "Part -> part_of -> Model",
        "status": status,
    }
    record.update(overrides)
    return record


def _master() -> dict:
    return {
        "nodes": [
            {"meta_name": "Harness Part", "layer": "L1"},
            {"meta_name": "Vehicle Model", "layer": "L1"},
        ],
        "instances": [],
        "relationships": [{"type": "part_of", "source": "Harness Part", "target": "Vehicle Model"}],
    }


# ---------------------------------------------------------------------------
# 1. Zero-regression: no CQ input → None everywhere
# ---------------------------------------------------------------------------


class ZeroRegressionContractTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_missing_catalog_returns_none_everywhere(self):
        missing = self.root / "nope.jsonl"
        self.assertIsNone(preflight_cq_summary(missing))
        self.assertIsNone(preflight_cq_catalog_artifact(self.root, missing))
        self.assertIsNone(audit_cq_validation(_master(), missing))
        self.assertIsNone(summarize_cq_review_artifact(self.root, missing))

    def test_empty_catalog_returns_none_everywhere(self):
        empty = _write_catalog(self.root / "cq" / "empty.jsonl", [])
        self.assertIsNone(preflight_cq_summary(empty))
        self.assertIsNone(preflight_cq_catalog_artifact(self.root, empty))
        self.assertIsNone(audit_cq_validation(_master(), empty))
        self.assertIsNone(summarize_cq_review_artifact(self.root, empty))

    def test_no_artifact_root_returns_none(self):
        catalog = _write_catalog(self.root / "cq" / "catalog.jsonl", [_cq_record()])
        self.assertIsNone(preflight_cq_catalog_artifact(None, catalog))
        self.assertIsNone(summarize_cq_review_artifact(None, catalog))

    def test_no_frozen_cqs_skips_audit_validation(self):
        catalog = _write_catalog(self.root / "cq" / "catalog.jsonl", [_cq_record(status="draft")])
        self.assertIsNone(audit_cq_validation(_master(), catalog))


# ---------------------------------------------------------------------------
# 2. preflight summary + catalog snapshot
# ---------------------------------------------------------------------------


class PreflightIntegrationTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.catalog = _write_catalog(
            self.root / "cq" / "catalog.jsonl",
            [_cq_record(), _cq_record(cq_id="CQ-9002", status="draft")],
        )

    def test_summary_counts_by_status(self):
        summary = preflight_cq_summary(self.catalog)
        self.assertIsNotNone(summary)
        self.assertEqual(summary["loaded_count"], 2)
        self.assertEqual(summary["frozen_count"], 1)
        self.assertEqual(summary["by_status"], {"frozen": 1, "draft": 1})
        self.assertTrue(summary["available"])

    def test_catalog_artifact_written(self):
        written = preflight_cq_catalog_artifact(self.root, self.catalog)
        self.assertIsNotNone(written)
        payload = json.loads(Path(written).read_text(encoding="utf-8"))
        self.assertEqual(payload["count"], 2)


# ---------------------------------------------------------------------------
# 3. audit read-only executability check
# ---------------------------------------------------------------------------


class AuditValidationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.catalog = _write_catalog(
            self.root / "cq" / "catalog.jsonl",
            [
                _cq_record(),  # Part/Model + part_of → all matched (leniently)
                _cq_record(cq_id="CQ-9003", entities=["Part", "Wiring Harness"], relations=["part_of", "affects"]),
            ],
        )

    def test_lenient_entity_match_and_exact_relation_match(self):
        validation = audit_cq_validation(_master(), self.catalog)
        self.assertIsNotNone(validation)
        self.assertEqual(validation["checked_count"], 2)
        by_id = {r["cq_id"]: r for r in validation["results"]}
        # "Part" ⊆ "Harness Part" (lenient containment); "part_of" exact (case-insensitive)
        self.assertTrue(by_id["CQ-9001"]["executable"])
        self.assertEqual(by_id["CQ-9001"]["missing_entities"], [])
        # CQ-9003: entity "Wiring Harness" missing? — "Wiring Harness" vs
        # "Harness Part" no containment either way → missing; relation "affects" missing.
        self.assertFalse(by_id["CQ-9003"]["executable"])
        self.assertEqual(by_id["CQ-9003"]["missing_relations"], ["affects"])

    def test_report_artifact_written_and_read_back(self):
        validation = audit_cq_validation(_master(), self.catalog)
        written = audit_cq_report_artifact(validation, self.root)
        self.assertIsNotNone(written)
        payload = json.loads(Path(written).read_text(encoding="utf-8"))
        self.assertEqual(payload["checked_count"], 2)
        self.assertEqual(payload["executable_count"], 1)

    def test_report_artifact_none_without_validation(self):
        self.assertIsNone(audit_cq_report_artifact(None, self.root))


# ---------------------------------------------------------------------------
# 4. summarize review sheet + bundle
# ---------------------------------------------------------------------------


class SummarizeReviewIntegrationTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.catalog = _write_catalog(
            self.root / "cq" / "catalog.jsonl",
            [_cq_record(), _cq_record(cq_id="CQ-9002")],
        )

    def test_review_counts_without_report_are_pending(self):
        info = summarize_cq_review_artifact(self.root, self.catalog)
        self.assertIsNotNone(info)
        self.assertEqual(info["total"], 2)
        self.assertEqual(info["executable"], 0)
        self.assertEqual(info["unanswered"], 0)
        self.assertEqual(info["pending_validation"], 2)
        self.assertTrue(Path(info["review_sheet"]).is_file())
        self.assertTrue(Path(info["bundle"]).is_file())

    def test_review_counts_with_audit_report(self):
        validation = audit_cq_validation(_master(), self.catalog)
        audit_cq_report_artifact(validation, self.root)
        info = summarize_cq_review_artifact(self.root, self.catalog)
        self.assertEqual(info["executable"], 2)
        self.assertEqual(info["unanswered"], 0)
        self.assertEqual(info["pending_validation"], 0)
        sheet = Path(info["review_sheet"]).read_text(encoding="utf-8")
        self.assertIn("CQ-9001", sheet)
        self.assertIn("✅", sheet)

    def test_bundle_metadata_counts(self):
        validation = audit_cq_validation(_master(), self.catalog)
        audit_cq_report_artifact(validation, self.root)
        info = summarize_cq_review_artifact(self.root, self.catalog, run_id="run-y")
        bundle = json.loads(Path(info["bundle"]).read_text(encoding="utf-8"))
        self.assertEqual(bundle["run_id"], "run-y")
        self.assertEqual(bundle["stage"], "cq")
        self.assertEqual(
            bundle["metadata"]["review_counts"],
            {"total": 2, "executable": 2, "unanswered": 0, "pending_validation": 0},
        )


# ---------------------------------------------------------------------------
# 5. Stage-level zero-regression: run_summarize
# ---------------------------------------------------------------------------


class SummarizeStageDelegationTest(unittest.TestCase):
    """The stage calls ``summarize_cq_review_artifact`` with the *default*
    seed catalog path, so tests redirect ``default_catalog_path`` instead of
    writing a per-run catalog."""

    def _run_stage(self, root: Path, default_catalog: Path) -> dict:
        import unittest.mock

        from src.pipeline.stages.summarize import run_summarize

        master_path = root / "master.json"
        master_path.write_text(json.dumps(_master()), encoding="utf-8")
        args = argparse.Namespace(
            master_path=str(master_path),
            output_dir=str(root / "out"),
            run_dir="",
            artifact_root=str(root / "artifacts"),
            run_id="stage-test",
        )
        with unittest.mock.patch(
            "src.pipeline.cq_integration.default_catalog_path", return_value=default_catalog
        ):
            result = run_summarize(args)
        self.assertTrue(result.success)
        return json.loads((root / "out" / "summary_report.json").read_text(encoding="utf-8"))

    def test_no_cq_input_keeps_summary_report_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = self._run_stage(root, default_catalog=root / "nope.jsonl")
            self.assertNotIn("cq_review", summary)

    def test_with_default_seed_adds_cq_review_to_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = _write_catalog(root / "seed.jsonl", [_cq_record()])
            summary = self._run_stage(root, default_catalog=seed)
            self.assertIn("cq_review", summary)
            self.assertEqual(summary["cq_review"]["total"], 1)
            review_sheet = Path(summary["cq_review"]["review_sheet"])
            self.assertTrue(review_sheet.is_file())
            self.assertEqual(review_sheet.parent.name, "cq")


if __name__ == "__main__":
    unittest.main()
