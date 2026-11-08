"""P1-4 (2026-09-01) characterization tests: CQ M2 artifact writers.

Locks the behavior that the three per-run CQ products land under
``<artifact_root>/cq/`` and register as one ArtifactBundle (stage=``cq``):

- ``cq_catalog.json`` / ``cq_validation_report.json`` / ``cq_review_sheet.md``
- bundle items only include files that exist on disk

All tests are hermetic: tmp_path file I/O only, no pipeline execution.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.application.cq.cq_artifacts import (
    write_cq_bundle,
    write_cq_catalog,
    write_cq_review_sheet,
    write_cq_validation_report,
)
from src.application.cq.cq_models import CQ, CQCatalog, CQValidationResult


def _cq(cq_id: str = "CQ-9001", status: str = "frozen") -> CQ:
    return CQ.model_validate(
        {
            "cq_id": cq_id,
            "cq_text": "哪些车型使用了该线束？",
            "entities": ["Part", "Model"],
            "relations": ["part_of"],
            "query_path": "Part -> part_of -> Model",
            "status": status,
            "open_questions": ["口径待确认"],
        }
    )


class CQArtifactWritersTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_catalog_snapshot_shape(self):
        path = write_cq_catalog(CQCatalog(items=[_cq()]), self.root)
        self.assertEqual(path, self.root / "cq" / "cq_catalog.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["items"][0]["cq_id"], "CQ-9001")

    def test_validation_report_shape(self):
        results = [
            CQValidationResult(cq_id="CQ-9001", executable=False, missing_entities=["Part"]),
            CQValidationResult(cq_id="CQ-9002", executable=True),
        ]
        path = write_cq_validation_report(results, self.root, meta={"stage": "audit"})
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["checked_count"], 2)
        self.assertEqual(payload["executable_count"], 1)
        self.assertEqual(payload["meta"], {"stage": "audit"})

    def test_review_sheet_lists_gaps_and_open_questions(self):
        catalog = CQCatalog(items=[_cq(), _cq(cq_id="CQ-9002")])
        results = [CQValidationResult(cq_id="CQ-9001", executable=False, missing_entities=["Part"])]
        path = write_cq_review_sheet(catalog, results, self.root)
        text = path.read_text(encoding="utf-8")
        self.assertIn("CQ Review Sheet", text)
        self.assertIn("| CQ-9001 | frozen |", text)
        self.assertIn("缺实体 Part", text)          # validation gap
        self.assertIn("口径待确认", text)            # open question carried through
        self.assertIn("❌", text)
        # CQ-9002 has no validation result → placeholder, not executable
        self.assertIn("| CQ-9002 | frozen |", text)
        self.assertIn("—", text)

    def test_bundle_registration_includes_existing_files_only(self):
        sheet = write_cq_review_sheet(CQCatalog(items=[_cq()]), [], self.root)
        bundle_path = write_cq_bundle(
            self.root,
            run_id="run-x",
            files={
                "cq_catalog": self.root / "cq" / "cq_catalog.json",  # not written → excluded
                "cq_review_sheet": sheet,
            },
            metadata={"k": 1},
        )
        self.assertEqual(bundle_path.name, "run-x-cq-bundle.json")
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["stage"], "cq")
        self.assertEqual(payload["run_id"], "run-x")
        self.assertEqual(sorted(payload["items"]), ["cq_review_sheet"])
        self.assertEqual(payload["metadata"], {"k": 1})


if __name__ == "__main__":
    unittest.main()
