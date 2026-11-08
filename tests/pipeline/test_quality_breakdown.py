"""Per-file extract scores stay separate from gate/master quality score."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.pipeline.status_quality import extract_stage_quality, quality_score_breakdown


class QualityBreakdownTest(unittest.TestCase):
    def _pipeline(self, files, gate=None):
        extract = MagicMock()
        extract.data = {
            "extraction_quality": {
                "status": "partial",
                "validation_passed": False,
                "files": files,
            }
        }
        pipeline = SimpleNamespace(
            _stage_results={"extract": extract},
            context=SimpleNamespace(metadata={"quality_gate": gate or {}}),
        )
        return pipeline

    def test_file_avg_differs_from_gate(self):
        pipeline = self._pipeline(
            [
                {"file": "a.pptx", "quality_score": 0.4, "validation_passed": False},
                {"file": "b.pptx", "quality_score": 0.8, "validation_passed": True},
            ],
            gate={"quality_score": 0.97, "validation_passed": True},
        )
        file_view = extract_stage_quality(pipeline)
        breakdown = quality_score_breakdown(pipeline)
        self.assertEqual(file_view["quality_score"], 0.6)
        self.assertEqual(breakdown["file_avg_score"], 0.6)
        self.assertEqual(breakdown["file_min_score"], 0.4)
        self.assertEqual(breakdown["gate_score"], 0.97)
        self.assertEqual(len(breakdown["files"]), 2)


if __name__ == "__main__":
    unittest.main()
