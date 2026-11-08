"""Failure memory adapter used by the unified extractor."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ontology_compiler.services.extraction.extraction_failure_memory import (
    apply_failure_memory_adjustments,
)
from ontology_compiler.services.failure_memory import (
    FailureMemory,
    bind_failure_memory,
    record_extraction_failure,
    record_json_parse_failure,
    record_schema_violation,
    reset_failure_memory,
)


class FailureMemoryAdapterTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.memory = FailureMemory(storage_path=Path(self._tmpdir.name))
        bind_failure_memory(self.memory)

    def tearDown(self):
        reset_failure_memory()
        self._tmpdir.cleanup()

    def test_single_failure_does_not_adjust(self):
        record_extraction_failure("boom", document_key="docs/a.pptx")
        adjustment = self.memory.check_should_adjust_strategy("docs/a.pptx")
        self.assertFalse(adjustment["should_adjust"])
        self.assertEqual(adjustment["recent_failure_count"], 1)

    def test_repeat_failure_enables_fast_mode_for_that_document(self):
        record_extraction_failure("boom", document_key="docs/a.pptx", severity="high")
        record_extraction_failure("boom again", document_key="docs/a.pptx", severity="critical")
        record_extraction_failure("other file", document_key="docs/b.pptx")

        ctx = {"prompt_strategy": "technical", "fast_mode": False}
        result = apply_failure_memory_adjustments("docs/a.pptx", ctx)
        self.assertTrue(result["fast_mode"])
        self.assertEqual(ctx["prompt_strategy"], "general")

        other = apply_failure_memory_adjustments("docs/b.pptx", {"prompt_strategy": "technical"})
        self.assertFalse(other["fast_mode"])

    def test_llm_call_kwargs_are_accepted(self):
        sample_id = record_extraction_failure(
            error_message="LLM call failed: timeout-shaped",
            failure_type="llm_call_failed",
            severity="high",
            document_key="docs/a.pptx",
        )
        self.assertTrue(sample_id.startswith("fail-"))
        stored = self.memory._failures[-1]
        self.assertEqual(stored.failure_type, "llm_call_failed")
        self.assertEqual(stored.severity, "high")

    def test_parse_and_schema_recorders_exist(self):
        record_json_parse_failure("bad json", document_type="llm_response", layer="L1")
        record_schema_violation("missing meta_class", document_key="docs/a.pptx")
        types = {sample.failure_type for sample in self.memory._failures}
        self.assertEqual(types, {"json_parse_error", "schema_violation"})

    def test_unresolved_repeats_still_recommend(self):
        record_json_parse_failure("bad", document_key="docs/a.pptx")
        record_json_parse_failure("bad again", document_key="docs/a.pptx")
        recs = self.memory.get_recommendations("extraction", "SUPPLY_CHAIN")
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["failure_type"], "json_parse_error")
        self.assertEqual(recs[0]["occurrence_count"], 2)
        self.assertEqual(recs[0]["recommended_strategy"], "strict_json")
        self.assertEqual(recs[0]["severity"], "medium")

    def test_json_repeats_switch_strategy_to_contract(self):
        record_json_parse_failure("bad", document_key="docs/a.pptx")
        record_json_parse_failure("bad again", document_key="docs/a.pptx")
        self.assertEqual(
            self.memory.get_adjusted_strategy("docs/a.pptx", "general"),
            "contract",
        )


if __name__ == "__main__":
    unittest.main()
