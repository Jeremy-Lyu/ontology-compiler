"""Guards for the three failures seen on the EBOM/MBOM deck.

A slide table must become Markdown, a JSON body cut off at the last character
must be marked truncated, and a partial graph must not be described as an
empty offline result.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ontology_compiler.agents.file_parser import FileParser, markdown_table
from ontology_compiler.services.extraction.llm_call import json_response_was_truncated
from ontology_compiler.services.extraction.unified_compilation_pipeline import (
    degraded_result_warning,
)


class MarkdownTableRenderTest(unittest.TestCase):
    def test_table_rows_are_complete_lines(self):
        lines = markdown_table([["Part", "Qty"], ["123", "2"]])
        self.assertEqual(lines[0], "| Part | Qty |")
        self.assertEqual(lines[2], "| 123 | 2 |")

    def test_pptx_table_renders_without_append_kwargs(self):
        from pptx import Presentation
        from pptx.util import Inches

        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        shape = slide.shapes.add_table(2, 2, Inches(0.5), Inches(0.5), Inches(6), Inches(1.5))
        shape.table.cell(0, 0).text = "Part"
        shape.table.cell(0, 1).text = "Address"
        shape.table.cell(1, 0).text = "12345678"
        shape.table.cell(1, 1).text = "Caliper"

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bom.pptx"
            prs.save(path)
            rendered = FileParser.extract_markdown(str(path))

        self.assertIn("| Part | Address |", rendered)
        self.assertIn("| 12345678 | Caliper |", rendered)

    def test_plain_text_reader_is_its_own_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "note.txt"
            path.write_text("EBOM becomes MBOM", encoding="utf-8")
            self.assertEqual(FileParser._read_text(path), "EBOM becomes MBOM")


class TruncatedJsonRetryTest(unittest.TestCase):
    def test_cutoff_at_the_last_character_is_truncation(self):
        body = '{"L0_entities": ['
        error = f"JSON parse error: Expecting value: line 1 column 18 (char {len(body)})"
        self.assertTrue(json_response_was_truncated(body, error))

    def test_prose_is_not_truncation(self):
        prose = "The model refused to emit JSON because the prompt was unclear. " * 20
        self.assertFalse(json_response_was_truncated(prose, "JSON parse error: Expecting value: line 1 column 1 (char 0)"))

    def test_syntax_error_in_the_middle_is_not_truncation(self):
        body = '{"L0_entities": [}, ' + '{"meta_id": "L0-002"}' * 40
        error = "JSON parse error: Expecting value: line 1 column 18 (char 17)"
        self.assertFalse(json_response_was_truncated(body, error))


class DegradedWarningTest(unittest.TestCase):
    def test_partial_graph_is_not_called_empty(self):
        warning = degraded_result_warning(
            {"nodes": [{"id": "L1-001"}], "instances": [], "relationships": []},
            {"degraded": True},
        )
        self.assertIn("partial result", warning)
        self.assertNotIn("empty", warning)
        self.assertNotIn("offline", warning)

    def test_empty_offline_result_keeps_the_offline_wording(self):
        warning = degraded_result_warning(
            {"nodes": [], "instances": [], "relationships": [], "offline_fallback": True},
            {},
        )
        self.assertIn("offline fallback", warning)
        self.assertIn("empty", warning)


if __name__ == "__main__":
    unittest.main()
