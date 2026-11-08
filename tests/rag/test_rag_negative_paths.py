"""RAG negative-path tests (plan P1 #5: negative path coverage).

Locks the failure behavior of RAG chunk-index persistence loading — the
DataContractError contract failures that the characterization tests do not
cover. Pure file-level paths, no I/O beyond temp files, no LLM.

Covered:
- load_inverted_index   (missing / invalid JSON / non-object / type filtering)
- load_chunks_jsonl     (missing / invalid line)
- load_ontology_json    (invalid JSON / non-object)
- _require_list         (missing key / non-list / non-dict item)
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.application.rag.chunk_indexing import (
    _require_list,
    load_chunks_jsonl,
    load_inverted_index,
    load_ontology_json,
)
from src.domain.errors import DataContractError


class LoadInvertedIndexNegativePath(unittest.TestCase):
    def test_missing_file_raises(self):
        with self.assertRaises(DataContractError) as ctx:
            load_inverted_index(Path("/nonexistent/inverted_index.json"))
        self.assertIn("inverted index not found", str(ctx.exception))

    def test_invalid_json_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "index.json"
            p.write_text("{not valid json", encoding="utf-8")
            with self.assertRaises(DataContractError) as ctx:
                load_inverted_index(p)
            self.assertIn("failed to load inverted index", str(ctx.exception))

    def test_non_object_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "index.json"
            p.write_text("[1, 2, 3]", encoding="utf-8")
            with self.assertRaises(DataContractError) as ctx:
                load_inverted_index(p)
            self.assertIn("is not an object", str(ctx.exception))

    def test_filters_non_string_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "index.json"
            p.write_text(
                json.dumps({"good": ["a", "b"], "bad": [1, 2], "num": 5}),
                encoding="utf-8",
            )
            result = load_inverted_index(p)
            self.assertEqual(result, {"good": ["a", "b"]})


class LoadChunksJsonlNegativePath(unittest.TestCase):
    def test_missing_file_raises(self):
        with self.assertRaises(DataContractError) as ctx:
            load_chunks_jsonl(Path("/nonexistent/chunks.jsonl"))
        self.assertIn("chunks file not found", str(ctx.exception))

    def test_invalid_json_line_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "chunks.jsonl"
            p.write_text("{broken json\n", encoding="utf-8")
            with self.assertRaises(DataContractError) as ctx:
                load_chunks_jsonl(p)
            self.assertIn("invalid JSON at line 1", str(ctx.exception))

    def test_missing_chunk_fields_raise_contract_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "chunks.jsonl"
            p.write_text(json.dumps({"chunk_id": "ok"}), encoding="utf-8")
            with self.assertRaises(DataContractError) as ctx:
                load_chunks_jsonl(p)
            self.assertIn("invalid chunk record at line 1", str(ctx.exception))


class LoadOntologyJsonNegativePath(unittest.TestCase):
    def test_invalid_json_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ontology.json"
            p.write_text("{oops", encoding="utf-8")
            with self.assertRaises(DataContractError) as ctx:
                load_ontology_json(p)
            self.assertIn("failed to load ontology json", str(ctx.exception))

    def test_non_object_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ontology.json"
            p.write_text('"just a string"', encoding="utf-8")
            with self.assertRaises(DataContractError) as ctx:
                load_ontology_json(p)
            self.assertIn("ontology json is not an object", str(ctx.exception))


class RequireListNegativePath(unittest.TestCase):
    def test_missing_key_raises(self):
        with self.assertRaises(DataContractError) as ctx:
            _require_list({}, "nodes", "source.json")
        self.assertIn("missing/invalid 'nodes' list", str(ctx.exception))

    def test_non_list_value_raises(self):
        with self.assertRaises(DataContractError) as ctx:
            _require_list({"nodes": "not a list"}, "nodes", "source.json")
        self.assertIn("missing/invalid 'nodes' list", str(ctx.exception))

    def test_non_dict_item_raises(self):
        with self.assertRaises(DataContractError) as ctx:
            _require_list({"nodes": [{"ok": 1}, "bad"]}, "nodes", "source.json")
        self.assertIn("invalid 'nodes[1]' record", str(ctx.exception))

    def test_valid_list_returns_items(self):
        items = _require_list({"nodes": [{"a": 1}, {"b": 2}]}, "nodes", "s.json")
        self.assertEqual(items, [{"a": 1}, {"b": 2}])


if __name__ == "__main__":
    unittest.main()