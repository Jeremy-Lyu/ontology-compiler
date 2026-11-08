"""MCP meaning tools stay read-only and curated."""
from __future__ import annotations

import json
import unittest

from mcp.ontology_mcp.server import dispatch
from mcp.ontology_mcp.tools import TOOLS, handle_tool


class McpToolsTest(unittest.TestCase):
    def test_allow_list_is_small_and_meaning_first(self):
        names = {t["name"] for t in TOOLS}
        self.assertEqual(
            names,
            {
                "search_entities",
                "get_entity",
                "check_answerable",
                "check_readiness",
                "ask_question",
            },
        )

    def test_dispatch_tools_list(self):
        resp = dispatch({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        self.assertEqual(resp["id"], 1)
        self.assertEqual(len(resp["result"]["tools"]), 5)

    def test_ask_question_refuses_when_not_answerable(self):
        from src.application.rag.semantic_surface import ask_question

        pkt = {
            "entities": {
                "L1": [{"id": "L1-1", "name": "UnrelatedThing", "layer": "L1", "type": "Data"}],
            },
            "relations": [],
        }
        out = ask_question("MBOM 如何由 EBOM 派生", packet=pkt)
        self.assertFalse(out["answerable"])
        self.assertTrue(out["refused"])
        self.assertIn("缺口", out["answer"])

    def test_handle_search_with_inline_packet(self):
        # search_entities hits live governed packet when present; just ensure call shape.
        # Use empty query → empty hits without requiring packet.
        from src.application.rag import semantic_surface as ss

        pkt = {
            "entities": {
                "L1": [{"id": "L1-1", "name": "MBOM", "layer": "L1", "type": "Data"}],
                "L0": [{"id": "L0-1", "name": "Asset", "layer": "L0", "type": "Asset"}],
                "L2": [{"id": "L2-1", "name": "Module", "layer": "L2", "type": "Module"}],
            },
            "relations": [],
        }
        hits = ss.search_entities("MBOM", packet=pkt)
        self.assertTrue(hits)
        self.assertEqual(hits[0].name, "MBOM")

    def test_unknown_tool(self):
        with self.assertRaises(KeyError):
            handle_tool("dump_master_json", {})


if __name__ == "__main__":
    unittest.main()
