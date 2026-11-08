"""P1-2 (2026-09-01): QueryRewriter LLM-expansion restoration characterization.

Locks the wiring that brings the Dify branch back into the LLM expansion path:

1. ``rag.dify_client`` shim -> ``rag_answerer.DIFY_CLIENT_AVAILABLE`` is True
   and ``rag.validator`` imports cleanly (the module moved to
   ``infrastructure/dify_client.py`` but the relative imports were never
   updated, so the Dify branch silently dead-ended).
2. ``DifyCallLLMBackend.call_llm`` routes a keyword-extraction prompt through
   ``ask_dify_kb_qa`` + ``get_dify_answer_text``.
3. ``RAGAnswerer._rewrite_query`` injects the Dify backend when no
   UnifiedLLMProvider/Client is present and a Dify config is set, so
   ``QueryRewriter._llm_expand`` actually runs (previously dead code).

Pure unit tests: no LLM, no network, no filesystem reads.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from src.application.rag import rag_answerer
from src.application.rag.rag_answerer import (
    DifyCallLLMBackend,
    RAGAnswerer,
)


def _answerer(**attrs):
    a = object.__new__(RAGAnswerer)
    a.model = "test-model"
    a.rag_dir = None
    a.dify_config = None
    a.llm_provider = None
    a.llm_client = None
    for k, v in attrs.items():
        setattr(a, k, v)
    return a


def _dify_config(**attrs) -> SimpleNamespace:
    cfg = SimpleNamespace(
        app_type="workflow",
        base_url="http://dify.test",
        api_key="test-key",
        response_mode="blocking",
        timeout_s=30,
    )
    for k, v in attrs.items():
        setattr(cfg, k, v)
    return cfg


class DifyShimImportTest(unittest.TestCase):
    """Lock the shim that repairs the broken ``from .dify_client import ...``."""

    def test_dify_client_available_is_true(self):
        self.assertTrue(rag_answerer.DIFY_CLIENT_AVAILABLE)

    def test_rag_dify_client_shim_exports_symbols(self):
        from src.application.rag import dify_client  # noqa: F401

        for name in (
            "DifyConfig",
            "load_dify_config_from_env",
            "extract_cap_node_ids",
            "ask_dify_kb_qa",
            "get_dify_answer_text",
        ):
            self.assertTrue(hasattr(dify_client, name), name)

    def test_validator_imports_dify_symbols(self):
        from src.application.rag.validator import (  # noqa: F401
            extract_cap_node_ids,
            get_dify_answer_text,
        )
        self.assertTrue(callable(extract_cap_node_ids))


class DifyCallLLMBackendTest(unittest.TestCase):
    def test_call_llm_routes_prompt_through_dify(self):
        calls = {}

        def fake_ask(query, inputs, config):
            calls["query"] = query
            calls["inputs"] = inputs
            calls["config"] = config
            return {"answer": "Production Planning ARO"}

        backend = DifyCallLLMBackend(_dify_config())
        with mock.patch.object(rag_answerer, "ask_dify_kb_qa", fake_ask), \
             mock.patch.object(rag_answerer, "get_dify_answer_text",
                               lambda resp: "Production Planning ARO"):
            out = backend.call_llm("BOM 变更管理", max_tokens=50, temperature=0.0)

        self.assertEqual(out, "Production Planning ARO")
        self.assertEqual(calls["query"], "BOM 变更管理")
        self.assertEqual(calls["config"].app_type, "workflow")
        self.assertIn("full_prompt", calls["inputs"])

    def test_call_llm_empty_extract_returns_empty_string(self):
        backend = DifyCallLLMBackend(_dify_config())
        with mock.patch.object(rag_answerer, "ask_dify_kb_qa", lambda *a, **k: {}), \
             mock.patch.object(rag_answerer, "get_dify_answer_text", lambda resp: ""):
            self.assertEqual(backend.call_llm("q"), "")


class RewriteQueryDifyWiringTest(unittest.TestCase):
    """Locks the ``_rewrite_query`` -> ``DifyCallLLMBackend`` injection."""

    def tearDown(self):
        # get_rewriter is a module singleton; reset between tests to avoid
        # cross-test backend leakage.
        from src.application.rag.domain import query_core
        query_core._default_rewriter = None

    def test_rewrite_query_uses_dify_backend_when_no_other_llm(self):
        a = _answerer(dify_config=_dify_config())
        with mock.patch.object(rag_answerer, "ask_dify_kb_qa",
                               return_value={"answer": "Fixed Planning ARO"}), \
             mock.patch.object(rag_answerer, "get_dify_answer_text",
                               return_value="Fixed Planning ARO") as extract_mock:
            out = a._rewrite_query("ARO 计划")

        # QueryRewriter._llm_expand Route 1 ran against the Dify backend.
        self.assertIn("Fixed Planning", out)
        extract_mock.assert_called_once()

    def test_rewrite_query_no_llm_without_dify_config(self):
        a = _answerer(dify_config=None)
        with mock.patch.object(rag_answerer, "ask_dify_kb_qa") as ask_mock:
            out = a._rewrite_query("BOM 变更")
        ask_mock.assert_not_called()
        # No backend -> no expansion; the original query is always preserved.
        self.assertIn("BOM 变更", out)


if __name__ == "__main__":
    unittest.main()
