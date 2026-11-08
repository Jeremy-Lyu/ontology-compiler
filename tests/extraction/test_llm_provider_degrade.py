"""LLM provider transient failures degrade instead of aborting."""
from __future__ import annotations

import asyncio
import unittest
from unittest.mock import MagicMock

from ontology_compiler.services.extraction.llm_call import (
    call_llm,
    is_transient_provider_error,
)


class TransientProviderErrorTest(unittest.TestCase):
    def test_markers(self):
        self.assertTrue(is_transient_provider_error("nodename nor servname provided"))
        self.assertTrue(is_transient_provider_error(TimeoutError("timed out")))
        self.assertTrue(is_transient_provider_error("429 rate limit"))
        self.assertFalse(is_transient_provider_error("invalid api key"))


class DegradedOnFailureTest(unittest.IsolatedAsyncioTestCase):
    async def test_transient_exhaustion_returns_degraded(self):
        llm = MagicMock()

        async def boom(*_a, **_k):
            raise ConnectionError("nodename nor servname provided")

        llm.call_with_retry = boom
        llm.__class__.__name__ = "FakeLLM"
        llm.model = "fake"

        import os

        os.environ["LLM_MAX_JSON_RETRIES"] = "0"
        os.environ["LLM_DEGRADED_ON_FAILURE"] = "1"
        os.environ["LLM_RETRY_DELAY_SECONDS"] = "0"
        try:
            result = await call_llm(
                llm,
                max_tokens=256,
                temperature=0.0,
                timeout=5,
                current_run_id=None,
                tracer=None,
                get_metadata=lambda: MagicMock(name="skill"),
                prompt='{"hello":1}',
            )
        finally:
            os.environ.pop("LLM_MAX_JSON_RETRIES", None)
            os.environ.pop("LLM_DEGRADED_ON_FAILURE", None)
            os.environ.pop("LLM_RETRY_DELAY_SECONDS", None)

        self.assertTrue(result.get("degraded"))
        self.assertTrue(result.get("parse_success"))
        self.assertIn("L0_entities", result.get("response", ""))


if __name__ == "__main__":
    unittest.main()
