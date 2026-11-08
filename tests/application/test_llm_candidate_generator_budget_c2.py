"""C-2 tests: LLMCandidateGenerator budget policy integration.

Characterizes the wiring that makes the candidate generator's
max_completion_tokens / temperature / truncation_retry_max come from a
PromptBudgetPolicy profile instead of env-only defaults.
"""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from src.application.services.prompt_budget_policy import (
    PromptBudget,
    PromptBudgetPolicy,
)
from src.infrastructure.llm.llm_candidate_generator import (
    LLMCandidateGenerator,
    LLMOutputType,
)


class _RecordingClient:
    """Minimal async LLM client that records call kwargs and returns canned JSON."""

    def __init__(self, response: str = '{"entities":[]}'):
        self._response = response
        self.calls: list[dict] = []

    async def call(self, prompt, max_tokens=None, temperature=None, **kw):
        self.calls.append({
            "max_tokens": max_tokens,
            "temperature": temperature,
            "prompt_len": len(prompt),
        })
        return self._response


def _make_policy(max_completion=1000, temperature=0.1, retry_max=5) -> PromptBudgetPolicy:
    path = Path(tempfile.mkstemp(suffix=".yaml")[1])
    path.write_text(f"""
defaults:
  max_completion_tokens: 8000
  max_prompt_tokens: 12000
  total_budget: 20000
  temperature: 0.3
  truncation_retry_max: 2
  json_repair_max_retries: 3
tasks:
  scene_classification:
    max_completion_tokens: {max_completion}
    temperature: {temperature}
    truncation_retry_max: {retry_max}
  entity_candidates:
    max_completion_tokens: 3500
    temperature: 0.2
    truncation_retry_max: 2
  relation_candidates:
    max_completion_tokens: 3000
    temperature: 0.2
    truncation_retry_max: 2
""", encoding="utf-8")
    return PromptBudgetPolicy.from_yaml(path)


class LLMCandidateGeneratorBudgetWiringTest(unittest.TestCase):
    def test_resolve_budget_uses_policy_completion_cap(self):
        policy = _make_policy(max_completion=1200)
        gen = LLMCandidateGenerator(budget_policy=policy)
        resolved = gen._resolve_budget(task="scene_classification")
        self.assertLessEqual(resolved, 1200)
        self.assertGreater(resolved, 0)

    def test_resolve_budget_without_policy_falls_back(self):
        gen = LLMCandidateGenerator()
        resolved = gen._resolve_budget(task="scene_classification", max_tokens=9000)
        self.assertGreater(resolved, 0)

    def test_resolve_truncation_retry_max_from_policy(self):
        policy = _make_policy(retry_max=5)
        gen = LLMCandidateGenerator(budget_policy=policy)
        self.assertEqual(gen._resolve_truncation_retry_max(task="scene_classification"), 5)

    def test_resolve_truncation_retry_max_default(self):
        gen = LLMCandidateGenerator()
        self.assertEqual(gen._resolve_truncation_retry_max(task=None), 2)

    def test_resolve_temperature_from_policy(self):
        policy = _make_policy(temperature=0.05)
        gen = LLMCandidateGenerator(budget_policy=policy)
        self.assertAlmostEqual(gen._resolve_temperature(task="scene_classification"), 0.05)

    def test_resolve_temperature_default(self):
        gen = LLMCandidateGenerator()
        self.assertAlmostEqual(gen._resolve_temperature(task=None), 0.2)

    def test_generate_scene_classification_passes_budget_task(self):
        policy = _make_policy(max_completion=1500, temperature=0.15)
        client = _RecordingClient(response='{"scene_type":"technical","confidence":0.9}')
        gen = LLMCandidateGenerator(llm_client=client, budget_policy=policy)
        asyncio.run(gen.generate_scene_classification("content"))
        self.assertEqual(len(client.calls), 1)
        self.assertLessEqual(client.calls[0]["max_tokens"], 1500)
        self.assertAlmostEqual(client.calls[0]["temperature"], 0.15)

    def test_generate_entity_candidates_passes_budget_task(self):
        policy = _make_policy()
        client = _RecordingClient(response='{"entities":[]}')
        gen = LLMCandidateGenerator(llm_client=client, budget_policy=policy)
        asyncio.run(gen.generate_entity_candidates("content", entity_types=["System"]))
        self.assertEqual(len(client.calls), 1)
        # entity_candidates profile has 3500 completion tokens
        self.assertLessEqual(client.calls[0]["max_tokens"], 3500)
        self.assertAlmostEqual(client.calls[0]["temperature"], 0.2)

    def test_generate_relation_candidates_passes_budget_task(self):
        policy = _make_policy()
        client = _RecordingClient(response='{"relationships":[]}')
        gen = LLMCandidateGenerator(llm_client=client, budget_policy=policy)
        asyncio.run(gen.generate_relation_candidates("content", entities=[{"type":"System","name":"SAP"}]))
        self.assertEqual(len(client.calls), 1)
        self.assertLessEqual(client.calls[0]["max_tokens"], 3000)

    def test_document_type_routed_budget(self):
        """When document_type is provided, the policy resolves by doc-type routing."""
        path = Path(tempfile.mkstemp(suffix=".yaml")[1])
        path.write_text("""
tasks:
  extraction_general:
    max_completion_tokens: 6000
  scene_classification:
    max_completion_tokens: 1000
document_type_profiles:
  md:
    default: extraction_general
""", encoding="utf-8")
        policy = PromptBudgetPolicy.from_yaml(path)
        gen = LLMCandidateGenerator(budget_policy=policy)
        resolved = gen._resolve_budget(task=None, document_type="md")
        self.assertLessEqual(resolved, 6000)


class LLMCandidateGeneratorDocumentTypeParamTest(unittest.TestCase):
    """The public generate_* methods accept a document_type kwarg."""

    def test_generate_methods_accept_document_type(self):
        import inspect
        for method_name in (
            "generate_scene_classification",
            "generate_entity_candidates",
            "generate_relation_candidates",
        ):
            sig = inspect.signature(getattr(LLMCandidateGenerator, method_name))
            self.assertIn("document_type", sig.parameters)


if __name__ == "__main__":
    unittest.main()
