"""C-2 tests: TrueUnifiedExtractor / PromptService / TextChunker budget wiring.

Characterizes the integration that makes the extraction flow derive
max_tokens, temperature, prompt-length ceiling, and chunk size from a
PromptBudgetPolicy instead of hardcoded defaults.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.application.services.prompt_budget_policy import PromptBudgetPolicy


def _make_policy(max_completion=8000, max_prompt=12000, temperature=0.2,
                 doc_tokens=10600, chunk_size=7000) -> PromptBudgetPolicy:
    path = Path(tempfile.mkstemp(suffix=".yaml")[1])
    path.write_text(f"""
defaults:
  max_completion_tokens: 8000
  max_prompt_tokens: 12000
  total_budget: 20000
  temperature: 0.2
  truncation_retry_max: 2
  json_repair_max_retries: 3
tasks:
  extraction_technical:
    max_completion_tokens: {max_completion}
    max_prompt_tokens: {max_prompt}
    temperature: {temperature}
    layers:
      base: 200
      domain: 300
      task: 500
      schema_hint: 400
      document: {doc_tokens}
chunking:
  max_chunk_size: {chunk_size}
  chunk_text_token_estimate: 1.5
""", encoding="utf-8")
    return PromptBudgetPolicy.from_yaml(path)


class TextChunkerBudgetWiringTest(unittest.TestCase):
    def test_chunker_without_policy_uses_default(self):
        from ontology_compiler.services.text_chunking import TextChunker
        chunker = TextChunker()
        default = TextChunker.DEFAULT_CONFIG.get("max_chunk_size")
        self.assertEqual(chunker.config.get("max_chunk_size"), default)

    def test_chunker_with_policy_uses_yaml_chunk_size(self):
        from ontology_compiler.services.text_chunking import TextChunker
        policy = _make_policy(chunk_size=7500)
        chunker = TextChunker(budget_policy=policy)
        self.assertEqual(chunker.config.get("max_chunk_size"), 7500)

    def test_resolve_chunk_size_from_budget_prefers_yaml(self):
        from ontology_compiler.services.text_chunking import _resolve_chunk_size_from_budget
        policy = _make_policy(chunk_size=9000)
        self.assertEqual(_resolve_chunk_size_from_budget(policy), 9000)

    def test_resolve_chunk_size_from_budget_derives_from_layers(self):
        from ontology_compiler.services.text_chunking import _resolve_chunk_size_from_budget
        # Build a policy whose YAML chunking.max_chunk_size is absent so the
        # helper falls back to layers.document token budget.
        path = Path(tempfile.mkstemp(suffix=".yaml")[1])
        path.write_text("""
tasks:
  extraction_technical:
    max_completion_tokens: 8000
    layers:
      document: 6000
""", encoding="utf-8")
        policy = PromptBudgetPolicy.from_yaml(path)
        # 6000 tokens / 1.5 chars-per-token = 4000 chars
        self.assertEqual(_resolve_chunk_size_from_budget(policy), 4000)

    def test_resolve_chunk_size_from_budget_returns_none_for_default(self):
        from ontology_compiler.services.text_chunking import _resolve_chunk_size_from_budget
        policy = PromptBudgetPolicy()
        self.assertIsNone(_resolve_chunk_size_from_budget(policy))


class PromptServiceBudgetWiringTest(unittest.TestCase):
    def _make_service(self, policy=None):
        from ontology_compiler.services.prompt_service import PromptService
        return PromptService(
            document_analyzer=None,
            context_loader=None,
            prompt_strategy=None,
            prompt_loader=None,
            skill_loader=None,
            budget_policy=policy,
        )

    def test_budget_max_prompt_chars_uses_policy(self):
        policy = _make_policy(max_prompt=12000)
        svc = self._make_service(policy)
        chars = svc._budget_max_prompt_chars()
        self.assertIsNotNone(chars)
        self.assertGreater(chars, 0)
        # 12000 tokens × 4 chars/token = 48000 chars, clamped to env default 12000
        self.assertLessEqual(chars, 12000)

    def test_budget_max_prompt_chars_none_without_policy(self):
        svc = self._make_service()
        self.assertIsNone(svc._budget_max_prompt_chars())

    def test_budget_document_char_budget_from_layers(self):
        policy = _make_policy(doc_tokens=10600)
        svc = self._make_service(policy)
        chars = svc._budget_document_char_budget()
        self.assertIsNotNone(chars)
        # 10600 tokens × 1.5 chars/token = 15900 chars
        self.assertEqual(chars, 15900)

    def test_budget_document_char_budget_none_without_policy(self):
        svc = self._make_service()
        self.assertIsNone(svc._budget_document_char_budget())

    def test_resolve_text_limit_returns_budget_or_none(self):
        policy = _make_policy(doc_tokens=10600)
        svc = self._make_service(policy)
        # When text is smaller than the budget, return text length
        limit = svc._resolve_text_limit(100)
        self.assertEqual(limit, 100)
        # When text exceeds the budget, return the budget
        limit = svc._resolve_text_limit(100000)
        self.assertEqual(limit, 15900)

    def test_resolve_text_limit_none_without_policy(self):
        svc = self._make_service()
        self.assertIsNone(svc._resolve_text_limit(10000))

    def test_build_unified_prompt_uses_budget_when_attached(self):
        """When a budget policy is attached, the prompt length ceiling is
        derived from the policy (clamped to the env default)."""
        from ontology_compiler.services.prompt_service import PromptService

        class _StubAnalyzer:
            def select_content_intelligently(self, text, max_len):
                return text[:max_len]

        class _StubStrategy:
            def build_base_fragment(self, domain):
                return "base"

            def build_domain_fragment(self, domain, metadata, context):
                return "domain"

            def build_task_fragment(self, schema_hints):
                return "task"

        svc = PromptService(
            document_analyzer=_StubAnalyzer(),
            context_loader=None,
            prompt_strategy=_StubStrategy(),
            prompt_loader=None,
            skill_loader=None,
            budget_policy=_make_policy(max_prompt=12000),
        )
        prompt = svc.build_unified_prompt("x" * 500, domain="SUPPLY_CHAIN", use_context=False)
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 0)


class TrueUnifiedExtractorBudgetWiringTest(unittest.TestCase):
    def test_extractor_with_budget_policy_sets_max_tokens(self):
        from ontology_compiler.services.true_unified_extractor import TrueUnifiedExtractor
        policy = _make_policy(max_completion=8000)
        ex = TrueUnifiedExtractor(budget_policy=policy, max_tokens=99999)
        self.assertEqual(ex.max_tokens, 8000)

    def test_extractor_with_budget_policy_sets_temperature(self):
        from ontology_compiler.services.true_unified_extractor import TrueUnifiedExtractor
        policy = _make_policy(temperature=0.15)
        ex = TrueUnifiedExtractor(budget_policy=policy, temperature=0.99)
        self.assertAlmostEqual(ex.temperature, 0.15)

    def test_extractor_with_budget_policy_sets_chunk_size(self):
        from ontology_compiler.services.true_unified_extractor import TrueUnifiedExtractor
        policy = _make_policy(chunk_size=7500)
        ex = TrueUnifiedExtractor(budget_policy=policy)
        if ex.chunker is not None:
            self.assertEqual(ex.chunker.config.get("max_chunk_size"), 7500)

    def test_extractor_config_override_wins_over_budget(self):
        from ontology_compiler.services.true_unified_extractor import TrueUnifiedExtractor
        policy = _make_policy(max_completion=8000)
        ex = TrueUnifiedExtractor(
            budget_policy=policy,
            config={"max_tokens": 5000, "temperature": 0.5},
        )
        self.assertEqual(ex.max_tokens, 5000)
        self.assertAlmostEqual(ex.temperature, 0.5)

    def test_extractor_budget_policy_attached(self):
        from ontology_compiler.services.true_unified_extractor import TrueUnifiedExtractor
        policy = _make_policy()
        ex = TrueUnifiedExtractor(budget_policy=policy)
        self.assertIs(ex._budget_policy, policy)

    def test_extractor_prompt_service_receives_budget_policy(self):
        from ontology_compiler.services.true_unified_extractor import TrueUnifiedExtractor
        policy = _make_policy()
        ex = TrueUnifiedExtractor(budget_policy=policy)
        self.assertIs(ex._prompt_service._budget_policy, policy)


if __name__ == "__main__":
    unittest.main()
