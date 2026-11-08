"""C-2 tests: PromptBudgetPolicy characterization.

Covers:
- default policy returns safe defaults when no YAML is present
- from_yaml loads tasks and document_type_profiles
- resolve_profile picks explicit task > document-type routing > fallback
- resolve returns a PromptBudget with correct caps
- prompt_tokens override shrinks completion allowance
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.application.services.prompt_budget_policy import (
    PromptBudget,
    PromptBudgetPolicy,
    get_prompt_budget_policy,
)


class PromptBudgetPolicyDefaultTest(unittest.TestCase):
    def test_default_policy_has_safe_defaults(self):
        policy = PromptBudgetPolicy()
        budget = policy.resolve(task="anything")
        self.assertGreater(budget.max_completion_tokens, 0)
        self.assertGreater(budget.max_prompt_tokens, 0)
        self.assertGreater(budget.total_budget, 0)
        self.assertEqual(budget.source, "default")

    def test_default_policy_fallback_profile(self):
        policy = PromptBudgetPolicy()
        budget = policy.resolve()
        self.assertTrue(budget.profile)


class PromptBudgetPolicyYamlTest(unittest.TestCase):
    def _write_yaml(self, content: str) -> Path:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
        f.write(content)
        f.close()
        return Path(f.name)

    def test_from_yaml_loads_tasks(self):
        path = self._write_yaml("""
defaults:
  max_completion_tokens: 6000
  max_prompt_tokens: 10000
  total_budget: 16000
  temperature: 0.2
  truncation_retry_max: 2
  json_repair_max_retries: 3

tasks:
  extraction_technical:
    max_completion_tokens: 8000
    max_prompt_tokens: 12000
    temperature: 0.2
  scene_classification:
    max_completion_tokens: 1000
    max_prompt_tokens: 5000
    temperature: 0.2

document_type_profiles:
  pptx:
    default: extraction_technical
    business: extraction_technical
  md:
    default: scene_classification
""")
        policy = PromptBudgetPolicy.from_yaml(path)
        self.assertIn("extraction_technical", policy.tasks)
        self.assertIn("scene_classification", policy.tasks)
        self.assertIn("pptx", policy.document_type_profiles)

    def test_from_yaml_missing_file_returns_default(self):
        policy = PromptBudgetPolicy.from_yaml("/nonexistent/path.yaml")
        self.assertEqual(len(policy.tasks), 0)
        budget = policy.resolve()
        self.assertEqual(budget.source, "default")

    def test_resolve_profile_explicit_task_wins(self):
        path = self._write_yaml("""
tasks:
  extraction_technical:
    max_completion_tokens: 8000
  scene_classification:
    max_completion_tokens: 1000
document_type_profiles:
  pptx:
    default: extraction_technical
""")
        policy = PromptBudgetPolicy.from_yaml(path)
        profile = policy.resolve_profile(task="scene_classification", document_type="pptx")
        self.assertEqual(profile, "scene_classification")

    def test_resolve_profile_document_type_routing(self):
        path = self._write_yaml("""
tasks:
  extraction_technical:
    max_completion_tokens: 8000
  extraction_business:
    max_completion_tokens: 6000
document_type_profiles:
  pptx:
    default: extraction_technical
    business: extraction_business
""")
        policy = PromptBudgetPolicy.from_yaml(path)
        profile = policy.resolve_profile(document_type="pptx", domain="business")
        self.assertEqual(profile, "extraction_business")

    def test_resolve_profile_fallback(self):
        path = self._write_yaml("""
tasks:
  extraction_general:
    max_completion_tokens: 6000
  extraction_technical:
    max_completion_tokens: 8000
""")
        policy = PromptBudgetPolicy.from_yaml(path)
        profile = policy.resolve_profile(document_type="unknown_type")
        self.assertEqual(profile, "extraction_general")

    def test_resolve_returns_correct_caps(self):
        path = self._write_yaml("""
defaults:
  max_completion_tokens: 8000
  max_prompt_tokens: 12000
  total_budget: 20000
  temperature: 0.2
  truncation_retry_max: 2
  json_repair_max_retries: 3
tasks:
  scene_classification:
    max_completion_tokens: 1000
    max_prompt_tokens: 5000
    temperature: 0.2
""")
        policy = PromptBudgetPolicy.from_yaml(path)
        budget = policy.resolve(task="scene_classification")
        self.assertEqual(budget.max_completion_tokens, 1000)
        self.assertEqual(budget.max_prompt_tokens, 5000)
        self.assertEqual(budget.temperature, 0.2)
        self.assertEqual(budget.truncation_retry_max, 2)
        self.assertEqual(budget.json_repair_max_retries, 3)
        self.assertEqual(budget.source, "yaml")

    def test_prompt_tokens_override_shrinks_completion(self):
        path = self._write_yaml("""
defaults:
  max_completion_tokens: 8000
  max_prompt_tokens: 12000
  total_budget: 20000
tasks:
  extraction_technical:
    max_completion_tokens: 8000
    max_prompt_tokens: 12000
""")
        policy = PromptBudgetPolicy.from_yaml(path)
        budget = policy.resolve(task="extraction_technical", prompt_tokens=18000)
        self.assertLess(budget.max_completion_tokens, 8000)
        self.assertGreaterEqual(budget.max_completion_tokens, 0)

    def test_completion_tokens_override_caps_result(self):
        path = self._write_yaml("""
defaults:
  max_completion_tokens: 8000
tasks:
  extraction_technical:
    max_completion_tokens: 8000
""")
        policy = PromptBudgetPolicy.from_yaml(path)
        budget = policy.resolve(task="extraction_technical", completion_tokens=2000)
        self.assertEqual(budget.max_completion_tokens, 2000)

    def test_max_tokens_alias(self):
        budget = PromptBudget(
            profile="test",
            max_prompt_tokens=100,
            max_completion_tokens=50,
            total_budget=150,
            temperature=0.0,
            truncation_retry_max=1,
            json_repair_max_retries=1,
        )
        self.assertEqual(budget.max_tokens, 50)


class PromptBudgetPolicySingletonTest(unittest.TestCase):
    def test_get_prompt_budget_policy_returns_instance(self):
        policy = get_prompt_budget_policy()
        self.assertIsInstance(policy, PromptBudgetPolicy)

    def test_get_prompt_budget_policy_cached(self):
        p1 = get_prompt_budget_policy()
        p2 = get_prompt_budget_policy()
        self.assertIs(p1, p2)


if __name__ == "__main__":
    unittest.main()
