from __future__ import annotations

"""C-5 characterization tests: cost-aware model routing policy.

Covers:
- task kind inference from prompt text
- complexity classification: LIGHT / STANDARD / HEAVY
- conflict / low-confidence escalation to HEAVY
- prompt-length and entity-count driven escalation
- tier resolution: light -> cheap model, heavy -> strong model
- route() produces a RouteDecision with signals + reason
- task_kind str alias accepted by gateway.route()
"""

import unittest

from src.application.services.model_router import (
    ModelRoutingConfig,
    ModelRoutingPolicy,
    ModelTier,
    RouteDecision,
    TaskComplexity,
    TaskKind,
    extract_routing_signals,
    get_model_routing_policy,
    infer_task_kind,
    set_model_routing_policy,
)


class TaskKindInferenceTest(unittest.TestCase):
    def test_json_repair_inferred(self):
        self.assertEqual(infer_task_kind("Please repair this JSON and return only valid JSON"), TaskKind.JSON_REPAIR)

    def test_classification_inferred(self):
        self.assertEqual(infer_task_kind("Classify the scene type"), TaskKind.CLASSIFICATION)

    def test_entity_extraction_inferred(self):
        self.assertEqual(infer_task_kind("Extract entity candidates from text"), TaskKind.ENTITY_EXTRACTION)

    def test_generic_when_no_match(self):
        self.assertEqual(infer_task_kind("hello world"), TaskKind.GENERIC)


class SignalExtractionTest(unittest.TestCase):
    def test_signals_include_prompt_chars_and_tokens(self):
        sig = extract_routing_signals("a" * 1000)
        self.assertEqual(sig["prompt_chars"], 1000)
        self.assertEqual(sig["approx_tokens"], 250)

    def test_requires_json_detected(self):
        sig = extract_routing_signals("Respond in JSON format")
        self.assertTrue(sig["requires_json"])

    def test_conflict_marker_detected(self):
        sig = extract_routing_signals("two candidates conflict with each other")
        self.assertTrue(sig["has_conflict"])

    def test_low_confidence_marker_detected(self):
        sig = extract_routing_signals("the result is uncertain and low-confidence")
        self.assertTrue(sig["low_confidence_inferred"])

    def test_explicit_conflict_overrides_inference(self):
        sig = extract_routing_signals("clean prompt", has_conflict=True)
        self.assertTrue(sig["has_conflict"])


class ComplexityClassificationTest(unittest.TestCase):
    def setUp(self):
        self.policy = ModelRoutingPolicy()

    def test_light_task_stays_light(self):
        c, _ = self.policy.classify_complexity("Classify scene type", task_kind=TaskKind.CLASSIFICATION)
        self.assertEqual(c, TaskComplexity.LIGHT)

    def test_light_task_escalates_to_standard_when_prompt_long(self):
        long_prompt = "Classify scene type. " * 1000
        c, _ = self.policy.classify_complexity(long_prompt, task_kind=TaskKind.CLASSIFICATION)
        self.assertEqual(c, TaskComplexity.STANDARD)

    def test_conflict_always_heavy(self):
        c, _ = self.policy.classify_complexity(
            "short text", task_kind=TaskKind.ENTITY_EXTRACTION, has_conflict=True
        )
        self.assertEqual(c, TaskComplexity.HEAVY)

    def test_low_confidence_always_heavy(self):
        c, _ = self.policy.classify_complexity(
            "short text", task_kind=TaskKind.ENTITY_EXTRACTION, confidence_hint=0.3
        )
        self.assertEqual(c, TaskComplexity.HEAVY)

    def test_standard_task_downgrades_to_light_when_small(self):
        c, _ = self.policy.classify_complexity("short", task_kind=TaskKind.ENTITY_EXTRACTION)
        self.assertEqual(c, TaskComplexity.LIGHT)

    def test_standard_task_escalates_to_heavy_when_very_long(self):
        very_long = "x" * 20000
        c, _ = self.policy.classify_complexity(very_long, task_kind=TaskKind.ENTITY_EXTRACTION)
        self.assertEqual(c, TaskComplexity.HEAVY)

    def test_standard_task_escalates_to_heavy_when_many_entities(self):
        c, _ = self.policy.classify_complexity(
            "x" * 5000, task_kind=TaskKind.ENTITY_EXTRACTION, entity_count_hint=100
        )
        self.assertEqual(c, TaskComplexity.HEAVY)

    def test_json_requirement_prevents_light_downgrade(self):
        # JSON repair is light by default, but requires_json forces STANDARD minimum.
        c, _ = self.policy.classify_complexity(
            "Please respond in JSON schema",
            task_kind=TaskKind.ENTITY_EXTRACTION,
        )
        self.assertNotEqual(c, TaskComplexity.LIGHT)


class TierResolutionTest(unittest.TestCase):
    def setUp(self):
        self.policy = ModelRoutingPolicy()

    def test_light_uses_light_tier(self):
        tier = self.policy.resolve_model_tier(TaskComplexity.LIGHT)
        self.assertEqual(tier.model, "deepseek-chat")
        self.assertEqual(tier.max_tokens, 1024)

    def test_heavy_uses_heavy_tier(self):
        tier = self.policy.resolve_model_tier(TaskComplexity.HEAVY)
        self.assertEqual(tier.model, "deepseek-reasoner")

    def test_custom_config_override(self):
        custom = ModelRoutingConfig(
            light=ModelTier(provider="openai", model="gpt-4o-mini", max_tokens=512, temperature=0.0),
        )
        policy = ModelRoutingPolicy(custom)
        tier = policy.resolve_model_tier(TaskComplexity.LIGHT)
        self.assertEqual(tier.model, "gpt-4o-mini")
        self.assertEqual(tier.max_tokens, 512)


class RouteTest(unittest.TestCase):
    def setUp(self):
        self.policy = ModelRoutingPolicy()

    def test_route_returns_decision_with_signals(self):
        decision = self.policy.route("Classify scene type", task_kind=TaskKind.CLASSIFICATION)
        self.assertIsInstance(decision, RouteDecision)
        self.assertEqual(decision.tier, TaskComplexity.LIGHT)
        self.assertIn("task=classification", decision.reason)
        self.assertIn("complexity=light", decision.reason)
        self.assertEqual(decision.signals["task_kind"], "classification")

    def test_route_escalates_on_conflict(self):
        decision = self.policy.route(
            "Resolve contradiction between two candidates", task_kind=TaskKind.CONFLICT_RESOLUTION
        )
        self.assertEqual(decision.tier, TaskComplexity.HEAVY)
        self.assertIn("conflict-detected", decision.reason)

    def test_route_attaches_entity_hint_in_signals(self):
        decision = self.policy.route(
            "extract entities", task_kind=TaskKind.ENTITY_EXTRACTION, entity_count_hint=20
        )
        self.assertEqual(decision.signals["entity_count_hint"], 20)


class PolicySingletonTest(unittest.TestCase):
    def tearDown(self):
        set_model_routing_policy(None)  # reset

    def test_default_singleton_returns_policy(self):
        policy = get_model_routing_policy()
        self.assertIsInstance(policy, ModelRoutingPolicy)

    def test_inject_custom_policy(self):
        custom = ModelRoutingPolicy(ModelRoutingConfig())
        set_model_routing_policy(custom)
        self.assertIs(get_model_routing_policy(), custom)


if __name__ == "__main__":
    unittest.main()
