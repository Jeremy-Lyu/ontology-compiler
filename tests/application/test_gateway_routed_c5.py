from __future__ import annotations

"""C-5 wiring tests: LLMGateway.execute_routed integration.

Verifies the gateway:
- accepts task_kind as str alias
- produces RouteDecision attached to LLMCallOutcome
- routed_model=True on routed outcomes
- explicit overrides win over routed defaults
- route() can be called standalone (no execution)
"""

import unittest
from typing import Any

from src.application.bootstrap.llm_gateway import LLMCallOutcome, LLMGateway
from src.application.services.model_router import (
    ModelRoutingConfig,
    ModelRoutingPolicy,
    ModelTier,
    RouteDecision,
    TaskComplexity,
    TaskKind,
    set_model_routing_policy,
)


class _FakeClient:
    """Minimal client that returns a fixed string and records call kwargs."""

    def __init__(self, response: str = "ok") -> None:
        self.response = response
        self.last_kwargs: dict[str, Any] = {}

    def call(self, prompt: str, temperature: Any = None, max_tokens: Any = None) -> str:
        self.last_kwargs = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        return self.response


class _FakeBootstrap:
    """Stand-in bootstrap that returns a fake client regardless of provider."""

    def __init__(self, client: _FakeClient) -> None:
        self._client = client
        self.config = type(
            "Cfg",
            (),
            {"primary_provider": __import__("src.application.bootstrap.llm_bootstrap", fromlist=["LLMProviderType"]).LLMProviderType.DEEPSEEK},
        )()

    def get_client(self, provider: Any) -> Any:
        return self._client


class GatewayRouteTest(unittest.TestCase):
    def setUp(self):
        self.client = _FakeClient("answer-text")
        self.gateway = LLMGateway(bootstrap=_FakeBootstrap(self.client))  # type: ignore[arg-type]

    def tearDown(self):
        set_model_routing_policy(None)

    def test_route_standalone_returns_decision(self):
        decision = self.gateway.route("Classify scene type", task_kind="classification")
        self.assertIsInstance(decision, RouteDecision)
        self.assertEqual(decision.tier, TaskComplexity.LIGHT)

    def test_execute_routed_attaches_decision(self):
        outcome = self.gateway.execute_routed("Classify scene type", task_kind="classification")
        self.assertIsInstance(outcome, LLMCallOutcome)
        self.assertTrue(outcome.routed_model)
        self.assertIsNotNone(outcome.route_decision)
        self.assertEqual(outcome.route_decision.tier, TaskComplexity.LIGHT)
        # The fake client was called.
        self.assertEqual(self.client.last_kwargs["prompt"], "Classify scene type")

    def test_execute_routed_uses_light_tier_max_tokens(self):
        outcome = self.gateway.execute_routed("Classify scene type", task_kind="classification")
        self.assertEqual(outcome.route_decision.model_tier.max_tokens, 1024)
        self.assertEqual(self.client.last_kwargs["max_tokens"], 1024)

    def test_override_max_tokens_wins(self):
        outcome = self.gateway.execute_routed(
            "Classify scene type", task_kind="classification", max_tokens_override=2048
        )
        self.assertEqual(self.client.last_kwargs["max_tokens"], 2048)

    def test_override_temperature_wins(self):
        outcome = self.gateway.execute_routed(
            "Classify scene type", task_kind="classification", temperature_override=0.7
        )
        self.assertEqual(self.client.last_kwargs["temperature"], 0.7)

    def test_execute_routed_escalates_on_conflict(self):
        outcome = self.gateway.execute_routed(
            "two candidates conflict", task_kind=TaskKind.ENTITY_EXTRACTION, has_conflict=True
        )
        self.assertEqual(outcome.route_decision.tier, TaskComplexity.HEAVY)
        self.assertEqual(outcome.route_decision.model_tier.model, "deepseek-reasoner")

    def test_injected_policy_used(self):
        # Inject a custom policy whose heavy tier uses a different model.
        custom_cfg = ModelRoutingConfig(
            heavy=ModelTier(provider="openai", model="gpt-4o", max_tokens=4096, temperature=0.5)
        )
        custom_policy = ModelRoutingPolicy(custom_cfg)
        set_model_routing_policy(custom_policy)
        outcome = self.gateway.execute_routed(
            "two candidates conflict", task_kind=TaskKind.ENTITY_EXTRACTION, has_conflict=True
        )
        self.assertEqual(outcome.route_decision.model_tier.model, "gpt-4o")

    def test_explicit_policy_param_overrides_singleton(self):
        # Pass policy directly; should not consult the singleton.
        local_cfg = ModelRoutingConfig(
            standard=ModelTier(provider="openai", model="gpt-4o-mini", max_tokens=256, temperature=0.0)
        )
        local_policy = ModelRoutingPolicy(local_cfg)
        # Use entity_count_hint=10 to stay in STANDARD (avoids LIGHT downgrade
        # since 10 > light_max_entity_hint=8, and avoids HEAVY since 10 <= 40).
        outcome = self.gateway.execute_routed(
            "extract entities",
            task_kind=TaskKind.ENTITY_EXTRACTION,
            entity_count_hint=10,
            policy=local_policy,
        )
        # Should be STANDARD -> gpt-4o-mini
        self.assertEqual(outcome.route_decision.model_tier.model, "gpt-4o-mini")


if __name__ == "__main__":
    unittest.main()
