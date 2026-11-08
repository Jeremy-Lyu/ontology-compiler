from __future__ import annotations

"""C-1 tests: unified LLM gateway + pricing registry + hard budget.

Covers:
- pricing registry: exact model lookup, provider fallback, unknown => price_known=False
- cost estimation: input/output split, cache-read discount, never silent 0
- budget ledger: per-call record, cap enforcement (cost/calls/tokens), reset
- monitor: unknown-priced calls are flagged, not silently counted as $0
- gateway: budget gate blocks before provider contact; execute records budget
"""

import asyncio
import unittest
from typing import Any, Dict

from src.application.services.llm_budget import (
    LLMBudgetExceededError,
    LLMBudgetLedger,
    LLMBudgetPolicy,
)
from src.application.services.llm_monitor import LLMMonitor
from src.application.services.llm_pricing import (
    CostEstimate,
    estimate_cost,
    resolve_price,
)


class LlmPricingTest(unittest.TestCase):
    def test_exact_model_lookup(self):
        self.assertEqual(resolve_price("deepseek", "deepseek-chat"), (0.27, 1.10))

    def test_provider_family_fallback(self):
        self.assertEqual(resolve_price("deepseek", "some-new-model"), (0.55, 2.19))

    def test_unknown_provider_returns_none(self):
        self.assertIsNone(resolve_price("nonsense-provider", None))
        self.assertIsNone(resolve_price("nonsense-provider", "unknown-model"))

    def test_estimate_cost_known_prices(self):
        est = estimate_cost("deepseek", "deepseek-chat", prompt_tokens=1000, completion_tokens=1000)
        self.assertTrue(est.price_known)
        self.assertAlmostEqual(est.usd, (1000 * 0.27 + 1000 * 1.10) / 1_000_000, places=6)

    def test_estimate_cost_unknown_price_is_flagged_not_zero_cost(self):
        est = estimate_cost("mystery", "model-x", prompt_tokens=5000, completion_tokens=5000)
        self.assertFalse(est.price_known)
        self.assertEqual(est.usd, 0.0)
        # Explicit contract: callers must inspect price_known, never infer "free".
        self.assertIsInstance(est, CostEstimate)

    def test_cache_read_tokens_discounted(self):
        est = estimate_cost(
            "openai", "gpt-4o",
            prompt_tokens=100, completion_tokens=100,
            cache_create_tokens=200, cache_read_tokens=300,
        )
        self.assertTrue(est.price_known)
        expected = (
            100 * 2.50 / 1_000_000
            + 200 * 2.50 / 1_000_000
            + 300 * 2.50 * 0.1 / 1_000_000
            + 100 * 10.00 / 1_000_000
        )
        self.assertAlmostEqual(est.usd, expected, places=6)

    def test_zero_usage_is_zero_but_known(self):
        est = estimate_cost("deepseek", "deepseek-chat", 0, 0)
        self.assertTrue(est.price_known)
        self.assertEqual(est.usd, 0.0)


class LlmBudgetLedgerTest(unittest.TestCase):
    def test_default_policy_is_inactive(self):
        ledger = LLMBudgetLedger()
        self.assertFalse(ledger.policy.is_active)
        ledger.check()  # must not raise

    def test_call_cap_blocks_next_call(self):
        policy = LLMBudgetPolicy(max_calls=2)
        ledger = LLMBudgetLedger(policy=policy)
        ledger.record(cost_usd=0.0, prompt_tokens=1, completion_tokens=1)
        ledger.record(cost_usd=0.0, prompt_tokens=1, completion_tokens=1)
        with self.assertRaises(LLMBudgetExceededError) as ctx:
            ledger.check()
        self.assertIn("call budget", ctx.exception.reason)

    def test_cost_cap_blocks_next_call(self):
        policy = LLMBudgetPolicy(max_cost_usd=1.0)
        ledger = LLMBudgetLedger(policy=policy)
        ledger.record(cost_usd=0.6, prompt_tokens=10, completion_tokens=10)
        ledger.check()  # still under
        ledger.record(cost_usd=0.5, prompt_tokens=10, completion_tokens=10)
        with self.assertRaises(LLMBudgetExceededError):
            ledger.check()

    def test_token_cap_blocks_next_call(self):
        policy = LLMBudgetPolicy(max_total_tokens=100)
        ledger = LLMBudgetLedger(policy=policy)
        ledger.record(cost_usd=0.0, prompt_tokens=60, completion_tokens=40)
        with self.assertRaises(LLMBudgetExceededError):
            ledger.check()

    def test_record_state_and_snapshot(self):
        ledger = LLMBudgetLedger()
        ledger.record(cost_usd=0.001, prompt_tokens=10, completion_tokens=5)
        self.assertEqual(ledger.state.total_calls, 1)
        self.assertEqual(ledger.state.total_tokens, 15)
        snap = ledger.snapshot()
        self.assertEqual(snap["state"]["total_calls"], 1)
        self.assertEqual(snap["call_count_recorded"], 1)

    def test_reset_clears_state(self):
        policy = LLMBudgetPolicy(max_calls=1)
        ledger = LLMBudgetLedger(policy=policy)
        ledger.record(cost_usd=0.0, prompt_tokens=1, completion_tokens=1)
        ledger.reset()
        ledger.check()  # no longer blocked
        self.assertEqual(ledger.state.total_calls, 0)

    def test_on_exceed_hook_invoked(self):
        seen: list[str] = []
        policy = LLMBudgetPolicy(max_calls=1, on_exceed=seen.append)
        ledger = LLMBudgetLedger(policy=policy)
        ledger.record(cost_usd=0.0, prompt_tokens=1, completion_tokens=1)
        with self.assertRaises(LLMBudgetExceededError):
            ledger.check()
        self.assertEqual(len(seen), 1)
        self.assertIn("call budget", seen[0])


class LlmMonitorPricingTest(unittest.TestCase):
    def setUp(self):
        self.monitor = LLMMonitor()
        self.monitor.enable()

    def test_unknown_provider_price_flagged_in_summary(self):
        self.monitor.record_call(
            provider="mystery", model="x", prompt_tokens=10, completion_tokens=10, total_tokens=20
        )
        self.assertEqual(self.monitor.get_summary()["unknown_price_calls"], 1)
        record = self.monitor.last_record()
        self.assertFalse(record.price_known)

    def test_known_model_price_used_for_cost(self):
        self.monitor.record_call(
            provider="deepseek", model="deepseek-chat",
            prompt_tokens=1_000_000, completion_tokens=0, total_tokens=1_000_000,
        )
        record = self.monitor.last_record()
        self.assertTrue(record.price_known)
        self.assertAlmostEqual(record.cost_estimate, 0.27, places=6)

    def test_legacy_provider_rate_takes_precedence(self):
        self.monitor.set_provider_cost_rate("custom", 0.01)  # per-1k-total legacy
        self.monitor.record_call(
            provider="custom", model="m", prompt_tokens=500, completion_tokens=500, total_tokens=1000
        )
        record = self.monitor.last_record()
        self.assertTrue(record.price_known)
        self.assertAlmostEqual(record.cost_estimate, (1000 / 1000) * 0.01, places=6)

    def test_reset_clears_unknown_counter(self):
        self.monitor.record_call(provider="mystery", model="x", total_tokens=1)
        self.monitor.reset()
        self.assertEqual(self.monitor.get_summary()["unknown_price_calls"], 0)


class _FakeSyncClient:
    """Synchronous client that mimics the bootstrap wrapper's monitoring behavior."""

    def __init__(self) -> None:
        self.calls: list[Dict[str, Any]] = []

    def call(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        return f"echo:{prompt[:16]}"


class _FakeBootstrap:
    class _Config:
        primary_provider = None

    def __init__(self) -> None:
        self.config = self._Config()
        self._client = _FakeSyncClient()

    def get_client(self, provider: Any) -> Any:
        return self._client

    def get_unified_provider(self) -> Any:
        raise NotImplementedError

    def get_rigor_client(self) -> Any:
        raise NotImplementedError


class LlmGatewayBudgetTest(unittest.TestCase):
    def setUp(self):
        from src.application.bootstrap.llm_gateway import LLMGateway

        self.bootstrap = _FakeBootstrap()
        self.gateway = LLMGateway(bootstrap=self.bootstrap)

    def test_execute_without_budget_policy(self):
        outcome = self.gateway.execute("hello world", provider="local", model="local")
        self.assertTrue(outcome.success)
        self.assertEqual(outcome.text, "echo:hello world")
        self.assertEqual(len(self.bootstrap._client.calls), 1)

    def test_budget_blocks_before_provider_contact(self):
        self.gateway.set_budget_policy(max_calls=1)
        self.gateway.execute("first", provider="local", model="local")
        with self.assertRaises(LLMBudgetExceededError):
            self.gateway.execute("second", provider="local", model="local")
        # The second call must never reach the provider.
        self.assertEqual(len(self.bootstrap._client.calls), 1)

    def test_budget_state_accumulates(self):
        self.gateway.set_budget_policy(max_calls=10)
        for _ in range(3):
            self.gateway.execute("x", provider="local", model="local")
        self.assertEqual(self.gateway.budget.state.total_calls, 3)

    def test_budget_replacement_resets_state(self):
        self.gateway.set_budget_policy(max_calls=1)
        self.gateway.execute("first", provider="local", model="local")
        self.gateway.set_budget_policy(max_calls=1)  # new run
        self.gateway.execute("second", provider="local", model="local")
        self.assertEqual(self.gateway.budget.state.total_calls, 1)

    def test_execute_outcome_fields(self):
        outcome = self.gateway.execute("hi", provider="local", model="local", stage="test")
        self.assertEqual(outcome.provider, "local")
        self.assertEqual(outcome.model, "local")
        self.assertTrue(outcome.prompt_hash)
        self.assertEqual(outcome.error_type, "")


class LlmGatewayAsyncTest(unittest.TestCase):
    def test_execute_async_with_sync_client(self):
        from src.application.bootstrap.llm_gateway import LLMGateway

        bootstrap = _FakeBootstrap()
        gateway = LLMGateway(bootstrap=bootstrap)

        async def run():
            return await gateway.execute_async("hi async", provider="local", model="local")

        outcome = asyncio.run(run())
        self.assertTrue(outcome.success)
        self.assertEqual(outcome.text, "echo:hi async")
        self.assertEqual(len(bootstrap._client.calls), 1)


if __name__ == "__main__":
    unittest.main()