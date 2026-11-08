"""P2-1 (2026-10-16): LLM provider circuit breaker.

Root issue O-2: transient provider errors (DeepSeek/Dify/DNS) make the same
command succeed or fail run to run. Locks:

- consecutive-failure threshold trips the breaker (open)
- a success resets it (closed)
- cooldown elapsed => half-open (allows again)
- killswitch (enabled=False) => full no-op, nothing persisted
- cross-process status file carries consecutive_failures + alert
- ``allow`` only blocks when ``enforce`` is on (default off => zero behavior change)
- the llm_monitor funnel feeds the breaker, but a bare parse-failure
  (success=False with no error_message) must NOT trip it
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.application.services.provider_circuit_breaker import (
    DEFAULT_COOLDOWN_SECONDS,
    ProviderCircuitBreaker,
    ProviderCircuitOpenError,
    get_circuit_breaker,
    reset_circuit_breaker_for_tests,
)


class _FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _breaker(tmp: Path, clock: _FakeClock, **kw) -> ProviderCircuitBreaker:
    return ProviderCircuitBreaker(
        threshold=kw.pop("threshold", 3),
        cooldown_seconds=kw.pop("cooldown_seconds", 60.0),
        status_path=tmp / "state.json",
        clock=clock,
        **kw,
    )


class ThresholdTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.clock = _FakeClock()
        self.b = _breaker(self.tmp, self.clock)

    def test_trips_only_at_threshold(self):
        for _ in range(2):
            self.b.record_failure("deepseek", "connection reset")
            self.assertFalse(self.b.is_open("deepseek"))
        self.b.record_failure("deepseek", "connection reset")
        self.assertTrue(self.b.is_open("deepseek"))

    def test_success_resets_consecutive_count(self):
        for _ in range(3):
            self.b.record_failure("deepseek", "x")
        self.assertTrue(self.b.is_open("deepseek"))
        self.b.record_success("deepseek")
        self.assertFalse(self.b.is_open("deepseek"))
        # needs a full fresh run of failures to trip again
        self.b.record_failure("deepseek", "x")
        self.assertFalse(self.b.is_open("deepseek"))

    def test_providers_are_independent(self):
        for _ in range(3):
            self.b.record_failure("deepseek", "x")
        self.assertTrue(self.b.is_open("deepseek"))
        self.assertFalse(self.b.is_open("dify"))

    def test_records_last_error(self):
        self.b.record_failure("deepseek", "nodename nor servname provided")
        snap = self.b.snapshot()
        self.assertIn("nodename", snap.providers["deepseek"]["last_error"])


class CooldownTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.clock = _FakeClock()
        self.b = _breaker(self.tmp, self.clock, cooldown_seconds=60.0)

    def test_half_open_after_cooldown_elapses(self):
        for _ in range(3):
            self.b.record_failure("deepseek", "x")
        self.assertTrue(self.b.is_open("deepseek"))
        self.clock.advance(59.0)
        self.assertTrue(self.b.is_open("deepseek"))
        self.clock.advance(2.0)
        self.assertFalse(self.b.is_open("deepseek"))


class KillswitchTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_disabled_is_full_noop(self):
        b = _breaker(self.tmp, _FakeClock(), enabled=False)
        for _ in range(10):
            b.record_failure("deepseek", "x")
        self.assertFalse(b.is_open("deepseek"))
        self.assertFalse((self.tmp / "state.json").exists())


class EnforceTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.b = _breaker(self.tmp, _FakeClock())

    def test_allow_by_default_even_when_open(self):
        for _ in range(3):
            self.b.record_failure("deepseek", "x")
        self.assertTrue(self.b.allow("deepseek"))  # enforce off => never blocks

    def test_allow_blocks_when_enforcing(self):
        self.b.enforce = True
        for _ in range(3):
            self.b.record_failure("deepseek", "x")
        self.assertFalse(self.b.allow("deepseek"))
        self.assertTrue(self.b.allow("dify"))

    def test_open_error_is_runtime_error(self):
        self.assertTrue(issubclass(ProviderCircuitOpenError, RuntimeError))


class PersistenceTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_status_file_carries_state_and_alert(self):
        b = _breaker(self.tmp, _FakeClock())
        for _ in range(3):
            b.record_failure("deepseek", "boom")
        payload = json.loads((self.tmp / "state.json").read_text(encoding="utf-8"))
        self.assertTrue(payload["alert"])
        self.assertEqual(payload["providers"]["deepseek"]["consecutive_failures"], 3)
        self.assertEqual(payload["providers"]["deepseek"]["total_failures"], 3)

    def test_persist_failure_is_swallowed(self):
        b = _breaker(self.tmp, _FakeClock())
        b.status_path = Path("/proc/definitely/not/writable/state.json")
        b.record_failure("deepseek", "x")  # must not raise
        self.assertEqual(b.snapshot().providers["deepseek"]["consecutive_failures"], 1)

    def test_snapshot_alert_false_when_closed(self):
        b = _breaker(self.tmp, _FakeClock())
        b.record_failure("deepseek", "x")
        self.assertFalse(b.snapshot().alert)


class SingletonEnvTest(unittest.TestCase):
    def tearDown(self):
        reset_circuit_breaker_for_tests()

    def test_env_configures_singleton(self):
        reset_circuit_breaker_for_tests()
        with mock.patch.dict(
            "os.environ",
            {"LLM_BREAKER_THRESHOLD": "5", "LLM_BREAKER_COOLDOWN": "42", "LLM_BREAKER_ENFORCE": "1"},
            clear=False,
        ):
            b = get_circuit_breaker()
        self.assertEqual(b.threshold, 5)
        self.assertEqual(b.cooldown_seconds, 42.0)
        self.assertTrue(b.enforce)
        self.assertEqual(DEFAULT_COOLDOWN_SECONDS, 300.0)

    def test_env_killswitch(self):
        reset_circuit_breaker_for_tests()
        with mock.patch.dict("os.environ", {"LLM_BREAKER_ENABLED": "0"}, clear=False):
            b = get_circuit_breaker()
        self.assertFalse(b.enabled)


class MonitorWiringTest(unittest.TestCase):
    """The llm_monitor funnel must feed the breaker — and NOT on parse failures."""

    def setUp(self):
        reset_circuit_breaker_for_tests()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp = Path(self._tmp.name)
        self.breaker = _breaker(tmp, _FakeClock())
        with mock.patch(
            "src.application.services.provider_circuit_breaker.get_circuit_breaker",
            return_value=self.breaker,
        ):
            from src.application.services.llm_monitor import LLMMonitor

            self.monitor = LLMMonitor()
            for _ in range(3):
                self.monitor.record_call(
                    provider="deepseek", model="m", success=False, error_message="connection reset"
                )
        self.assertTrue(self.breaker.is_open("deepseek"))

    def test_connection_failure_trips(self):
        self.assertTrue(self.breaker.is_open("deepseek"))

    def test_parse_failure_without_error_message_does_not_trip(self):
        breaker = _breaker(Path(self._tmp.name), _FakeClock())
        with mock.patch(
            "src.application.services.provider_circuit_breaker.get_circuit_breaker",
            return_value=breaker,
        ):
            from src.application.services.llm_monitor import LLMMonitor

            m = LLMMonitor()
            for _ in range(5):
                m.record_call(provider="deepseek", model="m", success=False)
        self.assertFalse(breaker.is_open("deepseek"))

    def test_success_resets(self):
        with mock.patch(
            "src.application.services.provider_circuit_breaker.get_circuit_breaker",
            return_value=self.breaker,
        ):
            self.monitor.record_call(provider="deepseek", model="m", success=True)
        self.assertFalse(self.breaker.is_open("deepseek"))


class PersistPolicyTest(unittest.TestCase):
    """record_success must not add per-call IO when nothing changed."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.b = _breaker(self.tmp, _FakeClock())

    def test_success_without_prior_failure_writes_nothing(self):
        self.b.record_success("deepseek")
        self.assertFalse((self.tmp / "state.json").exists())

    def test_success_after_failure_clears_and_persists(self):
        for _ in range(3):
            self.b.record_failure("deepseek", "x")
        self.assertTrue((self.tmp / "state.json").exists())
        self.b.record_success("deepseek")
        payload = json.loads((self.tmp / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["providers"]["deepseek"]["consecutive_failures"], 0)
        self.assertFalse(payload["alert"])


if __name__ == "__main__":
    unittest.main()
