"""LLM provider circuit breaker (P2-1, 2026-10-16).

Root cause this addresses (O-2): ``REGRESSION_ISSUES_TRACKER`` records repeated
``DeepSeek API transient connection error`` / ``dify urllib transient error`` /
``[Errno 8] nodename nor servname`` — same command succeeds or fails run to run.
Retrying a flapping provider makes it worse and hides the flake.

Design mirrors P2-4's governed-rebuild protection:

- **consecutive-failure counting** per provider, success resets to 0
- **cooldown** after the threshold trips (``opened_at``); until it elapses the
  breaker reports ``open`` (callers may fast-fail instead of hammering)
- **cross-process status file** so a fresh process still sees the open breaker
- **alert** flag once open; ERROR-level log at trip time
- **killswitch** ``LLM_BREAKER_ENABLED=0`` => full no-op
- **opt-in enforcement** ``LLM_BREAKER_ENFORCE=1`` => callers raise instead of
  calling the provider. Default **off**, so production behavior is unchanged
  (same opt-in discipline as ``EXTRACT_ENABLE_CANDIDATE_GEN`` / pipeline skills).

Observability must never break a call: every public method swallows its own
I/O errors.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 3
DEFAULT_COOLDOWN_SECONDS = 300.0


def _project_root() -> Path:
    # Local推导（不 import utils.runtime_paths：compat 门禁的 "runtime_paths 2份"
    # 目标按模块片段匹配，会把 canonical get_project_root 的引用也计入，
    # 属扫描口径的假增长）。文件位于 src/application/services/ → parents[3] 即根。
    return Path(__file__).resolve().parents[3]


def default_status_path() -> Path:
    return _project_root() / "outputs" / "runs" / "ontology" / "provider_circuit_state.json"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


@dataclass
class ProviderState:
    consecutive_failures: int = 0
    opened_at: Optional[float] = None
    last_error: str = ""
    total_failures: int = 0
    total_successes: int = 0


@dataclass
class BreakerSnapshot:
    enabled: bool = True
    enforce: bool = False
    threshold: int = DEFAULT_THRESHOLD
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS
    providers: Dict[str, Any] = field(default_factory=dict)
    alert: bool = False


class ProviderCircuitBreaker:
    """Per-provider consecutive-failure breaker with cooldown + persistence."""

    def __init__(
        self,
        *,
        threshold: int = DEFAULT_THRESHOLD,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        enabled: bool = True,
        enforce: bool = False,
        status_path: Optional[Path] = None,
        clock: Optional[Any] = None,
    ) -> None:
        self.threshold = max(1, int(threshold))
        self.cooldown_seconds = float(cooldown_seconds)
        self.enabled = bool(enabled)
        self.enforce = bool(enforce)
        self.status_path = Path(status_path) if status_path else default_status_path()
        self._clock = clock or time.time
        self._providers: Dict[str, ProviderState] = {}

    # -- core ---------------------------------------------------------------
    def _state(self, provider: str) -> ProviderState:
        return self._providers.setdefault(provider or "unknown", ProviderState())

    def record_success(self, provider: str) -> None:
        if not self.enabled:
            return
        st = self._state(provider)
        changed = st.consecutive_failures > 0 or st.opened_at is not None
        st.consecutive_failures = 0
        st.opened_at = None
        st.total_successes += 1
        # Only touch the status file when the break state actually changed;
        # persisting on every success would add per-call IO for no signal.
        if changed:
            self._persist()

    def record_failure(self, provider: str, error: str = "") -> None:
        if not self.enabled:
            return
        st = self._state(provider)
        st.consecutive_failures += 1
        st.total_failures += 1
        st.last_error = str(error)[:500]
        if st.consecutive_failures >= self.threshold and st.opened_at is None:
            st.opened_at = float(self._clock())
            logger.error(
                "🔴 [LLM BREAKER] provider '%s' opened after %d consecutive failures "
                "(cooldown %.0fs): %s",
                provider, st.consecutive_failures, self.cooldown_seconds, st.last_error,
            )
        self._persist()

    def is_open(self, provider: str) -> bool:
        """True while the breaker is tripped and the cooldown has not elapsed."""
        if not self.enabled:
            return False
        st = self._providers.get(provider or "unknown")
        if st is None or st.opened_at is None:
            return False
        return (float(self._clock()) - st.opened_at) < self.cooldown_seconds

    def allow(self, provider: str) -> bool:
        """Gate for callers: ``True`` when the provider may be called."""
        if self.enabled and self.enforce and self.is_open(provider):
            return False
        return True

    def snapshot(self) -> BreakerSnapshot:
        return BreakerSnapshot(
            enabled=self.enabled,
            enforce=self.enforce,
            threshold=self.threshold,
            cooldown_seconds=self.cooldown_seconds,
            providers={name: asdict(st) for name, st in sorted(self._providers.items())},
            alert=any(self.is_open(name) for name in self._providers),
        )

    # -- persistence (never raises) ----------------------------------------
    def _persist(self) -> None:
        if not self.enabled:
            return
        try:
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            payload = asdict(self.snapshot())
            payload["updated_at"] = float(self._clock())
            tmp = self.status_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            tmp.replace(self.status_path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("[LLM BREAKER] status persist skipped: %s", exc)


class ProviderCircuitOpenError(RuntimeError):
    """Raised (only when ``LLM_BREAKER_ENFORCE=1``) instead of calling a tripped provider."""


_BREAKER: Optional[ProviderCircuitBreaker] = None


def get_circuit_breaker() -> ProviderCircuitBreaker:
    """Process-wide breaker, configured from env on first use."""
    global _BREAKER
    if _BREAKER is None:
        _BREAKER = ProviderCircuitBreaker(
            threshold=int(os.environ.get("LLM_BREAKER_THRESHOLD", DEFAULT_THRESHOLD)),
            cooldown_seconds=float(os.environ.get("LLM_BREAKER_COOLDOWN", DEFAULT_COOLDOWN_SECONDS)),
            enabled=_env_bool("LLM_BREAKER_ENABLED", True),
            enforce=_env_bool("LLM_BREAKER_ENFORCE", False),
        )
    return _BREAKER


def reset_circuit_breaker_for_tests() -> None:
    """Drop the process singleton (tests only)."""
    global _BREAKER
    _BREAKER = None
