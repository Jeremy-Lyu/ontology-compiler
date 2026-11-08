"""LLM hard budget policy (C-1).

A run-scoped, deterministic budget that *blocks* further LLM calls once any
limit is reached. Unlike the monitor (which records after the fact), this is a
pre-call gate: callers invoke :meth:`LLMBudgetPolicy.check` before sending a
request and receive :class:`LLMBudgetExceededError` when the budget is spent.

The default policy is unlimited-safe (``max_cost_usd=0`` means "no cost cap",
``max_calls=0`` means "no call cap", ``max_total_tokens=0`` means "no token
cap"), so enabling the gateway in an unconfigured environment never changes
behavior — the budget becomes *active* only when a caller configures limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


class LLMBudgetExceededError(RuntimeError):
    """Raised when a configured LLM budget limit is reached.

    Callers should treat this as a *block* signal: degrade to local/rule-based
    handling or fail the run visibly — never silently retry the same expensive
    call (that would defeat the budget).
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class LLMBudgetState:
    """Mutable per-run budget consumption."""

    total_cost_usd: float = 0.0
    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, *, cost_usd: float, prompt_tokens: int, completion_tokens: int) -> None:
        self.total_cost_usd += max(0.0, cost_usd)
        self.total_calls += 1
        self.total_prompt_tokens += max(0, prompt_tokens)
        self.total_completion_tokens += max(0, completion_tokens)
        self.total_tokens += max(0, prompt_tokens) + max(0, completion_tokens)


@dataclass
class LLMBudgetPolicy:
    """Pre-call budget limit set.

    All limits default to 0 (inactive). A configured limit *blocks* the next
    call once the consumed budget meets/exceeds it.

    Attributes:
        max_cost_usd: USD cost cap for the run. 0 = no cap.
        max_calls: Call-count cap for the run. 0 = no cap.
        max_total_tokens: Total token cap (prompt+completion) for the run.
            0 = no cap.
        on_exceed: Optional hook ``(reason: str) -> None`` invoked when a
            check fails (logging / metrics / tracing), before the exception
            is raised.
    """

    max_cost_usd: float = 0.0
    max_calls: int = 0
    max_total_tokens: int = 0
    on_exceed: Optional[object] = None

    def check(self, state: LLMBudgetState) -> None:
        """Raise :class:`LLMBudgetExceededError` if any cap is reached."""
        reason: Optional[str] = None
        if self.max_calls > 0 and state.total_calls >= self.max_calls:
            reason = f"LLM call budget exceeded: {state.total_calls} >= {self.max_calls} calls"
        elif self.max_cost_usd > 0 and state.total_cost_usd >= self.max_cost_usd:
            reason = (
                f"LLM cost budget exceeded: ${state.total_cost_usd:.4f} >= ${self.max_cost_usd:.4f}"
            )
        elif self.max_total_tokens > 0 and state.total_tokens >= self.max_total_tokens:
            reason = f"LLM token budget exceeded: {state.total_tokens} >= {self.max_total_tokens} tokens"
        if reason is not None:
            if self.on_exceed is not None and callable(self.on_exceed):
                self.on_exceed(reason)
            raise LLMBudgetExceededError(reason)

    def describe(self) -> dict[str, object]:
        """Machine-readable description for manifests/traces."""
        return {
            "max_cost_usd": self.max_cost_usd,
            "max_calls": self.max_calls,
            "max_total_tokens": self.max_total_tokens,
            "active": self.is_active,
        }

    @property
    def is_active(self) -> bool:
        return self.max_cost_usd > 0 or self.max_calls > 0 or self.max_total_tokens > 0


@dataclass
class LLMBudgetLedger:
    """Reference implementation of a budgeted LLM executor (pure logic).

    Combines policy + state and offers the two methods a gateway needs:
    ``check()`` before a call and ``record()`` after it. Intentionally free of
    I/O so it is trivially testable and reusable across async/sync callers.
    """

    policy: LLMBudgetPolicy = field(default_factory=LLMBudgetPolicy)
    state: LLMBudgetState = field(default_factory=LLMBudgetState)
    history: List[dict[str, object]] = field(default_factory=list)

    def check(self) -> None:
        self.policy.check(self.state)

    def record(
        self,
        *,
        cost_usd: float,
        prompt_tokens: int,
        completion_tokens: int,
        provider: str = "",
        model: str = "",
        call_id: str = "",
    ) -> None:
        self.state.add(
            cost_usd=cost_usd,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        if len(self.history) < 10_000:
            self.history.append(
                {
                    "provider": provider,
                    "model": model,
                    "call_id": call_id,
                    "cost_usd": cost_usd,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }
            )

    def reset(self) -> None:
        self.state = LLMBudgetState()
        self.history.clear()

    def snapshot(self) -> dict[str, object]:
        return {
            "state": {
                "total_cost_usd": self.state.total_cost_usd,
                "total_calls": self.state.total_calls,
                "total_tokens": self.state.total_tokens,
                "total_prompt_tokens": self.state.total_prompt_tokens,
                "total_completion_tokens": self.state.total_completion_tokens,
            },
            "policy": self.policy.describe(),
            "call_count_recorded": len(self.history),
        }