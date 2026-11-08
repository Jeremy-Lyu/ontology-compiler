"""Lightweight unified LLM gateway facade.

This module provides a small access layer over LLMBootstrap so callers in
application, agent, and infrastructure code can share a consistent retrieval
path without importing bootstrap internals everywhere.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from src.application.bootstrap.llm_bootstrap import (
    LLMBootstrap,
    LLMProviderType,
    initialize_llm_bootstrap,
)
from src.application.services.llm_budget import LLMBudgetLedger, LLMBudgetPolicy
from src.application.services.llm_monitor import get_llm_monitor
from src.application.services.model_router import (
    ModelRoutingPolicy,
    RouteDecision,
    TaskKind,
    get_model_routing_policy,
)
from src.domain.models.core import TraceContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMCallOutcome:
    """Normalized outcome of a gateway LLM execution (C-1).

    The gateway is the *only* place that translates a raw provider client
    response into a consistent shape so callers never need to reach into
    provider-specific response objects.

    C-5 additions:
        routed_model: True when the call's provider/model was selected by
            :meth:`LLMGateway.execute_routed` rather than supplied explicitly.
        route_decision: The :class:`RouteDecision` (or ``None``) attached by
            routing. Carries tier + signals for trace / telemetry.
    """

    text: str
    provider: str
    model: str
    call_id: str
    prompt_hash: str
    success: bool
    cost_usd: float
    price_known: bool
    duration_ms: int
    error_type: str = ""
    routed_model: bool = False
    route_decision: Optional[RouteDecision] = None


class LLMGateway:
    """Facade for accessing configured LLM clients and providers."""

    def __init__(self, bootstrap: LLMBootstrap | None = None) -> None:
        self._bootstrap = bootstrap or initialize_llm_bootstrap()
        self._budget: LLMBudgetLedger = LLMBudgetLedger()

    @property
    def bootstrap(self) -> LLMBootstrap:
        return self._bootstrap

    def get_client(self, provider: str | LLMProviderType) -> Any:
        if isinstance(provider, str):
            provider = LLMProviderType(provider.lower())
        return self._bootstrap.get_client(provider)

    def get_monitor(self):
        return get_llm_monitor()

    # ------------------------------------------------------------------
    # C-1: budget policy + execution entry point
    # ------------------------------------------------------------------

    def set_budget_policy(
        self,
        *,
        max_cost_usd: float = 0.0,
        max_calls: int = 0,
        max_total_tokens: int = 0,
        on_exceed: Any = None,
    ) -> None:
        """Configure the run-scoped hard budget (0 = limit inactive).

        Idempotent; call at the start of a run/stage/session. The previous
        consumption ledger is replaced so a new run starts clean.
        """
        policy = LLMBudgetPolicy(
            max_cost_usd=max_cost_usd,
            max_calls=max_calls,
            max_total_tokens=max_total_tokens,
            on_exceed=on_exceed,
        )
        self._budget = LLMBudgetLedger(policy=policy)

    @property
    def budget(self) -> LLMBudgetLedger:
        """Access the active budget ledger (state + policy)."""
        return self._budget

    def execute(
        self,
        prompt: str,
        *,
        provider: str | LLMProviderType | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stage: str | None = None,
        call_id: str = "",
    ) -> LLMCallOutcome:
        """Synchronously execute an LLM call through the configured client.

        Single choke point (C-1): checks the hard budget *before* the call,
        invokes the monitored client, then attaches the cost estimate. When
        the budget is exhausted, raises ``LLMBudgetExceededError`` and the
        provider is never contacted.

        Args:
            prompt: The prompt text.
            provider: Provider key; defaults to the bootstrap primary.
            model: Optional model label used only for pricing/telemetry (the
                provider client owns actual model selection).
            temperature / max_tokens: Optional per-call overrides.
            stage: Optional stage label recorded in the monitor.
            call_id: Optional caller-provided call identifier.

        Returns:
            A normalized :class:`LLMCallOutcome`.

        Raises:
            LLMBudgetExceededError: When a configured budget cap is reached.
        """
        provider_type = self._resolve_provider(provider)
        return self._execute(
            prompt,
            provider_type=provider_type,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stage=stage,
            call_id=call_id,
            call_fn=lambda **kw: self.get_client(provider_type).call(**kw),
        )

    # ------------------------------------------------------------------
    # C-5: cost-aware model routing
    # ------------------------------------------------------------------

    def route(
        self,
        prompt: str,
        *,
        task_kind: TaskKind | str | None = None,
        entity_count_hint: int = 0,
        confidence_hint: float | None = None,
        has_conflict: bool | None = None,
        policy: ModelRoutingPolicy | None = None,
    ) -> RouteDecision:
        """Produce a :class:`RouteDecision` without executing the call.

        Exposed so callers (e.g. pipeline stages) can log / audit the routing
        choice before deciding whether to invoke :meth:`execute_routed`.
        """
        if isinstance(task_kind, str):
            task_kind = TaskKind(task_kind)
        active_policy = policy or get_model_routing_policy()
        return active_policy.route(
            prompt,
            task_kind=task_kind,
            entity_count_hint=entity_count_hint,
            confidence_hint=confidence_hint,
            has_conflict=has_conflict,
        )

    def execute_routed(
        self,
        prompt: str,
        *,
        task_kind: TaskKind | str | None = None,
        entity_count_hint: int = 0,
        confidence_hint: float | None = None,
        has_conflict: bool | None = None,
        stage: str | None = None,
        call_id: str = "",
        policy: ModelRoutingPolicy | None = None,
        temperature_override: float | None = None,
        max_tokens_override: int | None = None,
    ) -> LLMCallOutcome:
        """Execute an LLM call with cost-aware model routing (C-5).

        Routing picks (provider, model, max_tokens, temperature) from the
        :class:`ModelRoutingPolicy`; explicit overrides win when supplied.
        When routing selects a provider that has no registered client (e.g.
        ``deepseek-reasoner`` while only ``dify`` is configured), the gateway
        falls back to the bootstrap primary provider but keeps the routed
        model label so pricing/telemetry still reflect intent.

        Returns a :class:`LLMCallOutcome` whose ``routed_model=True`` and
        ``route_decision`` carries the routing signals for trace.
        """
        decision = self.route(
            prompt,
            task_kind=task_kind,
            entity_count_hint=entity_count_hint,
            confidence_hint=confidence_hint,
            has_conflict=has_conflict,
            policy=policy,
        )
        tier = decision.model_tier
        provider = tier.provider
        model = tier.model
        temperature = temperature_override if temperature_override is not None else tier.temperature
        max_tokens = max_tokens_override if max_tokens_override is not None else tier.max_tokens

        # C-5 后续增强 (2026-09-01): record the routing decision so the monitor
        # summary (and thus the manifest / regression baseline) carries
        # ``route_decisions`` telemetry. Best-effort: never break the call.
        try:
            get_llm_monitor().record_route(
                task_kind=decision.task_kind.value,
                tier=decision.tier.value,
                reason=decision.reason,
            )
        except Exception:  # pragma: no cover - defensive
            pass

        try:
            outcome = self.execute(
                prompt,
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stage=stage,
                call_id=call_id,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(
                "execute_routed: routed provider=%s failed (%s); "
                "falling back to bootstrap primary",
                provider,
                exc,
            )
            outcome = self.execute(
                prompt,
                provider=None,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stage=stage,
                call_id=call_id,
            )

        # Attach routing metadata (outcome is frozen -> rebuild).
        return LLMCallOutcome(
            text=outcome.text,
            provider=outcome.provider,
            model=outcome.model,
            call_id=outcome.call_id,
            prompt_hash=outcome.prompt_hash,
            success=outcome.success,
            cost_usd=outcome.cost_usd,
            price_known=outcome.price_known,
            duration_ms=outcome.duration_ms,
            error_type=outcome.error_type,
            routed_model=True,
            route_decision=decision,
        )

    async def execute_async(
        self,
        prompt: str,
        *,
        provider: str | LLMProviderType | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stage: str | None = None,
        call_id: str = "",
    ) -> LLMCallOutcome:
        """Async variant of :meth:`execute` (see its docstring)."""
        provider_type = self._resolve_provider(provider)

        async def call(client: Any, **kw: Any) -> Any:
            result = client.call(**kw)
            if hasattr(result, "__await__"):
                return await result
            return result

        return await self._execute_async(
            prompt,
            provider_type=provider_type,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stage=stage,
            call_id=call_id,
            call_fn=lambda **kw: call(self.get_client(provider_type), **kw),
        )

    def _execute(
        self,
        prompt: str,
        *,
        provider_type: LLMProviderType,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
        stage: str | None,
        call_id: str,
        call_fn: Any,
    ) -> LLMCallOutcome:
        provider_key = provider_type.value

        # Hard gate: refuse to spend more after the budget is exhausted.
        self._budget.check()

        from src.application.services.llm_trace import hash_text

        prompt_hash = hash_text(prompt or "")
        call_id = call_id or prompt_hash

        monitor = get_llm_monitor()
        prev_stage = getattr(monitor, "_current_stage", None)
        records_before = len(getattr(monitor, "_call_records", []))
        if stage:
            monitor.set_stage(stage)

        import time

        start = time.time()
        try:
            result = call_fn(prompt=prompt, temperature=temperature, max_tokens=max_tokens)

            import inspect

            if inspect.iscoroutine(result) or inspect.isawaitable(result):
                raise TypeError(
                    "gateway.execute() is synchronous; use execute_async() for async "
                    "clients (got an awaitable from client.call)"
                )
            duration_ms = int((time.time() - start) * 1000)
            text = str(result or "")

            record = monitor.last_record() if len(monitor._call_records) > records_before else None
            prompt_tokens = record.prompt_tokens if record else 0
            completion_tokens = record.completion_tokens if record else 0
            price_known = record.price_known if record else False

            cost_usd = self._cost_for(
                record, provider_key, model, prompt_tokens, completion_tokens
            )
            self._budget.record(
                cost_usd=cost_usd,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                provider=provider_key,
                model=model or "",
                call_id=call_id,
            )
            return LLMCallOutcome(
                text=text,
                provider=provider_key,
                model=model or "",
                call_id=call_id,
                prompt_hash=prompt_hash,
                success=True,
                cost_usd=cost_usd,
                price_known=price_known,
                duration_ms=duration_ms,
            )
        except Exception as exc:  # noqa: BLE001 - gateway reports all failures
            duration_ms = int((time.time() - start) * 1000)
            logger.warning("LLM gateway %s call failed: %s", provider_key, exc)
            return LLMCallOutcome(
                text="",
                provider=provider_key,
                model=model or "",
                call_id=call_id,
                prompt_hash=prompt_hash,
                success=False,
                cost_usd=0.0,
                price_known=False,
                duration_ms=duration_ms,
                error_type=type(exc).__name__,
            )
        finally:
            if stage:
                monitor.set_stage(prev_stage)

    async def _execute_async(
        self,
        prompt: str,
        *,
        provider_type: LLMProviderType,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
        stage: str | None,
        call_id: str,
        call_fn: Any,
    ) -> LLMCallOutcome:
        provider_key = provider_type.value

        # Hard gate: refuse to spend more after the budget is exhausted.
        self._budget.check()

        from src.application.services.llm_trace import hash_text

        prompt_hash = hash_text(prompt or "")
        call_id = call_id or prompt_hash

        monitor = get_llm_monitor()
        prev_stage = getattr(monitor, "_current_stage", None)
        records_before = len(getattr(monitor, "_call_records", []))
        if stage:
            monitor.set_stage(stage)

        import time

        start = time.time()
        try:
            result = await call_fn(prompt=prompt, temperature=temperature, max_tokens=max_tokens)
            duration_ms = int((time.time() - start) * 1000)
            text = str(result or "")

            record = monitor.last_record() if len(monitor._call_records) > records_before else None
            prompt_tokens = record.prompt_tokens if record else 0
            completion_tokens = record.completion_tokens if record else 0
            price_known = record.price_known if record else False

            cost_usd = self._cost_for(
                record, provider_key, model, prompt_tokens, completion_tokens
            )
            self._budget.record(
                cost_usd=cost_usd,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                provider=provider_key,
                model=model or "",
                call_id=call_id,
            )
            return LLMCallOutcome(
                text=text,
                provider=provider_key,
                model=model or "",
                call_id=call_id,
                prompt_hash=prompt_hash,
                success=True,
                cost_usd=cost_usd,
                price_known=price_known,
                duration_ms=duration_ms,
            )
        except Exception as exc:  # noqa: BLE001 - gateway reports all failures
            duration_ms = int((time.time() - start) * 1000)
            logger.warning("LLM gateway %s call failed: %s", provider_key, exc)
            return LLMCallOutcome(
                text="",
                provider=provider_key,
                model=model or "",
                call_id=call_id,
                prompt_hash=prompt_hash,
                success=False,
                cost_usd=0.0,
                price_known=False,
                duration_ms=duration_ms,
                error_type=type(exc).__name__,
            )
        finally:
            if stage:
                monitor.set_stage(prev_stage)

    def _resolve_provider(self, provider: str | LLMProviderType | None) -> LLMProviderType:
        if isinstance(provider, LLMProviderType):
            return provider
        if provider:
            return LLMProviderType(provider.lower())
        return self._bootstrap.config.primary_provider

    @staticmethod
    def _cost_for(record: Any, provider_key: str, model: str | None, prompt_tokens: int, completion_tokens: int) -> float:
        """Return a cost estimate for a finished call (pricing-registry aware).

        Prefers the monitor record's estimate (already pricing-aware) and falls
        back to a direct registry lookup for raw synchronous clients that do
        not emit usage records.
        """
        if record is not None:
            return float(record.cost_estimate or 0.0)
        try:
            from src.application.services.llm_pricing import estimate_cost

            return estimate_cost(provider_key, model, prompt_tokens, completion_tokens).usd
        except Exception:
            return 0.0

    def build_trace_context(
        self,
        *,
        run_id: str = "",
        session_id: str = "",
        stage_id: str = "",
        task_id: str = "",
        artifact_id: str = "",
        document_id: str = "",
        chunk_id: str = "",
        prompt_id: str = "",
        skill_id: str = "",
        model_call_id: str = "",
        parent_trace_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TraceContext:
        return TraceContext(
            run_id=run_id,
            session_id=session_id,
            stage_id=stage_id,
            task_id=task_id,
            artifact_id=artifact_id,
            document_id=document_id,
            chunk_id=chunk_id,
            prompt_id=prompt_id,
            skill_id=skill_id,
            model_call_id=model_call_id,
            parent_trace_id=parent_trace_id,
            metadata=metadata or {},
        )

    def get_primary_client(self) -> Any:
        return self._bootstrap.get_client(self._bootstrap.config.primary_provider)

    def get_unified_provider(self) -> Any:
        return self._bootstrap.get_unified_provider()

    def get_rigor_client(self) -> Any:
        return self._bootstrap.get_rigor_client()


_gateway: LLMGateway | None = None


def get_llm_gateway() -> LLMGateway:
    """Return the process-wide gateway instance."""
    global _gateway
    if _gateway is None:
        _gateway = LLMGateway()
    return _gateway
