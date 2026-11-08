"""LLM governance arming extracted from MainPipeline (P0-3 batch 8).

C-1 last-resort safety net: arm the LLMGateway budget + degraded mode from
context metadata or environment, using the established
``pipeline``-first-parameter pattern (same as trace_provenance, stage_plumbing,
regression_metadata, result_helpers).

Contents:
- :func:`arm_governance_from_context` — from
  ``MainPipeline._arm_governance_from_context``

Behavior: defensive-only, never lets budget arming kill the run. The gateway
import stays lazy (function-level) so sandbox/test environments without the
bootstrap module keep working.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hints only, avoids circular import
    from .main_pipeline import MainPipeline

logger = logging.getLogger(__name__)


def arm_governance_from_context(pipeline: "MainPipeline") -> None:
    """Arm the LLMGateway budget policy as a last-resort safety net.

    Priority order (highest first):
    1. Explicit context.metadata.governance (from CLI / run_batch args)
    2. Environment variables (LLM_MAX_COST_USD / LLM_MAX_CALLS /
       LLM_MAX_TOTAL_TOKENS / LLM_OFFLINE_MODE)
    3. Do nothing (gateway stays with caller-supplied or unlimited policy)

    Why overwrite: this runs *after* the callers above, so it only kicks in
    when the caller forgot to arm — the policy is the same or *tighter*
    (never looser). The gateway.set_budget_policy replaces the previous
    policy object entirely; if both caller and here set identical values,
    behavior is unchanged.
    """
    try:
        from src.application.bootstrap.llm_gateway import get_llm_gateway
    except Exception:  # pragma: no cover - gateway unavailable in sandbox
        return

    env = os.environ
    gov = (
        pipeline.context.metadata.get("governance", {})
        if isinstance(pipeline.context.metadata, dict)
        else {}
    )

    # --- resolve values (context > env > 0/unlimited) ---
    max_cost_usd = float(
        gov.get("max_cost_usd")
        if gov.get("max_cost_usd")
        else env.get("LLM_MAX_COST_USD", "0") or 0.0
    )
    max_calls = int(
        gov.get("max_calls")
        if gov.get("max_calls")
        else env.get("LLM_MAX_CALLS", "0") or 0
    )
    max_total_tokens = int(
        gov.get("max_total_tokens")
        if gov.get("max_total_tokens")
        else env.get("LLM_MAX_TOTAL_TOKENS", "0") or 0
    )

    budget_active = max_cost_usd > 0 or max_calls > 0 or max_total_tokens > 0
    if not budget_active:
        return  # nothing to arm — caller policy or unlimited stays in effect

    try:
        gateway = get_llm_gateway()
        gateway.set_budget_policy(
            max_cost_usd=max_cost_usd,
            max_calls=max_calls,
            max_total_tokens=max_total_tokens,
        )
        logger.info(
            "[C-1][BudgetGate] Armed (pipeline safety net): "
            "max_cost=$%.4f, max_calls=%d, max_tokens=%d",
            max_cost_usd, max_calls, max_total_tokens,
        )
    except Exception as exc:  # pragma: no cover - defensive — never let budget arming kill the run
        logger.warning("[C-1][BudgetGate] Failed to arm (non-fatal): %s", exc)
