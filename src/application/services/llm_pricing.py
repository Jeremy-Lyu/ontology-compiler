"""LLM pricing registry + per-call cost estimation (C-1).

Core rule from the C-1 batch:
  unknown prices must NOT silently become 0.0 USD.

This module owns the *model-level* price table (input/output per 1M tokens)
and the fallback chain (model → provider family → ``None``). Any caller that
needs a USD cost goes through :func:`estimate_cost` which returns
``(usd, price_known)``; ``price_known=False`` tells the monitor/trace to flag
the call as unknown-priced instead of recording a silent zero.

Prices are intentional *estimates*, used for budget/cost tracking rather than
billing. They are frozen in code so the budget gate is deterministic across
environments and does not depend on network lookups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

# USD per 1M tokens per model family (input, output).
# Values are public list prices kept intentionally conservative (overestimates
# are safer for a hard-budget gate than underestimates).
MODEL_PRICES_USD_PER_1M: dict[str, Tuple[float, float]] = {
    # DeepSeek
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    # OpenAI (fine-tuned defaults)
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-3.5-turbo": (0.50, 1.50),
    # Anthropic
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
    # Local / free backends
    "local": (0.0, 0.0),
}

# Provider-family fallback used when a specific model is unknown.
# Chosen as the *most conservative* member of each family (higher price) so
# an unknown model never silently underprices an actual call.
PROVIDER_FALLBACK_USD_PER_1M: dict[str, Tuple[float, float]] = {
    "deepseek": (0.55, 2.19),
    "dify": (2.50, 10.00),      # Dify gates an upstream model; assume worst case
    "openai": (2.50, 10.00),
    "azure": (2.50, 10.00),
    "anthropic": (3.00, 15.00),
    "local": (0.0, 0.0),
}


@dataclass(frozen=True)
class CostEstimate:
    """Cost estimate for a single LLM call.

    Attributes:
        usd: Estimated cost in US dollars (0.0 when unknown).
        price_known: False when neither the model nor the provider family has
            a registered price — the caller must flag this instead of treating
            ``usd == 0.0`` as a real zero-cost call.
    """

    usd: float
    price_known: bool


def resolve_price(provider: str, model: Optional[str]) -> Optional[Tuple[float, float]]:
    """Return (input, output) USD-per-1M price for a provider/model pair.

    Lookup order: exact model → provider family fallback → ``None``.
    """
    if model:
        normalized = model.strip().lower()
        if normalized in MODEL_PRICES_USD_PER_1M:
            return MODEL_PRICES_USD_PER_1M[normalized]
    family = (provider or "").strip().lower()
    return PROVIDER_FALLBACK_USD_PER_1M.get(family)


def estimate_cost(
    provider: str,
    model: Optional[str],
    prompt_tokens: int,
    completion_tokens: int,
    *,
    cache_create_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> CostEstimate:
    """Estimate the USD cost of a token-usage tuple.

    ``cache_read_tokens`` is priced at 10% of the input rate; ``cache_create``
    is priced at the full input rate (provider convention). A missing price
    yields ``CostEstimate(0.0, price_known=False)`` — callers decide how to
    surface that (monitor flags it; a hard-budget gate must not mis-count it
    as free).
    """
    price = resolve_price(provider, model)
    if price is None:
        return CostEstimate(usd=0.0, price_known=False)
    if prompt_tokens <= 0 and completion_tokens <= 0:
        return CostEstimate(usd=0.0, price_known=True)
    input_rate, output_rate = price
    input_tokens = max(0, prompt_tokens) + max(0, cache_create_tokens)
    read_tokens = max(0, cache_read_tokens)
    usd = (
        input_tokens * input_rate / 1_000_000
        + read_tokens * input_rate * 0.1 / 1_000_000
        + max(0, completion_tokens) * output_rate / 1_000_000
    )
    return CostEstimate(usd=round(usd, 6), price_known=True)