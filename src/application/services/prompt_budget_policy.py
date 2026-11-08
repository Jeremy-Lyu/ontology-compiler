"""Prompt budget policy for LLM calls (C-2).

This module turns the existing YAML budget hints into a canonical runtime
policy object that can be injected into candidate generators, extraction flows,
and other LLM callers.

Design goals:
- One place for prompt / completion / retry defaults
- Deterministic profile selection from task/document metadata
- Safe fallback to environment defaults when config is unavailable
- No hidden hardcoded max_tokens assumptions in call sites
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except Exception:  # pragma: no cover - yaml is available in repo, but keep safe
    yaml = None


@dataclass(frozen=True)
class PromptBudget:
    """Resolved prompt budget for a single task/profile."""

    profile: str
    max_prompt_tokens: int
    max_completion_tokens: int
    total_budget: int
    temperature: float
    truncation_retry_max: int
    json_repair_max_retries: int
    source: str = "default"

    @property
    def max_tokens(self) -> int:
        """Compatibility alias for callers that only consume completion cap."""
        return self.max_completion_tokens


@dataclass
class PromptBudgetPolicy:
    """Canonical prompt budget policy for LLM-driven tasks.

    The policy can be built from the repository YAML config file and then used
    to resolve a task-specific budget with deterministic fallbacks.
    """

    defaults: Dict[str, Any] = field(default_factory=lambda: {
        "max_completion_tokens": 8000,
        "max_prompt_tokens": 12000,
        "total_budget": 20000,
        "temperature": 0.2,
        "truncation_retry_max": 2,
        "json_repair_max_retries": 3,
    })
    tasks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    document_type_profiles: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # C-2: chunking config from the YAML (max_chunk_size, chunk_text_token_estimate).
    chunking: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PromptBudgetPolicy":
        path = Path(path)
        if yaml is None or not path.exists():
            return cls()
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            return cls()
        if not isinstance(payload, dict):
            return cls()
        return cls(
            defaults=dict(payload.get("defaults") or {}),
            tasks=dict(payload.get("tasks") or {}),
            document_type_profiles=dict(payload.get("document_type_profiles") or {}),
            chunking=dict(payload.get("chunking") or {}),
        )

    def resolve_profile(
        self,
        *,
        task: str | None = None,
        document_type: str | None = None,
        domain: str | None = None,
        fallback: str = "extraction_general",
    ) -> str:
        """Pick the effective profile name.

        Order:
        1. explicit task
        2. document-type routing
        3. domain-aware fallback (if a profile exposes a matching alias)
        4. generic fallback
        """
        if task and task in self.tasks:
            return task

        if document_type:
            doc_type = document_type.lower().strip()
            profile_map = self.document_type_profiles.get(doc_type)
            if isinstance(profile_map, dict):
                if domain and domain.lower().strip() in profile_map:
                    return profile_map[domain.lower().strip()]
                if "default" in profile_map:
                    return profile_map["default"]

        if task and task in self.tasks:
            return task

        if fallback in self.tasks:
            return fallback

        if self.tasks:
            return next(iter(self.tasks.keys()))

        return fallback

    def resolve(
        self,
        *,
        task: str | None = None,
        document_type: str | None = None,
        domain: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> PromptBudget:
        """Resolve a task/document budget into a concrete PromptBudget."""
        profile = self.resolve_profile(task=task, document_type=document_type, domain=domain)
        profile_cfg = dict(self.tasks.get(profile) or {})
        defaults = self.defaults

        max_prompt_tokens = int(profile_cfg.get("max_prompt_tokens", defaults.get("max_prompt_tokens", 12000)))
        max_completion_tokens = int(profile_cfg.get("max_completion_tokens", defaults.get("max_completion_tokens", 8000)))
        total_budget = int(profile_cfg.get("total_budget", defaults.get("total_budget", max_prompt_tokens + max_completion_tokens)))
        temperature = float(profile_cfg.get("temperature", defaults.get("temperature", 0.2)))
        truncation_retry_max = int(profile_cfg.get("truncation_retry_max", defaults.get("truncation_retry_max", 2)))
        json_repair_max_retries = int(profile_cfg.get("json_repair_max_retries", defaults.get("json_repair_max_retries", 3)))

        if prompt_tokens is not None and prompt_tokens > 0:
            # Keep the prompt within budget by shrinking completion allowance.
            remaining = max(0, total_budget - prompt_tokens)
            max_completion_tokens = min(max_completion_tokens, remaining)

        if completion_tokens is not None and completion_tokens > 0:
            max_completion_tokens = min(max_completion_tokens, int(completion_tokens))

        return PromptBudget(
            profile=profile,
            max_prompt_tokens=max_prompt_tokens,
            max_completion_tokens=max_completion_tokens,
            total_budget=total_budget,
            temperature=temperature,
            truncation_retry_max=truncation_retry_max,
            json_repair_max_retries=json_repair_max_retries,
            source="yaml" if self.tasks else "default",
        )


_default_policy: PromptBudgetPolicy | None = None


def get_prompt_budget_policy(path: str | Path = "config/prompts/prompt_token_budget.yaml") -> PromptBudgetPolicy:
    global _default_policy
    if _default_policy is None:
        _default_policy = PromptBudgetPolicy.from_yaml(path)
    return _default_policy
