"""Model role triplets (endpoint / api_key / model) with shared defaults.

Borrowing Auto Ontology's env pattern without NVIDIA NIM coupling:

    DEFAULT_MODELS_{ENDPOINT,API_KEY,MODEL}
    EXTRACT_{ENDPOINT,API_KEY,MODEL}     — heavy extraction / reasoning
    LIGHT_{ENDPOINT,API_KEY,MODEL}       — normalize / rewrite / reviewer light
    EMBED_{ENDPOINT,API_KEY,MODEL}
    RERANK_{ENDPOINT,API_KEY,MODEL}

Any unset field falls back to DEFAULT_MODELS_*, then to legacy single-key
envs already used in this repo (OPENAI_API_KEY, DIFY_*, EMBEDDING_*).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

__all__ = ["ModelTriplet", "resolve_model_role", "ROLE_PREFIXES"]

ROLE_PREFIXES = ("DEFAULT_MODELS", "EXTRACT", "LIGHT", "EMBED", "RERANK")


@dataclass(frozen=True)
class ModelTriplet:
    role: str
    endpoint: str
    api_key: str
    model: str

    def to_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "endpoint": self.endpoint,
            "api_key": "***" if self.api_key else "",
            "model": self.model,
        }


def _env(*names: str) -> str:
    for name in names:
        val = os.environ.get(name)
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


def resolve_model_role(role: str = "EXTRACT") -> ModelTriplet:
    """Resolve one role; unknown roles fall back to DEFAULT_MODELS / EXTRACT."""
    role_key = (role or "EXTRACT").strip().upper()
    if role_key not in ROLE_PREFIXES:
        role_key = "EXTRACT"

    def field(suffix: str, *legacy: str) -> str:
        return _env(
            f"{role_key}_{suffix}",
            f"DEFAULT_MODELS_{suffix}",
            *legacy,
        )

    endpoint = field(
        "ENDPOINT",
        "OPENAI_BASE_URL",
        "DIFY_API_URL",
        "EMBEDDING_ENDPOINT" if role_key == "EMBED" else "",
    )
    api_key = field(
        "API_KEY",
        "OPENAI_API_KEY",
        "DIFY_API_KEY",
        "NVIDIA_API_KEY",
        "EMBEDDING_API_KEY" if role_key == "EMBED" else "",
    )
    model = field(
        "MODEL",
        "OPENAI_MODEL",
        "DIFY_MODEL",
        "EMBEDDING_MODEL" if role_key == "EMBED" else "",
    )
    return ModelTriplet(role=role_key, endpoint=endpoint, api_key=api_key, model=model)
