"""Resolved RAG run configuration and provider snapshot.

[P2 RAG 收口 2026-09-03]: Centralises the RAG-specific fields resolved from
``CompilerConfig`` so that every RAG entry point consumes the same config
snapshot rather than each calling ``get_config()`` independently.

Adapter boundary (clarified):
  ┌─────────────────────────────────────────────────────┐
  │ domain/          — data models (ChunkRecord, etc.)  │
  │ application/rag  — use cases (validator, answerer) │
  │   core/          — adapter wrappers & factory       │
  │   infrastructure/— concrete clients (dify, neo4j,   │
  │                     vector, embedding)              │
  │ llm_provider.py  — unified LLM provider (root)      │
  └─────────────────────────────────────────────────────┘

Every RAG run should call ``resolve_rag_config()`` at start and pass the
result down so that provider/backend selection is deterministic and auditable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ontology_compiler.config import get_config

logger = logging.getLogger(__name__)

# Known retriever adapter types (maps to RetrieverFactory registry).
_KNOWN_ADAPTERS = ("local", "neo4j", "dify", "asset_packet")

# Known LLM backends (maps to LLMBackend enum in llm_provider.py).
_KNOWN_LLM_BACKENDS = ("deepseek", "openai", "azure", "dify")


@dataclass(frozen=True)
class ResolvedRAGConfig:
    """Immutable snapshot of RAG-relevant config at run start.

    All fields are resolved from ``CompilerConfig`` (which itself reads env
    vars at instantiation).  This ensures every downstream consumer sees the
    same values within one run.
    """

    # --- LLM provider selection ---
    llm_provider: str
    llm_fallback_providers: str
    ontology_rag_llm_provider: str

    # --- RAG mode / behaviour ---
    ontology_rag_mode: str           # "a" (deterministic) | "b" / "llm_first"
    ontology_rag_strict_b: bool
    ontology_rag_require_l1: bool

    # --- Adapter inventory ---
    available_adapters: tuple = _KNOWN_ADAPTERS
    available_llm_backends: tuple = _KNOWN_LLM_BACKENDS

    # --- Resolved provider (effective, after fallback logic) ---
    @property
    def effective_llm_provider(self) -> str:
        """The LLM provider that will actually be used for this run."""
        return (self.ontology_rag_llm_provider or self.llm_provider or "dify").strip().lower()

    @property
    def is_llm_first(self) -> bool:
        """True when B-mode (LLM-first) is active."""
        return self.ontology_rag_mode.strip().lower() in {"b", "llm_first"}

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dict for snapshot persistence."""
        d = asdict(self)
        d["effective_llm_provider"] = self.effective_llm_provider
        d["is_llm_first"] = self.is_llm_first
        return d


def resolve_rag_config() -> ResolvedRAGConfig:
    """Create a ``ResolvedRAGConfig`` from the canonical ``CompilerConfig``.

    This is the single entry point for RAG config resolution.  All RAG use
    cases should call this once at run start and thread the result through.
    """
    cfg = get_config()
    return ResolvedRAGConfig(
        llm_provider=(cfg.llm_provider or "dify").strip().lower(),
        llm_fallback_providers=(cfg.llm_fallback_providers or "").strip(),
        ontology_rag_llm_provider=(cfg.ontology_rag_llm_provider or "").strip().lower(),
        ontology_rag_mode=(cfg.ontology_rag_mode or "a").strip().lower(),
        ontology_rag_strict_b=bool(cfg.ontology_rag_strict_b),
        ontology_rag_require_l1=bool(cfg.ontology_rag_require_l1),
    )


def emit_rag_snapshot(
    config: ResolvedRAGConfig,
    output_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build (and optionally persist) a resolved-config + provider snapshot.

    Args:
        config: The resolved RAG config for this run.
        output_dir: If given, write ``rag_config_snapshot.json`` there.
        run_id: Optional run identifier included in the snapshot.

    Returns:
        A JSON-compatible dict capturing the resolved config and provider
        inventory at the moment the RAG run started.
    """
    snapshot: Dict[str, Any] = {
        "run_id": run_id,
        "config": config.to_dict(),
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "rag_config_snapshot.json"
        try:
            path.write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("RAG config snapshot written to %s", path)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to write RAG config snapshot: %s", exc)

    return snapshot
