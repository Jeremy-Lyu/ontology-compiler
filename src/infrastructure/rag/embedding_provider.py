"""Embedding SDK providers (B-4, 2026-08-17).

Owns the ``openai`` / ``cohere`` / ``sentence_transformers`` SDK imports so the
application layer never imports them directly. Mirrors the lazy-import
semantics the old code used: SDKs are imported only when a client is actually
created, and each returns None / raises a clear error on missing dependency.
"""

from __future__ import annotations

from typing import Any, Optional


def create_openai_client(api_key: str, timeout: float) -> Any:
    """Create an OpenAI client (lazy import; raises RuntimeError if SDK missing)."""
    import openai

    return openai.OpenAI(api_key=api_key, timeout=timeout)


def create_cohere_client(api_key: str, timeout: float) -> Any:
    """Create a Cohere client (lazy import; raises RuntimeError if SDK missing)."""
    import cohere

    return cohere.Client(api_key, timeout=timeout)


def load_sentence_transformer(model_name: str, device: str = "cpu", cache_folder: Optional[str] = None) -> Any:
    """Load a SentenceTransformer model (lazy import; raises if SDK missing)."""
    from sentence_transformers import SentenceTransformer

    kwargs: dict[str, Any] = {"device": device}
    if cache_folder:
        kwargs["cache_folder"] = cache_folder
    return SentenceTransformer(model_name, **kwargs)


def is_openai_available() -> bool:
    try:
        import openai  # noqa: F401

        return True
    except ImportError:
        return False


def is_cohere_available() -> bool:
    try:
        import cohere  # noqa: F401

        return True
    except ImportError:
        return False


def is_sentence_transformers_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False