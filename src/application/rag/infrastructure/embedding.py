"""Embedding utilities.

This module provides a canonical embedder configuration plus a backward-
compatible legacy config shim. The goal is to keep provider selection explicit
and avoid runtime alias rebinding between old and new config types.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EmbedderConfig:
    """Canonical embedder configuration."""

    provider: str = "local"  # local, openai, cohere, sentence_transformers
    api_key: Optional[str] = None
    model: Optional[str] = None
    timeout: float = 60.0
    batch_size: int = 100
    dimension: int = 384
    device: str = "cpu"
    cache_dir: Optional[str] = None
    extra_headers: Optional[Dict[str, str]] = None


@dataclass
class LegacyEmbeddingConfig:
    """Legacy config preserved for backward compatibility.

    Older call sites may still expect fields like `use_openai` or the old
    DeepSeek-oriented defaults. We keep this type explicitly distinct and
    convert it into `EmbedderConfig` via `_coerce_embedding_config`.
    """

    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "local"
    timeout: float = 60.0
    batch_size: int = 100
    proxy: Optional[str] = None
    no_proxy: bool = True
    use_openai: bool = False


class LocalEmbedder:
    """Local TF-IDF style embedder with no external API dependency."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self._is_fitted = False

    def _tokenize(self, text: str) -> List[str]:
        english = re.findall(r"[a-zA-Z]+", (text or "").lower())
        tokens: List[str] = list(english)
        # Chinese: adjacent bigrams instead of single characters. Single
        # characters dominated the top-k vocabulary without carrying word
        # meaning, so every Chinese query embedded to a near-identical vector
        # (fixed hit set regardless of query). Bigrams preserve word-level
        # semantics; a lone character segment still yields itself.
        for segment in re.findall(r"[\u4e00-\u9fff]+", text or ""):
            if len(segment) == 1:
                tokens.append(segment)
            else:
                tokens.extend(
                    dict.fromkeys(segment[i : i + 2] for i in range(len(segment) - 1))
                )
        return tokens

    def fit(self, texts: List[str]):
        doc_freq = Counter()
        all_tokens: List[str] = []

        for text in texts:
            tokens = set(self._tokenize(text))
            doc_freq.update(tokens)
            all_tokens.extend(self._tokenize(text))

        most_common = Counter(all_tokens).most_common(self.dimension)
        self.vocab = {word: idx for idx, (word, _) in enumerate(most_common)}

        n_docs = len(texts)
        for word, freq in doc_freq.items():
            if word in self.vocab:
                self.idf[word] = np.log((n_docs + 1) / (freq + 1)) + 1

        self._is_fitted = True
        return self

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        if not self._is_fitted:
            self.fit(texts)

        embeddings: List[np.ndarray] = []
        for text in texts:
            tokens = self._tokenize(text)
            token_counts = Counter(tokens)

            vec = np.zeros(len(self.vocab), dtype=np.float32)
            for token, count in token_counts.items():
                if token in self.vocab:
                    idx = self.vocab[token]
                    tf = count / len(tokens) if tokens else 0.0
                    vec[idx] = tf * self.idf.get(token, 1.0)

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)

        return np.array(embeddings, dtype=np.float32)

    def cosine_similarity(self, query_vector: np.ndarray, document_vectors: np.ndarray) -> np.ndarray:
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        return np.dot(document_vectors, query_vector.T).flatten()

    def save(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vocab": self.vocab,
            "idf": self.idf,
            "dimension": self.dimension,
            "is_fitted": self._is_fitted,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path: Path | str) -> "LocalEmbedder":
        path = Path(path)
        if not path.exists():
            return self
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = data.get("vocab", {})
        self.idf = {k: float(v) for k, v in data.get("idf", {}).items()}
        self.dimension = int(data.get("dimension", self.dimension))
        self._is_fitted = bool(data.get("is_fitted", False))
        return self


class OpenAIEmbedder:
    """OpenAI embedding client with local fallback."""

    DEFAULT_MODEL = "text-embedding-3-small"

    def __init__(self, config: Optional[EmbedderConfig] = None):
        self.config = config or EmbedderConfig(provider="openai", model=self.DEFAULT_MODEL)
        self._dimension: Optional[int] = None
        self._client = None

    def _get_client(self):
        if self._client is None:
            # B-4 (2026-08-17): SDK import via infrastructure provider.
            from src.infrastructure.rag.embedding_provider import create_openai_client

            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key.")

            try:
                self._client = create_openai_client(api_key, timeout=self.config.timeout)
            except ImportError:
                raise RuntimeError("OpenAI SDK not installed. Run: pip install openai") from None
        return self._client

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.get_dimension()), dtype=np.float32)

        client = self._get_client()
        model = self.config.model or self.DEFAULT_MODEL
        batch_size = max(1, int(self.config.batch_size))

        try:
            all_embeddings: List[List[float]] = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch = [t[:8000] if len(t) > 8000 else t for t in batch]
                response = client.embeddings.create(model=model, input=batch)
                all_embeddings.extend(item.embedding for item in response.data)

            return np.array(all_embeddings, dtype=np.float32)
        except Exception as exc:
            warnings.warn(f"OpenAI Embedding API failed: {exc}. Falling back to local embedder.")
            local = LocalEmbedder(dimension=self.config.dimension)
            return local.embed(texts)

    def get_dimension(self) -> int:
        if self._dimension is None:
            probe = self.embed(["probe"])
            self._dimension = int(probe.shape[1]) if probe.ndim == 2 and probe.shape[1] else self.config.dimension
        return self._dimension


class EmbedderBase(ABC):
    """Abstract base class for all embedders."""

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_dimension(self) -> int:
        raise NotImplementedError

    def cosine_similarity(self, query_vector: np.ndarray, document_vectors: np.ndarray) -> np.ndarray:
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        return np.dot(document_vectors, query_vector.T).flatten()

    def normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms


class CohereEmbedder(EmbedderBase):
    """Cohere Embed API client."""

    DEFAULT_MODEL = "embed-multilingual-v3"
    DEFAULT_DIMENSION = 1024

    def __init__(self, config: Optional[EmbedderConfig] = None):
        base_config = config or EmbedderConfig(provider="cohere")
        self.config = EmbedderConfig(**base_config.__dict__) if not isinstance(base_config, EmbedderConfig) else base_config
        self.model = self.config.model or self.DEFAULT_MODEL
        self._client = None
        self._dimension: Optional[int] = None

    def _get_client(self):
        if self._client is None:
            # B-4 (2026-08-17): SDK import via infrastructure provider.
            from src.infrastructure.rag.embedding_provider import create_cohere_client

            api_key = self.config.api_key or os.getenv("COHERE_API_KEY")
            if not api_key:
                raise RuntimeError("Cohere API key required. Set COHERE_API_KEY env var or pass api_key.")
            try:
                self._client = create_cohere_client(api_key, timeout=self.config.timeout)
            except ImportError:
                raise RuntimeError("Cohere SDK not installed. Run: pip install cohere") from None
        return self._client

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.get_dimension()), dtype=np.float32)

        client = self._get_client()
        all_embeddings: List[List[float]] = []
        batch_size = max(1, int(self.config.batch_size))

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch = [t[:2000] if len(t) > 2000 else t for t in batch]
            response = client.embed(
                texts=batch,
                model=self.model,
                input_type="search_document",
                embedding_types=["float"],
            )
            all_embeddings.extend(response.embeddings.float)

        vectors = np.array(all_embeddings, dtype=np.float32)
        return self.normalize(vectors)

    def get_dimension(self) -> int:
        if self._dimension is None:
            dummy = self.embed(["probe"])
            self._dimension = int(dummy.shape[1]) if dummy.ndim == 2 and dummy.shape[1] else self.DEFAULT_DIMENSION
        return self._dimension


class SentenceTransformersEmbedder(EmbedderBase):
    """Local SentenceTransformers embedder."""

    DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, config: Optional[EmbedderConfig] = None):
        base_config = config or EmbedderConfig(provider="sentence_transformers")
        self.config = EmbedderConfig(**base_config.__dict__) if not isinstance(base_config, EmbedderConfig) else base_config
        self.model_name = self.config.model or self.DEFAULT_MODEL
        self._model = None
        self._dimension: Optional[int] = None

    def _get_model(self):
        if self._model is None:
            # B-4 (2026-08-17): SDK import via infrastructure provider.
            from src.infrastructure.rag.embedding_provider import load_sentence_transformer

            try:
                self._model = load_sentence_transformer(
                    self.model_name,
                    device=self.config.device,
                    cache_folder=self.config.cache_dir,
                )
            except ImportError:
                raise RuntimeError("sentence-transformers not installed. Run: pip install sentence-transformers") from None
        return self._model

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.get_dimension()), dtype=np.float32)

        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=max(1, int(self.config.batch_size)),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def get_dimension(self) -> int:
        if self._dimension is None:
            model = self._get_model()
            self._dimension = int(model.get_sentence_embedding_dimension())
        return self._dimension

    def save_model(self, path: Optional[str] = None) -> str:
        model = self._get_model()
        save_path = path or f"./models/{self.model_name.replace('/', '_')}"
        model.save(save_path)
        return save_path


def _coerce_embedding_config(config: Any | None = None, **overrides: Any) -> EmbedderConfig:
    """Normalize legacy/new config inputs into a single canonical config object."""

    if isinstance(config, EmbedderConfig):
        merged = EmbedderConfig(**config.__dict__)
    elif isinstance(config, LegacyEmbeddingConfig):
        provider = "openai" if config.use_openai else (config.model or "local")
        merged = EmbedderConfig(
            provider="openai" if provider == "openai" else "local",
            api_key=config.api_key or None,
            model=config.model,
            timeout=config.timeout,
            batch_size=config.batch_size,
        )
    elif isinstance(config, dict):
        provider = str(config.get("provider") or config.get("EMBEDDING_PROVIDER") or "local")
        if str(config.get("use_openai", False)).lower() in {"1", "true", "yes"}:
            provider = "openai"
        merged = EmbedderConfig(
            provider=provider,
            api_key=config.get("api_key") or config.get("OPENAI_API_KEY"),
            model=config.get("model"),
            timeout=float(config.get("timeout", 60.0)),
            batch_size=int(config.get("batch_size", 100)),
            dimension=int(config.get("dimension", 384)),
            device=str(config.get("device", "cpu")),
            cache_dir=config.get("cache_dir"),
            extra_headers=config.get("extra_headers"),
        )
    else:
        merged = EmbedderConfig()

    for key, value in overrides.items():
        if hasattr(merged, key):
            setattr(merged, key, value)

    return merged


def create_embedder(config: Optional[EmbedderConfig] = None, **kwargs: Any) -> EmbedderBase:
    """Factory function to create an embedder from config."""

    coerced = _coerce_embedding_config(config, **kwargs)
    provider = coerced.provider.lower()

    if provider == "local":
        return LocalEmbedder(dimension=coerced.dimension)

    if provider == "openai":
        api_key = coerced.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            warnings.warn("OPENAI_API_KEY not found, falling back to local embedder")
            return LocalEmbedder(dimension=coerced.dimension)
        return OpenAIEmbedder(
            EmbedderConfig(
                provider="openai",
                api_key=api_key,
                model=coerced.model or OpenAIEmbedder.DEFAULT_MODEL,
                timeout=coerced.timeout,
                batch_size=coerced.batch_size,
                dimension=coerced.dimension,
                device=coerced.device,
                cache_dir=coerced.cache_dir,
                extra_headers=coerced.extra_headers,
            )
        )

    if provider == "cohere":
        return CohereEmbedder(coerced)

    if provider == "sentence_transformers":
        return SentenceTransformersEmbedder(coerced)

    raise ValueError(
        f"Unknown embedding provider: {provider}. Supported: local, openai, cohere, sentence_transformers"
    )


def create_embedder_from_config(config: Any | None = None, **overrides: Any) -> EmbedderBase:
    """Backward-compatible factory wrapper."""

    return create_embedder(config, **overrides)


# Backward-compatible aliases
get_embedder = create_embedder
EmbeddingConfig = LegacyEmbeddingConfig
EmbedderConfigLegacy = LegacyEmbeddingConfig
# G3 (2026-08-17): ExtendedEmbedderConfig was a field-for-field duplicate of
# EmbedderConfig — removed the class, kept the name as a deprecated alias so
# older imports keep working with a single canonical config.
ExtendedEmbedderConfig = EmbedderConfig


__all__ = [
    "LocalEmbedder",
    "OpenAIEmbedder",
    "CohereEmbedder",
    "SentenceTransformersEmbedder",
    "EmbedderBase",
    "EmbedderConfig",
    "EmbeddingConfig",
    "EmbedderConfigLegacy",
    "ExtendedEmbedderConfig",
    "create_embedder",
    "create_embedder_from_config",
    "get_embedder",
]
