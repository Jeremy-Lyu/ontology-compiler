"""Embedding single-config characterization (G3, 2026-08-17).

Locks the embedding config normalization after removing the duplicate
``ExtendedEmbedderConfig`` class: there is exactly one canonical config
``EmbedderConfig`` (Extended removed as a field-for-field duplicate, kept only
as a deprecated alias), legacy aliases map to ``LegacyEmbeddingConfig``, and
``_coerce_embedding_config`` remains the single assembly point.
"""

from __future__ import annotations

import unittest

from src.application.rag.infrastructure.embedding import (
    EmbedderConfig,
    EmbeddingConfig,
    EmbedderConfigLegacy,
    ExtendedEmbedderConfig,
    LegacyEmbeddingConfig,
    _coerce_embedding_config,
    create_embedder,
    LocalEmbedder,
)


class EmbeddingSingleConfigTest(unittest.TestCase):
    def test_extended_is_canonical_alias(self):
        """ExtendedEmbedderConfig is a deprecated alias of EmbedderConfig."""
        self.assertIs(ExtendedEmbedderConfig, EmbedderConfig)

    def test_legacy_aliases_map_to_legacy(self):
        self.assertIs(EmbeddingConfig, LegacyEmbeddingConfig)
        self.assertIs(EmbedderConfigLegacy, LegacyEmbeddingConfig)

    def test_coerce_dict_returns_canonical(self):
        c = _coerce_embedding_config({"provider": "cohere", "model": "m", "batch_size": 32})
        self.assertIsInstance(c, EmbedderConfig)
        self.assertEqual(c.provider, "cohere")
        self.assertEqual(c.model, "m")
        self.assertEqual(c.batch_size, 32)

    def test_coerce_legacy_use_openai(self):
        cfg = _coerce_embedding_config(EmbeddingConfig(api_key="k", use_openai=True))
        self.assertEqual(cfg.provider, "openai")
        self.assertEqual(cfg.api_key, "k")

    def test_coerce_canonical_passthrough(self):
        src = EmbedderConfig(provider="sentence_transformers", model="m", batch_size=16)
        cfg = _coerce_embedding_config(src)
        self.assertIsNot(cfg, src)  # defensive copy
        self.assertEqual(cfg.provider, "sentence_transformers")
        self.assertEqual(cfg.model, "m")

    def test_create_local_embedder(self):
        embedder = create_embedder({"provider": "local"})
        self.assertIsInstance(embedder, LocalEmbedder)

    def test_defaults_stable(self):
        self.assertEqual(EmbedderConfig().provider, "local")
        self.assertEqual(EmbedderConfig().batch_size, 100)
        self.assertEqual(EmbedderConfig().dimension, 384)


if __name__ == "__main__":
    unittest.main()