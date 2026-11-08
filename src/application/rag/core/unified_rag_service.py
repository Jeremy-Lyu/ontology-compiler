"""Unified RAG service compatibility core.

This module hosts the canonical service/factory/adapter layer during migration.
It is intentionally self-contained to avoid circular imports with the legacy
`rag_search_service` compatibility module.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type
import warnings

from src.domain.models.core import RetrievalAssetPacket

from src.application.ports.rag_ports import RAGSearchPort, RAGSearchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1-5 (2026-08-04) — Retrieval fusion weights.  Configurable via env so
# operators can tune hybrid weights without touching Python.
#
# Usage:
#   RAG_VECTOR_WEIGHT=0.5    # BM25+vector fusion: hybrid = 0.5
#   RAG_STRUCTURED_WEIGHT=0.3 # structured index contribution
#   RAG_RULE_WEIGHT=0.2       # deterministic rule / synonym contribution
#   RAG_SECONDARY_RETRIEVERS_WEIGHT=0.7  # first fallback weight (geometric decay)
#   RAG_GRAPH_WEIGHT=0.4     # neo4j/graph retriever contribution
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RetrievalWeightConfig:
    """Named weights for the multi-channel retrieval stack."""

    vector: float = 0.5
    structured: float = 0.3
    rule: float = 0.2
    secondary_retrievers_weight: float = 0.7  # first fallback; 0.7^(n+1) for fallbacks[n]
    graph_retriever_weight: float = 0.4       # neo4j / graph retriever

    @classmethod
    def from_env(cls) -> "RetrievalWeightConfig":
        def _f(name: str, default: float) -> float:
            val = os.environ.get(name, "").strip()
            if not val:
                return default
            try:
                return max(0.0, min(1.0, float(val)))
            except (TypeError, ValueError):
                return default
        return cls(
            vector=_f("RAG_VECTOR_WEIGHT", 0.5),
            structured=_f("RAG_STRUCTURED_WEIGHT", 0.3),
            rule=_f("RAG_RULE_WEIGHT", 0.2),
            secondary_retrievers_weight=_f("RAG_SECONDARY_RETRIEVERS_WEIGHT", 0.7),
            graph_retriever_weight=_f("RAG_GRAPH_WEIGHT", 0.4),
        )

    def validate(self) -> None:
        total = self.vector + self.structured + self.rule
        if total > 1.0001:
            warnings.warn(
                f"Retrieval weights vector({self.vector})+structured({self.structured})+"
                f"rule({self.rule}) = {total:.3f} > 1.0.  Weights are informational-only "
                "(different channels use different score scales, hard normalisation can "
                "hurt recall), but consider tuning RAG_VECTOR_WEIGHT / "
                "RAG_STRUCTURED_WEIGHT / RAG_RULE_WEIGHT if retrieval quality degrades.",
                stacklevel=2,
            )


# ---------------------------------------------------------------------------
# PHASE 1-5 H18 FIX (2026-08-04): async adapters MUST NOT block the event loop.
# ---------------------------------------------------------------------------
# Each adapter below runs its synchronous payload (BM25+vector NumPy / Neo4j
# driver synchronous calls / AssetPacket file index lookups) inside
# ``asyncio.to_thread``.  Without this, a single 100ms hybrid_search on a
# 100k-chunk index starves every other concurrent HTTP request in FastAPI.
# ---------------------------------------------------------------------------


class LocalRetrieverAdapter(RAGSearchPort):
    """Hybrid (BM25 + Vector) retriever wrapped for non-blocking I/O."""

    def __init__(self, chunks_path: Optional[Path] = None, rag_dir: Optional[str | Path] = None, **kwargs: Any):
        # Phase B: default is always the governed packet directory.  HybridIndexManager
        # itself enforces legacy-dir hard constraints, so we defer resolution and
        # simply pass None when nothing was given.
        from ..governed_packets import resolve_governed_rag_dir

        if rag_dir is not None:
            self._rag_dir: Optional[Path] = resolve_governed_rag_dir(Path(rag_dir))
        elif chunks_path is not None:
            self._rag_dir = resolve_governed_rag_dir(Path(chunks_path))
        else:
            self._rag_dir = None  # manager will default to governed
        self._kwargs = kwargs
        self._initialized = False
        self._manager = None

    async def _init_retriever(self):
        if self._initialized:
            return
        from ..hybrid_rag import HybridIndexManager
        kw = dict(self._kwargs)
        if self._rag_dir is not None:
            kw["rag_dir"] = self._rag_dir
        self._manager = HybridIndexManager(**kw)
        self._initialized = True

    async def search(self, query: str, top_k: int = 10, **kwargs: Any) -> list[RAGSearchResult]:
        await self._init_retriever()
        weights = RetrievalWeightConfig.from_env()
        manager = self._manager
        assert manager is not None

        def _run_sync(q: str, topk: int) -> list:
            return manager.hybrid_search(
                query=q,
                topk=topk,
                vector_weight=kwargs.get("vector_weight", weights.vector),
                keyword_weight=kwargs.get("keyword_weight", max(0.05, 1.0 - weights.vector)),
            )

        results = await asyncio.to_thread(_run_sync, query, top_k)
        out: list[RAGSearchResult] = []
        for r in results:
            chunk = getattr(r, "chunk", None)
            cid = str(getattr(chunk, "chunk_id", getattr(r, "chunk_id", "")))
            content = getattr(chunk, "content", str(r))
            src = getattr(chunk, "chunk_type", "hybrid")
            scr = float(getattr(r, "hybrid_score", 0.0))
            meta: Dict[str, Any] = {}
            layer = getattr(chunk, "layer", None)
            src_file = getattr(chunk, "source_file", None)
            if layer: meta["layer"] = layer
            if src_file: meta["source_file"] = src_file
            out.append(RAGSearchResult(chunk_id=cid, content=content, source=src, score=scr, metadata=meta))
        return out

    async def batch_search(self, queries: list[str], top_k: int = 10, **kwargs: Any) -> list[list[RAGSearchResult]]:
        return [await self.search(q, top_k, **kwargs) for q in queries]

    def get_retriever_type(self) -> str:
        return "local"


class Neo4jRetrieverAdapter(RAGSearchPort):
    """Graph-retriever adapter — runs synchronous graph driver calls off-loop."""

    def __init__(self, packet_path: Optional[str] = None, packet: Optional[Dict[str, Any] | RetrievalAssetPacket] = None, **kwargs: Any):
        self._kwargs = kwargs
        self._packet_path = packet_path
        self._packet = RetrievalAssetPacket.model_validate(packet) if isinstance(packet, dict) else packet
        self._retriever = None

    async def _init_retriever(self):
        if self._retriever is not None:
            return
        from ..neo4j_retriever import Neo4jRetriever
        self._retriever = Neo4jRetriever(**self._kwargs)

    async def search(self, query: str, top_k: int = 10, **kwargs: Any) -> list[RAGSearchResult]:
        await self._init_retriever()
        ret = self._retriever
        assert ret is not None

        def _run_sync(q: str, limit: int) -> list:
            return ret.keyword_search(q, limit=limit)

        results = await asyncio.to_thread(_run_sync, query, top_k)
        return [
            RAGSearchResult(
                chunk_id=str(getattr(r, "id", "")),
                content=getattr(r, "description", ""),
                source=getattr(r, "type", "neo4j"),
                score=float(getattr(r, "score", 0.0)),
                metadata={
                    "properties": getattr(r, "properties", {}),
                    "name": getattr(r, "name", ""),
                },
            )
            for r in results
        ]

    async def batch_search(self, queries: list[str], top_k: int = 10, **kwargs: Any) -> list[list[RAGSearchResult]]:
        return [await self.search(q, top_k, **kwargs) for q in queries]

    def get_retriever_type(self) -> str:
        return "neo4j"


class DifyRetrieverAdapter(RAGSearchPort):
    """Dify knowledge-base retriever adapter.

    NOTE: DifyClient.search is already a coroutine doing HTTP I/O via the
    urllib/httpx transports; it correctly yields the event loop.  No
    ``asyncio.to_thread`` wrapping is necessary here.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self._api_key = api_key
        self._base_url = base_url
        self._client = None

    async def _init_client(self):
        if self._client is not None:
            return
        from ..infrastructure.dify_client import DifyClient
        self._client = DifyClient(api_key=self._api_key, base_url=self._base_url)

    async def search(self, query: str, top_k: int = 10, **kwargs: Any) -> list[RAGSearchResult]:
        await self._init_client()
        assert self._client is not None
        results = await self._client.search(query, top_k=top_k, **kwargs)
        return [
            RAGSearchResult(
                chunk_id=r.get("id", ""),
                content=r.get("content", ""),
                source=r.get("source", "dify"),
                score=r.get("score", 0.0),
                metadata=r.get("metadata", {}),
            )
            for r in results
        ]

    async def batch_search(self, queries: list[str], top_k: int = 10, **kwargs: Any) -> list[list[RAGSearchResult]]:
        return [await self.search(q, top_k, **kwargs) for q in queries]

    def get_retriever_type(self) -> str:
        return "dify"


class AssetPacketServiceRetrieverAdapter(RAGSearchPort):
    """Governed-asset-packet retriever (H6 FIX 2026-08-04).

    Previously ``RetrieverFactory`` knew only ``{local,neo4j,dify}`` so the
    ``AssetPacketRetriever`` implementation (which already implements
    ``RAGSearchPort``) was unreachable through ``from_config`` even though
    operators listed ``{"type":"asset_packet"}`` as a fallback.  Now exposed
    via type = ``"asset_packet"``.
    """

    def __init__(self, packet_path: Optional[str] = None, packet: Optional[Dict[str, Any]] = None, **kwargs: Any):
        self._kwargs = kwargs
        self._packet_path = packet_path
        self._packet = packet
        self._retriever = None

    async def _init_retriever(self):
        if self._retriever is not None:
            return
        from src.application.rag.asset_packet_retriever import AssetPacketRetriever
        self._retriever = AssetPacketRetriever(
            packet_path=self._packet_path,
            packet=self._packet,
        )

    async def search(self, query: str, top_k: int = 10, **kwargs: Any) -> list[RAGSearchResult]:
        await self._init_retriever()
        ret = self._retriever
        assert ret is not None

        async def _run_async(q: str, topk: int) -> list:
            return await ret.search(q, top_k=topk, **kwargs)

        results = await _run_async(query, top_k)
        out: list[RAGSearchResult] = []
        for r in results:
            if isinstance(r, dict):
                out.append(RAGSearchResult(
                    chunk_id=str(r.get("chunk_id", "")),
                    content=r.get("content", ""),
                    source=r.get("source", "asset_packet"),
                    score=float(r.get("score", 0.0)),
                    metadata=r.get("metadata", {}) or {},
                ))
            else:
                out.append(RAGSearchResult(
                    chunk_id=str(getattr(r, "chunk_id", "")),
                    content=getattr(r, "content", ""),
                    source=getattr(r, "source", "asset_packet"),
                    score=float(getattr(r, "score", 0.0)),
                    metadata=(getattr(r, "metadata", None) or {}),
                ))
        return out

    async def batch_search(self, queries: list[str], top_k: int = 10, **kwargs: Any) -> list[list[RAGSearchResult]]:
        return [await self.search(q, top_k, **kwargs) for q in queries]

    def get_retriever_type(self) -> str:
        return "asset_packet"


class StructuredRetrieverAdapter(RAGSearchPort):
    """Structured (entity/relation/evidence) retriever adapter — runs synchronous index searches off-loop."""

    def __init__(self, index_path: Optional[str] = None, index: Optional["StructuredIndex"] = None, **kwargs: Any):
        self._kwargs = kwargs
        self._index_path = index_path
        self._index = index
        self._retriever = None

    async def _init_retriever(self):
        if self._retriever is not None:
            return
        from .structured_index import StructuredIndex
        if self._index is not None:
            self._retriever = self._index
        else:
            self._retriever = StructuredIndex()
            # TODO: Implement index loading from file when persistence is added
            # if self._index_path:
            #     self._retriever.load(self._index_path)

    async def search(self, query: str, top_k: int = 10, **kwargs: Any) -> list[RAGSearchResult]:
        await self._init_retriever()
        ret = self._retriever
        assert ret is not None

        search_type = kwargs.get("search_type", "entity")
        results = []

        def _run_sync(q: str, limit: int, st: str):
            if st == "relation":
                return ret.search_relations(query=q, limit=limit)
            elif st == "evidence":
                return ret.search_evidences(query=q, limit=limit)
            else:
                # Default to entity search
                return ret.search_entities(query=q, limit=limit)

        raw_results = await asyncio.to_thread(_run_sync, query, top_k, search_type)

        out: list[RAGSearchResult] = []
        for item in raw_results:
            if isinstance(item, tuple):
                # Relation search returns (RelationIndex, from_name, to_name)
                relation, from_name, to_name = item
                content = f"{from_name} {relation.relation_type} {to_name}: {relation.properties}"
                chunk_id = relation.relation_id
                source = "structured-relation"
                score = float(relation.confidence)
                metadata = {
                    "relation_type": relation.relation_type,
                    "from_entity": from_name,
                    "to_entity": to_name,
                    "properties": relation.properties,
                    "confidence": relation.confidence
                }
            elif hasattr(item, "entity_id"):
                # Entity search returns EntityIndex
                content = f"{item.name}: {item.description or 'No description'}"
                chunk_id = item.entity_id
                source = "structured-entity"
                score = float(item.confidence)
                metadata = {
                    "entity_type": item.entity_type,
                    "layer": item.layer,
                    "aliases": item.aliases,
                    "attributes": item.attributes,
                    "confidence": item.confidence
                }
            elif hasattr(item, "evidence_id"):
                # Evidence search returns EvidenceIndex
                content = item.content or "No content"
                chunk_id = item.evidence_id
                source = "structured-evidence"
                score = float(item.confidence)
                metadata = {
                    "entity_id": item.entity_id,
                    "source": item.source,
                    "source_type": item.source_type,
                    "confidence": item.confidence
                }
            else:
                continue
            out.append(RAGSearchResult(
                chunk_id=chunk_id,
                content=content,
                source=source,
                score=score,
                metadata=metadata
            ))
        return out

    async def batch_search(self, queries: list[str], top_k: int = 10, **kwargs: Any) -> list[list[RAGSearchResult]]:
        return [await self.search(q, top_k, **kwargs) for q in queries]

    def get_retriever_type(self) -> str:
        return "structured"


class RetrieverFactory:
    _retrievers: Dict[str, Type[RAGSearchPort]] = {
        "local": LocalRetrieverAdapter,
        "neo4j": Neo4jRetrieverAdapter,
        "dify": DifyRetrieverAdapter,
        "asset_packet": AssetPacketServiceRetrieverAdapter,  # H6 FIX 2026-08-04
        "structured": StructuredRetrieverAdapter,
    }

    @classmethod
    def register_retriever(cls, name: str, retriever_class: Type[RAGSearchPort]) -> None:
        cls._retrievers[name] = retriever_class

    @classmethod
    def create(cls, retriever_type: str, **kwargs: Any) -> RAGSearchPort:
        if retriever_type not in cls._retrievers:
            raise ValueError(f"Unknown retriever type: {retriever_type}. Available types: {list(cls._retrievers.keys())}")
        return cls._retrievers[retriever_type](**kwargs)


class UnifiedRAGService:
    def __init__(self, primary_retriever: RAGSearchPort, fallback_retrievers: Optional[List[RAGSearchPort]] = None, neo4j_retriever: Optional[RAGSearchPort] = None, structured_retriever: Optional[RAGSearchPort] = None):
        self._primary = primary_retriever
        self._fallbacks = fallback_retrievers or []
        self._neo4j = neo4j_retriever
        self._structured = structured_retriever

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'UnifiedRAGService':
        primary_config = config.get("primary", {})
        primary_type = primary_config.pop("type", "local")
        primary = RetrieverFactory.create(primary_type, **primary_config)
        neo4j = None
        if config.get("neo4j"):
            neo4j_config = config.get("neo4j", {})
            neo4j_type = neo4j_config.pop("type", "neo4j")
            neo4j = RetrieverFactory.create(neo4j_type, **neo4j_config)
        structured = None
        if config.get("structured"):
            structured_config = config.get("structured", {})
            structured_type = structured_config.pop("type", "structured")
            structured = RetrieverFactory.create(structured_type, **structured_config)
        fallbacks = []
        for fallback_config in config.get("fallbacks", []):
            fallback_type = fallback_config.pop("type", "local")
            fallbacks.append(RetrieverFactory.create(fallback_type, **fallback_config))
        return cls(primary, fallbacks, neo4j, structured)

    @classmethod
    def from_packet(cls, packet_path: Optional[str] = None, packet: Optional[Dict[str, Any] | RetrievalAssetPacket] = None) -> "UnifiedRAGService":
        primary = RetrieverFactory.create("asset_packet", packet_path=packet_path, packet=packet)
        return cls(primary)

    async def search(self, query: str, top_k: int = 10, use_hybrid: bool = False, use_graph: bool = False, use_structured: bool = False, **kwargs: Any) -> list[RAGSearchResult]:
        """Run a unified RAG search with primary → fallback → optional graph fusion.

        RAG-P0-2 BUGFIX (2026-08-04):
        - Old implementation only returned the primary retriever output and
          silently ignored ``fallback_retrievers`` and ``neo4j_retriever``, so
          configuration like ``fallbacks=[{"type":"dify"}]`` was dead code.
        - New flow:
              1. run primary; if it returns results ``_merge_results`` with
                 each fallback in order (carrying over a dedup'd union) so a
                 fallback can contribute novel documents that the primary missed
              2. if primary yielded nothing, try fallbacks in order, taking the
                 first non-empty result (no need to further merge empties)
              3. if ``use_graph`` and a neo4j retriever is configured, merge its
                 results as a secondary pass (lower default weight because the
                 graph retriever returns structured IDs, not passage chunks)
              4. if ``use_structured`` and a structured retriever is configured, merge its
                 results as a secondary pass (uses configured structured weight)
        """
        # --- Phase 1: primary + fallbacks ---
        try:
            primary_results = await self._primary.search(query, top_k=top_k, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Primary RAG retriever failed: %s", exc)
            primary_results = []

        merged: list[RAGSearchResult] = list(primary_results) if primary_results else []
        weights = RetrievalWeightConfig.from_env()
        weights.validate()

        if len(merged) == 0:
            # Empty primary → try fallbacks in order; stop at the first one that
            # returns anything.  This gives the "failover" semantics users expect
            # when they list fallbacks.
            for fb in self._fallbacks:
                try:
                    hits = await fb.search(query, top_k=top_k, **kwargs)
                except Exception as exc:
                    logger.warning("Fallback RAG retriever failed (%s): %s", fb.get_retriever_type(), exc)
                    continue
                if hits:
                    merged = list(hits)
                    break
        elif self._fallbacks and use_hybrid:
            # Primary has results + user asked for hybrid aggregation.  Run each
            # fallback and merge with decreasing weight (each subsequent fallback
            # is considered marginally less authoritative than the previous).
            for idx, fb in enumerate(self._fallbacks):
                try:
                    hits = await fb.search(query, top_k=top_k, **kwargs)
                except Exception as exc:
                    logger.warning("Fallback RAG retriever failed (%s): %s", fb.get_retriever_type(), exc)
                    continue
                # attenuate weight: 1st fallback 0.7, 2nd ~0.49, 3rd ~0.34…
                weight = max(0.2, weights.secondary_retrievers_weight ** (idx + 1))
                merged = self._merge_results(merged, hits, secondary_weight=weight)

        # --- Phase 2: graph retriever (optional) ---
        if use_graph and self._neo4j is not None:
            try:
                neo4j_hits = await self._neo4j.search(query, top_k=max(5, top_k // 2), **kwargs)
                merged = self._merge_results(merged, neo4j_hits, secondary_weight=weights.graph_retriever_weight)
            except Exception as exc:
                logger.warning("Neo4j RAG retriever failed (ignored): %s", exc)

        # --- Phase 3: structured retriever (optional) ---
        if use_structured and self._structured is not None:
            try:
                structured_hits = await self._structured.search(query, top_k=max(5, top_k // 2), **kwargs)
                merged = self._merge_results(merged, structured_hits, secondary_weight=weights.structured)
            except Exception as exc:
                logger.warning("Structured RAG retriever failed (ignored): %s", exc)

        # --- Cap to top_k ---
        return merged[:top_k]

    async def batch_search(self, queries: list[str], top_k: int = 10, **kwargs: Any) -> list[list[RAGSearchResult]]:
        return [await self.search(q, top_k, **kwargs) for q in queries]

    def _merge_results(self, primary: list[RAGSearchResult], secondary: list[RAGSearchResult], secondary_weight: float = 0.7) -> list[RAGSearchResult]:
        """Weighted concatenation of two result sets with final dedup + sorting.

        RAG-P0-2 BUGFIX (2026-08-04):
        - Old implementation was a literal ``primary + secondary`` concatenation
          with zero deduplication, so the same ``chunk_id`` could appear twice
          (once from each retriever).  The sibling method ``_deduplicate`` was
          defined but never called.
        - New flow:
              1. ``secondary`` scores are multiplied by ``secondary_weight`` (a
                 float in (0,1]); without a full cross-retriever score
                 normalisation this is the pragmatic per-retriever attenuation
                 recommended in RAG fusion literature.
              2. concatenate primary (unmodified weight) + attenuated secondary
              3. run through ``_deduplicate`` (first hit wins, which is from the
                 higher-precision primary); this also guarantees no chunk_id dupes
              4. re-sort by ``score`` descending so top results stay on top

        Args:
            primary: Results from the primary (higher-trust) retriever.  Scores
                are preserved verbatim.
            secondary: Results from a secondary / fallback / graph retriever.
                Scores are multiplied by ``secondary_weight`` before concat.
            secondary_weight: Multiplier ∈ (0,1] applied to secondary scores.

        Returns:
            Dedup'd list sorted by score desc, with contributions from both
            retrievers where they brought unique chunk_id values.
        """
        if not secondary:
            # No-op path; still run dedup just in case primary itself contained
            # a duplicate (unlikely but safe).
            return self._deduplicate(list(primary))

        # Apply cross-retriever score normalization
        normalized_primary = self._normalize_scores(primary)
        normalized_secondary = self._normalize_scores(secondary)

        # Apply weight to secondary.  Use dataclass-replace semantics via
        # dataclasses.replace when available (RAGSearchResult is dataclass in
        # ports/all_ports.py; fall back to manual rebuild otherwise).
        from dataclasses import replace as _dc_replace, is_dataclass

        def _attenuate(item: RAGSearchResult) -> RAGSearchResult:
            raw_score = getattr(item, "score", 0.0) or 0.0
            try:
                new_score = float(raw_score) * float(secondary_weight)
            except (TypeError, ValueError):
                new_score = 0.0
            if is_dataclass(item):
                try:
                    return _dc_replace(item, score=new_score)  # type: ignore[arg-type]
                except Exception:
                    pass
            # Generic fallback: mutate via dict if supported (ports use slots or
            # frozen dataclasses, so we construct a fresh instance).
            try:
                return item.__class__(**{**item.__dict__, "score": new_score})
            except Exception:
                # Last resort: leave as-is (dedup still handles duplicates)
                return item

        attenuated_secondary = [_attenuate(r) for r in normalized_secondary]
        concatenated = list(normalized_primary) + attenuated_secondary

        deduped = self._deduplicate(concatenated)

        # Sort by score desc.  Items missing a numeric score sort to the tail.
        def _score_key(item: RAGSearchResult) -> float:
            val = getattr(item, "score", None)
            try:
                return float(val) if val is not None else -1.0
            except (TypeError, ValueError):
                return -1.0

        deduped.sort(key=_score_key, reverse=True)
        return deduped

    def _deduplicate(self, results: list[RAGSearchResult]) -> list[RAGSearchResult]:
        """Remove duplicates by chunk_id, first-hit-wins."""
        seen: set[str] = set()
        unique: list[RAGSearchResult] = []
        for result in results:
            cid = getattr(result, "chunk_id", "") or ""
            if not cid:
                # No usable id → keep but don't contribute to dedup set.
                unique.append(result)
                continue
            if cid not in seen:
                seen.add(cid)
                unique.append(result)
        return unique

    def _normalize_scores(self, results: list[RAGSearchResult]) -> list[RAGSearchResult]:
        """Normalize scores to [0, 1] range for cross-retriever fusion."""
        if not results:
            return results
        
        from dataclasses import replace as _dc_replace, is_dataclass
        
        scores: List[float] = []
        for r in results:
            score = getattr(r, "score", 0.0)
            try:
                scores.append(float(score))
            except (TypeError, ValueError):
                scores.append(0.0)
        
        if not scores:
            return results
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            # All scores are the same, no need to normalize
            return results
        
        normalized: List[RAGSearchResult] = []
        for r in results:
            score = getattr(r, "score", 0.0)
            try:
                norm_score = (float(score) - min_score) / (max_score - min_score)
            except (TypeError, ValueError):
                norm_score = 0.0
            
            if is_dataclass(r):
                try:
                    normalized.append(_dc_replace(r, score=norm_score))
                    continue
                except Exception:
                    pass
            
            # Fallback for non-dataclass objects
            try:
                obj_dict = r.__dict__.copy()
                obj_dict["score"] = norm_score
                normalized.append(r.__class__(**obj_dict))
            except Exception:
                # Last resort: keep original
                normalized.append(r)
        
        return normalized


_default_service: Optional[UnifiedRAGService] = None


def get_rag_service(config: Optional[Dict[str, Any]] = None) -> UnifiedRAGService:
    global _default_service
    if _default_service is None or config is not None:
        if config is None:
            config = {"primary": {"type": "local"}}
        _default_service = UnifiedRAGService.from_config(config)
    return _default_service


__all__ = [
    "LocalRetrieverAdapter",
    "Neo4jRetrieverAdapter",
    "DifyRetrieverAdapter",
    "RetrieverFactory",
    "UnifiedRAGService",
    "get_rag_service",
]
