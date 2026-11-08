"""Canonical BM25 keyword retrieval implementation.

[BATCH 3 - 2026-06-29]: This module provides the canonical BM25/keyword retrieval
functions used by rag_answerer.py, validator.py, and retrieval/__init__.py.

Note: load_inverted_index has been migrated to chunk_indexing.py for unified indexing.

For hybrid retrieval (BM25 + Vector), use HybridIndexManager from hybrid_rag.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .chunk_indexing import ChunkRecord, load_chunks_jsonl, load_idf, tokenize, load_inverted_index
from .hybrid_rag import HybridIndexManager
from .models import RetrievedChunk

logger = logging.getLogger(__name__)


def build_chunk_by_id(chunks: Sequence[ChunkRecord]) -> Dict[str, ChunkRecord]:
    """Build a chunk_id -> ChunkRecord lookup dict.
    
    Utility for efficient chunk lookup during retrieval.
    """
    return {c.chunk_id: c for c in chunks if c.chunk_id}


def retrieve_bm25(
    query: str,
    chunks: Sequence[ChunkRecord],
    inverted_index: Dict[str, List[str]],
    idf: Dict[str, float],
    topk: int = 10,
) -> List[RetrievedChunk]:
    """BM25 keyword retrieval algorithm (standard Okapi BM25).
    
    Uses standard BM25 formula with k1=1.5, b=0.75 for better ranking quality.
    """
    if topk <= 0:
        raise ValueError("topk must be positive")

    chunk_by_id = build_chunk_by_id(chunks)
    
    # Precompute chunk token frequencies and lengths
    chunk_tf: Dict[str, Dict[str, int]] = {}
    chunk_len: Dict[str, int] = {}
    for chunk in chunks:
        tokens = tokenize(chunk.content)
        chunk_len[chunk.chunk_id] = max(len(tokens), 1)
        tf: Dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        chunk_tf[chunk.chunk_id] = tf
    
    avgdl = sum(chunk_len.values()) / max(len(chunk_len), 1)
    
    # Standard BM25 parameters
    k1 = 1.5
    b = 0.75
    
    use_idf = bool(idf)
    scores: Dict[str, float] = {}
    
    query_tokens = tokenize(query)
    for token in query_tokens:
        weight = idf.get(token, 1.0) if use_idf else 1.0
        for chunk_id in inverted_index.get(token, []):
            tf_val = chunk_tf.get(chunk_id, {}).get(token, 0)
            if tf_val == 0:
                continue
            dl = chunk_len.get(chunk_id, avgdl)
            # Standard BM25 score
            tf_component = (tf_val * (k1 + 1)) / (tf_val + k1 * (1 - b + b * dl / avgdl))
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight * tf_component
    
    # Boost chunks that contain multiple query terms
    query_token_set = set(query_tokens)
    if len(query_token_set) > 1:
        for chunk_id, tf_map in chunk_tf.items():
            overlap = len(query_token_set & set(tf_map.keys()))
            if overlap > 1:
                scores[chunk_id] = scores.get(chunk_id, 0.0) * (1.0 + 0.2 * overlap)
    
    out: List[RetrievedChunk] = []
    for chunk_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:topk]:
        chunk = chunk_by_id.get(chunk_id)
        if chunk:
            out.append(RetrievedChunk(chunk=chunk, score=score))
    return out


def retrieve_keyword(
    query: str,
    chunks: Sequence[ChunkRecord],
    inverted_index: Dict[str, List[str]],
    topk: int = 10,
    idf: Dict[str, float] | None = None,
) -> List[RetrievedChunk]:
    """Keyword-based retrieval wrapper using BM25.
    
    Primary entry point for keyword retrieval in rag_answerer.py and validator.py.
    """
    return retrieve_bm25(query, chunks, inverted_index, idf or {}, topk)


def retrieve_hybrid(
    query: str,
    chunks: Sequence[ChunkRecord],
    inverted_index: Dict[str, List[str]],
    topk: int = 10,
    idf: Optional[Dict[str, float]] = None,
    vector_weight: float = 0.5,
    keyword_weight: float = 0.5,
) -> List[RetrievedChunk]:
    """**DEPRECATED placeholder** — name implies BM25+Vector fusion but is pure BM25.

    P0 Transition Cleanup (2026-08-04):
    - This function was originally written as a shim pending the real hybrid
      index; it never performed vector fusion.  Its signature exposes weights
      (vector_weight / keyword_weight) that are silently ignored, which is a
      contract lie for any caller expecting true hybrid behaviour.
    - **Zero confirmed runtime callers** in the codebase (grep across source
      files: only the function definition and QA report references matched).
    - **Correct replacements** (use these instead):
        * ``HybridIndexManager.hybrid_search(...)`` — in-process BM25+Vector
          with RRF fusion, backed by VectorIndexManager + inverted index on disk.
        * ``retrieve_from_rag_dir(...)`` — convenience one-liner for the same.
        * ``retrieve_keyword(...)`` — if you genuinely only want BM25.

    Behaviour of this shim is UNCHANGED: it falls through to ``retrieve_bm25``
    (identical to retrieve_keyword) so any hypothetical legacy caller still
    works.  A PendingDeprecationWarning is emitted once per import via
    ``warnings.warn``; callers should migrate as soon as possible.
    """
    import warnings
    warnings.warn(
        "retrieve_hybrid() is DEPRECATED: its behaviour is pure BM25 (ignores "
        "vector_weight/keyword_weight). Use HybridIndexManager.hybrid_search "
        "or retrieve_from_rag_dir for true BM25+Vector fusion. "
        "See retriever.py docstring for details.",
        DeprecationWarning,
        stacklevel=2,
    )
    logger.debug(
        "[RAG] retrieve_hybrid deprecated-shim: query=%s topk=%s "
        "vector_weight=%s keyword_weight=%s -> falling back to BM25",
        query, topk, vector_weight, keyword_weight,
    )
    return retrieve_bm25(query, chunks, inverted_index, idf or {}, topk)


def retrieve_from_rag_dir(
    query: str,
    rag_dir: Path | str,
    topk: int = 10,
    vector_weight: float = 0.5,
    keyword_weight: float = 0.5,
) -> List[RetrievedChunk]:
    """Convenience retrieval from a RAG directory with hybrid search.
    
    Uses HybridIndexManager for full hybrid retrieval (BM25 + Vector).
    Falls back to keyword-only retrieval if hybrid search fails.
    
    Args:
        query: Search query string
        rag_dir: Directory containing chunks.jsonl, inverted_index.json, vectors.npy
        topk: Maximum results to return
        vector_weight: Weight for vector similarity (0-1)
        keyword_weight: Weight for BM25 keyword matching (0-1)
    
    Returns:
        List of RetrievedChunk sorted by hybrid score
    """
    rag_dir = Path(rag_dir)
    try:
        results = HybridIndexManager(rag_dir=rag_dir).hybrid_search(
            query=query,
            topk=topk,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
        )
        return [RetrievedChunk(chunk=item.chunk, score=item.hybrid_score) for item in results]
    except Exception as exc:
        logger.warning("[RAG] hybrid retrieval failed: %s; falling back to keyword retrieval", exc)
        chunks = load_chunks_jsonl(rag_dir / "chunks.jsonl")
        inverted_index = load_inverted_index(rag_dir / "inverted_index.json")
        idf = load_idf(rag_dir / "idf.json") if (rag_dir / "idf.json").exists() else {}
        return retrieve_keyword(query, chunks, inverted_index, topk, idf)


__all__ = [
    "load_inverted_index",
    "retrieve_keyword",
    "retrieve_bm25",
    "build_chunk_by_id",
    "retrieve_hybrid",
    "retrieve_from_rag_dir",
]
