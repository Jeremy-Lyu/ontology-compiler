"""Packet-to-RAG index bridge (ROADMAP T4.1).

Keeps OntologyPacket as the extraction contract and converts its derived RAG
chunks into the existing ChunkRecord format consumed by HybridIndexManager.
The bridge is deliberately explicit: no hidden file I/O and no embedder
assumptions. Callers choose when to persist/build the actual hybrid index.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .chunk_indexing import ChunkRecord
from .packet_derivation import derive_rag_objects

__all__ = [
    "rag_chunks_from_packet",
    "write_packet_chunks",
    "build_hybrid_index_from_packet",
]


def _chunk_content(chunk: Dict[str, Any]) -> str:
    """Render a stable, human-readable retrieval document."""
    lines = [f"{chunk.get('title') or ''}".strip()]
    summary = str(chunk.get("summary") or "").strip()
    if summary:
        lines.append(summary)
    keywords = [str(x).strip() for x in chunk.get("keywords") or [] if str(x).strip()]
    if keywords:
        lines.append(f"Keywords: {', '.join(keywords)}")
    path = [str(x).strip() for x in chunk.get("parent_path") or [] if str(x).strip()]
    if path:
        lines.append(f"Hierarchy: {' > '.join(path)}")
    return "\n".join(x for x in lines if x)


def rag_chunks_from_packet(packet: Dict[str, Any]) -> List[ChunkRecord]:
    """Convert packet-derived RAG chunks to existing ``ChunkRecord`` objects."""
    derived = derive_rag_objects(packet)
    chunks: List[ChunkRecord] = []
    for item in derived["rag_chunks"]:
        chunks.append(ChunkRecord(
            chunk_id=item["chunk_id"],
            chunk_type=f"ENTITY_{item.get('layer') or 'UNKNOWN'}",
            content=_chunk_content(item),
            node_ids=[item["entity_id"]] if item.get("entity_id") else [],
            linked_node_ids=[],
            meta={
                "entity_id": item.get("entity_id", ""),
                "layer": item.get("layer", ""),
                "type": item.get("type", ""),
                "source_file": item.get("source_file", ""),
                "evidence_ref": item.get("evidence_ref", ""),
                "parent_path": item.get("parent_path", []),
                "valid_from": item.get("valid_from", ""),
                "valid_to": item.get("valid_to", ""),
            },
        ))
    return chunks


def write_packet_chunks(packet: Dict[str, Any], output_path: Path | str) -> List[ChunkRecord]:
    """Derive packet chunks and persist them using the canonical JSONL writer."""
    from .chunk_indexing import write_chunks_jsonl
    chunks = rag_chunks_from_packet(packet)
    write_chunks_jsonl(chunks, Path(output_path))
    return chunks


def build_hybrid_index_from_packet(
    packet: Dict[str, Any],
    rag_dir: Path | str,
    *,
    embedder: Any = None,
    force_rebuild: bool = False,
) -> Any:
    """Explicit packet→chunks→hybrid-index write path.

    Importing ``HybridIndexManager`` locally avoids a module cycle and keeps
    packet derivation usable in lightweight/offline RAG tests.
    """
    from .hybrid_rag import HybridIndexManager
    manager = HybridIndexManager(rag_dir=rag_dir, embedder=embedder)
    return manager.build_hybrid_index(
        rag_chunks_from_packet(packet), force_rebuild=force_rebuild
    )
