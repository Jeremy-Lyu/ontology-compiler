"""统一索引构建模块 - 整合 Chunk 构建与索引工具

本模块整合了原 chunk_builder.py 和 _indexing_utils.py 的功能，提供：
1. ChunkRecord 数据模型与构建逻辑
2. Token 化与倒排索引构建
3. IDF 计算
4. JSONL 读写

架构目标：
- G4: 统一知识消费层
- G5: 统一质量闭环

[Phase 3 - RAG 2026-06-26]: Merged from chunk_builder.py and _indexing_utils.py
[P2 RAG 收口 2026-09-03]: Domain builder extracted to domain_chunk_builder.py;
ChunkRecord unified to domain/models.py. This module re-exports both for
backward compatibility.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.domain.errors import DataContractError
from src.domain.models.evidence import Evidence

from .domain.models import ChunkRecord
from .domain_chunk_builder import (
    _make_chunk_id,
    _node_id,
    _require_list,
    build_chunks_from_domain_ontology,
    infer_capability_level,
    infer_capability_number,
    infer_parent_capability_ids,
)
from .models import (
    CHUNK_PREFIX_ACTOR,
    CHUNK_PREFIX_ASSET,
    CHUNK_PREFIX_CAP,
    CHUNK_PREFIX_DATA,
    CHUNK_PREFIX_PATH,
    CHUNK_PREFIX_PROC,
    CHUNK_PREFIX_SCENARIO,
    CHUNK_PREFIX_SYSMAP,
    CHUNK_TYPE_ACTOR_CARD,
    CHUNK_TYPE_ASSET_CARD,
    CHUNK_TYPE_CAPABILITY_CARD,
    CHUNK_TYPE_DATA_CARD,
    CHUNK_TYPE_PATH_CARD,
    CHUNK_TYPE_PROCESS_CARD,
    CHUNK_TYPE_SCENARIO_CARD, CHUNK_TYPE_SYSTEM_MAPPING_CARD,
    EDGE_TYPES_SYSTEM_TO_CAP as _EDGE_TYPES_SYSTEM_TO_CAP,
)


# =============================================================================
# Tokenization and Indexing Utilities
# =============================================================================

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_CHINESE_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+")

# Lazy import jieba to avoid loading it if not needed
_jieba = None

def _get_jieba():
    """Lazy load jieba for Chinese word segmentation."""
    global _jieba
    if _jieba is None:
        try:
            import jieba
            _jieba = jieba
        except ImportError:
            print("[tokenize] Warning: jieba not installed, falling back to character-level tokenization")
    return _jieba


def _chinese_bigrams(segment: str) -> List[str]:
    """Adjacent character bigrams for a Chinese segment (deduped, order kept).

    Jieba's dictionary misses domain terms (e.g. "主数据" -> ["主", "数据"]),
    which makes exact query terms unmatchable in the index. Bigrams guarantee
    that any term appearing contiguously in the text produces matching tokens
    regardless of how jieba segments either side.
    """
    if len(segment) < 2:
        return []
    return list(dict.fromkeys(segment[i : i + 2] for i in range(len(segment) - 1)))


def tokenize(text: str) -> List[str]:
    """Tokenize text into tokens, supporting both English and Chinese.

    English tokens: alphanumeric + underscore, lowercased
    Chinese tokens: word-level segmentation using jieba (if available),
                    otherwise character-level fallback, plus adjacent
                    character bigrams as a segmentation-agnostic fallback.
    """
    tokens: List[str] = []

    # Extract English tokens
    tokens.extend([t.lower() for t in _TOKEN_RE.findall(text or "") if t])

    # Extract Chinese tokens
    chinese_text = text or ""
    jieba = _get_jieba()

    if jieba:
        # Use jieba for word-level segmentation
        for match in _CHINESE_TOKEN_RE.finditer(chinese_text):
            chinese_segment = match.group(0)
            # Use jieba.lcut for precise word segmentation
            chinese_words = jieba.lcut(chinese_segment, cut_all=False)
            tokens.extend(chinese_words)
            tokens.extend(_chinese_bigrams(chinese_segment))
    else:
        # Fallback: character-level segmentation plus bigrams
        for match in _CHINESE_TOKEN_RE.finditer(chinese_text):
            chinese_word = match.group(0)
            tokens.extend(list(chinese_word))
            tokens.extend(_chinese_bigrams(chinese_word))

    return tokens


def build_inverted_index(chunks: Sequence[ChunkRecord]) -> Dict[str, List[str]]:
    """Build a simple inverted index from chunks."""

    index: Dict[str, List[str]] = {}
    for c in chunks:
        for tok in tokenize(c.content):
            index.setdefault(tok, []).append(c.chunk_id)
    # de-dup + stable ordering
    for tok, ids in list(index.items()):
        index[tok] = sorted(set(ids))
    return index


def incremental_update_inverted_index(
    existing_index: Dict[str, List[str]],
    new_chunks: Sequence[ChunkRecord],
    modified_chunks: Sequence[ChunkRecord],
    deleted_chunk_ids: List[str]
) -> Dict[str, List[str]]:
    """Incrementally update inverted index without full rebuild.

    [LT02 2026-06-14]: Optimized version that only processes changed chunks.

    Args:
        existing_index: The current inverted index
        new_chunks: Newly added chunks
        modified_chunks: Chunks that were modified (content changed)
        deleted_chunk_ids: IDs of chunks to delete

    Returns:
        Updated inverted index
    """
    # Create a copy to avoid mutating the original
    index: Dict[str, List[str]] = {k: list(v) for k, v in existing_index.items()}
    deleted_set = set(deleted_chunk_ids)

    # 1. Remove deleted chunks from all token postings
    for token, chunk_ids in list(index.items()):
        new_ids = [cid for cid in chunk_ids if cid not in deleted_set]
        if not new_ids:
            del index[token]  # Remove token with no postings
        else:
            index[token] = new_ids

    # 2. Process modified chunks - remove old content first
    for c in modified_chunks:
        if not c.chunk_id:
            continue
        # Remove this chunk from all tokens (old content)
        for token, chunk_ids in list(index.items()):
            if c.chunk_id in chunk_ids:
                index[token].remove(c.chunk_id)
                if not index[token]:
                    del index[token]

    # 3. Add new and modified chunks
    for c in list(modified_chunks) + list(new_chunks):
        if not c.chunk_id:
            continue
        for tok in tokenize(c.content):
            index.setdefault(tok, []).append(c.chunk_id)

    # 4. De-dup and sort
    for tok, ids in index.items():
        index[tok] = sorted(set(ids))

    return index


def compute_idf(inverted_index: Dict[str, List[str]], num_chunks: int) -> Dict[str, float]:
    """Compute IDF (Inverse Document Frequency) for each token.

    T7-A: Uses standard log-smoothed IDF formula:
        idf(t) = log((N + 1) / (df(t) + 1)) + 1
    where N = total number of chunks, df(t) = number of chunks containing token t.
    The +1 smoothing avoids zero division and ensures rare terms score > common ones.

    Args:
        inverted_index: token -> list of chunk_ids mapping.
        num_chunks: total number of chunks in the corpus.

    Returns:
        Dict mapping token -> idf float value.
    """
    N = max(num_chunks, 1)
    idf: Dict[str, float] = {}
    for token, chunk_ids in inverted_index.items():
        df = len(set(chunk_ids))
        idf[token] = math.log((N + 1) / (df + 1)) + 1.0
    return idf


def write_idf(idf: Dict[str, float], output_path: Path) -> None:
    """Write IDF dict to JSON.

    T7-A: Persisted alongside inverted_index.json so retriever can load at query time.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(idf, f, ensure_ascii=False)


def load_idf(path: Path) -> Dict[str, float]:
    """Load IDF dict from JSON. Returns empty dict if file not found.

    T7-A: Graceful fallback to plain TF scoring when idf.json is absent
    (e.g., index built by older version).
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
    except Exception:
        pass
    return {}


def write_inverted_index(index: Dict[str, List[str]], output_path: Path) -> None:
    """Write inverted index to JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def load_inverted_index(path: Path) -> Dict[str, List[str]]:
    """Load inverted index from JSON.

    This function was migrated from retriever.py to consolidate index loading utilities.
    """
    if not path.exists():
        raise DataContractError(f"inverted index not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        raise DataContractError(f"failed to load inverted index: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataContractError(f"inverted index is not an object: {path}")
    return {
        k: v
        for k, v in payload.items()
        if isinstance(k, str) and isinstance(v, list) and all(isinstance(x, str) for x in v)
    }


def load_chunks_jsonl(path: Path) -> List[ChunkRecord]:
    """Load ChunkRecord list from JSONL."""

    if not path.exists():
        raise DataContractError(f"chunks file not found: {path}")

    chunks: List[ChunkRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:  # pragma: no cover
                raise DataContractError(f"invalid JSON at line {i}: {exc}") from exc
            try:
                chunks.append(ChunkRecord.from_dict(obj))
            except (KeyError, TypeError) as exc:
                # B11/B14: surface record-level contract failures as a domain
                # DataContractError instead of a bare KeyError so consumers can
                # handle the bad line gracefully (2026-08-13).
                raise DataContractError(f"invalid chunk record at line {i}: {exc}") from exc
    return chunks


def write_chunks_jsonl(chunks: Sequence[ChunkRecord], output_path: Path) -> None:
    """Write ChunkRecord list to JSONL."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


# =============================================================================
# Ontology Loading and Utilities
# =============================================================================

def load_ontology_json(path: Path) -> Dict[str, Any]:
    """Load an ontology JSON artifact.

    Raises:
        DataContractError: if the file cannot be loaded as a JSON object.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        raise DataContractError(f"failed to load ontology json: {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise DataContractError(f"ontology json is not an object: {path}")
    return payload


# =============================================================================
# Process Ontology Chunk Building
# =============================================================================

def build_chunks_from_process_ontology(payload: Dict[str, Any], source_file: str) -> List[ChunkRecord]:
    """Build chunks from layer1_process_ontology.json payload."""

    nodes = _require_list(payload, "nodes", source_file)
    # R18 Fragment 模型标准化: 使用 'relationships' 作为标准字段名
    relationships = _require_list(payload, "relationships", source_file)

    node_by_id: Dict[str, Dict[str, Any]] = {}
    for n in nodes:
        node_by_id[_node_id(n, source_file)] = n

    # adjacency (both directions)
    adj: Dict[str, List[Dict[str, Any]]] = {}
    for rel in relationships:
        src = rel.get("source")
        tgt = rel.get("target")
        if isinstance(src, str) and isinstance(tgt, str):
            adj.setdefault(src, []).append(rel)
            adj.setdefault(tgt, []).append(rel)

    chunks: List[ChunkRecord] = []
    for nid, n in node_by_id.items():
        if n.get("meta_class") != "Process":
            continue

        name = str(n.get("meta_name") or "")
        desc = str(n.get("meta_description") or "")
        domain = str(n.get("sandbox_domain") or "")

        linked: List[str] = []
        rel_types: List[str] = []
        for rel in adj.get(nid, []):
            src = rel.get("source")
            tgt = rel.get("target")
            if isinstance(src, str) and src != nid:
                linked.append(src)
            if isinstance(tgt, str) and tgt != nid:
                linked.append(tgt)
            rtype = rel.get("type")
            if isinstance(rtype, str):
                rel_types.append(rtype)

        content_lines = [
            f"Process: {name}",
            f"Domain: {domain}",
            f"Description: {desc}",
            f"Linked nodes ({len(set(linked))}):",
        ]
        for lid in sorted(set(linked))[:50]:
            lnode = node_by_id.get(lid)
            lname = str(lnode.get("meta_name") or "") if lnode else ""
            content_lines.append(f"- {lid} {lname}")

        chunks.append(
            ChunkRecord(
                chunk_id=_make_chunk_id(CHUNK_PREFIX_PROC, nid),
                chunk_type=CHUNK_TYPE_PROCESS_CARD,
                content="\n".join(content_lines).strip(),
                node_ids=[nid],
                linked_node_ids=sorted(set(linked)),
                meta={
                    "domain": domain,
                    "source_file": source_file,
                    "relationship_types": sorted(set(rel_types)),
                    "meta_class": "Process",
                    "meta_subclass": n.get("meta_subclass"),
                },
            )
        )

    return chunks


def build_chunks(
    domain_path: Path,
    process_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> List[ChunkRecord]:
    """Build chunks from ontology JSON files.

    向后兼容：支持两种调用模式
    1. 旧模式（兼容原 chunk_builder）：build_chunks(domain_file, process_file=None)
    2. 新模式：build_chunks(ontology_dir) - 自动查找 layer1_domain_ontology.json 和 layer1_process_ontology.json

    Args:
        domain_path: 域本体文件路径或包含本体文件的目录路径
        process_path: 过程本体文件路径（仅旧模式使用）
        output_path: 可选的输出路径，用于写入 chunks.jsonl

    Returns:
        List of ChunkRecord objects.

    Raises:
        DataContractError: 如果指定的文件不存在
    """

    chunks: List[ChunkRecord] = []

    if domain_path.is_file():
        # 旧模式：直接传入文件路径
        payload = load_ontology_json(domain_path)
        chunks.extend(build_chunks_from_domain_ontology(payload, str(domain_path)))

        if process_path and process_path.exists():
            payload = load_ontology_json(process_path)
            chunks.extend(build_chunks_from_process_ontology(payload, str(process_path)))
    elif domain_path.is_dir():
        # 新模式：传入目录路径
        domain_file = domain_path / "layer1_domain_ontology.json"
        process_file = domain_path / "layer1_process_ontology.json"

        if domain_file.exists():
            payload = load_ontology_json(domain_file)
            chunks.extend(build_chunks_from_domain_ontology(payload, str(domain_file)))

        if process_file.exists():
            payload = load_ontology_json(process_file)
            chunks.extend(build_chunks_from_process_ontology(payload, str(process_file)))
    else:
        # 路径不存在，抛出异常（向后兼容）
        payload = load_ontology_json(domain_path)
        chunks.extend(build_chunks_from_domain_ontology(payload, str(domain_path)))

    if output_path:
        write_chunks_jsonl(chunks, output_path)

    return chunks


def build_chunks_with_evidence(
    ontology_dir: Path,
    output_path: Optional[Path] = None,
    master_ontology_path: Optional[Path] = None,
) -> List[ChunkRecord]:
    """Build chunks with evidence chain from ontology files.

    G26-Evidence-01: Enhanced version that traces each chunk back to source documents.

    Args:
        ontology_dir: Directory containing ontology JSON files
        output_path: Optional output path for chunks.jsonl
        master_ontology_path: Optional path to master ontology for evidence extraction

    Returns:
        List of ChunkRecord with evidence attached
    """
    # Build base chunks
    chunks = build_chunks(ontology_dir)

    # Attach evidence if master ontology available
    if master_ontology_path and master_ontology_path.exists():
        try:
            master_payload = load_ontology_json(master_ontology_path)
            chunks = _enrich_chunks_with_evidence(chunks, master_payload)
        except Exception as e:
            print(f"[chunk_builder] Warning: Could not attach evidence: {e}")

    if output_path:
        write_chunks_jsonl(chunks, output_path)

    return chunks


def _enrich_chunks_with_evidence(
    chunks: List[ChunkRecord],
    master_payload: Dict[str, Any]
) -> List[ChunkRecord]:
    """Enrich chunks with evidence from master ontology.

    G26-Evidence-01: Internal helper that extracts evidence from master ontology nodes
    and attaches them to corresponding chunks using unified Evidence model.
    """
    # Build node_id -> evidence mapping from master ontology
    nodes = master_payload.get("nodes", [])
    evidence_map: Dict[str, Evidence] = {}

    for node in nodes:
        node_id = node.get("meta_id")
        if not node_id:
            continue

        # Extract evidence fields using unified Evidence model
        evidence_data: Dict[str, Any] = {}
        if node.get("source_file"):
            evidence_data["source_file"] = node.get("source_file")
        if node.get("line_range"):
            evidence_data["line_range"] = node.get("line_range")
        if node.get("confidence"):
            evidence_data["confidence"] = node.get("confidence")
        if node.get("evidence"):
            evidence_data.update(node.get("evidence"))
        if node.get("context"):
            evidence_data["context"] = node.get("context")

        if evidence_data:
            evidence_map[node_id] = Evidence.from_dict(evidence_data)

    # Attach evidence to chunks
    enriched_chunks: List[ChunkRecord] = []
    for chunk in chunks:
        # Find evidence for any node in this chunk
        chunk_evidence: Evidence | None = None
        for node_id in chunk.node_ids:
            if node_id in evidence_map:
                chunk_evidence = evidence_map[node_id]
                break  # Use first matching evidence

        # Create enriched chunk (ChunkRecord is frozen, so we create a new one)
        enriched = ChunkRecord(
            chunk_id=chunk.chunk_id,
            chunk_type=chunk.chunk_type,
            content=chunk.content,
            node_ids=chunk.node_ids,
            linked_node_ids=chunk.linked_node_ids,
            meta=chunk.meta,
            evidence=chunk_evidence,
        )
        enriched_chunks.append(enriched)

    return enriched_chunks


__all__ = [
    # Data model
    "ChunkRecord",
    # Tokenization and indexing
    "tokenize",
    "build_inverted_index",
    "incremental_update_inverted_index",
    "compute_idf",
    "write_idf",
    "load_idf",
    "write_inverted_index",
    "load_inverted_index",
    # IO
    "load_chunks_jsonl",
    "write_chunks_jsonl",
    "load_ontology_json",
    # Capability inference
    "infer_capability_level",
    "infer_capability_number",
    "infer_parent_capability_ids",
    # Chunk building
    "build_chunks",
    "build_chunks_from_domain_ontology",
    "build_chunks_from_process_ontology",
    "build_chunks_with_evidence",
]
