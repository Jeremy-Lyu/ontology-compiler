"""Build the single governed RAG asset packet (Phase B hard constraint).

Run after ``master_ontology/`` has been refreshed.  Produces a timestamped
directory under ``outputs/runs/ontology/asset_packets/governed/<id>/`` with::

    chunks.jsonl / inverted_index.json / idf.json / vector_index/{...}
    packet.json (the full OntologyPacket)
    manifest.json (+ manifest.sha256) — provenance, checksums, master file hashes

Then rewrites ``asset_packets/latest_governed_packet.json`` to point at the
new packet so every consumer picks it up.

Exit code 1 with DataContractError when master source is missing.  Consumers
*must* read from the governed packet only unless ``RAG_ALLOW_LEGACY_DIR=1``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.application.rag.chunk_indexing import (
    ChunkRecord,
    build_inverted_index,
    compute_idf,
    write_chunks_jsonl,
    write_idf,
    write_inverted_index,
)

from src.application.rag.governed_packets import (
    ASSET_PACKETS_ROOT,
    DEFAULT_MASTER_DIR,
    GOVERNED_DIR,
    SCHEMA_VERSION,
    build_file_manifest,
    default_governed_rag_dir,
    file_sha256,
    write_latest_pointer,
)
from src.application.rag.publish_gate import evaluate_publish_gate, write_publish_receipt
from src.application.rag.infrastructure.embedding import LocalEmbedder
from src.application.rag.infrastructure.vector_index_manager import VectorIndexManager
from src.application.rag.master_ontology_adapter import (
    _BILINGUAL_ALIASES_PATH,
    _load_bilingual_aliases,
    load_master_ontology_packet,
)
from src.application.rag.packet_derivation import derive_rag_objects


VECTOR_DIMENSION = 4096


def _hash_rag_module_files() -> Dict[str, str]:
    """Snapshot source code fingerprints used in build so consumers can audit.

    Captures: tokenizer, embedder, adapter, derivation, vector manager,
    bilingual_aliases config.  Purpose: when retrieval degrades, we can tell
    whether a *code* change (not only a master ontology change) happened.
    """
    import src.application.rag.chunk_indexing as _ci
    import src.application.rag.infrastructure.embedding as _emb
    import src.application.rag.master_ontology_adapter as _ad
    import src.application.rag.packet_derivation as _der
    import src.application.rag.infrastructure.vector_index_manager as _vim

    files = {
        "chunk_indexing.py": Path(_ci.__file__),
        "embedding.py": Path(_emb.__file__),
        "master_ontology_adapter.py": Path(_ad.__file__),
        "packet_derivation.py": Path(_der.__file__),
        "vector_index_manager.py": Path(_vim.__file__),
        "entity_bilingual_aliases.json": _BILINGUAL_ALIASES_PATH,
    }
    return {name: file_sha256(path) for name, path in files.items() if path.exists()}


def _master_file_hashes(master_dir: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in ("nodes.json", "relationships.json"):
        p = master_dir / name
        out[name] = file_sha256(p) if p.exists() else ""
    return out


def build(
    master_dir: Path,
    *,
    force: bool = False,
    require_publishable: bool = False,
    certified: bool = False,
) -> Path:
    if not (master_dir / "nodes.json").exists() or not (master_dir / "relationships.json").exists():
        raise FileNotFoundError(
            f"[GovernedBuild] master_ontology missing nodes.json/relationships.json in {master_dir}"
        )

    master_hash_snapshot = _master_file_hashes(master_dir)
    latest_packet_id = None
    try:
        latest = default_governed_rag_dir()
        manifest_path = latest / "manifest.json"
        if manifest_path.exists() and not force:
            prev = json.loads(manifest_path.read_text(encoding="utf-8"))
            prev_mh = prev.get("master_hashes") or {}
            if (
                prev_mh.get("nodes.json") == master_hash_snapshot["nodes.json"]
                and prev_mh.get("relationships.json") == master_hash_snapshot["relationships.json"]
            ):
                # Master unchanged -> skip rebuild (unless --force).
                print(
                    f"[GovernedBuild] master unchanged; reusing governed packet {prev['packet_id']}.\n"
                    "                  Pass --force to rebuild anyway."
                )
                return latest
            latest_packet_id = prev.get("packet_id")
    except Exception:
        pass

    packet_id = f"GOV-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    packet_dir = GOVERNED_DIR / packet_id
    packet_dir.mkdir(parents=True, exist_ok=True)

    # 1) Adapter + derive (bilingual aliases already merged inside adapter)
    print(f"[GovernedBuild] [{packet_id}] adapter -> derive ...")
    packet = load_master_ontology_packet(master_dir)
    derived = derive_rag_objects(packet)
    # derive_rag_objects returns plain dicts; convert to ChunkRecord so the
    # canonical JSONL format carries the exact same content used by the
    # retrieval evaluator (title+summary+keywords+parent_path concatenation).
    chunk_records: List[ChunkRecord] = []
    for c in derived["rag_chunks"]:
        title = str(c.get("title") or "")
        summary = str(c.get("summary") or "")
        keywords = list(c.get("keywords") or [])
        parent_path = list(c.get("parent_path") or [])
        content = "\n".join([title, summary, *keywords, *parent_path]).strip()
        chunk_records.append(
            ChunkRecord(
                chunk_id=c.get("chunk_id") or f"ragc:{c.get('entity_id')}",
                chunk_type=str(c.get("type") or "entity"),
                content=content,
                node_ids=[str(c.get("entity_id") or "")],
                linked_node_ids=[],
                meta={
                    "title": title,
                    "keywords": keywords,
                    "parent_path": parent_path,
                },
                evidence=None,
            )
        )
    chunks = chunk_records
    # packet["entities"] from master_ontology_adapter is a dict {layer: [entity]}
    buckets = packet.get("entities") or {}
    if isinstance(buckets, dict):
        entities = [e for v in buckets.values() if isinstance(v, list) for e in v]
    elif isinstance(buckets, list):
        entities = list(buckets)
    else:
        entities = []
    bilingual_count = len(_load_bilingual_aliases())

    # 2) Persist packet.json for downstream answer-path consumers
    (packet_dir / "packet.json").write_text(
        json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 3) BM25 index
    print(f"[GovernedBuild] [{packet_id}] BM25 index ({len(chunks)} chunks)...")
    write_chunks_jsonl(chunks, packet_dir / "chunks.jsonl")
    inverted_index = build_inverted_index(chunks)
    write_inverted_index(inverted_index, packet_dir / "inverted_index.json")
    idf = compute_idf(inverted_index, len(chunks))
    write_idf(idf, packet_dir / "idf.json")

    # 4) Vector index
    print(f"[GovernedBuild] [{packet_id}] Vector index (dim={VECTOR_DIMENSION})...")
    vector_base = packet_dir  # VectorIndexManager appends vector_index/ itself
    dim = VECTOR_DIMENSION
    manager = VectorIndexManager(
        vector_base,
        embedder=LocalEmbedder(dimension=dim),
        dimension=dim,
    )
    manager.build_index(chunks, force_rebuild=True)

    # 5) Manifest + checksum
    print(f"[GovernedBuild] [{packet_id}] manifest ...")
    file_list: List[str] = [
        "chunks.jsonl",
        "inverted_index.json",
        "idf.json",
        "packet.json",
        "vector_index/vectors.npz",
        "vector_index/chunk_ids.json",
        "vector_index/metadata.json",
        "vector_index/vocab.json",
    ]
    files = build_file_manifest(packet_dir, file_list)
    vector_meta_path = packet_dir / "vector_index" / "metadata.json"
    vector_dim = (
        json.loads(vector_meta_path.read_text(encoding="utf-8")).get("dimension")
        if vector_meta_path.exists() else dim
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "packet_id": packet_id,
        "built_at": datetime.now().isoformat(),
        "master_dir": str(master_dir.resolve()),
        "master_hashes": master_hash_snapshot,
        "source_code_version": _hash_rag_module_files(),
        "previous_packet_id": latest_packet_id,
        "chunk_count": len(chunks),
        "entity_count": len(entities),
        "bilingual_alias_groups": bilingual_count,
        "vector_dimension": int(vector_dim),
        "files": [
            {"path": f.path, "size": f.size, "sha256": f.sha256} for f in files
        ],
    }
    manifest_path = packet_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (packet_dir / "manifest.sha256").write_text(file_sha256(manifest_path), encoding="utf-8")

    gate = evaluate_publish_gate(packet, certified=certified)
    promote = gate.publishable or not require_publishable
    if require_publishable and not gate.publishable:
        promote = False
        print(f"[GovernedBuild] [{packet_id}] NOT publishable — latest pointer left unchanged")
        for err in gate.errors:
            print(f"  ERROR: {err}")
    elif not gate.publishable:
        print(f"[GovernedBuild] [{packet_id}] WARN publish gate failed (pointer still updated; use --require-publishable to fail closed)")
        for err in gate.errors:
            print(f"  ERROR: {err}")
    for warn in gate.warnings:
        print(f"  WARN: {warn}")

    write_publish_receipt(
        packet_dir,
        gate,
        packet_id=packet_id,
        master_hashes=master_hash_snapshot,
        previous_packet_id=latest_packet_id,
        promoted=promote,
    )

    ASSET_PACKETS_ROOT.mkdir(parents=True, exist_ok=True)
    if promote:
        write_latest_pointer(packet_id, manifest_path)
    print(
        f"[GovernedBuild] [{packet_id}] DONE -> {packet_dir}\n"
        f"                  chunks={len(chunks)} entities={len(entities)} "
        f"bilingual_groups={bilingual_count} vector_dim={vector_dim} "
        f"publishable={gate.publishable} promoted={promote} certified={gate.certified}"
    )
    if require_publishable and not gate.publishable:
        raise RuntimeError(
            f"governed packet {packet_id} failed publish gate: {'; '.join(gate.errors)}"
        )
    return packet_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master-dir", type=Path, default=DEFAULT_MASTER_DIR)
    ap.add_argument("--force", action="store_true", help="Rebuild even when master files match latest")
    ap.add_argument(
        "--require-publishable",
        action="store_true",
        help="Fail closed: do not update latest pointer when publish gate errors",
    )
    ap.add_argument(
        "--certified",
        action="store_true",
        help="Mark receipt certified=true only when the publish gate also passes",
    )
    args = ap.parse_args()

    try:
        build(
            args.master_dir,
            force=args.force,
            require_publishable=args.require_publishable,
            certified=args.certified,
        )
    except RuntimeError as exc:
        print(f"[GovernedBuild] FAIL: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
