"""Single source of truth for the **governed RAG asset packet**.

Phase B hard constraint: every RAG consumer (retrieval evaluation,
HybridIndexManager answer path, LocalRAGAdapter, rag_answerer) MUST read from
the governed packet directory published here.  Legacy side-channels
(``outputs/runs/ontology/rag/*.jsonl`` written by earlier pipeline stages)
are explicitly unauthorised unless ``RAG_ALLOW_LEGACY_DIR=1`` is set.

Contract
--------
A governed packet is a directory under ``asset_packets/governed`` named by
its build timestamp.  It contains::

    chunks.jsonl          Canonical chunk payload (derived from adapter+derive,
                          with bilingual aliases merged into keywords).
    inverted_index.json   BM25 keyword index built over chunks.jsonl.
    idf.json              IDF table for BM25 (built at the same time).
    vector_index/         VectorIndexManager payload (vectors.npz, chunk_ids.json,
                          metadata.json, vocab.json) - dimension big enough to
                          cover the whole bigram vocabulary.
    packet.json           Full ``OntologyPacket`` (entities / buckets /
                          relations / evidence_bundles) for downstream answers.
    manifest.json         Provenance + shape + hashes for every file above,
                          plus the source master_dir path and its nodes/rels
                          file hashes so rebuilds can detect stale assets.
    manifest.sha256       Overall checksum for manifest.json itself.

Only the packet referenced by ``latest.json`` (``{asset_packets}/governed/``
sibling) is considered *current*; older timestamped dirs exist for audit.

"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.domain.errors import DataContractError

__all__ = [
    "ROOT",
    "ASSET_PACKETS_ROOT",
    "GOVERNED_DIR",
    "LATEST_POINTER",
    "LEGACY_RAG_DIR",
    "DEFAULT_MASTER_DIR",
    "SCHEMA_VERSION",
    "ALLOW_LEGACY",
    "FileManifest",
    "GovernedPacketManifest",
    "file_sha256",
    "json_bytes_sha256",
    "build_file_manifest",
    "default_governed_rag_dir",
    "resolve_governed_rag_dir",
    "require_governed_packet",
    "read_latest_manifest",
    "write_latest_pointer",
]

ROOT = Path(__file__).resolve().parents[3]  # project root
# Outputs mirror the other pipeline products (regression runs, master ontology).
ASSET_PACKETS_ROOT = ROOT / "outputs" / "runs" / "ontology" / "asset_packets"
GOVERNED_DIR = ASSET_PACKETS_ROOT / "governed"
LATEST_POINTER = ASSET_PACKETS_ROOT / "latest_governed_packet.json"
# Unauthorised legacy directory.  Consumers must not default to this path.
LEGACY_RAG_DIR = ROOT / "outputs" / "runs" / "ontology" / "rag"
DEFAULT_MASTER_DIR = ROOT / "outputs" / "runs" / "ontology" / "master" / "master_ontology"

SCHEMA_VERSION = "1.0"

# Escape hatch for tests / local scratch runs.  NEVER set this in CI/CD.
ALLOW_LEGACY: bool = os.environ.get("RAG_ALLOW_LEGACY_DIR", "0") in {"1", "true", "True", "yes"}
GOVERNED_ONLY: bool = os.environ.get("RAG_GOVERNED_ONLY", "1") not in {"0", "false", "False", "no"}

_GOVERNED_REQUIRED = (
    "chunks.jsonl",
    "inverted_index.json",
    "idf.json",
    "vector_index/metadata.json",
    "packet.json",
    "manifest.json",
    "manifest.sha256",
)


@dataclass
class FileManifest:
    path: str
    size: int
    sha256: str


@dataclass
class GovernedPacketManifest:
    schema_version: str
    packet_id: str
    built_at: str
    master_dir: str
    master_hashes: Dict[str, str]
    source_code_version: Dict[str, str]
    files: List[FileManifest]
    chunk_count: int
    entity_count: int
    bilingual_alias_groups: int
    vector_dimension: int


def file_sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def json_bytes_sha256(obj: Any) -> str:
    """Deterministic sha256 over a JSON-serializable object."""
    return hashlib.sha256(
        json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_file_manifest(root: Path, files: Iterable[str | Path]) -> List[FileManifest]:
    out: List[FileManifest] = []
    for rel in files:
        rel_p = Path(rel)
        abs_p = root / rel_p
        if not abs_p.exists():
            raise DataContractError(
                f"[GovernedPacket] missing governed file when building manifest: {abs_p}"
            )
        out.append(
            FileManifest(
                path=rel_p.as_posix(),
                size=abs_p.stat().st_size,
                sha256=file_sha256(abs_p),
            )
        )
    return out


def default_governed_rag_dir() -> Path:
    """Return the *current* governed rag_dir (vector_index lives under it)."""
    pointer = read_latest_pointer()
    if pointer is None:
        raise DataContractError(
            "[GovernedPacket] latest_governed_packet.json is missing; run "
            "`python scripts/build_governed_packet.py` before any RAG consumer."
        )
    packet_dir = GOVERNED_DIR / pointer["packet_id"]
    require_governed_packet(packet_dir)
    return packet_dir


def resolve_governed_rag_dir(requested: Optional[Path] = None) -> Path:
    """Resolve the authoritative rag_dir for consumers.

    * If ``requested`` is provided **and points inside GOVERNED_DIR**, use it
      (validation still runs).
    * If ``requested`` points to the legacy directory:
        - raise DataContractError unless ``RAG_ALLOW_LEGACY_DIR=1``.
    * Otherwise fall back to ``default_governed_rag_dir()``.
    """
    if requested is None:
        return default_governed_rag_dir()

    requested_abs = Path(requested).resolve()
    try:
        governed_parent = GOVERNED_DIR.resolve()
        inside_governed = requested_abs == governed_parent or requested_abs.is_relative_to(
            governed_parent
        )
    except ValueError:  # pathlib edge case for different drives
        inside_governed = False

    if inside_governed:
        # Latest pointer may not match explicit subdir - allow per-run audit.
        if (requested_abs / "manifest.json").exists():
            require_governed_packet(requested_abs)
            return requested_abs
        # Fall back if caller supplied governed_dir itself instead of packet dir.
        return default_governed_rag_dir()

    legacy = LEGACY_RAG_DIR.resolve()
    try:
        inside_legacy = requested_abs == legacy or requested_abs.is_relative_to(legacy)
    except ValueError:
        inside_legacy = False
    if inside_legacy and GOVERNED_ONLY and not ALLOW_LEGACY:
        raise DataContractError(
            "[GovernedPacket] HARD CONSTRAINT: legacy rag_dir is forbidden "
            f"({requested}).  All RAG retrieval must read the governed packet "
            f"at {GOVERNED_DIR}.  Build with `scripts/build_governed_packet.py`, "
            "or set RAG_ALLOW_LEGACY_DIR=1 ONLY for local scratch runs (never CI)."
        )
    if inside_legacy and ALLOW_LEGACY:
        return requested_abs
    # Unknown path - go governed, let the require fail loudly.
    return default_governed_rag_dir()


def require_governed_packet(packet_dir: Path) -> GovernedPacketManifest:
    """Validate the governed packet and return its manifest.

    Hard-fail if any required file is missing or the overall checksum is wrong.
    """
    missing = [name for name in _GOVERNED_REQUIRED if not (packet_dir / name).exists()]
    if missing:
        raise DataContractError(
            "[GovernedPacket] HARD CONSTRAINT: governed packet missing required files "
            f"in {packet_dir}: {missing}.  Run `python scripts/build_governed_packet.py`."
        )
    manifest_path = packet_dir / "manifest.json"
    checksum_path = packet_dir / "manifest.sha256"
    expected_checksum = checksum_path.read_text(encoding="utf-8").strip()
    actual_checksum = file_sha256(manifest_path)
    if actual_checksum != expected_checksum:
        raise DataContractError(
            f"[GovernedPacket] manifest checksum mismatch in {packet_dir}: "
            f"expected {expected_checksum}, got {actual_checksum}.  "
            "Packet may be corrupt; rebuild."
        )
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = [FileManifest(**item) for item in raw["files"]]
    for fm in files:
        abs_path = packet_dir / fm.path
        actual = file_sha256(abs_path)
        if actual != fm.sha256:
            raise DataContractError(
                f"[GovernedPacket] {fm.path} sha256 mismatch in {packet_dir}: "
                f"expected {fm.sha256}, got {actual}."
            )
    return GovernedPacketManifest(
        schema_version=raw["schema_version"],
        packet_id=raw["packet_id"],
        built_at=raw["built_at"],
        master_dir=raw["master_dir"],
        master_hashes=dict(raw["master_hashes"]),
        source_code_version=dict(raw.get("source_code_version") or {}),
        files=files,
        chunk_count=int(raw["chunk_count"]),
        entity_count=int(raw["entity_count"]),
        bilingual_alias_groups=int(raw.get("bilingual_alias_groups") or 0),
        vector_dimension=int(raw.get("vector_dimension") or 0),
    )


def read_latest_pointer() -> Optional[Dict[str, Any]]:
    if not LATEST_POINTER.exists():
        return None
    raw = json.loads(LATEST_POINTER.read_text(encoding="utf-8"))
    return raw


def write_latest_pointer(packet_id: str, manifest_path: Path) -> None:
    ASSET_PACKETS_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "packet_id": packet_id,
        "manifest": str(manifest_path.relative_to(ROOT)) if manifest_path.is_absolute() else str(manifest_path),
        "updated_at": datetime.now().isoformat(),
        "schema_version": SCHEMA_VERSION,
    }
    LATEST_POINTER.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# typing helper re-export
from typing import Iterable as _Iterable  # noqa: E402  (used above via string)
