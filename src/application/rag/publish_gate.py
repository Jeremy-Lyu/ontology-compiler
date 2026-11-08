"""Fail-closed publishability checks + publish receipt for governed packets.

Mirrors Auto Ontology's incomplete-catalog fail-closed habit and lightweight
certification: a packet may be built for audit, but the latest pointer should
not advance when structural gates fail (unless explicitly forced).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.application.rag.governed_packets import file_sha256, json_bytes_sha256
from src.application.rag.semantic_surface import flatten_entities

__all__ = [
    "PublishGateResult",
    "evaluate_publish_gate",
    "write_publish_receipt",
    "read_publish_receipt",
]

_REQUIRED_LAYERS_DEFAULT = ("L0", "L1", "L2")
# Instance-like identifiers that should not live on abstract layers.
_INSTANCE_CODE = re.compile(
    r"(?i)^(?:pn[-_]?|sku[-_]?|vin[-_]?|sn[-_]?)[a-z0-9._-]{3,}$"
    r"|^(?:[A-Z]{1,5}[-_]?\d{4,}|\d{6,})[A-Z0-9._-]*$"
)
_ABSTRACT_LAYERS = frozenset({"L0", "L1", "l0", "l1"})


@dataclass
class PublishGateResult:
    publishable: bool
    certified: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _required_field_ok(ent: Dict[str, Any]) -> bool:
    has_id = bool(ent.get("id") or ent.get("meta_id"))
    has_name = bool(ent.get("name") or ent.get("meta_name"))
    has_layer = bool(ent.get("layer"))
    has_type = bool(ent.get("type") or ent.get("meta_class"))
    return has_id and has_name and has_layer and has_type


def _looks_like_instance_code(ent: Dict[str, Any]) -> bool:
    code = str(ent.get("code") or "").strip()
    name = str(ent.get("name") or ent.get("meta_name") or "").strip()
    if code and _INSTANCE_CODE.match(code):
        return True
    if name and _INSTANCE_CODE.match(name):
        return True
    return False


def evaluate_publish_gate(
    packet: Dict[str, Any],
    *,
    required_layers: Sequence[str] = _REQUIRED_LAYERS_DEFAULT,
    max_missing_field_ratio: float = 0.15,
    allow_empty_l3: bool = True,
    certified: bool = False,
) -> PublishGateResult:
    """Structural gates before promoting a governed packet to *latest*."""
    errors: List[str] = []
    warnings: List[str] = []
    entities = flatten_entities(packet)
    relations = [r for r in (packet.get("relations") or []) if isinstance(r, dict)]

    by_layer: Dict[str, int] = {}
    for ent in entities:
        layer = str(ent.get("layer") or "unknown")
        by_layer[layer] = by_layer.get(layer, 0) + 1

    if not entities:
        errors.append("packet has zero entities")

    for layer in required_layers:
        if by_layer.get(layer, 0) <= 0:
            errors.append(f"required layer '{layer}' is empty")

    if not allow_empty_l3 and by_layer.get("L3", 0) <= 0:
        warnings.append("L3 instance layer is empty")

    missing_fields = [e for e in entities if not _required_field_ok(e)]
    missing_ratio = (len(missing_fields) / len(entities)) if entities else 1.0
    if missing_ratio > max_missing_field_ratio:
        errors.append(
            f"field completeness fail: {len(missing_fields)}/{len(entities)} "
            f"entities missing id/name/layer/type (ratio={missing_ratio:.2f} > {max_missing_field_ratio})"
        )
    elif missing_fields:
        warnings.append(
            f"{len(missing_fields)} entities missing required fields (within tolerance)"
        )

    misplaced = [
        e
        for e in entities
        if str(e.get("layer") or "") in _ABSTRACT_LAYERS and _looks_like_instance_code(e)
    ]
    if misplaced:
        sample = ", ".join(
            str(e.get("id") or e.get("name")) for e in misplaced[:5]
        )
        errors.append(
            f"{len(misplaced)} instance-like part codes on L0/L1 (must be L3): {sample}"
        )

    provenance_counts: Dict[str, int] = {}
    for rel in relations:
        prov = str(rel.get("provenance") or "unknown")
        provenance_counts[prov] = provenance_counts.get(prov, 0) + 1

    # Certification is never automatic from this gate.
    if certified and errors:
        errors.append("certified=true rejected because publish gate has errors")
        certified = False

    stats = {
        "entity_count": len(entities),
        "relation_count": len(relations),
        "layer_counts": by_layer,
        "missing_required_field_count": len(missing_fields),
        "missing_required_field_ratio": round(missing_ratio, 4),
        "misplaced_instance_codes": len(misplaced),
        "relation_provenance": provenance_counts,
    }
    return PublishGateResult(
        publishable=not errors,
        certified=bool(certified) and not errors,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )


def write_publish_receipt(
    packet_dir: Path,
    gate: PublishGateResult,
    *,
    packet_id: str,
    master_hashes: Optional[Dict[str, str]] = None,
    previous_packet_id: Optional[str] = None,
    promoted: bool = False,
) -> Path:
    """Write ``publish_receipt.json`` (+ sha256) beside the governed packet."""
    packet_path = packet_dir / "packet.json"
    manifest_path = packet_dir / "manifest.json"
    receipt = {
        "schema_version": "1.0",
        "packet_id": packet_id,
        "built_at": datetime.now().isoformat(),
        "promoted_to_latest": bool(promoted),
        "publishable": gate.publishable,
        "certified": gate.certified,
        "errors": list(gate.errors),
        "warnings": list(gate.warnings),
        "stats": dict(gate.stats),
        "master_hashes": dict(master_hashes or {}),
        "previous_packet_id": previous_packet_id,
        "artifacts": {
            "packet.json": file_sha256(packet_path) if packet_path.exists() else "",
            "manifest.json": file_sha256(manifest_path) if manifest_path.exists() else "",
        },
    }
    receipt["receipt_sha256"] = json_bytes_sha256(
        {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    )
    out = packet_dir / "publish_receipt.json"
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    (packet_dir / "publish_receipt.sha256").write_text(
        receipt["receipt_sha256"] + "\n", encoding="utf-8"
    )
    return out


def read_publish_receipt(packet_dir: Path) -> Dict[str, Any]:
    path = Path(packet_dir) / "publish_receipt.json"
    if not path.exists():
        raise FileNotFoundError(f"publish_receipt.json missing in {packet_dir}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    checksum_path = Path(packet_dir) / "publish_receipt.sha256"
    if checksum_path.exists():
        expected = checksum_path.read_text(encoding="utf-8").strip()
        body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
        actual = json_bytes_sha256(body)
        # Prefer embedded field if present; fall back to recomputation.
        embedded = str(receipt.get("receipt_sha256") or "")
        if embedded and embedded != expected:
            raise ValueError(
                f"publish_receipt checksum mismatch: file={expected} embedded={embedded}"
            )
        if actual != expected and actual != embedded:
            raise ValueError(
                f"publish_receipt content drift: expected={expected} recomputed={actual}"
            )
    return receipt
