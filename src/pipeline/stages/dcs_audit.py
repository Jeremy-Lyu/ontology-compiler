#!/usr/bin/env python3
"""Dimension Coverage Score Audit (步骤3: 维度覆盖度审计)

Computes 6-dimension coverage score for each entity:
  DCS = (S_str + S_sem + S_struct + E_evid + T_temp + R_rel) / 6

Each dimension is binary (0 or 1):
- S_str: Has unique canonical ID + aliases
- S_sem: Has L0-L3 meta_class/layer binding
- S_struct: Has parent_id/parent_class or BOM attachment
- E_evid: Has evidence with source_text, source_file, confidence
- T_temp: Has temporal info (valid_from, valid_to, version, timestamp)
- R_rel: Has at least 1 relationship (incoming or outgoing)

When DCS < 0.67, generates back-filling tasks for missing dimensions.
"""
from __future__ import annotations

import re
import time
from argparse import Namespace
from collections import Counter as _IdCounter
from datetime import datetime
from pathlib import Path
from typing import Any

from .common import (
    StageResult,
    _entity_layer_value,
    _iter_master_entities,
    _read_json,
    _write_json,
)

try:
    from src.pipeline.storage.master_store import MasterOntologyStore
except ImportError:
    MasterOntologyStore = None  # type: ignore[assignment]

try:
    from core.utils.relationship_helpers import relationship_endpoints
except ImportError:
    def relationship_endpoints(rel: dict[str, Any]) -> tuple[str, str]:
        if not isinstance(rel, dict):
            return ("", "")
        source_id = str(rel.get("source") or rel.get("from") or "").strip()
        target_id = str(rel.get("target") or rel.get("to") or "").strip()
        return source_id, target_id


DCS_THRESHOLD = 0.67
VALID_LAYERS = {"L0", "L1", "L2", "L2.5", "L3"}

# Keys that count as temporal coverage (checked at top-level / evidence / metadata / provenance).
_TEMPORAL_KEYS = (
    "valid_from", "valid_from_date", "valid_to", "valid_to_date",
    "validfrom", "validuntil", "valid_until",
    "effective_date", "effectivity", "effectivity_date", "expiry", "expires_at",
    "version", "version_id", "timestamp", "created_at", "updated_at", "extracted_at",
)

# run_id values often embed a timestamp such as "MAIN-20260729-141633".
_DATE_PATTERN = re.compile(r"\d{4}\d{2}\d{2}|\d{4}-\d{2}-\d{2}")

# Attribute names that indicate a BOM / org-tree attachment.
_BOM_ATTR_NAMES = {
    "bom", "bom_id", "bom_path", "assembly", "assembly_id",
    "parent", "parent_id", "parent_class", "org_unit", "org_id",
}


def _entity_id(entity: dict[str, Any]) -> str:
    return str(entity.get("meta_id") or entity.get("entity_id") or entity.get("id") or "")


def _entity_name(entity: dict[str, Any]) -> str:
    return str(entity.get("name") or entity.get("meta_name") or _entity_id(entity) or "")


def _evidence_dict(entity: dict[str, Any]) -> dict[str, Any]:
    ev = entity.get("evidence")
    return ev if isinstance(ev, dict) else {}


def _has_aliases(entity: dict[str, Any]) -> bool:
    aliases = entity.get("aliases")
    if isinstance(aliases, list):
        return any(str(a).strip() for a in aliases if a)
    if isinstance(aliases, dict):
        return len(aliases) > 0
    if isinstance(aliases, str):
        return bool(aliases.strip())
    return False


def _score_string_dimension(entity: dict[str, Any]) -> tuple[bool, str]:
    meta_id = _entity_id(entity)
    if not meta_id:
        return (False, "missing canonical meta_id")
    # `_dcs_meta_id_is_unique` is a transient flag set by run_dcs_audit based on
    # global meta_id counts (defaults to True when not set).
    if not entity.get("_dcs_meta_id_is_unique", True):
        return (False, f"meta_id '{meta_id}' is not unique (shared by multiple entities)")
    name = entity.get("name") or entity.get("meta_name")
    has_name = bool(name and str(name).strip())
    if has_name:
        return (True, f"unique meta_id '{meta_id}' with name")
    if _has_aliases(entity):
        return (True, f"unique meta_id '{meta_id}' with aliases")
    return (False, f"meta_id '{meta_id}' present but missing name and aliases")


def _score_semantic_dimension(entity: dict[str, Any]) -> tuple[bool, str]:
    meta_class = str(entity.get("meta_class") or "").strip()
    if not meta_class or meta_class.upper() == "UNKNOWN":
        return (False, f"meta_class is missing or UNKNOWN ('{meta_class}')")
    layer = str(_entity_layer_value(entity)).strip().upper()
    if layer not in VALID_LAYERS:
        return (False, f"layer '{layer}' not in L0-L3 binding")
    return (True, f"meta_class='{meta_class}', layer={layer}")


def _score_structural_dimension(entity: dict[str, Any]) -> tuple[bool, str]:
    layer = str(_entity_layer_value(entity)).strip().upper()
    if layer == "L0":
        return (True, "L0 root entity exempt from parent requirement")
    parent_value = (
        entity.get("parent_id")
        or entity.get("parent")
        or entity.get("parent_class")
    )
    if parent_value and str(parent_value).strip():
        return (True, f"has parent binding '{parent_value}'")
    for key in ("bom_id", "assembly_id", "bom_path", "org_unit", "org_id"):
        value = entity.get(key)
        if value and str(value).strip():
            return (True, f"attached via {key}='{value}'")
    attrs = entity.get("attributes")
    if isinstance(attrs, list):
        for attr in attrs:
            if not isinstance(attr, dict):
                continue
            aname = str(attr.get("name") or "").lower()
            if aname in _BOM_ATTR_NAMES and attr.get("value"):
                return (True, f"attached via attribute '{aname}'")
    return (False, "no parent_id/parent_class and no BOM/org attachment")


def _score_evidence_dimension(entity: dict[str, Any]) -> tuple[bool, str]:
    ev = _evidence_dict(entity)
    if not ev:
        return (False, "no evidence dict")
    source_text = (
        ev.get("source_text")
        or ev.get("content")
        or ev.get("citation")
        or ev.get("text")
    )
    if not source_text or not str(source_text).strip():
        return (False, "evidence missing source_text/citation")
    confidence = ev.get("confidence")
    try:
        conf_val = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        conf_val = None
    if conf_val is None:
        return (False, "evidence missing confidence value")
    if conf_val < 0.5:
        return (False, f"evidence confidence {conf_val:.2f} below 0.5")
    return (True, f"source_text present, confidence={conf_val:.2f}")


def _score_temporal_dimension(entity: dict[str, Any]) -> tuple[bool, str]:
    sources = [
        entity,
        _evidence_dict(entity),
        entity.get("metadata") if isinstance(entity.get("metadata"), dict) else {},
        entity.get("provenance") if isinstance(entity.get("provenance"), dict) else {},
    ]
    for src in sources:
        if not isinstance(src, dict):
            continue
        for key in _TEMPORAL_KEYS:
            value = src.get(key)
            if value and str(value).strip():
                return (True, f"has temporal field '{key}'")
    # Fall back to a timestamp embedded in run_id (e.g. provenance.run_id).
    provenance = entity.get("provenance")
    run_id = ""
    if isinstance(provenance, dict):
        run_id = str(provenance.get("run_id") or "")
    if not run_id:
        run_id = str(entity.get("run_id") or "")
    if run_id and _DATE_PATTERN.search(run_id):
        return (True, f"temporal info inferred from run_id '{run_id}'")
    return (False, "no valid_from/valid_to/version/timestamp found")


def _score_relationship_dimension(entity_id: str, relationship_set: set[str]) -> tuple[bool, str]:
    if not entity_id:
        return (False, "entity has no id, cannot match relationships")
    if entity_id in relationship_set:
        return (True, "entity participates in at least 1 relationship")
    return (False, "entity has no incoming or outgoing relationships")


def _compute_dcs_score(
    entity: dict[str, Any],
    rel_source_set: set[str],
    rel_target_set: set[str],
) -> dict[str, Any]:
    entity_id = _entity_id(entity)
    rel_participants = rel_source_set | rel_target_set

    checks = (
        ("S_str", lambda: _score_string_dimension(entity)),
        ("S_sem", lambda: _score_semantic_dimension(entity)),
        ("S_struct", lambda: _score_structural_dimension(entity)),
        ("E_evid", lambda: _score_evidence_dimension(entity)),
        ("T_temp", lambda: _score_temporal_dimension(entity)),
        ("R_rel", lambda: _score_relationship_dimension(entity_id, rel_participants)),
    )

    dimensions: dict[str, dict[str, Any]] = {}
    for name, func in checks:
        passed, reason = func()
        dimensions[name] = {"score": 1 if passed else 0, "reason": reason}

    total = sum(d["score"] for d in dimensions.values())
    dcs_score = round(total / 6.0, 4)
    missing = [name for name, d in dimensions.items() if d["score"] == 0]

    return {
        "entity_id": entity_id,
        "entity_name": _entity_name(entity),
        "layer": str(_entity_layer_value(entity)),
        "dcs_score": dcs_score,
        "dimensions": dimensions,
        "missing_dimensions": missing,
        "needs_backfill": dcs_score < DCS_THRESHOLD,
    }


_BACKFILL_ACTIONS = {
    "S_str": "Re-extract canonical meta_id and standard aliases for entity",
    "S_sem": "Assign L0-L3 meta_class and layer binding",
    "S_struct": "Link entity to parent_id or attach to BOM/org tree",
    "E_evid": "Re-extract source_text citation, source_file and confidence from source documents",
    "T_temp": "Search for effectivity date or version in source documents",
    "R_rel": "Establish at least one incoming or outgoing relationship for entity",
}


def _generate_backfill_tasks(dcs_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for result in dcs_results:
        if not result.get("needs_backfill"):
            continue
        entity_id = result.get("entity_id", "")
        for missing in result.get("missing_dimensions", []):
            tasks.append({
                "entity_id": entity_id,
                "entity_name": result.get("entity_name", ""),
                "missing": missing,
                "action": _BACKFILL_ACTIONS.get(
                    missing, f"Back-fill missing dimension {missing}"
                ),
            })
    return tasks


async def run_dcs_audit(args: Namespace) -> StageResult:
    """Main entry point for the Dimension Coverage Score Audit stage."""
    start = time.time()
    threshold = float(getattr(args, "dcs_threshold", DCS_THRESHOLD) or DCS_THRESHOLD)

    master_path = Path(args.master_path)
    if not master_path.exists():
        return StageResult(
            "dcs_audit",
            False,
            f"master not found: {master_path}",
            elapsed_seconds=round(time.time() - start, 3),
        )

    if MasterOntologyStore:
        try:
            store = MasterOntologyStore(str(master_path))
            master = store.read()
        except Exception:
            master = _read_json(master_path, {})
    else:
        master = _read_json(master_path, {})

    all_entities = _iter_master_entities(master)
    all_entities = [
        e for e in all_entities
        if isinstance(e, dict) and _entity_layer_value(e) in VALID_LAYERS
    ]

    relationships = master.get("relationships") or master.get("edges") or []

    rel_source_set: set[str] = set()
    rel_target_set: set[str] = set()
    for rel in relationships:
        if not isinstance(rel, dict):
            continue
        source_id, target_id = relationship_endpoints(rel)
        if source_id:
            rel_source_set.add(source_id)
        if target_id:
            rel_target_set.add(target_id)

    # Global meta_id uniqueness — applied via a transient flag on shallow copies
    # so the original master dicts are never mutated.
    id_counts: _IdCounter[str] = _IdCounter(
        _entity_id(e) for e in all_entities if _entity_id(e)
    )
    score_entities: list[dict[str, Any]] = []
    for e in all_entities:
        copy = dict(e)  # shallow copy; nested dicts are only read
        copy["_dcs_meta_id_is_unique"] = (id_counts.get(_entity_id(copy), 0) <= 1)
        score_entities.append(copy)

    dcs_results = [
        _compute_dcs_score(e, rel_source_set, rel_target_set) for e in score_entities
    ]

    backfill_tasks = _generate_backfill_tasks(dcs_results)

    total_entities = len(dcs_results)
    avg_dcs = (
        round(sum(r["dcs_score"] for r in dcs_results) / total_entities, 4)
        if total_entities else 0.0
    )
    entities_below_threshold = sum(1 for r in dcs_results if r["needs_backfill"])

    dimension_coverage: dict[str, dict[str, Any]] = {}
    for dim in ("S_str", "S_sem", "S_struct", "E_evid", "T_temp", "R_rel"):
        passed = sum(
            1 for r in dcs_results
            if r["dimensions"].get(dim, {}).get("score", 0) == 1
        )
        dimension_coverage[dim] = {
            "passed": passed,
            "total": total_entities,
            "coverage": round(passed / total_entities, 4) if total_entities else 0.0,
        }

    report = {
        "generated_at": datetime.now().isoformat(),
        "stage": "dcs_audit",
        "master_path": str(master_path),
        "threshold": threshold,
        "total_entities": total_entities,
        "avg_dcs": avg_dcs,
        "entities_below_threshold": entities_below_threshold,
        "backfill_task_count": len(backfill_tasks),
        "dimension_coverage": dimension_coverage,
        "entity_scores": dcs_results,
    }

    output_dir = Path(args.output_dir)
    report_path = output_dir / "dcs_audit_report.json"
    backfill_path = output_dir / "backfill_tasks.json"
    _write_json(report_path, report)
    _write_json(backfill_path, backfill_tasks)

    message = (
        f"dcs_audit completed (avg_dcs={avg_dcs:.3f}, "
        f"{entities_below_threshold}/{total_entities} entities below {threshold})"
    )

    return StageResult(
        "dcs_audit",
        True,
        message,
        {
            "total_entities": total_entities,
            "avg_dcs": avg_dcs,
            "entities_below_threshold": entities_below_threshold,
            "backfill_task_count": len(backfill_tasks),
            "threshold": threshold,
            "dimension_coverage": dimension_coverage,
            "dcs_audit_report_path": str(report_path),
            "backfill_tasks_path": str(backfill_path),
        },
        round(time.time() - start, 3),
        artifact_paths=[str(report_path), str(backfill_path)],
    )
