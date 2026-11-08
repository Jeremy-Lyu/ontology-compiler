"""CQ ↔ pipeline thin integration layer (M2 产物先行).

CQ_PIPELINE_IMPLEMENTATION_PLAN.md §5/§9-M2:

- preflight: load the catalog → ``cq_loaded_summary`` (skip when absent)
- audit: read-only executability check of frozen CQs → ``cq_validation_report.json``
- summarize: CQ review sheet (回译确认) → ``cq_review_sheet.md``

零回归约束：目录不存在 / 无 frozen CQ → 所有函数返回 ``None``，stage 不写
任何文件、不改任何输出（行为与无 CQ 时逐字节一致）。本层吞掉所有异常——
CQ 是增量增强，永远不能弄挂主线。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.application.cq.cq_artifacts import (
    write_cq_bundle,
    write_cq_catalog,
    write_cq_extract_coverage,
    write_cq_gap_report,
    write_cq_review_sheet,
    write_cq_validation_report,
)
from src.application.cq.cq_models import CQ, CQCatalog, CQValidationResult
from src.application.cq.cq_registry import CQRegistry, default_catalog_path, load_cq_mapping

_CQ_ARTIFACT_DIR = "cq"


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


def preflight_cq_summary(
    catalog_path: Union[str, Path, None] = None,
) -> Optional[Dict[str, Any]]:
    """Catalog summary for preflight; ``None`` when there is no usable catalog.

    A catalog that exists but is empty (or fully invalid) counts as "no CQ
    input" so empty-config runs stay byte-identical to the pre-CQ baseline.
    """
    try:
        path = Path(catalog_path) if catalog_path else default_catalog_path()
        registry = CQRegistry.load(path)
        if len(registry) == 0:
            return None
        status_counts: Dict[str, int] = {}
        for item in registry.items:
            status_counts[item.status] = status_counts.get(item.status, 0) + 1
        return {
            "available": True,
            "loaded_count": len(registry),
            "frozen_count": len(registry.frozen()),
            "by_status": status_counts,
            "source": str(path),
            "warnings": registry.warnings,
        }
    except Exception:  # noqa: BLE001 - CQ must never break preflight
        return None


def preflight_cq_catalog_artifact(
    artifact_root: Union[str, Path, None],
    catalog_path: Union[str, Path, None] = None,
) -> Optional[str]:
    """Write the per-run ``cq_catalog.json`` snapshot; ``None`` when no catalog."""
    try:
        if not artifact_root:
            return None
        path = Path(catalog_path) if catalog_path else default_catalog_path()
        registry = CQRegistry.load(path)
        if len(registry) == 0:
            return None
        written = write_cq_catalog(CQCatalog(items=registry.items), artifact_root)
        return str(written)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# extract (M4: CQ 关注词表 coverage + 漏抽提示; read-only, never scores
# candidates itself — 关键约束: CQ 只做约束/提示，不替代抽取器主逻辑)
# ---------------------------------------------------------------------------


def extract_cq_coverage(
    nodes: List[Dict[str, Any]],
    instances: Optional[List[Dict[str, Any]]] = None,
    relationships: Optional[List[Dict[str, Any]]] = None,
    catalog_path: Union[str, Path, None] = None,
) -> Optional[Dict[str, Any]]:
    """Compute CQ focus-vocabulary coverage over an extract fragment.

    Returns ``None`` when there is no CQ catalog / no active (non-deprecated)
    CQ, so extract stays byte-identical to the pre-CQ baseline. Matching is
    lenient containment (same M2/M3 policy). The result is pure statistics
    plus a deduped missing-term hint list; it never mutates fragment data or
    candidate scores.
    """
    try:
        path = Path(catalog_path) if catalog_path else default_catalog_path()
        registry = CQRegistry.load(path)
        active = registry.active()
        if not active:
            return None

        vocab = registry.focus_vocabulary()
        names, rel_types, attr_names = _master_vocabulary(
            {"nodes": list(nodes or []), "instances": list(instances or []),
             "relationships": list(relationships or [])}
        )

        per_cq: List[Dict[str, Any]] = []
        for cq in active:
            matched_entities = [e for e in cq.entities if _entity_in_master(e, names)]
            missing_entities = [e for e in cq.entities if not _entity_in_master(e, names)]
            matched_relations = [r for r in cq.relations if _relation_in_master(r, rel_types)]
            missing_relations = [r for r in cq.relations if not _relation_in_master(r, rel_types)]
            matched_attributes = [a for a in cq.attributes if _attribute_in_master(a, attr_names)]
            missing_attributes = [a for a in cq.attributes if not _attribute_in_master(a, attr_names)]
            denom_e = len(cq.entities) or 1
            denom_r = len(cq.relations) or 1
            per_cq.append({
                "cq_id": cq.cq_id,
                "status": cq.status,
                "entity_coverage": round(len(matched_entities) / denom_e, 4),
                "matched_entities": matched_entities,
                "missing_entities": missing_entities,
                "relation_coverage": round(len(matched_relations) / denom_r, 4),
                "matched_relations": matched_relations,
                "missing_relations": missing_relations,
                "matched_attributes": matched_attributes,
                "missing_attributes": missing_attributes,
            })

        focus_entities = vocab["entities"]
        focus_relations = vocab["relations"]
        focus_attributes = vocab["attributes"]
        present_entities = [t for t in focus_entities if _entity_in_master(t, names)]
        present_relations = [t for t in focus_relations if _relation_in_master(t, rel_types)]
        present_attributes = [t for t in focus_attributes if _attribute_in_master(t, attr_names)]
        missing_entities = [t for t in focus_entities if not _entity_in_master(t, names)]
        missing_relations = [t for t in focus_relations if not _relation_in_master(t, rel_types)]
        missing_attributes = [t for t in focus_attributes if not _attribute_in_master(t, attr_names)]

        def _ratio(present: List[str], total: List[str]) -> float:
            return round(len(present) / len(total), 4) if total else 1.0

        return {
            "focus_vocabulary": {
                "entity_count": len(focus_entities),
                "relation_count": len(focus_relations),
                "attribute_count": len(focus_attributes),
            },
            "cq_entity_coverage": _ratio(present_entities, focus_entities),
            "cq_relation_coverage": _ratio(present_relations, focus_relations),
            "cq_attribute_coverage": _ratio(present_attributes, focus_attributes),
            "present_focus_entities": present_entities,
            "missing_focus_entities": missing_entities,
            "present_focus_relations": present_relations,
            "missing_focus_relations": missing_relations,
            "missing_focus_attributes": missing_attributes,
            # 漏抽提示: term -> originating CQ ids (which business questions
            # the extraction could not answer yet)
            "missing_hints": [
                {"kind": kind, "term": term, "cq_ids": vocab["sources"].get(f"{kind}:{term.lower()}", [])}
                for kind, terms in (
                    ("entity", missing_entities),
                    ("relation", missing_relations),
                    ("attribute", missing_attributes),
                )
                for term in terms
            ],
            "per_cq": per_cq,
        }
    except Exception:  # noqa: BLE001 - CQ must never break extract
        return None


def extract_cq_coverage_artifact(
    coverage: Union[Dict[str, Any], None],
    artifact_root: Union[str, Path, None],
) -> Optional[str]:
    """Persist ``cq_extract_coverage.json`` from an :meth:`extract_cq_coverage` payload."""
    try:
        if not coverage or not artifact_root:
            return None
        written = write_cq_extract_coverage(coverage, artifact_root)
        return str(written)
    except Exception:  # noqa: BLE001
        return None


def extract_cq_stage_hint(
    nodes: List[Dict[str, Any]],
    instances: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    artifact_root: Union[str, Path, None],
) -> Dict[str, Any]:
    """Stage delegation helper (M4): compute coverage, persist artifact, and
    return the StageResult data keys. Empty dict when there is no CQ input —
    callers can ``stage_data.update(...)`` unconditionally (零回归)."""
    try:
        coverage = extract_cq_coverage(nodes, instances, relationships)
        if coverage is None:
            return {}
        data: Dict[str, Any] = {
            "cq_entity_coverage": coverage["cq_entity_coverage"],
            "cq_relation_coverage": coverage["cq_relation_coverage"],
            "cq_missing_hints": coverage["missing_hints"],
        }
        path = extract_cq_coverage_artifact(coverage, artifact_root)
        if path:
            data["cq_coverage_artifact"] = path
        return data
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------
# audit (read-only executability check)
# ---------------------------------------------------------------------------


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _entity_in_master(term: str, master_names: List[str]) -> bool:
    """Lenient containment match (CQ terms are business words, master names
    are model names) — the check is read-only/record-only, so generous
    matching avoids false alarms; real gap analysis lands in M3."""
    needle = _norm(term)
    if not needle:
        return True
    for name in master_names:
        haystack = _norm(name)
        if not haystack:
            continue
        if needle == haystack or needle in haystack or haystack in needle:
            return True
    return False


def _relation_in_master(term: str, master_relation_types: List[str]) -> bool:
    needle = _norm(term).upper().replace(" ", "_")
    if not needle:
        return True
    return any(needle == _norm(t).upper() for t in master_relation_types)


def _attribute_in_master(term: str, master_attribute_names: List[str]) -> bool:
    """Lenient containment match, same policy as entities (M2 note applies)."""
    needle = _norm(term)
    if not needle:
        return True
    for name in master_attribute_names:
        haystack = _norm(name)
        if not haystack:
            continue
        if needle == haystack or needle in haystack or haystack in needle:
            return True
    return False


def _master_vocabulary(master: Dict[str, Any]) -> tuple[List[str], List[str], List[str]]:
    """Return (entity names, relation types, attribute names) from a master doc."""
    entities = list(master.get("nodes") or []) + list(master.get("instances") or [])
    names: List[str] = []
    attr_names: List[str] = []
    for n in entities:
        if not isinstance(n, dict):
            continue
        names.append(str(n.get("meta_name") or n.get("name") or ""))
        for attr in n.get("attributes") or []:
            if isinstance(attr, dict):
                attr_names.append(str(attr.get("name") or ""))
            elif isinstance(attr, str):
                attr_names.append(attr)
    relations = master.get("relationships") or master.get("edges") or []
    rel_types: List[str] = []
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        rel_types.append(
            str(rel.get("type") or rel.get("relation_type") or rel.get("rel_type") or "")
        )
    return names, rel_types, attr_names


def parse_query_path(query_path: str) -> List[tuple[str, str]]:
    """Split a ``a -> rel -> b`` query path into (kind, term) segments.

    Position 0/2/4... are entities, 1/3/5... relations. Anything after ``|``
    (e.g. ``| filter: ...``) is not part of the chain and is dropped.
    """
    chain = str(query_path or "").split("|", 1)[0]
    segments = [seg.strip() for seg in chain.split("->") if seg.strip()]
    return [
        ("entity" if idx % 2 == 0 else "relation", seg)
        for idx, seg in enumerate(segments)
    ]


def audit_cq_validation(
    master: Dict[str, Any],
    catalog_path: Union[str, Path, None] = None,
) -> Optional[Dict[str, Any]]:
    """Read-only executability check over frozen CQs (M3 验证闭环).

    Per CQ, three read-only checks against the master vocabulary:
    1. entity / relation term presence (M2, lenient containment)
    2. attribute presence (M3: ``cq.attributes`` vs master attribute names)
    3. query_path chain walk (M3: per-segment break localization, e.g.
       ``position 3: relation 'affects' not found in master``)

    Returns ``None`` when there are no frozen CQs (caller skips the report).
    The check never mutates ``master`` and never blocks the audit stage.
    """
    try:
        path = Path(catalog_path) if catalog_path else default_catalog_path()
        registry = CQRegistry.load(path)
        frozen: List[CQ] = registry.frozen()
        if not frozen:
            return None

        names, rel_types, attr_names = _master_vocabulary(master or {})
        results: List[CQValidationResult] = []
        for cq in frozen:
            missing_entities = [
                e for e in cq.entities if not _entity_in_master(e, names)
            ]
            missing_relations = [
                r for r in cq.relations if not _relation_in_master(r, rel_types)
            ]
            missing_attributes = [
                a for a in cq.attributes if not _attribute_in_master(a, attr_names)
            ]
            chain_breaks = [
                f"position {idx}: {kind} '{term}' not found in master"
                for idx, (kind, term) in enumerate(parse_query_path(cq.query_path))
                if not (
                    _entity_in_master(term, names)
                    if kind == "entity"
                    else _relation_in_master(term, rel_types)
                )
            ]
            results.append(
                CQValidationResult(
                    cq_id=cq.cq_id,
                    executable=not (
                        missing_entities
                        or missing_relations
                        or missing_attributes
                        or chain_breaks
                    ),
                    missing_entities=missing_entities,
                    missing_relations=missing_relations,
                    missing_attributes=missing_attributes,
                    chain_breaks=chain_breaks,
                    notes="M3 read-only check (no gate)",
                )
            )
        return {
            "checked_count": len(results),
            "executable_count": sum(1 for r in results if r.executable),
            "results": [r.model_dump() for r in results],
        }
    except Exception:  # noqa: BLE001
        return None


def audit_cq_report_artifact(
    validation: Union[Dict[str, Any], None],
    artifact_root: Union[str, Path, None],
) -> Optional[str]:
    """Persist ``cq_validation_report.json`` from an :meth:`audit_cq_validation` payload."""
    try:
        if not validation or not artifact_root:
            return None
        results = [CQValidationResult.model_validate(r) for r in validation.get("results", [])]
        written = write_cq_validation_report(results, artifact_root)
        return str(written)
    except Exception:  # noqa: BLE001
        return None


def audit_cq_gap_report_artifact(
    validation: Union[Dict[str, Any], None],
    artifact_root: Union[str, Path, None],
    catalog_path: Union[str, Path, None] = None,
) -> Optional[str]:
    """Persist ``cq_gap_report.md`` (M3 验证闭环): human-readable gap report.

    ``None`` when there is no validation payload / no artifact root / no gap
    report writer failure — silent skip keeps audit byte-identical without CQ.
    """
    try:
        if not validation or not artifact_root:
            return None
        results = [CQValidationResult.model_validate(r) for r in validation.get("results", [])]
        path = Path(catalog_path) if catalog_path else default_catalog_path()
        registry = CQRegistry.load(path)
        mapping = load_cq_mapping()
        written = write_cq_gap_report(results, CQCatalog(items=registry.items), artifact_root, mapping=mapping)
        return str(written)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


def summarize_cq_review_artifact(
    artifact_root: Union[str, Path, None],
    catalog_path: Union[str, Path, None] = None,
    run_id: str = "cq-summary",
) -> Optional[Dict[str, Any]]:
    """Write the review sheet and register the ``cq`` ArtifactBundle.

    Reads back the validation report audit wrote (when present). Returns a
    small info dict for the summary payload, or ``None`` when there is no
    CQ input (caller skips — byte-identical behaviour).
    """
    try:
        if not artifact_root:
            return None
        path = Path(catalog_path) if catalog_path else default_catalog_path()
        registry = CQRegistry.load(path)
        if len(registry) == 0:
            return None

        catalog = CQCatalog(items=registry.items)
        report_path = Path(artifact_root) / _CQ_ARTIFACT_DIR / "cq_validation_report.json"
        results: List[CQValidationResult] = []
        if report_path.exists():
            import json

            payload = json.loads(report_path.read_text(encoding="utf-8"))
            results = [
                CQValidationResult.model_validate(r) for r in payload.get("results", [])
            ]

        sheet_path = write_cq_review_sheet(catalog, results, artifact_root)
        bundle_path = write_cq_bundle(
            artifact_root,
            run_id=run_id or "cq-summary",
            files={
                "cq_catalog": Path(artifact_root) / _CQ_ARTIFACT_DIR / "cq_catalog.json",
                "cq_extract_coverage": Path(artifact_root) / _CQ_ARTIFACT_DIR / "cq_extract_coverage.json",
                "cq_validation_report": report_path,
                "cq_gap_report": Path(artifact_root) / _CQ_ARTIFACT_DIR / "cq_gap_report.md",
                "cq_review_sheet": sheet_path,
            },
            metadata={"review_counts": {
                "total": len(catalog),
                "executable": sum(1 for r in results if r.executable),
                "unanswered": sum(1 for r in results if not r.executable),
                "pending_validation": max(0, len(catalog) - len(results)),
            }},
        )
        return {
            "total": len(catalog),
            "executable": sum(1 for r in results if r.executable),
            "unanswered": sum(1 for r in results if not r.executable),
            "pending_validation": max(0, len(catalog) - len(results)),
            "review_sheet": str(sheet_path),
            "bundle": str(bundle_path),
        }
    except Exception:  # noqa: BLE001
        return None
