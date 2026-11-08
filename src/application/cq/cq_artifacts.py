"""CQ artifact writers (M2 产物先行).

Writes the three per-run CQ products under ``<artifact_root>/cq/``:

- ``cq_catalog.json``        — the loaded CQ catalog snapshot (preflight)
- ``cq_validation_report.json`` — read-only executability check (audit)
- ``cq_review_sheet.md``     — business-facing review/回译 sheet (summarize)

and registers them as one ``ArtifactBundle`` (stage=``cq``) via
``src.pipeline.artifact_registry.ArtifactRegistry``.

All functions raise on IO errors; the *integration layer*
(``src.pipeline.cq_integration``) decides whether to call them and wraps the
calls defensively so a CQ failure can never break a pipeline run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

from src.application.cq.cq_models import CQ, CQCatalog, CQValidationResult

CQ_BUNDLE_STAGE = "cq"


def _cq_dir(artifact_root: Union[str, Path]) -> Path:
    path = Path(artifact_root) / "cq"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_cq_catalog(catalog: CQCatalog, artifact_root: Union[str, Path]) -> Path:
    """Write the CQ catalog snapshot; returns the written path."""
    out_dir = _cq_dir(artifact_root)
    path = out_dir / "cq_catalog.json"
    payload = {
        "items": [item.model_dump() for item in catalog.items],
        "count": len(catalog),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_cq_validation_report(
    results: Iterable[CQValidationResult],
    artifact_root: Union[str, Path],
    *,
    meta: Union[Dict[str, Any], None] = None,
) -> Path:
    """Write the read-only executability report; returns the written path."""
    out_dir = _cq_dir(artifact_root)
    path = out_dir / "cq_validation_report.json"
    items = list(results)
    payload: Dict[str, Any] = {
        "checked_count": len(items),
        "executable_count": sum(1 for r in items if r.executable),
        "results": [r.model_dump() for r in items],
    }
    if meta:
        payload["meta"] = meta
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_cq_review_sheet(
    catalog: CQCatalog,
    results: List[CQValidationResult],
    artifact_root: Union[str, Path],
) -> Path:
    """Write the business-facing review sheet (回译确认); returns the path."""
    out_dir = _cq_dir(artifact_root)
    path = out_dir / "cq_review_sheet.md"
    by_id = {r.cq_id: r for r in results}
    lines: List[str] = [
        "# CQ Review Sheet（口径回译确认）",
        "",
        "| cq_id | 状态 | 业务问题 | 可回答 | 缺口 / 待确认口径 |",
        "|---|---|---|---|---|",
    ]
    for item in catalog.items:
        result = by_id.get(item.cq_id)
        executable = "—" if result is None else ("✅" if result.executable else "❌")
        gaps = "; ".join(
            [f"缺实体 {e}" for e in (result.missing_entities if result else [])]
            + [f"缺关系 {r}" for r in (result.missing_relations if result else [])]
            + list(item.open_questions)
        )
        lines.append(
            f"| {item.cq_id} | {item.status} | {item.cq_text} | {executable} | {gaps or '—'} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_cq_extract_coverage(payload: Dict[str, Any], artifact_root: Union[str, Path]) -> Path:
    """Write the M4 extract-stage coverage snapshot; returns the path.

    ``cq_extract_coverage.json`` carries the focus-vocabulary coverage
    ratios (``cq_entity_coverage`` / ``cq_relation_coverage``) plus the
    per-CQ detail and the 漏抽提示 (missing focus terms with originating CQ
    ids). Read-only statistics — the extract stage writes this without
    changing fragment content.
    """
    out_dir = _cq_dir(artifact_root)
    path = out_dir / "cq_extract_coverage.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_cq_gap_report(
    results: Iterable[CQValidationResult],
    catalog: CQCatalog,
    artifact_root: Union[str, Path],
    *,
    mapping: Union[Dict[str, Dict[str, Any]], None] = None,
) -> Path:
    """Write the M3 gap report (验证闭环缺口报告); returns the path.

    Structure: a summary table (cq_id / bound sample / executability / gap
    digest) followed by a per-CQ detail section listing missing entities,
    relations, attributes, query_path chain breaks, and open questions.
    Only validated (i.e. frozen) CQs appear in the report.
    """
    out_dir = _cq_dir(artifact_root)
    path = out_dir / "cq_gap_report.md"
    by_id = {r.cq_id: r for r in results}
    mapping = mapping or {}
    gap_results = [r for r in results if not r.executable]
    lines: List[str] = [
        "# CQ Gap Report（验证闭环缺口报告）",
        "",
        f"**Checked**: {len(by_id)} frozen CQs / **Executable**: {len(by_id) - len(gap_results)} / **With gap**: {len(gap_results)}",
        "",
        "## 汇总",
        "",
        "| cq_id | 绑定样本 | 可回答 | 缺口摘要 |",
        "|---|---|---|---|",
    ]
    for item in catalog.items:
        result = by_id.get(item.cq_id)
        if result is None:
            continue
        binding = mapping.get(item.cq_id) or {}
        sample = str(binding.get("sample_id") or "—（unbound）")
        mark = "✅" if result.executable else "❌"
        digest = "; ".join(
            [f"缺实体 {e}" for e in result.missing_entities]
            + [f"缺关系 {r}" for r in result.missing_relations]
            + [f"缺属性 {a}" for a in result.missing_attributes]
            + [f"断链 {b}" for b in result.chain_breaks]
        )
        lines.append(f"| {item.cq_id} | {sample} | {mark} | {digest or '—'} |")
    lines.extend(["", "## 缺口明细", ""])
    for result in gap_results:
        item = catalog.get(result.cq_id)
        lines.append(f"### {result.cq_id}")
        if item is not None:
            lines.append(f"- 业务问题: {item.cq_text}")
            lines.append(f"- query_path: `{item.query_path}`")
        if result.missing_entities:
            lines.append(f"- 缺实体: {', '.join(result.missing_entities)}")
        if result.missing_relations:
            lines.append(f"- 缺关系: {', '.join(result.missing_relations)}")
        if result.missing_attributes:
            lines.append(f"- 缺属性: {', '.join(result.missing_attributes)}")
        for brk in result.chain_breaks:
            lines.append(f"- query_path 断链: {brk}")
        for question in (item.open_questions if item is not None else []):
            lines.append(f"- 待确认口径: {question}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_cq_bundle(
    artifact_root: Union[str, Path],
    run_id: str,
    files: Dict[str, Union[str, Path]],
    metadata: Union[Dict[str, Any], None] = None,
) -> Path:
    """Register the three CQ products as one ArtifactBundle (stage=``cq``).

    ``files`` maps logical names to written paths; only existing files are
    included. Returns the bundle JSON path.
    """
    from src.pipeline.artifact_registry import ArtifactBundle, ArtifactRegistry

    registry = ArtifactRegistry(str(artifact_root))
    items = {
        name: str(path)
        for name, path in files.items()
        if path and Path(path).exists()
    }
    bundle = ArtifactBundle(
        run_id=run_id,
        stage=CQ_BUNDLE_STAGE,
        artifact_root=str(artifact_root),
        audit_root=str(registry.audit_root),
        regression_root=str(registry.regression_root),
        items=items,
        metadata=metadata or {},
    )
    return Path(registry.register_bundle(bundle))
