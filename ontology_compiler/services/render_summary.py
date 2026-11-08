"""Report rendering functions for folder analysis output.

This module contains functions for rendering various markdown reports
from folder analysis results, separating presentation logic from
business logic.

P1-7 Refactoring (2026-05-15):
- Static text moved to config/report_templates.yaml
- Uses report_template_loader for configuration-driven rendering
- Maintains backward compatibility with existing API

P2 Migration (2026-05-26):
- Migrated remaining hardcoded Chinese text to YAML templates
- Added _render_checklist_section() helper for template-driven rendering
- Manual checklist, guard check, and publish steps now fully configurable
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List

from ontology_compiler.services.quality_assurance.quality_rules import (
    build_briefing_lines,
    calculate_dimensions,
    score_grade,
    stars,
)
from ontology_compiler.services.report_template_loader import (
    get_business_scenarios,
    get_checklist_items,
    get_core_results_header,
    get_data_analysis_footer,
    get_data_analysis_header,
    get_data_analysis_sections,
    get_entity_tables,
    get_guard_check_steps,
    get_multi_agent_analysis_footer,
    get_multi_agent_analysis_header,
    get_multi_agent_analysis_sections,
    get_multi_agent_analysis_subtitle,
    get_ontology_health_sections,
    get_publish_steps,
    get_risk_matrix,
    get_section_title,
    load_templates,
    render_section,
)

logger = logging.getLogger(__name__)


def _render_checklist_section() -> List[str]:
    """Render manual checklist section using YAML templates.

    P2 Migration (2026-05-26):
    - Moved hardcoded checklist items to report_templates.yaml
    - Supports dynamic item addition without code changes

    Returns:
        List of markdown lines for the checklist section.
    """
    title = get_section_title("core_results.manual_checklist.title", "## 人工检查清单（每次文件夹运行后）")
    items = get_checklist_items()

    lines = ["", title, ""]
    for item in items:
        if isinstance(item, str):
            lines.append(f"- [ ] {item}")
        elif isinstance(item, dict):
            desc = item.get("description", "")
            lines.append(f"- [ ] {desc}")
    return lines


def _render_guard_section() -> List[str]:
    """Render guard check section using YAML templates.

    P2 Migration (2026-05-26):
    - Moved hardcoded guard steps to report_templates.yaml
    - Supports dynamic step configuration without code changes

    Returns:
        List of markdown lines for the guard section.
    """
    title = get_section_title("core_results.guard_check.title", "## Guard 验收通过（操作步骤）")
    steps = get_guard_check_steps()

    lines = ["", title, ""]
    for idx, step in enumerate(steps, 1):
        if isinstance(step, dict):
            desc = step.get("description", "")
            command = step.get("command", "")
            lines.append(f"{idx}. {desc}")
            if command:
                lines.append(f"   - `{command}`")
        elif isinstance(step, str):
            lines.append(f"{idx}. {step}")
    return lines


def _render_publish_section() -> List[str]:
    """Render publish to master section using YAML templates.

    P2 Migration (2026-05-26):
    - Moved hardcoded publish steps to report_templates.yaml
    - Supports dynamic step configuration without code changes

    Returns:
        List of markdown lines for the publish section.
    """
    title = get_section_title("core_results.publish_to_master.title", "## 发布到 Master（publish_delta_to_master.py）")
    steps = get_publish_steps()

    lines = ["", title, ""]
    for idx, step in enumerate(steps, 1):
        if isinstance(step, dict):
            desc = step.get("description", "")
            command = step.get("command", "")
            details = step.get("details", [])
            lines.append(f"{idx}. {desc}")
            if command:
                lines.append(f"   - `{command}`")
            if isinstance(details, list):
                for detail in details:
                    if isinstance(detail, str):
                        lines.append(f"     - {detail}")
        elif isinstance(step, str):
            lines.append(f"{idx}. {step}")
    return lines


def render_l2_business_scenarios_markdown(payload: Dict[str, Any]) -> str:
    """Render L2 business scenarios as markdown.

    Args:
        payload: Scenario payload containing meta and scenarios.

    Returns:
        Markdown string with formatted business scenarios.
    """
    meta = payload.get("meta") or {}
    scenarios = payload.get("scenarios") or []

    lines = [
        "# L2 Business Scenarios",
        "",
        f"Generated at: {meta.get('generated_at')}",
        f"run_id: `{meta.get('run_id')}`",
        f"topic_id: `{meta.get('topic_id')}`",
        f"domain: `{meta.get('domain')}`",
        f"source_folder: `{meta.get('source_folder')}`",
        "",
        f"Total scenarios: {len(scenarios)}",
        "",
    ]

    for scn in scenarios:
        if not isinstance(scn, dict):
            continue
        lines.extend(
            [
                f"## {scn.get('title')} ({scn.get('scenario_id')})",
                "",
                f"- domain: `{scn.get('domain')}`",
                f"- anchor_data: `{((scn.get('anchor') or {}).get('data_meta_id'))}`",
                "",
                "### Business summary",
                *[f"- {line}" for line in (scn.get("business_summary") or [])],
                "",
                "### Evidence",
                *[
                    f"- `{p}`"
                    for p in ((scn.get("evidence") or {}).get("source_files") or [])
                ],
                "",
                "### Signals",
                f"- objects: {(scn.get('signals') or {}).get('objects')}",
                f"- relationships: {(scn.get('signals') or {}).get('relationships')}",
                f"- instances: {(scn.get('signals') or {}).get('instances')}",
                "",
            ]
        )

        objs = scn.get("objects") or []
        if objs:
            lines.append("### Objects")
            for o in objs:
                if isinstance(o, dict):
                    lines.append(
                        f"- `{o.get('meta_id')}` ({o.get('meta_class')}): {o.get('meta_name')}"
                    )
            lines.append("")

        rels = scn.get("relationships") or []
        if rels:
            lines.append("### Relationships")
            for r in rels:
                if isinstance(r, dict):
                    lines.append(
                        f"- `{r.get('source')}` -[{r.get('predicate')}]-> `{r.get('target')}`"
                    )
            lines.append("")

        insts = scn.get("instances") or []
        if insts:
            lines.append("### Instances")
            for i in insts:
                if isinstance(i, dict):
                    interpreted = i.get("interpreted_instance_values") or {}
                    extra = f" | interpreted={interpreted}" if interpreted else ""
                    lines.append(
                        f"- `{i.get('meta_id')}` ({i.get('instance_of')}): {i.get('instance_values')}{extra}"
                    )
            lines.append("")

    return "\n".join(lines)


def render_core_results_markdown(
    *,
    run_id: str,
    topic_id: str,
    domain: str,
    source_folder: str,
    started_at: str | None,
    finished_at: str | None,
    source_profile: Dict[str, Any] | None,
    status: str,
    quality: Dict[str, Any],
    summary: Dict[str, Any],
    artifacts: Dict[str, str],
    l1: Dict[str, Any] | None,
    l2: Dict[str, Any] | None,
    llm_usage: Dict[str, Any] | None,
    scenarios: Dict[str, Any] | None,
) -> str:
    """Render a minimal but information-dense, human-readable summary for a run.

    Args:
        run_id: Run identifier.
        topic_id: Topic identifier.
        domain: Domain name.
        source_folder: Source folder path.
        started_at: Start timestamp.
        finished_at: Finish timestamp.
        source_profile: Source profile information.
        status: Run status.
        quality: Quality metrics.
        summary: Summary statistics.
        artifacts: Artifact paths.
        l1: L1 domain sandbox.
        l2: L2 operational instances.
        llm_usage: LLM usage statistics.
        scenarios: L2 business scenarios.

    Returns:
        Markdown string with formatted core results.
    """
    quality_dict = quality.to_dict() if hasattr(quality, "to_dict") else quality
    q_warnings = quality_dict.get("warnings") or []
    lines = [
        f"# CORE RESULTS ({run_id})",
        "",
        f"- status: `{status}`",
        f"- topic_id: `{topic_id}`",
        f"- domain: `{domain}`",
        f"- source_folder: `{source_folder}`",
        *(
            [
                f"- started_at: `{started_at}`",
                f"- finished_at: `{finished_at}`",
            ]
            if started_at or finished_at
            else []
        ),
        "",
        "## Summary",
        f"- nodes: {summary.get('nodes')}",
        f"- instances: {summary.get('instances')}",
        f"- relationships: {summary.get('relationships')}",
        "",
        "**Quality Metrics**:",
        f"- grade: `{quality_dict.get('grade')}`",
        f"- l1_score: {quality_dict.get('l1_score')}",
        f"- l2_score: {quality_dict.get('l2_score')}",
        f"- warnings: {len(q_warnings)}",
    ]
    if q_warnings:
        lines.extend(["", "### Warnings", *[f"- {w}" for w in q_warnings]])

    if source_profile:
        lines.extend(
            [
                "",
                "## Input profile",
                f"- primary_type: `{source_profile.get('primary_type')}`",
                f"- has_pdf: `{source_profile.get('has_pdf')}`",
                f"- has_doc: `{source_profile.get('has_doc')}`",
                f"- has_excel: `{source_profile.get('has_excel')}`",
            ]
        )

    if llm_usage:
        lines.extend(
            [
                "",
                "## LLM usage",
                f"- llm_calls: {llm_usage.get('llm_calls')}",
                f"- prompt_tokens: {llm_usage.get('prompt_tokens')}",
                f"- completion_tokens: {llm_usage.get('completion_tokens')}",
                f"- total_tokens: {llm_usage.get('total_tokens')}",
                f"- cost: {llm_usage.get('cost')}",
            ]
        )

    sandbox = (l1 or {}).get("sandbox") if isinstance(l1, dict) else {}
    objects = (sandbox or {}).get("objects") if isinstance(sandbox, dict) else []
    rels = (sandbox or {}).get("relationships") if isinstance(sandbox, dict) else []
    if isinstance(objects, list) and objects:
        lines.extend(["", "## Key objects (L1)"])
        for obj in objects[:15]:
            if not isinstance(obj, dict):
                continue
            obj_id = obj.get("meta_id") or obj.get("id") or ""
            obj_name = obj.get("meta_name") or obj.get("name") or ""
            obj_class = obj.get("meta_class") or obj.get("class") or ""
            src = obj.get("source_file") or ""
            extra = f" (source: `{src}`)" if src else ""
            lines.append(f"- `{obj_id}` ({obj_class}): {obj_name}{extra}")

    if isinstance(rels, list) and rels:
        lines.extend(["", "## Key relationships (L1)"])
        for rel in rels[:20]:
            if not isinstance(rel, dict):
                continue
            src = rel.get("source") or ""
            pred = rel.get("predicate") or ""
            tgt = rel.get("target") or ""
            src_file = rel.get("source_file") or ""
            extra = f" (evidence: `{src_file}`)" if src_file else ""
            lines.append(f"- `{src}` -[{pred}]-> `{tgt}`{extra}")

    if scenarios and isinstance(scenarios, dict):
        scenario_list = scenarios.get("scenarios")
        if isinstance(scenario_list, list) and scenario_list:
            lines.extend(["", "## L2 business scenarios"])
            lines.append(f"- total: {len(scenario_list)}")
            for scn in scenario_list[:5]:
                if not isinstance(scn, dict):
                    continue
                sid = scn.get("scenario_id") or scn.get("id") or ""
                title = scn.get("title") or scn.get("name") or ""
                anchor = scn.get("anchor_data") or ""
                lines.append(f"- `{sid}`: {title} (anchor_data: `{anchor}`)")
                evidence = scn.get("evidence") or []
                if isinstance(evidence, list) and evidence:
                    for ev in evidence[:5]:
                        lines.append(f"  - evidence: `{ev}`")

    # --- Extraction Scorecard & Briefing ---
    sandbox = (l1 or {}).get("sandbox") if isinstance(l1, dict) else {}
    all_objects = (sandbox or {}).get("objects") or []
    node_count = summary.get("nodes", 0) or 0
    rel_count = summary.get("relationships", 0) or 0
    evidence_coverage = (sandbox or {}).get("evidence", {}).get("coverage", 0.0) if isinstance(sandbox, dict) else 0.0
    objects_with_src = sum(1 for o in all_objects if isinstance(o, dict) and o.get("source_file"))
    src_ratio = objects_with_src / max(node_count, 1)

    # Infer RIGOR iterations from fragments metadata (max across fragments)
    rigor_iters = 0
    run_dir_path = artifacts.get("run_dir") or ""
    if run_dir_path:
        try:
            from pathlib import Path
            import json as _json
            compilations_dir = Path(run_dir_path) / "compilations"
            for frag_file in compilations_dir.glob("*/fragments/*.json"):
                try:
                    frag_data = _json.loads(frag_file.read_text(encoding="utf-8"))
                    iters = frag_data.get("metadata", {}).get("rigor_iterations", 0) or 0
                    rigor_iters = max(rigor_iters, iters)
                except Exception:
                    pass
        except Exception:
            pass

    # Calculate quality dimensions using quality_scorer module
    dimensions = calculate_dimensions(
        evidence_coverage=evidence_coverage,
        node_count=node_count,
        rel_count=rel_count,
        rigor_iters=rigor_iters,
        src_ratio=src_ratio,
        l1_score=float(quality_dict.get("l1_score") or 0.0),
    )

    total_score = dimensions.total_score
    overall_grade = dimensions.grade

    # Build briefing sentences using quality_scorer
    briefing_lines = build_briefing_lines(
        node_count=node_count,
        rel_count=rel_count,
        rigor_iters=rigor_iters,
        src_ratio=src_ratio,
        q_warnings=q_warnings,
    )

    lines.extend([
        "",
        "## 提取评分 (Extraction Scorecard)",
        "",
        "| 维度 | 得分 | 评级 |",
        "|------|------|------|",
        f"| 证据覆盖度 (Evidence Coverage) | {stars(dimensions.coverage)} {dimensions.coverage:.2f} | {score_grade(dimensions.coverage)} |",
        f"| 本体规模 (Scale: nodes+rels) | {stars(dimensions.scale)} {dimensions.scale:.2f} | {score_grade(dimensions.scale)} |",
        f"| 语义深度 (RIGOR iterations={rigor_iters}) | {stars(dimensions.depth)} {dimensions.depth:.2f} | {score_grade(dimensions.depth)} |",
        f"| 来源多样性 (Source Traceability) | {stars(dimensions.diversity)} {dimensions.diversity:.2f} | {score_grade(dimensions.diversity)} |",
        f"| L1质量分 (L1 Score) | {stars(dimensions.l1_score)} {dimensions.l1_score:.2f} | {score_grade(dimensions.l1_score)} |",
        f"| **综合评分** | **{stars(total_score)} {total_score:.2f}** | **{overall_grade}** |",
        "",
        "## 简报 (Briefing)",
        "",
        *[f"> {line}" for line in briefing_lines],
        "",
    ])

    lines.extend(["", "## Artifacts"])
    for key in (
        "manifest",
        "l1",
        "l2",
        "l2_scenarios_md",
        "core_results_md",
    ):
        path = artifacts.get(key)
        if path:
            lines.append(f"- {key}: `{path}`")

    # Append full domain analysis report
    analysis_section = render_domain_analysis_report(
        run_id=run_id,
        domain=domain,
        source_folder=source_folder,
        l1=l1,
        l2=l2,
        summary=summary,
        quality=quality_dict,
        started_at=started_at,
        finished_at=finished_at,
    )
    lines.append(analysis_section)

    # P2 Migration (2026-05-26): Use template-driven rendering for checklist sections
    lines.extend(_render_checklist_section())
    lines.extend(_render_guard_section())
    lines.extend(_render_publish_section())

    return "\n".join(lines)


def render_domain_analysis_report(
    *,
    run_id: str,
    domain: str,
    source_folder: str,
    l1: Dict[str, Any] | None,
    l2: Dict[str, Any] | None,
    summary: Dict[str, Any],
    quality: Dict[str, Any],
    started_at: str | None = None,
    finished_at: str | None = None,
) -> str:
    """Render a comprehensive domain analysis report from ontology extraction results.

    Generates a deep-dive business analysis including concept taxonomy, process flows,
    relationship semantics, risk analysis, and actionable business scenarios.

    Args:
        run_id: Run identifier.
        domain: Domain name.
        source_folder: Source folder path.
        l1: L1 domain sandbox payload.
        l2: L2 operational instances payload.
        summary: Summary statistics (nodes, instances, relationships).
        quality: Quality metrics dict.
        started_at: Optional start timestamp string.
        finished_at: Optional finish timestamp string.

    Returns:
        Markdown string with full domain analysis report.
    """
    import json as _json  # noqa: F401
    from pathlib import Path as _Path
    from collections import defaultdict

    quality_dict = quality.to_dict() if hasattr(quality, "to_dict") else (quality or {})
    sandbox = (l1 or {}).get("sandbox") if isinstance(l1, dict) else {}
    all_objects: list = (sandbox or {}).get("objects") or [] if isinstance(sandbox, dict) else []
    all_rels: list = (sandbox or {}).get("relationships") or [] if isinstance(sandbox, dict) else []
    all_instances: list = (l2 or {}).get("instances") or [] if isinstance(l2, dict) else []

    node_count = summary.get("nodes", 0) or 0
    rel_count = summary.get("relationships", 0) or 0
    inst_count = summary.get("instances", 0) or 0

    # ── helpers ──────────────────────────────────────────────────────────────
    def _obj_line(o: dict) -> str:
        oid = o.get("meta_id", "")
        name = o.get("meta_name", "")
        cls = o.get("meta_class", "")
        sub = o.get("meta_subclass", "") or ""
        desc = o.get("meta_description", "") or ""
        src = o.get("source_file", "") or ""
        sub_str = f" / {sub}" if sub else ""
        src_str = f" *(src: `{_Path(src).name}`)*" if src else ""
        desc_str = f"\n  > {desc}" if desc else ""
        return f"- **`{oid}`** `{cls}{sub_str}` → **{name}**{src_str}{desc_str}"

    def _rel_line(r: dict) -> str:
        src = r.get("source", "")
        pred = r.get("predicate", "")
        tgt = r.get("target", "")
        sf = r.get("source_file", "") or ""
        ev = f" *(evidence: `{_Path(sf).name}`)*" if sf else ""
        return f"- `{src}` **-[{pred}]->** `{tgt}`{ev}"

    # ── group objects by meta_class (loaded from YAML) ───────────────────────
    domain_config = render_section("domain_classification") or {}
    groups_config = domain_config.get("groups", {}) or {}
    default_group = domain_config.get("default_group", "Other")

    class_keyword_map: Dict[str, list] = {}
    for grp_name, grp_data in groups_config.items():
        if isinstance(grp_data, dict) and "keywords" in grp_data:
            class_keyword_map[grp_name] = grp_data["keywords"]

    groups: Dict[str, list] = defaultdict(list)
    for obj in all_objects:
        if not isinstance(obj, dict):
            continue
        cls = (obj.get("meta_class") or "")
        sub = (obj.get("meta_subclass") or "")
        matched = False
        for grp, keywords in class_keyword_map.items():
            for kw in keywords:
                if kw.lower() in cls.lower() or kw.lower() in sub.lower():
                    groups[grp].append(obj)
                    matched = True
                    break
            if matched:
                break
        if not matched:
            groups[default_group].append(obj)

    # ── group relationships by source-file ───────────────────────────────────
    rel_by_src: Dict[str, list] = defaultdict(list)
    for r in all_rels:
        if not isinstance(r, dict):
            continue
        sf = r.get("source_file", "") or "（无文件证据）"
        rel_by_src[sf].append(r)

    # ── duration ─────────────────────────────────────────────────────────────
    duration_str = ""
    if started_at and finished_at:
        try:
            fmt = "%Y-%m-%dT%H:%M:%S.%f"
            t0 = datetime.strptime(started_at[:26], fmt)
            t1 = datetime.strptime(finished_at[:26], fmt)
            secs = int((t1 - t0).total_seconds())
            duration_str = f"{secs // 60}分{secs % 60}秒"
        except Exception:
            duration_str = ""

    # ── document instances ───────────────────────────────────────────────────
    doc_instances = [i for i in all_instances
                     if isinstance(i, dict) and i.get("meta_subclass") == "Document_Artifact"]
    ver_instances = [i for i in all_instances
                     if isinstance(i, dict) and i.get("meta_subclass") == "Version_Reference"]

    # ── evidence-backed key relationships (loaded from YAML) ─────────────────
    key_predicates_list = render_section("key_predicates") or []
    key_predicates = set(key_predicates_list) if isinstance(key_predicates_list, list) else {
        "is_basis_for", "references", "has_operations", "executes",
        "generates", "fulfills", "defines_components", "validatesOutput",
        "feedsDataTo", "suppliesMaterial",
    }
    key_rels = [r for r in all_rels
                if isinstance(r, dict)
                and r.get("predicate") in key_predicates
                and r.get("source_file")]

    # ═══════════════════════════════════════════════════════════════════════
    # BUILD REPORT
    # ═══════════════════════════════════════════════════════════════════════
    rlines: list[str] = [
        "",
        "---",
        "",
        "# 领域分析报告 (Domain Analysis Report)",
        "",
        f"**Run:** `{run_id}` | **Domain:** `{domain}` | **Source:** `{source_folder}`",
    ]
    if duration_str:
        rlines.append(
            f"**Duration:** {duration_str} | "
            f"**Started:** `{started_at}` → **Finished:** `{finished_at}`"
        )
    rlines.extend([
        "",
        "> 本报告由本体提取管道自动生成，基于 L1 领域沙箱（Domain Sandbox）和 L2 运营实例（Operational Instances）",
        "> 的全量数据，经 RIGOR LLM 精化迭代后综合分析得出。",
        "> 涵盖：概念体系分类、核心业务流程、关系语义全目录、属性深析、风险矩阵、可执行业务场景。",
        "",
        "---",
        "",
        "## 一、提取规模与质量总览",
        "",
        "| 指标 | 值 |",
        "|------|----|",
        f"| 领域概念节点 (L1 Objects) | **{node_count}** |",
        f"| 语义关系 (L1 Relationships) | **{rel_count}** |",
        f"| 操作实例 (L2 Instances) | **{inst_count}** |",
        f"| 文档实例 | **{len(doc_instances)}** |",
        f"| 版本引用实例 | **{len(ver_instances)}** |",
        f"| 质量等级 | **{quality_dict.get('grade', 'N/A')}** |",
        f"| L1评分 | **{float(quality_dict.get('l1_score') or 0):.4f}** |",
        f"| L2评分 | **{float(quality_dict.get('l2_score') or 0):.4f}** |",
        "",
    ])

    # ── Section 2: concept taxonomy (template-driven) ────────────────────────
    active_groups = [(g, objs) for g, objs in groups.items() if objs]
    domain_title = domain_config.get("title", "## 二、核心概念体系（按业务域分类）")
    domain_desc = domain_config.get("description", "共提取 **{node_count}** 个领域概念，分为 **{group_count}** 个业务域：")
    domain_desc = domain_desc.format(node_count=node_count, group_count=len(active_groups))
    
    rlines.extend([
        "---",
        "",
        domain_title,
        "",
        domain_desc,
        "",
    ])
    for idx, (grp_name, objs) in enumerate(active_groups, 1):
        rlines.extend([f"### 2.{idx} {grp_name}（{len(objs)}个节点）", ""])
        for obj in objs:
            if isinstance(obj, dict):
                rlines.append(_obj_line(obj))
        rlines.append("")

    # ── Section 3: relationships catalog ────────────────────────────────────
    rlines.extend([
        "---",
        "",
        "## 三、语义关系全目录（按证据文件分组）",
        "",
        f"共提取 **{rel_count}** 条有向语义关系，覆盖 **{len(rel_by_src)}** 个证据来源：",
        "",
    ])
    for sf, rels in sorted(rel_by_src.items(),
                           key=lambda x: (x[0] == "（无文件证据）", x[0])):
        label = _Path(sf).name if sf != "（无文件证据）" else "（无文件证据 — LLM推断）"
        rlines.extend([f"### 来源：`{label}`（{len(rels)}条）", ""])
        for r in rels:
            rlines.append(_rel_line(r))
        rlines.append("")

    # ── Section 4: business process flows (template-driven) ──────────────────
    process_flows_config = render_section("business_process_flows") or {}
    rlines.extend([
        "---",
        "",
        process_flows_config.get("title", "## 四、核心业务流程分析"),
        "",
    ])
    
    flows = process_flows_config.get("flows", {}) or {}
    for flow_key, flow_data in flows.items():
        if not isinstance(flow_data, dict):
            continue
        rlines.append(flow_data.get("title", f"### {flow_key}"))
        rlines.append("")
        
        if "description" in flow_data:
            rlines.append(f"> {flow_data['description']}")
            rlines.append("")
        
        if "diagram" in flow_data:
            rlines.append("```")
            rlines.append(flow_data["diagram"])
            rlines.append("```")
            rlines.append("")
        
        if flow_data.get("key_relationships") and key_rels:
            rlines.append("**关键证据关系：**")
            rlines.append("")
            for r in key_rels:
                rlines.append(_rel_line(r))
            rlines.append("")
        
        if "warning" in flow_data:
            for warning_line in flow_data["warning"].split("\n"):
                rlines.append(warning_line)
            rlines.append("")

    # ── Section 5: attribute deep-dive (template-driven) ─────────────────────
    entity_attrs_config = render_section("entity_attributes") or {}
    rlines.extend([
        "---",
        "",
        entity_attrs_config.get("title", "## 五、核心概念属性深析"),
        "",
    ])
    
    entities = entity_attrs_config.get("entities", {}) or {}
    for entity_key, entity_data in entities.items():
        if not isinstance(entity_data, dict):
            continue
        rlines.append(entity_data.get("title", f"### {entity_key}"))
        rlines.append("")
        
        columns = entity_data.get("columns", [])
        rows = entity_data.get("rows", [])
        
        if columns:
            rlines.append("| " + " | ".join(columns) + " |")
            rlines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        
        if rows:
            for row in rows:
                if isinstance(row, list):
                    rlines.append("| " + " | ".join(row) + " |")
        
        rlines.append("")

    # ── Section 6: risk matrix (template-driven) ──────────────────────────────
    risk_matrix_config = render_section("risk_matrix") or {}
    rlines.extend([
        "---",
        "",
        risk_matrix_config.get("title", "## 六、主要问题点与风险矩阵"),
        "",
    ])
    
    headers = risk_matrix_config.get("table_headers", ["风险等级", "问题", "涉及节点", "业务影响", "建议措施"])
    risks = risk_matrix_config.get("risks", [])
    
    rlines.append("| " + " | ".join(headers) + " |")
    rlines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    
    for risk in risks:
        if isinstance(risk, dict):
            row = [
                risk.get("level", ""),
                risk.get("problem", ""),
                risk.get("nodes", ""),
                risk.get("impact", ""),
                risk.get("action", ""),
            ]
            rlines.append("| " + " | ".join(row) + " |")
    
    rlines.append("")

    # ── Section 7: business scenarios (template-driven) ───────────────────────
    scenarios_config = render_section("business_scenarios") or {}
    rlines.extend([
        "---",
        "",
        scenarios_config.get("title", "## 七、可执行业务分析场景"),
        "",
    ])
    
    scenarios = scenarios_config.get("scenarios", [])
    for idx, scenario in enumerate(scenarios, 1):
        if not isinstance(scenario, dict):
            continue
        rlines.append(f"### 场景{idx}：{scenario.get('name', scenario.get('id', ''))}")
        rlines.append("")
        
        if "trigger" in scenario:
            rlines.append(f"**触发**：{scenario['trigger']}")
            rlines.append("")
        
        if "diagram" in scenario:
            rlines.append("```")
            rlines.append(scenario["diagram"])
            rlines.append("```")
            rlines.append("")
        
        if "output" in scenario:
            rlines.append(f"**预期输出**：{scenario['output']}")
            rlines.append("")

    # ── Section 8: document evidence timeline (template-driven) ───────────────
    doc_evidence_config = render_section("document_evidence") or {}
    doc_count = len(doc_instances)
    
    rlines.extend([
        "---",
        "",
        doc_evidence_config.get("title", "## 八、证据文档时间线（L2实例）"),
        "",
        doc_evidence_config.get("description", "共 **{doc_count}** 份源文档，覆盖制造BOM体系演进历史：").format(doc_count=doc_count),
        "",
        "| " + " | ".join(doc_evidence_config.get("table_headers", ["文档名称", "类型", "日期/版本", "说明"])) + " |",
        "| " + " | ".join(["---"] * 4) + " |",
    ])
    for inst in doc_instances:
        if not isinstance(inst, dict):
            continue
        name = inst.get("meta_name", "")
        vals = inst.get("instance_values", {}) or {}
        dtype = (vals.get("document_type") or "").upper()
        date_v = vals.get("date") or vals.get("version") or "—"
        author = vals.get("author") or ""
        author_str = f" (作者:{author[:30]})" if author and len(author) < 40 else ""
        rlines.append(f"| `{name}` | {dtype} | {date_v} | {author_str} |")
    rlines.extend([
        "",
    ])

    # ── Section 9: ontology health (template-driven) ──────────────────────────
    objects_with_src = sum(1 for o in all_objects
                           if isinstance(o, dict) and o.get("source_file"))
    src_pct = int(objects_with_src / max(node_count, 1) * 100)
    avg_rel = rel_count / max(node_count, 1)
    missing_src_pct = 100 - src_pct
    
    health_config = render_section("ontology_health") or {}
    health_sections = health_config.get("sections", {}) or {}
    
    rlines.extend([
        "---",
        "",
        health_config.get("title", "## 九、本体健康度评估与改进建议"),
        "",
    ])
    
    for section_key, section_data in health_sections.items():
        if not isinstance(section_data, dict):
            continue
        rlines.append(section_data.get("title", f"### {section_key}"))
        items = section_data.get("items", [])
        
        for item in items:
            if isinstance(item, str):
                item = item.format(
                    node_count=node_count,
                    rel_count=rel_count,
                    avg_rel=avg_rel,
                    src_pct=src_pct,
                    missing_src_pct=missing_src_pct,
                )
                rlines.append(f"- {item}")
        
        rlines.append("")

    return "\n".join(rlines)


def render_data_analysis_markdown(
    *,
    analysis_id: str,
    executed_at: str,
    data_source: str,
    record_count: int,
    summary_stats: Dict[str, Any],
    findings: List[Dict[str, Any]],
    recommendations: List[str],
    data_quality: Dict[str, float],
) -> str:
    """Render a data analysis report from CSV/Excel data.

    P4 Migration (2026-05-30):
    - Migrated from orchestrator.py hardcoded rendering
    - Uses YAML templates for configurable presentation
    - Separates presentation logic from business logic

    Args:
        analysis_id: Unique analysis identifier.
        executed_at: Execution timestamp string.
        data_source: Path or description of data source.
        record_count: Number of records analyzed.
        summary_stats: Dictionary of summary statistics.
        findings: List of key findings with 'title' and 'description'.
        recommendations: List of recommendation strings.
        data_quality: Dict with 'completeness', 'consistency', 'timeliness' scores.

    Returns:
        Markdown string with formatted data analysis report.
    """
    sections = get_data_analysis_sections()
    header = get_data_analysis_header()
    footer = get_data_analysis_footer()

    lines = [header, ""]

    # Overview Section
    overview_cfg = sections.get("overview", {})
    overview_title = overview_cfg.get("title", "## 一、分析概览")
    lines.extend([overview_title, ""])

    overview_metrics = overview_cfg.get("metrics", [])
    for metric in overview_metrics:
        label = metric.get("label", metric.get("key", ""))
        key = metric.get("key", "")
        format_type = metric.get("format", "")
        if key == "analysis_id":
            value = analysis_id
        elif key == "executed_at":
            value = executed_at
        elif key == "data_source":
            value = data_source
        elif key == "record_count":
            value = str(record_count)
        else:
            value = summary_stats.get(key, "")
        if format_type == "code":
            value = f"`{value}`"
        lines.append(f"- **{label}**: {value}")
    lines.append("")

    # Summary Statistics Section
    summary_cfg = sections.get("summary_stats", {})
    summary_title = summary_cfg.get("title", "## 二、统计摘要")
    summary_desc = summary_cfg.get("description", "关键指标统计：")
    lines.extend([summary_title, "", summary_desc, ""])
    for stat_name, stat_value in summary_stats.items():
        if isinstance(stat_value, (int, float)):
            lines.append(f"- **{stat_name}**: {stat_value}")
        else:
            lines.append(f"- **{stat_name}**: {stat_value}")
    lines.append("")

    # Findings Section
    findings_cfg = sections.get("findings", {})
    findings_title = findings_cfg.get("title", "## 三、关键发现")
    findings_desc = findings_cfg.get("description", "数据模式与异常识别：")
    lines.extend([findings_title, "", findings_desc, ""])
    for finding in findings:
        title = finding.get("title", "发现")
        description = finding.get("description", "")
        lines.append(f"### {title}")
        lines.append(f"{description}")
        lines.append("")

    # Recommendations Section
    recs_cfg = sections.get("recommendations", {})
    recs_title = recs_cfg.get("title", "## 四、业务建议")
    recs_desc = recs_cfg.get("description", "基于分析结果的可执行建议：")
    lines.extend([recs_title, "", recs_desc, ""])
    for rec in recommendations:
        lines.append(f"- {rec}")
    lines.append("")

    # Data Quality Section
    quality_cfg = sections.get("data_quality", {})
    quality_title = quality_cfg.get("title", "## 五、数据质量评估")
    lines.extend([quality_title, ""])
    quality_checks = quality_cfg.get("checks", [])
    for check in quality_checks:
        label = check.get("label", check.get("key", ""))
        key = check.get("key", "")
        score = data_quality.get(key, 0.0)
        score_pct = f"{score * 100:.1f}%"
        status = "✅" if score >= 0.8 else "⚠️" if score >= 0.6 else "❌"
        lines.append(f"- {status} **{label}**: {score_pct}")
    lines.append("")

    # Footer
    lines.append(footer)

    return "\n".join(lines)


def render_multi_agent_analysis_markdown(
    *,
    generated_at: str,
    ontology_run_dir: str,
    data_sources: List[str],
    total_rounds: int,
    converged: bool,
    background: str,
    question: str,
    history: List[Dict[str, Any]],
    unresolved_hypotheses: List[str],
) -> str:
    """Render a Multi-Agent iteration loop analysis report.

    P4 Migration (2026-05-30):
    - Migrated from orchestrator.py _render_markdown hardcoded rendering
    - Uses YAML templates for configurable presentation
    - Separates presentation logic from business logic

    Args:
        generated_at: Report generation timestamp.
        ontology_run_dir: Ontology run directory path.
        data_sources: List of data source names.
        total_rounds: Total number of iteration rounds.
        converged: Whether the analysis converged.
        background: User background context.
        question: Core analysis question.
        history: List of iteration round records with round_index, focus, confidence,
                 verifier_verdict, findings, gaps, hypotheses, contradictions, verifier_notes.
        unresolved_hypotheses: List of hypotheses that remain unresolved.

    Returns:
        Markdown string with formatted Multi-Agent analysis report.
    """
    sections = get_multi_agent_analysis_sections()
    header = get_multi_agent_analysis_header()
    subtitle = get_multi_agent_analysis_subtitle()

    lines = [header, "", subtitle, "", "---", ""]

    # Metadata Section
    metadata_cfg = sections.get("metadata", {})
    metadata_title = metadata_cfg.get("title", "## 分析元数据")
    lines.extend([metadata_title, ""])

    metadata_items = metadata_cfg.get("items", [])
    for item in metadata_items:
        label = item.get("label", item.get("key", ""))
        key = item.get("key", "")
        format_type = item.get("format", "")

        if key == "generated_at":
            value = generated_at
        elif key == "ontology_run_dir":
            value = ontology_run_dir
        elif key == "data_sources":
            value = ", ".join(data_sources)
        elif key == "total_rounds":
            value = str(total_rounds)
        elif key == "converged":
            value = "✅ 是" if converged else "⚠️ 否"
        else:
            value = ""

        if format_type == "code":
            value = f"`{value}`"
        lines.append(f"- **{label}**: {value}")
    lines.append("")

    # User Context Section
    user_ctx_cfg = sections.get("user_context", {})
    user_ctx_title = user_ctx_cfg.get("title", "## 用户背景")
    user_ctx_desc = user_ctx_cfg.get("description", "分析问题的业务背景")
    lines.extend([user_ctx_title, "", user_ctx_desc, "", background, ""])

    # Core Question Section
    question_cfg = sections.get("core_question", {})
    question_title = question_cfg.get("title", "## 核心分析问题")
    question_desc = question_cfg.get("description", "本次分析的核心问题")
    lines.extend([question_title, "", question_desc, "", question, "", "---", ""])

    # Iteration History Section
    history_cfg = sections.get("iteration_history", {})
    history_title = history_cfg.get("title", "## 迭代过程记录")
    lines.extend([history_title, ""])

    round_header_template = history_cfg.get("round_header", "### 第 {round_index} 轮 | 焦点: {focus}")
    round_metrics = history_cfg.get("round_metrics", [])
    round_sections = history_cfg.get("sections", {})
    empty_marker = history_cfg.get("empty_marker", "(无)")

    for round_data in history:
        round_index = round_data.get("round_index", 0)
        focus = round_data.get("focus", "")
        round_header = round_header_template.format(round_index=round_index, focus=focus)
        lines.extend(["", round_header, ""])

        # Round metrics
        for metric in round_metrics:
            label = metric.get("label", metric.get("key", ""))
            key = metric.get("key", "")
            format_type = metric.get("format", "")

            if key == "confidence":
                value = f"{round_data.get('confidence', 0.0):.2f}"
            elif key == "verifier_verdict":
                value = round_data.get("verifier_verdict", "")
            else:
                value = ""

            if format_type == "percentage":
                value = f"{value}"
            elif format_type == "code":
                value = f"`{value}`"

            lines.append(f"- **{label}**: {value}")
        lines.append("")

        # Round sections
        sections_labels = {
            "findings": round_sections.get("findings", "**Analyst 发现:**"),
            "gaps": round_sections.get("gaps", "**数据缺口:**"),
            "hypotheses": round_sections.get("hypotheses", "**提出的假设:**"),
            "contradictions": round_sections.get("contradictions", "**发现的矛盾:**"),
            "verifier_notes": round_sections.get("verifier_notes", "**Verifier 评语:**"),
        }

        lines.append(sections_labels["findings"])
        lines.append(round_data.get("findings", ""))
        lines.append("")

        lines.append(sections_labels["gaps"])
        gaps = round_data.get("gaps", [])
        lines.append("\n".join(f"- {g}" for g in gaps) if gaps else empty_marker)
        lines.append("")

        lines.append(sections_labels["hypotheses"])
        hypotheses = round_data.get("hypotheses", [])
        lines.append("\n".join(f"- {h}" for h in hypotheses) if hypotheses else empty_marker)
        lines.append("")

        lines.append(sections_labels["contradictions"])
        contradictions = round_data.get("contradictions", [])
        lines.append("\n".join(f"- {c}" for c in contradictions) if contradictions else empty_marker)
        lines.append("")

        lines.append(sections_labels["verifier_notes"])
        lines.append(round_data.get("verifier_notes", ""))
        lines.append("")
        lines.append("---")

    # Unresolved Hypotheses Section
    if unresolved_hypotheses:
        unresolved_cfg = sections.get("unresolved_hypotheses", {})
        unresolved_title = unresolved_cfg.get("title", "## 未解决假设")
        unresolved_desc = unresolved_cfg.get("description", "以下假设在当前分析中尚未得到充分验证：")
        lines.extend(["", unresolved_title, "", unresolved_desc, ""])
        for h in unresolved_hypotheses:
            lines.append(f"- {h}")
        lines.append("")

    return "\n".join(lines)
