from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class PipelineResultArtifactService:
    """Persist pipeline result artifacts and render human-readable summaries."""

    def _format_markdown_value(self, value: Any) -> str:
        if isinstance(value, dict):
            items = [f"{k}={v}" for k, v in list(value.items())[:12]]
            return "; ".join(items)
        if isinstance(value, list):
            return ", ".join(str(v) for v in value[:12])
        return str(value)

    def _json_default(self, value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(exclude_none=True)
        if hasattr(value, "dict"):
            try:
                return value.dict()
            except TypeError:
                pass
        if isinstance(value, Path):
            return str(value)
        return str(value)

    def _dump_json(self, path: Path, payload: Any) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=self._json_default),
            encoding="utf-8",
        )

    def render_result_markdown(self, result: Dict[str, Any], ontology: Optional[Dict[str, Any]] = None) -> str:
        context = result.get("context", {}) or {}
        stats = result.get("stats", {}) or {}
        metadata = result.get("metadata", {}) or {}
        ontology = ontology or {}
        extracted_metadata = metadata.get("extracted_metadata", {}) or {}
        structured_metadata = metadata.get("structured_metadata", {}) or {}
        stage_results = result.get("stage_results", {}) or {}

        lines = [
            "# Pipeline Execution Result",
            "",
            "## Pipeline Summary",
            "",
            f"- **Run ID**: {result.get('run_id', '')}",
            f"- **Domain**: {context.get('domain', '')}",
            f"- **Source**: {context.get('source_path', '')}",
            f"- **Status**: {result.get('status', '')}",
            f"- **Regression Mode**: {context.get('regression_mode', False)}",
            "",
            "## Statistics",
            "",
            f"- **Completed Stages**: {stats.get('completed_stages', 0)}/{stats.get('total_stages', 0)}",
            f"- **LLM Calls**: {stats.get('llm_calls', 0)}",
            f"- **Entity Count**: {stats.get('entity_count', 0)}",
            f"- **Relation Count**: {stats.get('relation_count', 0)}",
            f"- **Evidence Count**: {stats.get('evidence_count', 0)}",
            "",
            "## Ontology Summary",
            "",
            f"- **Total Nodes**: {ontology.get('total_nodes', stats.get('entity_count', 0))}",
            f"- **Total Relationships**: {ontology.get('total_relationships', stats.get('relation_count', 0))}",
            f"- **Total Instances**: {ontology.get('total_instances', 0)}",
            "",
        ]

        if extracted_metadata:
            lines.extend(["## Extracted Metadata", ""])
            for key, value in extracted_metadata.items():
                lines.append(f"- **{key}**: `{self._format_markdown_value(value)}`")
            lines.append("")

        if structured_metadata:
            lines.extend(["## Structured Metadata", ""])
            for key, value in structured_metadata.items():
                lines.append(f"- **{key}**: `{self._format_markdown_value(value)}`")
            lines.append("")
            lines.extend([
                "## Structured Metadata Summary",
                "",
                f"- **Field Count**: `{len(structured_metadata)}`",
                f"- **Keys**: `{', '.join(list(structured_metadata.keys())[:12])}`",
                "",
            ])
            if "rewrite_context" in structured_metadata:
                lines.extend([
                    "## Rewrite Context",
                    "",
                    f"- **rewrite_context**: `{self._format_markdown_value(structured_metadata.get('rewrite_context'))}`",
                    "",
                ])
            for extra_key in ("time_windows", "percentages", "ranges", "ontology"):
                if extra_key in structured_metadata:
                    lines.append(f"- **{extra_key}**: `{self._format_markdown_value(structured_metadata.get(extra_key))}`")
            lines.append("")
            rewrite_ctx = structured_metadata.get("rewrite_context") or {}
            type_counts = rewrite_ctx.get("type_counts") if isinstance(rewrite_ctx, dict) else None
            if isinstance(type_counts, dict):
                total_rel = sum(type_counts.values())
                related_count = type_counts.get("RELATED_TO", 0)
                ratio = round(related_count / total_rel, 4) if total_rel else 0.0
                lines.extend([
                    "## RELATED_TO Residual",
                    "",
                    f"- **Total Rewritten Relations**: `{total_rel}`",
                    f"- **RELATED_TO Residual**: `{related_count}`",
                    f"- **RELATED_TO Ratio**: `{ratio}`",
                    "",
                ])

        entity_profile_summaries = []
        for stage_name, stage_result in stage_results.items():
            if isinstance(stage_result, dict):
                summary = stage_result.get("data", {}).get("entity_profile_summary")
                if summary:
                    entity_profile_summaries.append((stage_name, summary))
        if entity_profile_summaries:
            lines.extend(["## Entity Profile Summary", ""])
            for stage_name, summary in entity_profile_summaries:
                lines.append(f"- **{stage_name}**: `{self._format_markdown_value(summary)}`")
            lines.append("")
            if isinstance(entity_profile_summaries[0][1], dict):
                top_summary = entity_profile_summaries[0][1]
                if top_summary.get("with_provenance") is not None:
                    layers = top_summary.get("layers", {}) or {}
                    total_layers = sum(layers.values()) or 1
                    lines.extend([
                        "## Entity Profile Quality",
                        "",
                        f"- **With Provenance**: `{top_summary.get('with_provenance', 0)}`",
                        f"- **With Evidence**: `{top_summary.get('with_evidence', 0)}`",
                        f"- **With Structured Metadata**: `{top_summary.get('with_structured_metadata', 0)}`",
                        f"- **With Confidence**: `{top_summary.get('with_confidence', 0)}`",
                        f"- **With Validation State**: `{top_summary.get('with_validation_state', 0)}`",
                        f"- **Profile Completeness**: `{top_summary.get('profile_completeness', 0)}`",
                        f"- **Type Count**: `{len(top_summary.get('types', {}))}`",
                        f"- **Layer Count**: `{len(layers)}`",
                        f"- **L2 Count**: `{layers.get('L2', 0)}`",
                        f"- **L2.5 Count**: `{layers.get('L2.5', 0)}`",
                        f"- **L3 Count**: `{layers.get('L3', 0)}`",
                        f"- **L3 Ratio**: `{round(layers.get('L3', 0) / total_layers, 4)}`",
                        "",
                    ])

        if metadata.get("classification_metrics"):
            lines.extend(["## Classification Metrics", ""])
            for key, value in (metadata.get("classification_metrics") or {}).items():
                lines.append(f"- **{key}**: `{value}`")
            lines.append("")
        if metadata.get("artifact_bundles"):
            lines.extend(["## Artifact Bundles", ""])
            for item in metadata.get("artifact_bundles", []):
                lines.append(f"- `{item}`")
            lines.append("")
        return "\n".join(lines)

    def persist_result_bundle(
        self,
        *,
        save_dir: str,
        result: Any,
        result_dict: Dict[str, Any],
        core_result: Any,
        validation_report: Any,
        artifact_envelope: Any,
        contract_artifacts: Dict[str, Any],
        manifest_payload: Dict[str, Any],
        markdown_text: str,
    ) -> Dict[str, str]:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        result_path = save_path / "pipeline_result.json"
        core_result_path = save_path / "pipeline_execution_result.contract.json"
        validation_report_path = save_path / "validation_report.contract.json"
        artifact_envelope_path = save_path / "artifact_envelope.contract.json"
        manifest_path = save_path / "run_manifest.json"
        markdown_path = save_path / "pipeline_result.md"

        result_payload = dict(result_dict)
        result_payload["contract_artifacts"] = contract_artifacts
        result_payload["manifest_path"] = str(manifest_path)

        self._dump_json(result_path, result_payload)
        self._dump_json(core_result_path, core_result)
        self._dump_json(validation_report_path, validation_report)
        self._dump_json(artifact_envelope_path, artifact_envelope)
        self._dump_json(manifest_path, manifest_payload)
        markdown_path.write_text(markdown_text, encoding="utf-8")

        return {
            "result_path": str(result_path),
            "core_result_path": str(core_result_path),
            "validation_report_path": str(validation_report_path),
            "artifact_envelope_path": str(artifact_envelope_path),
            "manifest_path": str(manifest_path),
            "markdown_path": str(markdown_path),
        }


_SERVICE: Optional[PipelineResultArtifactService] = None


def get_pipeline_result_artifact_service() -> PipelineResultArtifactService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = PipelineResultArtifactService()
    return _SERVICE
