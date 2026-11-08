"""Convert fragment + summary_report into a compare_pipeline_results.py compatible pipeline_result.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _build_pipeline_result(fragment_path: Path, summary_path: Path) -> dict[str, Any]:
    fragment = json.loads(fragment_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    nodes = fragment.get("nodes", [])
    rels = fragment.get("relationships", [])

    total_nodes = summary.get("nodes", len(nodes))
    total_instances = summary.get("instances", 0)
    total_rels = summary.get("relationships", len(rels))

    l0_nodes = [n for n in nodes if n.get("layer") == "L0"]
    auto_supplemented_l0 = sum(
        1 for n in l0_nodes
        if any(a.get("name") == "auto_generated" and a.get("value") == "true" for a in n.get("attributes", []))
    )

    nodes_with_provenance = sum(1 for n in nodes if n.get("provenance") or n.get("source_file"))
    nodes_with_evidence = sum(1 for n in nodes if n.get("evidence"))
    node_types = {}
    node_layers = {}
    for n in nodes:
        t = n.get("meta_class", "Unknown")
        l = n.get("layer", "Unknown")
        node_types[t] = node_types.get(t, 0) + 1
        node_layers[l] = node_layers.get(l, 0) + 1

    rels_with_type = [r for r in rels if r.get("type")]
    type_counts: dict[str, int] = {}
    for r in rels_with_type:
        t = str(r.get("type", "")).upper()
        type_counts[t] = type_counts.get(t, 0) + 1

    structured_metadata: dict[str, Any] = {}
    for key in ["mbom", "bom", "bop", "occurrence", "position", "variant", "ontology", "security_classes"]:
        if any(key in str(n).lower() for n in nodes):
            structured_metadata[key] = {"detected": True}

    has_ontology = any("ontology" in str(n).lower() or "concept" in str(n).lower() for n in nodes)
    has_position = any("position" in str(n).lower() for n in nodes)
    has_variant = any("variant" in str(n).lower() for n in nodes)
    has_occurrence = any("occurrence" in str(n).lower() for n in nodes)

    for flag, key in [(has_ontology, "ontology_hint"), (has_position, "position_hint"),
                       (has_variant, "variant_hint"), (has_occurrence, "occurrence_hint")]:
        if flag:
            structured_metadata[key] = True

    rewrite_context = {
        "relationship_types": list(type_counts.keys()),
        "type_counts": type_counts,
    }

    structured_metadata["rewrite_context"] = rewrite_context

    dedup_audit_path = Path(str(summary.get("master_path", "")).replace(".json", ".dedup_audit_20260729_143316.json"))
    entities_before = total_nodes
    entities_after = total_nodes
    rels_before = total_rels
    rels_after = total_rels
    computed_quality = 0.0
    if dedup_audit_path.exists():
        try:
            audit = json.loads(dedup_audit_path.read_text(encoding="utf-8"))
            entities_before = audit.get("entities_before", entities_before)
            entities_after = audit.get("entities_after", entities_after)
            rels_before = audit.get("relationships_before", rels_before)
            rels_after = audit.get("relationships_after", rels_after)
            computed_quality = audit.get("quality_score", 0.0)
        except (json.JSONDecodeError, KeyError):
            pass

    if computed_quality <= 0:
        has_evidence_ratio = nodes_with_evidence / len(nodes) if nodes else 0.0
        has_provenance_ratio = nodes_with_provenance / len(nodes) if nodes else 0.0
        dedup_quality = 1.0 - (entities_before - entities_after) / entities_before if entities_before > 0 else 0.0
        computed_quality = round(0.4 * has_evidence_ratio + 0.3 * has_provenance_ratio + 0.3 * max(0, dedup_quality), 4)

    return {
        "run_id": fragment.get("run_id", ""),
        "ontology_statistics": {
            "total_nodes": total_nodes,
            "total_relationships": total_rels,
            "total_instances": total_instances,
            "layer_distribution": summary.get("layer_distribution", {}),
        },
        "audit_details": {
            "quality_score": computed_quality,
            "total_entities": total_nodes,
            "total_relations": total_rels,
        },
        "extraction_details": {
            "auto_supplemented_l0_base_classes": auto_supplemented_l0,
        },
        "deduplication_results": {
            "entities_before": entities_before,
            "entities_after": entities_after,
            "relationships_before": rels_before,
            "relationships_after": rels_after,
        },
        "relationships": rels,
        "stage_results": {
            "extract": {
                "data": {
                    "entity_profile_summary": {
                        "count": len(nodes),
                        "with_provenance": nodes_with_provenance,
                        "with_evidence": nodes_with_evidence,
                        "with_structured_metadata": len(structured_metadata),
                        "types": node_types,
                        "layers": node_layers,
                    }
                }
            }
        },
        "metadata": {
            "structured_metadata": structured_metadata,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert fragment+summary into pipeline_result.json")
    parser.add_argument("--fragment", required=True, help="Path to fragment_*.json")
    parser.add_argument("--summary", required=True, help="Path to summary_report.json")
    parser.add_argument("--output", required=True, help="Output path for pipeline_result.json")
    args = parser.parse_args()

    fragment_path = Path(args.fragment)
    summary_path = Path(args.summary)
    output_path = Path(args.output)

    result = _build_pipeline_result(fragment_path, summary_path)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
