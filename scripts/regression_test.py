#!/usr/bin/env python3
"""Regression Test Script (P2-7: 回归测试基线)

Compares pipeline output metrics against expected baseline values.
Usage:
    python scripts/regression_test.py --master outputs/runs/ontology/master_ontology_unified.json
    python scripts/regression_test.py --master ... --update-baseline
"""
import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter


# Expected baseline metrics (golden sample thresholds)
# These define the acceptable ranges for a "good" pipeline run
#
# [2026-08-06] Baseline relaxed after batch3 (1122 nodes / 2761 rels / 168 L3):
#   node_count.max       1200 -> 2500   (预留 ~8+ batch 的增长空间，每 batch 预计 +80~150)
#   relationship_count.max 3000 -> 6000 (关系增速略快于节点，预留同样倍数)
#   instance_count.max    200 -> 600    (每 batch 新增实例 ~15-30)
# 保留 min 下限不变（防止实体提取大面积失败时被误判为 PASS）
BASELINE = {
    "node_count": {"min": 500, "max": 2500, "description": "Total node count"},
    "relationship_count": {"min": 400, "max": 6000, "description": "Total relationship count"},
    "instance_count": {"min": 30, "max": 600, "description": "L3 instance count"},
    "dcs_avg": {"min": 0.90, "max": 1.0, "description": "Average DCS score"},
    "cross_layer_violations": {"min": 0, "max": 10, "description": "Cross-layer violations"},
    "dangling_relationships": {"min": 0, "max": 5, "description": "Dangling relationships"},
    "orphan_nodes": {"min": 0, "max": 10, "description": "Non-L3 nodes with no relationships"},
    "high_quality_ratio": {"min": 0.90, "max": 1.0, "description": "Ratio of entities with quality_score >= 0.8"},
    "s_str_coverage": {"min": 0.99, "max": 1.0, "description": "String dimension coverage"},
    "s_sem_coverage": {"min": 0.99, "max": 1.0, "description": "Semantic dimension coverage"},
    "s_struct_coverage": {"min": 0.95, "max": 1.0, "description": "Structural dimension coverage"},
    "e_evid_coverage": {"min": 0.95, "max": 1.0, "description": "Evidence dimension coverage"},
    "t_temp_coverage": {"min": 0.95, "max": 1.0, "description": "Temporal dimension coverage"},
    "r_rel_coverage": {"min": 0.95, "max": 1.0, "description": "Relationship dimension coverage (excl. L3)"},
}


_LAYER_ORDER = ("L0", "L1", "L2", "L2.5", "L3")


def _load_allowed_cross_layer() -> set[tuple[str, str]]:
    """Load allowed cross-layer directions from config/ontology/relationship_rules.yaml.

    Mirrors global_optimizer._cross_layer_direction_allowed and the QualityGate
    layer_pair_rules: a direction (A, B) is allowed when pair key "A-B" is
    explicitly allowed, or - when "A-B" is absent - the reversed key "B-A" is
    explicitly allowed. Falls back to forward-only L0->L1->L2->L3 if the config
    cannot be loaded.
    """
    rule_path = os.environ.get(
        "ONTOLOGY_RELATIONSHIP_RULES",
        "config/ontology/relationship_rules.yaml",
    )
    try:
        import yaml

        with open(rule_path, "r", encoding="utf-8") as f:
            pair_rules = (yaml.safe_load(f) or {}).get("layer_pair_rules", {})
    except Exception as exc:  # pragma: no cover - config is present in-repo
        print(f"[regression] WARNING cross-layer rules unavailable ({exc}); "
              f"falling back to forward-only L0->L1->L2->L3")
        pair_rules = {
            "L0-L1": {"allowed": True},
            "L1-L2": {"allowed": True},
            "L2-L3": {"allowed": True},
            "L2.5-L3": {"allowed": True},
        }

    allowed: set[tuple[str, str]] = set()
    for src in _LAYER_ORDER:
        for tgt in _LAYER_ORDER:
            if src == tgt:
                continue
            pair = f"{src}-{tgt}"
            if pair in pair_rules:
                ok = bool(pair_rules[pair].get("allowed", True))
            elif f"{tgt}-{src}" in pair_rules:
                ok = bool(pair_rules[f"{tgt}-{src}"].get("allowed", True))
            else:
                ok = True
            if ok:
                allowed.add((src, tgt))
    return allowed


def compute_metrics(master_path: str) -> dict:
    """Compute all metrics from a master ontology file."""
    with open(master_path, "r", encoding="utf-8") as f:
        master = json.load(f)

    nodes = master.get("nodes", [])
    rels = master.get("relationships", [])
    instances = master.get("instances", [])
    total = len(nodes)

    # Build lookup sets
    node_ids = {n.get("meta_id", "") for n in nodes if isinstance(n, dict)}
    layer_map = {n.get("meta_id", ""): n.get("layer", "") for n in nodes if isinstance(n, dict)}

    # Relationship endpoint set
    rel_all = set()
    for r in rels:
        for k in ("source", "from", "target", "to"):
            val = r.get(k) if isinstance(r, dict) else None
            if val:
                rel_all.add(str(val))

    # DCS dimensions
    s_str = sum(1 for n in nodes if isinstance(n, dict) and (n.get("meta_name") or n.get("name")))
    s_sem = sum(1 for n in nodes if isinstance(n, dict) and n.get("layer") and n.get("meta_class", "") != "UNKNOWN")
    s_struct = sum(1 for n in nodes if isinstance(n, dict) and (n.get("parent_id") or n.get("parent_class") or n.get("layer") == "L0"))
    e_evid = sum(1 for n in nodes if isinstance(n, dict) and isinstance(n.get("evidence"), dict) and (n.get("evidence", {}).get("content") or n.get("evidence", {}).get("source_text")))
    t_temp = sum(1 for n in nodes if isinstance(n, dict) and (n.get("valid_from") or n.get("valid_to") or n.get("version")))
    r_rel = sum(1 for n in nodes if isinstance(n, dict) and n.get("meta_id", "") in rel_all)

    # Cross-layer violations (direction rules loaded from relationship_rules.yaml)
    allowed = _load_allowed_cross_layer()
    violations = 0
    for r in rels:
        if not isinstance(r, dict):
            continue
        src = r.get("source", "")
        tgt = r.get("target", "")
        sl = layer_map.get(src, "")
        tl = layer_map.get(tgt, "")
        if sl and tl and sl != tl and (sl, tl) not in allowed:
            violations += 1

    # Dangling relationships
    dangling = sum(1 for r in rels if isinstance(r, dict) and (r.get("source", "") not in node_ids or r.get("target", "") not in node_ids))

    # Orphan nodes (exclude L3 instances which are leaf nodes)
    orphans = sum(1 for n in nodes if isinstance(n, dict) and n.get("meta_id", "") not in rel_all and n.get("layer") != "L3")

    # Quality scores
    qs_list = [n.get("quality_score", 0) for n in nodes if isinstance(n, dict) and n.get("quality_score") is not None]
    high_q = sum(1 for c in qs_list if c >= 0.8) if qs_list else 0

    # Non-L3 nodes count for coverage calculations (L3 instances are leaf nodes)
    non_l3_nodes = [n for n in nodes if isinstance(n, dict) and n.get("layer") != "L3"]
    non_l3_total = len(non_l3_nodes)

    return {
        "node_count": len(nodes),
        "relationship_count": len(rels),
        "instance_count": len(instances),
        "dcs_avg": round((s_str + s_sem + s_struct + e_evid + t_temp + r_rel) / (6 * total), 4) if total else 0,
        "cross_layer_violations": violations,
        "dangling_relationships": dangling,
        "orphan_nodes": orphans,
        "high_quality_ratio": round(high_q / len(qs_list), 4) if qs_list else 0,
        "s_str_coverage": round(s_str / total, 4) if total else 0,
        "s_sem_coverage": round(s_sem / total, 4) if total else 0,
        "s_struct_coverage": round(s_struct / total, 4) if total else 0,
        "e_evid_coverage": round(e_evid / total, 4) if total else 0,
        "t_temp_coverage": round(t_temp / total, 4) if total else 0,
        "r_rel_coverage": round(sum(1 for n in non_l3_nodes if n.get("meta_id", "") in rel_all) / non_l3_total, 4) if non_l3_total else 0,
    }


def run_regression_test(master_path: str) -> bool:
    """Run regression test and return True if all metrics pass."""
    print("=" * 60)
    print("  Regression Test Report")
    print("=" * 60)
    print()

    metrics = compute_metrics(master_path)
    all_pass = True

    print(f"  {'Metric':<30} {'Value':>10} {'Range':>20} {'Status':>8}")
    print(f"  {'-'*30} {'-'*10} {'-'*20} {'-'*8}")

    for key, expected in BASELINE.items():
        value = metrics.get(key, 0)
        lo = expected["min"]
        hi = expected["max"]
        passed = lo <= value <= hi
        status = "PASS" if passed else "FAIL"
        range_str = f"[{lo}, {hi}]"
        print(f"  {expected['description']:<30} {value:>10} {range_str:>20} {status:>8}")
        if not passed:
            all_pass = False

    print()
    print("=" * 60)
    if all_pass:
        print("  [PASS] ALL TESTS PASSED")
    else:
        print("  [FAIL] SOME TESTS FAILED -- review metrics above")
    print("=" * 60)

    # Save report
    report_path = Path(master_path).parent / "regression_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "baseline": BASELINE, "passed": all_pass}, f, indent=2, ensure_ascii=False)
    print(f"\n  Report saved: {report_path}")

    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Regression test for ontology pipeline")
    parser.add_argument("--master", required=True, help="Path to master_ontology_unified.json")
    parser.add_argument("--update-baseline", action="store_true", help="Update baseline from current metrics")
    args = parser.parse_args()

    if not Path(args.master).exists():
        print(f"Error: Master file not found: {args.master}")
        sys.exit(1)

    success = run_regression_test(args.master)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
