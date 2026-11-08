from __future__ import annotations

"""Compare two regression summaries and emit deltas.

The runner invokes this script with before/after summary JSON paths and an output
path for the comparison JSON. The compare report focuses on the core baseline
metrics used by the regression manifest.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class MetricRule:
    key: str
    label: str
    better: str  # "higher" or "lower"


CORE_METRICS: tuple[MetricRule, ...] = (
    MetricRule("quality_score", "Quality Score", "higher"),
    MetricRule("evidence_coverage", "Evidence Coverage", "higher"),
    MetricRule("classification_consistency", "Classification Consistency", "higher"),
    MetricRule("related_to_share", "RELATED_TO Share", "lower"),
    MetricRule("mismerge_rate", "Dedup Mismerge Rate", "lower"),
    MetricRule("rollback_success", "Rollback Success", "higher"),
    MetricRule("relation_count", "Relationship Count", "higher"),
    MetricRule("entity_count", "Entity Count", "higher"),
    MetricRule("golden_entity_recall", "Golden Entity Recall", "higher"),
    MetricRule("golden_relation_recall", "Golden Relation Recall", "higher"),
    MetricRule("golden_f1", "Golden Semantic F1", "higher"),
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Summary not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Summary must be a JSON object: {path}")
    return payload


def _summary_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    results = summary.get("results") or []
    if not isinstance(results, list):
        results = []

    # Aggregate the first available observed metrics across samples.
    aggregate: dict[str, list[float]] = {rule.key: [] for rule in CORE_METRICS}
    overall_pass = 0
    overall_fail = 0

    for item in results:
        if not isinstance(item, dict):
            continue
        if item.get("pass_fail") == "PASS":
            overall_pass += 1
        else:
            overall_fail += 1

        observed = (((item.get("evaluation") or {}).get("observed")) or {})
        if not isinstance(observed, dict):
            observed = {}
        for rule in CORE_METRICS:
            value = observed.get(rule.key)
            if isinstance(value, (int, float)):
                aggregate[rule.key].append(float(value))

        # E-2 (2026-08-15): golden semantic metrics live in
        # evaluation.golden, not observed. Pull them in so the before/after
        # compare covers semantic recall/precision/F1, not just thresholds.
        golden = (((item.get("evaluation") or {}).get("golden")) or {})
        if isinstance(golden, dict):
            golden_mapping = {
                "golden_entity_recall": golden.get("entity_recall"),
                "golden_relation_recall": golden.get("relation_recall"),
                "golden_f1": golden.get("f1"),
            }
            for rule_key, value in golden_mapping.items():
                if isinstance(value, (int, float)):
                    aggregate[rule_key].append(float(value))

    def _avg(values: list[float]) -> float | None:
        if not values:
            return None
        return round(sum(values) / len(values), 6)

    return {
        "sample_count": len(results),
        "passed_count": overall_pass,
        "failed_count": overall_fail,
        "overall_status": summary.get("overall_status"),
        "metrics": {rule.key: _avg(values) for rule, values in ((rule, aggregate[rule.key]) for rule in CORE_METRICS)},
        "manifest": summary.get("manifest"),
        "executed_at": summary.get("executed_at"),
    }


def _delta(after: float | None, before: float | None) -> float | None:
    if after is None or before is None:
        return None
    return round(after - before, 6)


def _direction(rule: MetricRule, delta: float | None) -> str:
    if delta is None:
        return "unknown"
    if delta == 0:
        return "flat"
    if rule.better == "higher":
        return "improved" if delta > 0 else "regressed"
    return "improved" if delta < 0 else "regressed"


def _build_comparison(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_metrics = before["metrics"]
    after_metrics = after["metrics"]

    metric_rows = []
    for rule in CORE_METRICS:
        before_value = before_metrics.get(rule.key)
        after_value = after_metrics.get(rule.key)
        diff = _delta(after_value, before_value)
        metric_rows.append(
            {
                "key": rule.key,
                "label": rule.label,
                "better": rule.better,
                "before": before_value,
                "after": after_value,
                "delta": diff,
                "direction": _direction(rule, diff),
            }
        )

    return {
        "before": before,
        "after": after,
        "sample_count_delta": after["sample_count"] - before["sample_count"],
        "passed_count_delta": after["passed_count"] - before["passed_count"],
        "failed_count_delta": after["failed_count"] - before["failed_count"],
        "metric_deltas": metric_rows,
    }


def _render_markdown(comparison: dict[str, Any], before_path: Path, after_path: Path) -> str:
    lines = [
        "# Pipeline Comparison",
        "",
        f"- Before: `{before_path}`",
        f"- After: `{after_path}`",
        f"- Sample delta: `{comparison['sample_count_delta']}`",
        f"- Passed delta: `{comparison['passed_count_delta']}`",
        f"- Failed delta: `{comparison['failed_count_delta']}`",
        "",
        "## Metric Deltas",
        "",
        "| Metric | Before | After | Delta | Direction |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in comparison["metric_deltas"]:
        before_value = row["before"]
        after_value = row["after"]
        delta = row["delta"]
        lines.append(
            f"| {row['label']} | {before_value if before_value is not None else 'N/A'} | "
            f"{after_value if after_value is not None else 'N/A'} | "
            f"{delta if delta is not None else 'N/A'} | {row['direction']} |"
        )
    return "\n".join(lines)


def _render_html(comparison: dict[str, Any], before_path: Path, after_path: Path) -> str:
    rows = []
    for row in comparison["metric_deltas"]:
        rows.append(
            "<tr>"
            f"<td>{row['label']}</td>"
            f"<td>{row['before'] if row['before'] is not None else 'N/A'}</td>"
            f"<td>{row['after'] if row['after'] is not None else 'N/A'}</td>"
            f"<td>{row['delta'] if row['delta'] is not None else 'N/A'}</td>"
            f"<td>{row['direction']}</td>"
            "</tr>"
        )
    rows_html = "\n".join(rows)
    return f"""<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Pipeline Comparison</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; color: #111827; background: #f9fafb; }}
    .card {{ background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px; margin-bottom: 16px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 10px 12px; text-align: left; font-size: 14px; }}
    th {{ background: #f3f4f6; }}
    .improved {{ color: #166534; font-weight: 700; }}
    .regressed {{ color: #991b1b; font-weight: 700; }}
    .flat {{ color: #6b7280; font-weight: 700; }}
    code {{ background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <div class=\"card\">
    <h1>Pipeline Comparison</h1>
    <p>Before: <code>{before_path}</code></p>
    <p>After: <code>{after_path}</code></p>
    <p>Sample delta: <code>{comparison['sample_count_delta']}</code> | Passed delta: <code>{comparison['passed_count_delta']}</code> | Failed delta: <code>{comparison['failed_count_delta']}</code></p>
  </div>
  <div class=\"card\">
    <table>
      <thead>
        <tr><th>Metric</th><th>Before</th><th>After</th><th>Delta</th><th>Direction</th></tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>
</body>
</html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two pipeline regression summaries")
    parser.add_argument("--before", required=True, help="Path to baseline summary JSON")
    parser.add_argument("--after", required=True, help="Path to candidate summary JSON")
    parser.add_argument("--output", required=True, help="Output comparison JSON path")
    parser.add_argument("--html", action="store_true", help="Also emit HTML and Markdown reports")
    args = parser.parse_args()

    before_path = Path(args.before)
    after_path = Path(args.after)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    before_summary = _summary_metrics(_read_json(before_path))
    after_summary = _summary_metrics(_read_json(after_path))
    comparison = _build_comparison(before_summary, after_summary)
    comparison["before_path"] = str(before_path)
    comparison["after_path"] = str(after_path)

    output_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.html:
        md_path = output_path.with_suffix(".md")
        html_path = output_path.with_suffix(".html")
        md_path.write_text(_render_markdown(comparison, before_path, after_path), encoding="utf-8")
        html_path.write_text(_render_html(comparison, before_path, after_path), encoding="utf-8")

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
