from __future__ import annotations

"""Summarize audit artifacts for a run_id.

Produces JSON, Markdown, and optional HTML for human review.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _collect_json_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*.json") if p.is_file()])


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Audit Summary",
        "",
        f"- Run root: `{summary['run_root']}`",
        f"- Executed at: `{summary['executed_at']}`",
        f"- Artifact files: `{summary['counts']['artifact_files']}`",
        f"- Audit files: `{summary['counts']['audit_files']}`",
        f"- Regression files: `{summary['counts']['regression_files']}`",
        "",
        "## Bundle Overview",
        "",
    ]
    for item in summary.get("bundle_overview", []):
        lines.extend([
            f"### {item['stage'] or 'unknown'}",
            f"- Path: `{item['path']}`",
            f"- Run ID: `{item['run_id']}`",
            f"- Items: `{item['item_count']}`",
            "",
        ])
    return "\n".join(lines)


def _render_html(summary: dict[str, Any]) -> str:
    rows = []
    for item in summary.get("bundle_overview", []):
        rows.append(
            "<tr>"
            f"<td>{item['stage']}</td>"
            f"<td>{item['run_id']}</td>"
            f"<td>{item['item_count']}</td>"
            f"<td><code>{item['path']}</code></td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Audit Summary</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; color: #111827; background: #f9fafb; }}
    .card {{ background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px; margin-bottom: 16px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 10px 12px; text-align: left; font-size: 14px; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    code {{ background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Audit Summary</h1>
    <p>Run root: <code>{summary['run_root']}</code></p>
    <p>Executed at: <code>{summary['executed_at']}</code></p>
    <p>Artifact files: <code>{summary['counts']['artifact_files']}</code> | Audit files: <code>{summary['counts']['audit_files']}</code> | Regression files: <code>{summary['counts']['regression_files']}</code></p>
  </div>
  <div class="card">
    <table>
      <thead>
        <tr><th>Stage</th><th>Run ID</th><th>Items</th><th>Path</th></tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </div>
</body>
</html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize audit artifacts per run_id")
    parser.add_argument("--run-root", required=True, help="Path to the run root")
    parser.add_argument("--output", default="", help="Optional summary output path")
    parser.add_argument("--html", action="store_true", help="Also write HTML summary")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    artifact_root = run_root / "artifacts"
    audit_root = artifact_root / "audit"
    regression_root = artifact_root / "regression"

    bundles = _collect_json_files(artifact_root)
    audit_files = _collect_json_files(audit_root)
    regression_files = _collect_json_files(regression_root)

    summary = {
        "run_root": str(run_root),
        "artifact_root": str(artifact_root),
        "audit_root": str(audit_root),
        "regression_root": str(regression_root),
        "executed_at": datetime.now().isoformat(),
        "counts": {
            "artifact_files": len(bundles),
            "audit_files": len(audit_files),
            "regression_files": len(regression_files),
        },
        "files": {
            "artifact_files": [str(p) for p in bundles],
            "audit_files": [str(p) for p in audit_files],
            "regression_files": [str(p) for p in regression_files],
        },
        "bundle_overview": [],
    }

    for path in bundles:
        data = _read_json(path)
        summary["bundle_overview"].append({
            "path": str(path),
            "stage": data.get("stage", ""),
            "run_id": data.get("run_id", ""),
            "item_count": len(data.get("items", {}) or {}),
        })

    output = Path(args.output) if args.output else run_root / "artifacts" / f"audit_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = output.with_suffix(".md")
    md_path.write_text(_render_markdown(summary), encoding="utf-8")

    if args.html:
        html_path = output.with_suffix(".html")
        html_path.write_text(_render_html(summary), encoding="utf-8")
        summary["html_report"] = str(html_path)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
