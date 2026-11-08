#!/usr/bin/env python3
"""Compat usage gate (P0-6): fail when any tracked compat target grows.

Compares the current scan (same engine as scripts/compat_usage_stats.py)
against the committed baseline outputs/audits/compat_usage_stats.json:

- total_references strictly non-increasing per target (retirement progress)
- new referencing files are reported (informational, non-failing)

Usage:
    python3 scripts/check_compat_usage_gate.py              # verify, exit 0/1
    python3 scripts/check_compat_usage_gate.py --update-baseline   # refresh baseline (P0-6 batches only)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = PROJECT_ROOT / "outputs" / "audits" / "compat_usage_stats.json"

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from compat_usage_stats import scan  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-baseline", action="store_true", help="refresh the committed baseline (P0-6 batches only)")
    args = parser.parse_args()

    current = scan()
    if args.update_baseline:
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8")) if BASELINE_PATH.exists() else {"scan_roots": [], "stats": {}, "line_counts": {}}
        baseline["stats"] = current
        BASELINE_PATH.write_text(json.dumps(baseline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Baseline refreshed: {BASELINE_PATH}")
        return 0

    if not BASELINE_PATH.exists():
        print(f"FAIL: baseline missing: {BASELINE_PATH}")
        return 1
    baseline_stats = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))["stats"]

    failures: list[str] = []
    for name, base in baseline_stats.items():
        cur = current.get(name, {"total_references": 0, "referencing_files": 0, "files": []})
        if cur["total_references"] > base["total_references"]:
            failures.append(
                f"{name}: {base['total_references']} -> {cur['total_references']} refs "
                f"({base['referencing_files']} -> {cur['referencing_files']} files)"
            )
        new_files = sorted(set(cur.get("files", [])) - set(base.get("files", [])))
        if new_files:
            print(f"[info] {name}: new referencing files: {', '.join(new_files[:5])}" + (" ..." if len(new_files) > 5 else ""))

    if failures:
        print("FAIL: compat usage grew (retirement regression):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"Compat usage gate: {len(baseline_stats)} targets, all non-increasing. OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
