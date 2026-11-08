#!/usr/bin/env python3
"""Print input / injected / master_reused breakdown for a master_ontology dir.

Usage::

    python scripts/analyze_master_origins.py
    python scripts/analyze_master_origins.py --master-dir outputs/runs/ontology/master_ontology
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from regression_artifact_diagnosis import master_source_breakdown  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--master-dir",
        type=Path,
        default=ROOT / "outputs" / "runs" / "ontology" / "master_ontology",
    )
    ap.add_argument("--json", action="store_true", help="machine-readable only")
    args = ap.parse_args()
    if not args.master_dir.exists():
        print(f"FAIL: master dir missing: {args.master_dir}", file=sys.stderr)
        return 2
    summary = master_source_breakdown(args.master_dir)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    print(f"master: {summary['master_dir']}")
    print(
        f"entities={summary['entity_total']} "
        f"by_source={summary['entities_by_source']} "
        f"extracted_share={summary['extracted_entity_share']} "
        f"injected_share={summary['injected_entity_share']}"
    )
    print(
        f"relations={summary['relation_total']} "
        f"by_source={summary['relations_by_source']} "
        f"extracted_share={summary['extracted_relation_share']} "
        f"injected_share={summary['injected_relation_share']}"
    )
    print(
        f"RELATED_TO count={summary['related_to_count']} "
        f"share={summary['related_to_share']}"
    )
    top = sorted(
        (summary.get("relations_by_type") or {}).items(),
        key=lambda kv: (-kv[1], kv[0]),
    )[:8]
    print("top relation types:", ", ".join(f"{k}:{v}" for k, v in top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
