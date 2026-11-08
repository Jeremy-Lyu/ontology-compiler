#!/usr/bin/env python3
"""Gate A freshness check (no LLM required).

Compares the live master ontology + governed meaning surface against the
committed regression baseline expectations:

* RELATED_TO share vs manifest ``max_related_to_share`` (strictest sample)
* origin_kind coverage on master (input/injected/master_reused/derived)
* optional CQ answerability smoke for key manufacturing questions

Does **not** replace a full ``make regression`` LLM run — it only flags drift
signals so "mechanism green ≠ business green" is visible between full runs.

Usage::

    python scripts/gate_a_freshness_check.py
    python scripts/gate_a_freshness_check.py --require-answerable
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from regression_artifact_diagnosis import master_source_breakdown  # noqa: E402

DEFAULT_MASTER = ROOT / "outputs" / "runs" / "ontology" / "master_ontology"
DEFAULT_BASELINE = (
    ROOT / "outputs" / "runs" / "ontology" / "baselines" / "regression_summary_baseline.json"
)
DEFAULT_MANIFEST = ROOT / "config" / "regression_samples.json"

SMOKE_QUESTIONS = (
    "MBOM 如何由 EBOM 派生",
    "什么是 MBOM",
    "BOM 与制造有什么关系",
)


def _manifest_related_to_caps(manifest: Dict[str, Any]) -> List[float]:
    caps: List[float] = []
    for group in manifest.get("sample_groups") or []:
        for sample in group.get("samples") or []:
            expected = sample.get("expected") or {}
            if "max_related_to_share" in expected:
                try:
                    caps.append(float(expected["max_related_to_share"]))
                except (TypeError, ValueError):
                    continue
    return caps


# Live aggregated master is not a single-sample gate; sample caps (often 0.05)
# are WARN thresholds. Hard FAIL uses a master-level ceiling.
DEFAULT_MASTER_RELATED_TO_CAP = 0.15


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master-dir", type=Path, default=DEFAULT_MASTER)
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument(
        "--require-answerable",
        action="store_true",
        help="fail if any smoke question is not answerable on governed packet",
    )
    ap.add_argument(
        "--max-related-to-share",
        type=float,
        default=DEFAULT_MASTER_RELATED_TO_CAP,
        help=f"hard FAIL ceiling for live master RELATED_TO share (default {DEFAULT_MASTER_RELATED_TO_CAP})",
    )
    args = ap.parse_args()

    failures: List[str] = []
    warnings: List[str] = []

    if not args.master_dir.exists():
        failures.append(f"master dir missing: {args.master_dir}")
        print("FAIL (Gate A freshness):")
        for f in failures:
            print(f"  - {f}")
        return 2

    summary = master_source_breakdown(args.master_dir)
    print("[gate-a] master origin breakdown")
    print(
        f"  entities={summary['entity_total']} {summary['entities_by_source']} "
        f"extracted={summary['extracted_entity_share']} injected={summary['injected_entity_share']}"
    )
    print(
        f"  relations={summary['relation_total']} {summary['relations_by_source']} "
        f"RELATED_TO share={summary['related_to_share']}"
    )

    if summary["entity_source_coverage"] < 0.5:
        failures.append(
            f"entity origin_kind coverage too low: {summary['entity_source_coverage']}"
        )
    if summary["relation_source_coverage"] < 0.5:
        failures.append(
            f"relation origin_kind coverage too low: {summary['relation_source_coverage']}"
        )

    sample_warn = None
    if args.manifest.exists():
        try:
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            caps = _manifest_related_to_caps(manifest)
            if caps:
                sample_warn = min(caps)
        except Exception as exc:
            warnings.append(f"manifest unreadable ({exc})")

    share = summary["related_to_share"]
    hard_cap = float(args.max_related_to_share)
    if share > hard_cap:
        failures.append(f"RELATED_TO share {share} exceeds master cap {hard_cap}")
    else:
        print(f"[gate-a] RELATED_TO share OK vs master cap {hard_cap}")
        if sample_warn is not None and share > sample_warn:
            warnings.append(
                f"RELATED_TO share {share} above strictest sample cap {sample_warn} "
                f"(informational; single-sample gates ≠ live master)"
            )

    if args.baseline.exists():
        try:
            baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
            print(
                f"[gate-a] baseline file: passed={baseline.get('passed_count')} "
                f"failed={baseline.get('failed_count')} at={baseline.get('executed_at')}"
            )
            age_note = (
                "baseline is from 2026-09-09 era — schedule a full LLM regression "
                "(`make regression`) when provider is available to refresh Gate A"
            )
            if str(baseline.get("executed_at") or "").startswith("2026-09"):
                warnings.append(age_note)
        except Exception as exc:
            warnings.append(f"baseline unreadable: {exc}")
    else:
        warnings.append(f"baseline missing: {args.baseline}")

    # Meaning-surface smoke (governed packet).
    try:
        from src.application.rag.semantic_surface import check_answerable

        for q in SMOKE_QUESTIONS:
            report = check_answerable(q)
            status = "OK" if report.answerable else "GAP"
            print(
                f"[gate-a] answerable[{status}] conf={report.confidence:.2f} q={q!r}"
            )
            if args.require_answerable and not report.answerable:
                failures.append(f"not answerable: {q} gaps={report.gaps}")
    except Exception as exc:
        msg = f"meaning surface unavailable ({exc})"
        if args.require_answerable:
            failures.append(msg)
        else:
            warnings.append(msg)

    for w in warnings:
        print(f"[gate-a] WARN: {w}")
    if failures:
        print("FAIL (Gate A freshness):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("Gate A freshness check: OK (structural; full LLM regression still recommended)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
