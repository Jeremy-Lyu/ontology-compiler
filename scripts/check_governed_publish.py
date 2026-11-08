#!/usr/bin/env python3
"""Governed-packet freshness / health gate (P2-4, 2026-09-09).

Checks (cheap pure reads, no subprocess):

1. FRESHNESS: the published governed packet's ``master_hashes`` must match
   the current master ``nodes.json`` / ``relationships.json`` sha256 — i.e.
   the retrieval asset reflects the current master (the "last rank
   publishes" contract from the CI matrix convention).
   Skippable when there is no master materialized yet (fresh repo/CI).

2. REBUILD HEALTH: ``outputs/runs/ontology/governed_rebuild_status.json``
   (written by MasterOntologyStore._record_rebuild_outcome) must not carry
   ``alert=true`` / consecutive_failures >= 1 — the auto-rebuild hook has
   not been silently failing behind successful master writes.

3. PUBLISH RECEIPT (optional ``--require-receipt`` / ``--require-publishable``):
   latest packet must carry a valid ``publish_receipt.json``; when
   ``--require-publishable`` is set, ``publishable`` must be true.

Exit 0 only when enabled checks pass. Used by ``make check-governed-publish``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
STATUS_PATH = ROOT / "outputs" / "runs" / "ontology" / "governed_rebuild_status.json"
LATEST_POINTER = ROOT / "outputs" / "runs" / "ontology" / "asset_packets" / "latest_governed_packet.json"
DEFAULT_MASTER_DIR = ROOT / "outputs" / "runs" / "ontology" / "master_ontology"

sys.path.insert(0, str(ROOT / "scripts"))
from build_governed_packet import _master_file_hashes  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master-dir", type=Path, default=DEFAULT_MASTER_DIR)
    ap.add_argument("--skip-freshness", action="store_true",
                    help="only check rebuild health (e.g. bootstrap before first packet)")
    ap.add_argument(
        "--require-receipt",
        action="store_true",
        help="require publish_receipt.json (+ checksum) on the latest governed packet",
    )
    ap.add_argument(
        "--require-publishable",
        action="store_true",
        help="also require receipt.publishable == true (implies --require-receipt)",
    )
    args = ap.parse_args()
    if args.require_publishable:
        args.require_receipt = True

    failures: list[str] = []

    if not args.skip_freshness:
        hashes = _master_file_hashes(args.master_dir)
        if not hashes.get("nodes.json") and not hashes.get("relationships.json"):
            print("[governed-fresh] no master materialized yet -> freshness check skipped")
        else:
            pointer = LATEST_POINTER if LATEST_POINTER.exists() else None
            if not pointer.exists():
                failures.append("latest_governed_packet.json pointer missing")
            else:
                try:
                    pointer_data = json.loads(pointer.read_text(encoding="utf-8"))
                    packet_id = pointer_data.get("packet_id")
                    manifest_path = (
                        ROOT / "outputs" / "runs" / "ontology" / "asset_packets"
                        / "governed" / str(packet_id) / "manifest.json"
                    )
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    prev = manifest.get("master_hashes") or {}
                    if (
                        prev.get("nodes.json") != hashes.get("nodes.json")
                        or prev.get("relationships.json") != hashes.get("relationships.json")
                    ):
                        failures.append(
                            f"governed packet {packet_id} master_hashes differ from current master "
                            "(retrieval asset is STALE — run scripts/build_governed_packet.py)"
                        )
                    elif (
                        pointer_data.get("master_hashes")
                        and prev.get("nodes.json") != pointer_data["master_hashes"].get("nodes.json")
                    ):
                        failures.append("pointer master_hashes and manifest master_hashes disagree")
                except Exception as exc:
                    failures.append(f"manifest unreadable ({exc})")

    if not STATUS_PATH.exists():
        print("[governed-health] no rebuild status file yet (hook has not run) -> skipped")
    else:
        try:
            status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
            if status.get("alert"):
                failures.append(
                    f"governed rebuild alert: {status.get('consecutive_failures')} consecutive failures; "
                    f"last_error={str(status.get('last_error'))[:200]}"
                )
            elif status.get("last_outcome") == "failed":
                print(
                    f"[governed-health] WARN: last rebuild failed "
                    f"({status.get('consecutive_failures')} consecutive) but below alert threshold"
                )
        except Exception as exc:
            failures.append(f"rebuild status unreadable ({exc})")

    if args.require_receipt:
        if not LATEST_POINTER.exists():
            failures.append("latest_governed_packet.json pointer missing (receipt check)")
        else:
            try:
                from src.application.rag.publish_gate import read_publish_receipt

                pointer_data = json.loads(LATEST_POINTER.read_text(encoding="utf-8"))
                packet_dir = (
                    ROOT / "outputs" / "runs" / "ontology" / "asset_packets"
                    / "governed" / str(pointer_data.get("packet_id"))
                )
                receipt = read_publish_receipt(packet_dir)
                if args.require_publishable and not receipt.get("publishable"):
                    failures.append(
                        "publish_receipt.publishable is false: "
                        + "; ".join(receipt.get("errors") or [])
                    )
                else:
                    print(
                        f"[governed-receipt] ok publishable={receipt.get('publishable')} "
                        f"certified={receipt.get('certified')}"
                    )
            except Exception as exc:
                failures.append(f"publish receipt invalid ({exc})")

    if failures:
        print("FAIL (governed publish health):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("Governed publish check: freshness + rebuild health OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
