#!/usr/bin/env python3
"""Execute dated shim retirements when their calendar date is reached.

Schedule (from PROJECT_CURRENT_STATUS_AND_ROADMAP §4-4):

* 2026-11-15 — text_utils re-export shims (DONE early 2026-11-07 → tombstones)
* 2026-11-30 — skills.skill_registry facade (DONE 2026-11-08 → harness-backed router)
* 2027-01-31 — DualPipeline observation window ends (archive if still 0 callers)

Usage::

    python scripts/retire_dated_shims.py            # report status
    python scripts/retire_dated_shims.py --apply    # act on due items
"""
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archived" / "retired_dated"


@dataclass(frozen=True)
class ShimItem:
    key: str
    due: date
    paths: tuple[str, ...]
    status: str  # pending | done | observe
    note: str


ITEMS: tuple[ShimItem, ...] = (
    ShimItem(
        key="text_utils",
        due=date(2026, 11, 15),
        paths=(
            "ontology_compiler/services/text_utils.py",
            "ontology_compiler/services/extraction/text_utils.py",
        ),
        status="done",
        note="Tombstones raise ImportError; bodies in archived/retired_20261107/",
    ),
    ShimItem(
        key="skills.skill_registry",
        due=date(2026, 11, 30),
        paths=("skills/skill_registry.py",),
        status="done",
        note="Archived 2026-11-08; SkillsRouter uses harness; ImportError tombstone remains",
    ),
    ShimItem(
        key="dual_pipeline",
        due=date(2027, 1, 31),
        paths=("src/pipeline/dual_pipeline.py",),
        status="observe",
        note="Archive after 3 observation rounds with 0 production callers",
    ),
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="archive due observe items")
    ap.add_argument("--today", type=str, default="", help="override today YYYY-MM-DD")
    args = ap.parse_args()
    today = date.fromisoformat(args.today) if args.today else date.today()

    print(f"Dated shim status (today={today.isoformat()}):")
    due_apply = []
    for item in ITEMS:
        due_flag = "DUE" if today >= item.due else "WAIT"
        print(f"  [{due_flag}] {item.key} due={item.due} status={item.status}")
        print(f"         {item.note}")
        for rel in item.paths:
            exists = (ROOT / rel).exists()
            print(f"         - {rel} {'OK' if exists else 'MISSING'}")
        if item.status == "observe" and today >= item.due:
            due_apply.append(item)

    if not args.apply:
        return 0

    if not due_apply:
        print("Nothing to apply.")
        return 0

    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for item in due_apply:
        stamp = ARCHIVE / f"{item.key}_{today.isoformat()}"
        stamp.mkdir(parents=True, exist_ok=True)
        for rel in item.paths:
            src = ROOT / rel
            if not src.exists():
                continue
            dest = stamp / Path(rel).name
            shutil.move(str(src), str(dest))
            print(f"Archived {rel} -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
