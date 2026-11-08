"""Outputs retention for pipeline run directories (F-2, 2026-08-15).

Policy: keep the newest N run directories (MAIN-*/REG-*/EXTRACT-*/PIPELINE-*)
under the pipeline runs root, plus any run marked with a ``.keep`` marker
file; delete older ones. Dry-run by default — pass ``--apply`` to remove.

Regression baselines under ``baselines/`` and the governed master ontology are
NOT touched (approved baselines are protected). Use:
    python scripts/cleanup_outputs_retention.py            # dry run
    python scripts/cleanup_outputs_retention.py --keep 12  # dry run keep 12
    python scripts/cleanup_outputs_retention.py --keep 12 --apply
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_PIPELINE_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "runs" / "ontology" / "pipeline"

_RUN_PREFIX_RE = re.compile(r"^(MAIN|EXTRACT|PIPELINE|REG)-")


def collect_runs(root: Path) -> list[Path]:
    """Return run directories (matching MAIN-/EXTRACT-/PIPELINE- prefix)."""
    if not root.exists():
        return []
    runs = []
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and _RUN_PREFIX_RE.match(entry.name):
            runs.append(entry)
    # newest first (dir names are timestamp-ordered)
    runs.sort(key=lambda p: p.name, reverse=True)
    return runs


def keep_set(runs: list[Path], keep: int) -> set[Path]:
    """Newest ``keep`` runs plus anything carrying a ``.keep`` marker."""
    kept: set[Path] = set(runs[:keep])
    for run in runs:
        if (run / ".keep").exists():
            kept.add(run)
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_PIPELINE_ROOT)
    parser.add_argument("--keep", type=int, default=10, help="newest N runs to keep (default 10)")
    parser.add_argument("--apply", action="store_true", help="actually delete; default is dry-run")
    args = parser.parse_args()

    runs = collect_runs(args.root)
    if not runs:
        print(f"No run directories found under {args.root}")
        return 0

    kept = keep_set(runs, args.keep)
    victims = [r for r in runs if r not in kept]

    kept_size = sum(_dir_size(r) for r in kept)
    victim_size = sum(_dir_size(r) for r in victims)

    print(f"Runs under {args.root}: total={len(runs)} keep={len(kept)} candidate-to-delete={len(victims)}")
    print(f"Estimated size: kept={kept_size/1024:.1f} MB candidates={victim_size/1024:.1f} MB")

    for run in victims[:20]:
        size = _dir_size(run)
        action = "[DELETE]" if args.apply else "[would delete]"
        print(f"  {action} {run.name} ({size/1024:.1f} MB)")
    if len(victims) > 20:
        print(f"  ... and {len(victims) - 20} more")

    if victims and not args.apply:
        print("\nDry-run: nothing deleted. Re-run with --apply to remove candidates.")
    if victims and args.apply:
        # 2026-10-13: --apply 此前只打印不删除（静默空操作）。补齐真实删除，
        # 删全部 victims（旧版只 print victims[:20]，且从不 rmtree）。
        freed = 0
        for run in victims:
            freed += _dir_size(run)
            shutil.rmtree(run, ignore_errors=True)
        print(f"\nDeleted {len(victims)} run dirs, freed {freed/1024/1024:.1f} MB.")
    if not victims:
        print("\nNothing to delete (all below retention threshold or marked .keep).")
    return 0


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


if __name__ == "__main__":
    sys.exit(main())