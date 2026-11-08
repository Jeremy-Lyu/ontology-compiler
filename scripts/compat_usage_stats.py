#!/usr/bin/env python3
"""Compatibility-layer usage statistics (9.10 P3 第一步：调用量统计).

Scans active code (src/, ontology_compiler/, skills/, scripts/, core/, tests/)
excluding archived/, docs/, outputs/, .venv and counts, per tracked
compatibility target, the number of referencing files and total references.

The result is printed as a review table and, when --json is given, written to
outputs/audits/compat_usage_stats.json for later diffing (retirement progress).

Exit code: 0 always (reporting tool, not a gate).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOTS = ("src", "ontology_compiler", "skills", "scripts", "core", "tests")
SKIP_DIRPARTS = {"archived", ".archive", "outputs", "node_modules", ".venv", "__pycache__", ".git"}

# name -> (module path tokens to match as dotted target, symbol tokens)
# Each target is tracked by matching module-or-symbol references.
TARGETS = {
    "dual_pipeline.DualPipeline": (("pipeline.dual_pipeline", "dual_pipeline"), ("DualPipeline",)),
    "all_ports facade": (("ports.all_ports", "application.ports.all_ports"), ("all_ports",)),
    "skill_registry.harness": (("harness.skill_registry", "ontology_compiler.harness"), ("build_default_skill_registry",)),
    "skill_registry.legacy(skills)": (("skills.skill_registry",), ("get_skill_registry",)),
    "skill_registry.R38-QA": (("skills.registry",), ("Skill",)),
    "skill_registry.agent": (("agent.orchestration.skill_registry",), ("ORCHESTRATION_SKILLS",)),
    "asset_packet_service": (("asset_packet_service",), ("AssetPacketService", "RetrievalAssetPacketService", "get_asset_packet_service")),
    "use_cases.legacy stack": (("pipeline_use_cases", "_stage_execution", "stage_loop_with_hooks_use_case"), ("StageCommandResult",)),
    "runtime_paths 2份": (("utils.runtime_paths", "config.get_runtime_paths"), ("get_runtime_paths",)),
    "normalize_text 多份": (("utils.json_parser", "text_utils"), ("normalize_text",)),
    "tokenize 多份": (("utils.json_parser", "chunk_indexing", "unified_query"), ("tokenize",)),
    # 2026-10-16 诊断 P0-1: extraction 旧路径垫片 9 个（到期 2026-11-30 删除，只减不增）。
    "extraction 旧路径垫片": (
        ("services.entity_normalizer", "services.extraction_packet_adapter",
         "services.extraction_postprocessing", "services.extraction_steps",
         "services.llm_response_parser", "services.prompt_service",
         "services.text_chunking", "services.text_utils",
         "services.true_unified_extractor"),
        (),
    ),
}

IMPORT_RE = re.compile(r"^\s*(?:from\s+(\S+)\s+import|import\s+(\S+))", re.MULTILINE)


def _skip(path: Path) -> bool:
    return any(part in SKIP_DIRPARTS for part in path.parts)


def _python_files() -> list[Path]:
    files: list[Path] = []
    self_path = PROJECT_ROOT / "scripts" / "compat_usage_stats.py"
    for root in SCAN_ROOTS:
        base = PROJECT_ROOT / root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if not _skip(path) and path != self_path:
                files.append(path)
    return files


def _dotted_module(targets: tuple[str, ...]) -> str:
    """Normalise a target module fragment for 'from x import' matching."""
    return targets[0] if targets else ""


# Files that ARE the compat target itself (re-export shims): their own
# docstrings / mapping tables mention the symbols by necessity. Counting them
# as "usage" punishes consolidation, so they are excluded per-target.
# Files that ARE the compat target itself (re-export shims / canonical owners):
# their own docstrings / mapping tables / definitions mention the symbols by
# necessity. Counting them as "usage" punishes consolidation, so they are
# excluded per-target.
SELF_FILES: dict[str, set[str]] = {
    "normalize_text 多份": {
        "ontology_compiler/services/text_utils.py",
        "ontology_compiler/services/extraction/text_utils.py",
    },
    "skill_registry.harness": {"ontology_compiler/harness/skill_registry.py"},
    "skill_registry.legacy(skills)": {"skills/skill_registry.py"},
    # P0-6-5: canonical tokenize owner + lazy shim mention the symbol in
    # docstrings/mapping tables; all real call sites already import canonical.
    "tokenize 多份": {
        "src/application/rag/chunk_indexing.py",
        "ontology_compiler/services/text_utils.py",
        "ontology_compiler/services/extraction/text_utils.py",
    },
    # Retirement scheduler names DualPipeline in its schedule table only.
    "dual_pipeline.DualPipeline": {"scripts/retire_dated_shims.py"},
}


def scan() -> dict[str, dict]:
    stats: dict[str, dict] = {}
    files = _python_files()
    for name, (mods, symbols) in TARGETS.items():
        referencing: list[str] = []
        test_referencing: list[str] = []
        total_refs = 0
        test_refs = 0
        for path in files:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            count = 0
            for line in text.splitlines():
                if line.lstrip().startswith(("#", "//")):
                    continue
                matched = False
                for mod in mods:
                    leaf = mod.split(".")[-1]
                    # import leaf / from leaf import ... / from a.b.leaf import ...
                    if re.search(rf"(?:^|\s)(?:import|from)\s+[\.\w]*{re.escape(leaf)}(?:\s|\.|$|,)", line) and mod in line:
                        matched = True
                        break
                if not matched:
                    for sym in symbols:
                        if re.search(rf"\b{re.escape(sym)}\b", line):
                            matched = True
                            break
                if matched:
                    count += 1
            if count:
                rel = str(path.relative_to(PROJECT_ROOT))
                if rel in SELF_FILES.get(name, set()):
                    continue
                # 2026-10-16: 测试代码无法“迁移离开”被测 API（测试就是要调用它），
                # 因此 tests/ 的引用不计入 retirement-debt（total_references）。
                # 它们仍被报告（test_files/test_references）以便可见，但不影响
                # 门禁的“只减不增”判定。生产引用继续严格计数。
                if rel.startswith("tests/"):
                    test_referencing.append(rel)
                    test_refs += count
                    continue
                referencing.append(rel)
                total_refs += count
        stats[name] = {
            "referencing_files": len(referencing),
            "total_references": total_refs,
            "files": referencing,
            "test_files": test_referencing,
            "test_references": test_refs,
        }
    return stats


def line_counts() -> dict[str, int]:
    """Current line totals for the two main code directories (net-reduction baseline)."""
    result: dict[str, int] = {}
    for root in ("src", "ontology_compiler", "core", "skills"):
        total = 0
        base = PROJECT_ROOT / root
        if base.exists():
            for path in sorted(base.rglob("*")):
                if path.is_file() and not _skip(path) and path.suffix in {".py", ".yaml", ".yml", ".json", ".toml", ".ini", ".md"}:
                    try:
                        total += len(path.read_text(encoding="utf-8", errors="replace").splitlines())
                    except Exception:
                        pass
        result[root] = total
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="write machine-readable report")
    args = parser.parse_args()

    stats = scan()
    line_stats = line_counts()

    print("=" * 100)
    print("Compatibility-layer usage statistics (import/reference scan)")
    print("=" * 100)
    for name, s in stats.items():
        refs = s["total_references"]
        marker = "ZERO-CALLER" if refs == 0 else f"{refs} refs"
        print(f"\n[{name}] -> {marker} (files: {s['referencing_files']})")
        for f in s["files"]:
            print(f"    {f}")

    print("\n" + "-" * 100)
    print("Line-count baseline (net-reduction tracking)")
    print("-" * 100)
    for root, total in line_stats.items():
        print(f"  {root:<20} {total:>6} lines")
    grand = sum(line_stats.values())
    print(f"  {'TOTAL':<20} {grand:>6} lines")

    if args.json:
        out_dir = PROJECT_ROOT / "outputs" / "audits"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "compat_usage_stats.json"
        payload = {"scan_roots": list(SCAN_ROOTS), "stats": stats, "line_counts": line_stats}
        out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nReport written to {out_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())