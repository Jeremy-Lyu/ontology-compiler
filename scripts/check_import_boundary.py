from __future__ import annotations

"""Import boundary gate (execution plan Phase 0 / Phase 2).

Blocks NEW cross-layer violations while tolerating the historical ones that
are listed in a checked-in baseline file. Baseline semantics follow
docs/NEXT_PHASE_ARCHITECTURE_OPTIMIZATION_EXECUTION_PLAN_20260811.md (7.1):

    ontology_compiler -> src.application | src.pipeline | src.infrastructure : forbidden
    src.domain        -> src.application  | src.infrastructure                 : forbidden
    src.pipeline      -> ontology_compiler                                     : allowed only via facade (reported)

Usage:
    python3 scripts/check_import_boundary.py               # verify (fails on NEW violations)
    python3 scripts/check_import_boundary.py --write-baseline   # snapshot current state
    python3 scripts/check_import_boundary.py --report     # print all violations, exit 0
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "config" / "quality" / "import_boundary_baseline.json"

SCAN_DIRS = ["ontology_compiler", "src", "core"]
IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([a-zA-Z_][\w.]*)")

# (module_root_pattern, forbidden_prefix) -> rule name
RULES = [
    (r"^ontology_compiler($|\.)", "src.application"),
    (r"^ontology_compiler($|\.)", "src.pipeline"),
    (r"^ontology_compiler($|\.)", "src.infrastructure"),
    (r"^src\.domain($|\.)", "src.application"),
    (r"^src\.domain($|\.)", "src.infrastructure"),
]

REPORT_ONLY_RULES = [
    (r"^src\.pipeline($|\.)", "ontology_compiler"),
]


def _scan() -> dict[str, list[str]]:
    """Return {violation_key: [file:line relative paths]} for all rules.

    A finding is recorded when the *importing file's package root* matches the
    rule's source pattern AND the *imported module* starts with the forbidden
    prefix (e.g. ontology_compiler/*.py importing src.application.*).
    """
    findings: dict[str, list[str]] = {}
    for root_dir in SCAN_DIRS:
        base = ROOT / root_dir
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts or "/archived" in str(path) or "/deprecated" in str(path):
                continue
            rel = str(path.relative_to(ROOT).as_posix())
            source_root = path.relative_to(ROOT).parts[0]
            for lineno, line in enumerate(
                path.open(encoding="utf-8", errors="ignore"), start=1
            ):
                m = IMPORT_RE.match(line)
                if not m:
                    continue
                imported = m.group(1)
                for pattern, forbidden in RULES:
                    if re.match(pattern, source_root) and imported.startswith(forbidden):
                        rule = f"{pattern}|{forbidden}"
                        findings.setdefault(rule, []).append(f"{rel}:{lineno}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Import boundary gate")
    parser.add_argument("--write-baseline", action="store_true", help="snapshot current violations as baseline")
    parser.add_argument("--report", action="store_true", help="print all violations and exit 0")
    args = parser.parse_args()

    findings = _scan()

    if args.write_baseline:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(
            json.dumps({k: sorted(v) for k, v in sorted(findings.items())}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote baseline with {sum(len(v) for v in findings.values())} violations to {BASELINE}")
        return 0

    baseline: dict[str, list[str]] = {}
    if BASELINE.exists():
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

    new_violations: list[str] = []
    print(f"Import boundary check: {len(findings)} rule/finding buckets in current scan")
    for rule in sorted(set(list(findings) + list(baseline))):
        current = sorted(set(findings.get(rule, [])))
        known = sorted(set(baseline.get(rule, [])))
        for item in current:
            if item not in known:
                new_violations.append(f"  {rule}: {item}")

    if args.report:
        for rule, items in sorted(findings.items()):
            for item in items:
                print(f"  {rule}: {item}")
        return 0

    if new_violations:
        print("NEW import boundary violations (not in baseline):")
        print("\n".join(new_violations))
        print("\nIf approved, refresh baseline with --write-baseline.")
        return 1
    print("OK: no new import boundary violations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())