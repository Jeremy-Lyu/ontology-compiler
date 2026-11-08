#!/usr/bin/env python3
"""Check markdown docs for broken relative links and non-existent backtick code paths.

Part of docs governance (PROJECT_OPTIMIZATION_EXECUTION_CHECKLIST.md 9.9 P2):
- Increase Markdown relative link check.
- Increase backtick code path existence check.

Scope:
- ``active`` (default, wired into ``make check``): scans ``docs/``.
  That directory now holds only the contracts the runtime still cites.
- ``--full``: same tree; kept so existing invocations stay valid.

Rules:
1. Markdown relative links that point to a local file/image are resolved
   against the doc's directory, then the docs root, then known base prefixes.
2. Backtick code spans that look like a physical file path are resolved the
   same way. Only paths that exist nowhere are reported as missing.
3. Multiline spans (shell snippets, ASCII tree diagrams), wildcard paths
   (*_adapter.py), and pure concept names without an extension are skipped.
4. Windows `file:///` links (carried from another machine) are warnings.

Exit code: 0 when no issues in the scanned scope.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
ACTIVE_FILES = (
    PROJECT_ROOT / "docs",
)
TRACKED_EXTENSIONS = {
    ".py", ".md", ".yaml", ".yml", ".json", ".txt", ".sh", ".toml", ".cfg",
    ".ini", ".env", ".csv", ".tsv", ".xlsx", ".xls", ".docx", ".doc", ".pptx",
    ".ppt", ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".html", ".css",
    ".js", ".ts", ".sql", ".rst", ".drawio",
}
BASE_PREFIXES = ("", "src", "ontology_compiler", "core", "docs", "scripts", "config", "skills", "data")

LINK_RE = re.compile(r"(?:^|\s)\[([^\]]*)\]\(([^)]+)\)")
CODE_PATH_RE = re.compile(r"`([^`]+)`")
WINDOWS_FILE_LINK_RE = re.compile(r"^file:///", re.IGNORECASE)
SHELL_NOISE_RE = re.compile(r"^(bash|sh|python|cd|ls|pip|make|\./|\$ )", re.IGNORECASE)


def active_docs() -> list[Path]:
    docs: list[Path] = []
    for entry in ACTIVE_FILES:
        if entry.is_dir():
            docs.extend(sorted(entry.rglob("*.md")))
        elif entry.exists():
            docs.append(entry)
    return docs


def all_docs() -> list[Path]:
    return sorted(DOCS_DIR.rglob("*.md"))


def resolve_path(candidate: str, doc: Path | None = None) -> Path | None:
    """Resolve a repo-relative candidate against base prefixes."""
    stripped = candidate.strip().strip("`\"'<>")
    if not stripped or stripped in {"/", "\\"}:
        return None
    if any(token in stripped for token in ("*", "?", "{", "}", "<", ">")):
        return None
    if SHELL_NOISE_RE.match(stripped):
        return None
    path = Path(stripped)
    if path.is_absolute():
        return None
    # Relative to the referencing doc first (e.g. ../DOCS_INDEX.md).
    if doc is not None:
        resolved = (doc.parent / path).resolve()
        if resolved.exists():
            return resolved
    for prefix in BASE_PREFIXES:
        resolved = PROJECT_ROOT / prefix / path if prefix else PROJECT_ROOT / path
        if resolved.exists():
            return resolved
    return None


def check_markdown_links(doc: Path, issues: list[str], warnings: list[str]) -> None:
    try:
        text = doc.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        issues.append(f"{doc.relative_to(PROJECT_ROOT)}: unreadable: {e}")
        return

    for match in LINK_RE.finditer(text):
        target = match.group(2).strip()
        if not target or target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        if WINDOWS_FILE_LINK_RE.match(target):
            warnings.append(f"{doc.relative_to(PROJECT_ROOT)}: windows file:// link")
            continue
        target = target.split()[0].split("#")[0]
        if not target:
            continue
        if resolve_path(target, doc) is not None:
            continue
        issues.append(
            f"{doc.relative_to(PROJECT_ROOT)}: broken relative link -> {target}"
        )


def check_backtick_paths(doc: Path, issues: list[str]) -> None:
    try:
        text = doc.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return

    for match in CODE_PATH_RE.finditer(text):
        cand = match.group(1).strip()
        if "\n" in cand or "\r" in cand:
            continue
        if "/" not in cand and "\\" not in cand:
            continue
        if any(token in cand for token in ("*", "?", "{", "}", "<", ">")):
            continue
        # Planned-artifact markers: paths expected to be created by an
        # upcoming milestone (e.g. "`src/x.py`（计划产物）"), not broken refs.
        # Also accept "（旧路径）" annotations that document a historical
        # (non-existent) path by design.
        after = text[match.end():match.end() + 12]
        if "计划产物" in after or "旧路径" in after or "planned" in after.lower():
            continue
        suffix = Path(cand).suffix.lower()
        if suffix and suffix not in TRACKED_EXTENSIONS:
            continue
        if not suffix:
            continue
        if resolve_path(cand, doc) is not None:
            continue
        issues.append(
            f"{doc.relative_to(PROJECT_ROOT)}: missing code path `{cand}`"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="scan all docs including frozen areas")
    args = parser.parse_args()

    issues: list[str] = []
    warnings: list[str] = []
    docs = all_docs() if args.full else active_docs()

    for doc in docs:
        check_markdown_links(doc, issues, warnings)
        check_backtick_paths(doc, issues)

    check_source_of_truth_uniqueness(docs, issues)

    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for warning in sorted(set(warnings)):
            print(f"  - {warning}")

    scope = "full" if args.full else "active"
    if issues:
        print(f"Found {len(issues)} doc issue(s) in {scope} scope:")
        for issue in issues:
            print(f"  - {issue}")
        print(f"Docs link/path check: FAIL ({len(docs)} docs scanned)")
        return 1

    print(f"Docs link/path check: OK ({scope} scope, {len(docs)} docs scanned)")
    return 0


def check_source_of_truth_uniqueness(docs: list[Path], issues: list[str]) -> None:
    """Ensure the active surface keeps a single current doc per topic.

    Only the H1 title of each document counts as a role claim, so navigation
    prose mentioning a role is not misflagged.
    """
    titles: dict[str, list[str]] = {}
    roles: dict[str, list[str]] = {"execution checklist": [], "status snapshot": [], "navigation map": [], "docs index": []}

    def _role(key: str, markers: tuple[str, ...], h1: str) -> None:
        if any(marker in h1.lower() for marker in markers):
            roles[key].append(str(rel))

    for doc in docs:
        try:
            text = doc.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        rel = doc.relative_to(PROJECT_ROOT)
        head = "\n".join(text.splitlines()[:8]).upper()
        if any(marker in head for marker in ("SUPERSEDED", "HISTORICAL", "DEPRECATED", "已过期", "已退役", "CONSOLIDATED", "已收敛")):
            continue
        h1 = ""
        for line in text.splitlines():
            if line.startswith("# "):
                h1 = line[2:].strip()
                break
        titles.setdefault(h1.lower(), []).append(str(rel))
        _role("execution checklist", ("execution checklist", "执行清单"), h1)
        _role("status snapshot", ("status snapshot", "状态快照"), h1)
        _role("navigation map", ("docs map", "navigation", "导航图"), h1)
        _role("docs index", ("documentation index", "docs index", "文档索引"), h1)

    for title, files in titles.items():
        if len(files) > 1 and title:
            issues.append(
                f"source-of-truth: duplicate title '{title}' in {', '.join(files)}"
            )
    for role, files in roles.items():
        if len(files) > 1:
            issues.append(
                f"source-of-truth: multiple docs claim '{role}' role: {', '.join(files)}"
            )


if __name__ == "__main__":
    sys.exit(main())