from __future__ import annotations

"""Utilities for extracting simple structured metadata from raw text."""

import re
from dataclasses import dataclass, asdict
from typing import Any


DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
FREQUENCY_RE = re.compile(r"\b(?:every\s+\d+\s+(?:weeks?|months?)|per\s+week|per\s+month|weekly|monthly|quarterly|daily|annually|yearly|biweekly)\b", re.I)
SECURITY_RE = re.compile(r"\b(Proprietary|Confidential|Internal|Public|Restricted|Sensitive)\b", re.I)
CONDITION_RE = re.compile(r"\b(?:if needed|ad-?hoc|when needed|optional|as required|where applicable|if applicable)\b", re.I)
TIME_WINDOW_RE = re.compile(r"\b(?:by\s+\d{4}|q[1-4]|quarter\s+\d|month\s+\d+|week\s+\d+)\b", re.I)
NUMBERED_FORMULA_RE = re.compile(r"\b[A-Z][A-Z0-9_]{1,}\s*[=:]\s*[^\n]+")
PERCENTAGE_RE = re.compile(r"\b\d+(?:\.\d+)?%\b")
RANGE_RE = re.compile(r"\b\d+\s*(?:to|-|–)\s*\d+\b")


SOURCE_HINTS = (
    "wip",
    "git",
    "ibp",
    "inventory",
    "meeting",
    "forum",
    "bom",
    "variant",
    "occurrence",
    "position",
    "mbom",
    "bop",
    "plant",
    "kpi",
    "report",
    "benchmark",
    "safety stock",
    "supply",
    "factory",
    "site",
    "compliance",
)


@dataclass
class ExtractedMetadata:
    effective_dates: list[str]
    security_classes: list[str]
    frequencies: list[str]
    conditions: list[str]
    formulas: list[str]
    source_hints: list[str]
    time_windows: list[str]
    percentages: list[str]
    ranges: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_metadata(text: str) -> ExtractedMetadata:
    text = text or ""
    effective_dates = sorted(set(DATE_RE.findall(text)))
    security_classes = sorted(set(m.group(1).title() for m in SECURITY_RE.finditer(text)))
    frequencies = sorted(set(m.group(0) for m in FREQUENCY_RE.finditer(text)))
    conditions = sorted(set(m.group(0) for m in CONDITION_RE.finditer(text)))
    formulas = sorted(set(m.strip() for m in re.findall(r"[^\n=]{3,}=\s*[^\n]+", text)))
    formulas.extend(sorted(set(m.group(0).strip() for m in NUMBERED_FORMULA_RE.finditer(text))))
    formulas = sorted(set(formulas))
    time_windows = sorted(set(m.group(0) for m in TIME_WINDOW_RE.finditer(text)))
    percentages = sorted(set(PERCENTAGE_RE.findall(text)))
    ranges = sorted(set(m.group(0) for m in RANGE_RE.finditer(text)))
    source_hints = []
    text_lower = text.lower()
    for hint in SOURCE_HINTS:
        if hint in text_lower:
            source_hints.append(hint)
    ontology_hint = any(token in text_lower for token in ("position", "occurrence", "variant", "mbom", "bop"))
    if ontology_hint:
        source_hints.append("ontology")
    metadata = ExtractedMetadata(
        effective_dates=effective_dates,
        security_classes=security_classes,
        frequencies=frequencies,
        conditions=conditions,
        formulas=formulas,
        source_hints=sorted(set(source_hints)),
        time_windows=time_windows,
        percentages=percentages,
        ranges=ranges,
    )
    return metadata
