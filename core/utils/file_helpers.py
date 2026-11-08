from __future__ import annotations

import functools
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def slug_text(text: str) -> str:
    return "-".join(
        part
        for part in "".join(ch.lower() if ch.isalnum() else "-" for ch in text).split("-")
        if part
    )


@functools.lru_cache(maxsize=256)
def load_json(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return json.loads(p.read_text(encoding="utf-8"))


def load_json_optional(path: str | Path) -> Any | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


@functools.lru_cache(maxsize=256)
def _load_with_mtime(p_str: str, mtime: float) -> Any:
    return json.loads(Path(p_str).read_text(encoding="utf-8"))


def load_json_cached(path: str | Path) -> Any | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        mtime = p.stat().st_mtime
        return _load_with_mtime(str(p), mtime)
    except Exception:
        return None


def load_json_required(path: str | Path) -> dict[str, Any]:
    result = load_json_optional(path)
    if result is None:
        raise FileNotFoundError(f"File not found or invalid JSON: {path}")
    return result if isinstance(result, dict) else {}


def load_json_object(path: str | Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"json root is not an object: {path}")
    return payload


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def resolve_path(
    value: str | Path,
    *,
    base_dir: str | Path | None = None,
    must_exist: bool = False,
) -> Path:
    p = Path(value)
    if not p.is_absolute() and base_dir is not None:
        p = Path(base_dir) / p
    p = p.resolve()
    if must_exist and not p.exists():
        raise FileNotFoundError(str(p))
    return p


def strip_timestamp_suffix(text: str) -> str:
    return re.sub(r"-\d{14}$", "", text)


def detect_source_profile_from_paths(paths: Any) -> dict[str, object]:
    excel_suffixes = {".xlsx", ".xls", ".csv"}
    doc_suffixes = {".md", ".docx", ".pptx", ".txt"}
    pdf_suffixes = {".pdf"}
    normalized_paths = [Path(p) for p in paths]
    has_excel = any(p.suffix.lower() in excel_suffixes for p in normalized_paths)
    has_doc = any(p.suffix.lower() in doc_suffixes for p in normalized_paths)
    has_pdf = any(p.suffix.lower() in pdf_suffixes for p in normalized_paths)
    primary = "doc"
    if has_excel and has_doc:
        primary = "mixed"
    elif has_excel:
        primary = "excel"
    elif has_pdf:
        primary = "pdf"
    return {
        "has_excel": has_excel,
        "has_doc": has_doc,
        "has_pdf": has_pdf,
        "primary_type": primary,
    }


def deduplicate_by_key(items: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        item_id = item.get(key)
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        deduped.append(item)
    return deduped


def deduplicate_relationships(
    relationships: List[Dict[str, Any]],
    source_key: str = "source",
    predicate_key: str = "predicate",
    target_key: str = "target",
) -> List[Dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for rel in relationships:
        identity = (
            str(rel.get(source_key) or ""),
            str(rel.get(predicate_key) or ""),
            str(rel.get(target_key) or ""),
        )
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(rel)
    return deduped
