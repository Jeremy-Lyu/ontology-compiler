"""Schema-driven output evaluators for discovered skills (P1-2, 2026-10-16).

Rationale
---------
Every discoverable skill ships a ``.ontology-skill/skill.json`` that already
declares an ``output_schema`` (a small JSON-Schema subset: ``type`` /
``required`` / ``enum`` / ``minimum`` / ``maximum`` / ``properties`` /
``items``). Hand-writing one bespoke evaluator per skill would duplicate that
declaration and drift from it.

This module derives the evaluator **from the declared contract** instead — the
same "schema is the single source of truth" rule the packet schemas follow. A
skill becomes evaluable the moment its ``skill.json`` declares
``output_schema``; no per-skill code is required.

Contract
--------
``evaluate_output(schema, output) -> list[str]``
    Pure. Returns a list of human-readable violations (empty == conforms).
    Unknown/absent schema => ``[]`` (nothing to check, not a failure).

Deliberately *not* a full JSON-Schema implementation: only the keywords the
skill metadata actually uses are honored, and unknown keywords are ignored
(forward-compatible, never a false failure).
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional

__all__ = [
    "evaluate_output",
    "evaluate_skill_document",
    "is_evaluable",
    "SUPPORTED_KEYWORDS",
]

SUPPORTED_KEYWORDS = frozenset(
    {"type", "required", "enum", "minimum", "maximum", "properties", "items"}
)

_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
}


def _type_ok(value: Any, declared: str) -> bool:
    expected = _TYPE_MAP.get(declared)
    if expected is None:
        return True  # unknown type name: do not judge
    if declared in ("number", "integer") and isinstance(value, bool):
        return False  # bool is an int subclass in Python; never a number here
    return isinstance(value, expected)


def _fmt(path: str) -> str:
    return path or "$"


def _as_properties(schema: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """Return the effective properties map for a skill output schema.

    The discovered-skill metadata uses two shapes for ``output_schema``:

    1. JSON-Schema-ish — carries keywords (``type`` / ``properties`` /
       ``required`` / ``enum`` / ...). Handled by :func:`_check` directly.
    2. **Flat field map** — ``{"gate_id": {"type": "string"}, ...}`` with no
       wrapper. This is what most ``skill.json`` files ship, so it must be
       recognized or every output would silently pass.
    """
    if any(k in schema for k in SUPPORTED_KEYWORDS):
        return schema.get("properties") if isinstance(schema.get("properties"), Mapping) else None
    return schema


def _check(schema: Mapping[str, Any], value: Any, path: str, out: List[str]) -> None:
    if not isinstance(schema, Mapping):
        return

    flat = _as_properties(schema)
    if flat is schema and not isinstance(value, Mapping):
        # Flat field map declared, but caller passed a non-mapping output.
        out.append(f"{_fmt(path)}: expected object, got {type(value).__name__}")
        return

    declared = schema.get("type")
    if isinstance(declared, str) and not _type_ok(value, declared):
        out.append(f"{_fmt(path)}: expected {declared}, got {type(value).__name__}")
        return  # further keyword checks are meaningless on a type mismatch

    enum = schema.get("enum")
    if isinstance(enum, list) and enum and value not in enum:
        out.append(f"{_fmt(path)}: {value!r} not in enum {enum}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        lo, hi = schema.get("minimum"), schema.get("maximum")
        if isinstance(lo, (int, float)) and value < lo:
            out.append(f"{_fmt(path)}: {value} < minimum {lo}")
        if isinstance(hi, (int, float)) and value > hi:
            out.append(f"{_fmt(path)}: {value} > maximum {hi}")

    if isinstance(value, dict):
        required = schema.get("required")
        if isinstance(required, list):
            for key in required:
                if key not in value:
                    out.append(f"{_fmt(path)}: missing required key {key!r}")
        props = flat if isinstance(flat, Mapping) else schema.get("properties")
        if isinstance(props, Mapping):
            for key, sub in props.items():
                if key in value:
                    _check(sub, value[key], f"{path}.{key}" if path else key, out)

    if isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, Mapping):
            for i, item in enumerate(value):
                _check(items, item, f"{path}[{i}]", out)


def evaluate_output(
    schema: Optional[Mapping[str, Any]],
    output: Any,
) -> List[str]:
    """Validate *output* against a skill's declared ``output_schema``.

    Returns violations; ``[]`` when the output conforms **or** when there is no
    schema to check against.
    """
    if not schema:
        return []
    violations: List[str] = []
    _check(schema, output, "", violations)
    return violations


def evaluate_skill_document(
    skill_document: Mapping[str, Any],
    output: Any,
) -> List[str]:
    """Convenience wrapper taking a whole parsed ``skill.json`` document."""
    return evaluate_output(skill_document.get("output_schema"), output)


def is_evaluable(skill_document: Mapping[str, Any]) -> bool:
    """A skill is evaluable iff its metadata declares a non-empty schema."""
    schema = skill_document.get("output_schema")
    return isinstance(schema, Mapping) and bool(schema)
