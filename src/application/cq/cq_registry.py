"""CQ registry (M1 种子 CQ).

Loads a CQ catalog from JSONL with *tolerant* semantics: malformed lines are
skipped and reported via ``warnings`` instead of raising, so a single bad
seed can never break a pipeline run (M2 零回归约束).

Strict loading lives in :class:`src.application.cq.cq_models.CQCatalog`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.application.cq.cq_models import CQ, CQStatus

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def default_catalog_path(project_root: Union[str, Path, None] = None) -> Path:
    """Return the seed CQ catalog path (``config/cq/seed_cq.jsonl``)."""
    root = Path(project_root) if project_root else PROJECT_ROOT
    return root / "config" / "cq" / "seed_cq.jsonl"


def default_mapping_path(project_root: Union[str, Path, None] = None) -> Path:
    """Return the CQ ↔ regression sample mapping path (``config/cq/cq_mapping.json``)."""
    root = Path(project_root) if project_root else PROJECT_ROOT
    return root / "config" / "cq" / "cq_mapping.json"


def load_cq_mapping(
    path: Union[str, Path, None] = None,
    project_root: Union[str, Path, None] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load CQ ↔ regression sample bindings (M3); ``{}`` when missing/malformed.

    Expected JSON shape::

        {"bindings": [{"cq_id": "CQ-0004", "sample_id": "mbom-du-v2",
                       "status": "bound", "note": "..."}, ...]}

    Returns a ``{cq_id: binding}`` dict. Tolerant (same M1 semantics as the
    catalog loader): a missing or malformed file yields ``{}`` — the mapping
    is auxiliary metadata and must never break a run.
    """
    try:
        path = Path(path) if path else default_mapping_path(project_root)
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        bindings = payload.get("bindings") if isinstance(payload, dict) else None
        mapping: Dict[str, Dict[str, Any]] = {}
        for binding in bindings or []:
            if isinstance(binding, dict) and str(binding.get("cq_id") or "").strip():
                mapping[str(binding["cq_id"]).strip()] = binding
        return mapping
    except Exception:  # noqa: BLE001 - auxiliary metadata, never break a run
        return {}


class CQRegistry:
    """In-memory view over a CQ catalog with frozen-only filters for gating."""

    def __init__(self, items: Optional[List[CQ]] = None, warnings: Optional[List[str]] = None):
        self.items: List[CQ] = list(items or [])
        self.warnings: List[str] = list(warnings or [])

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path: Union[str, Path]) -> "CQRegistry":
        """Load from JSONL; skip blank/malformed lines (recorded in ``warnings``).

        A missing file yields an empty registry (with a warning) rather than
        an exception — callers treat "no CQ input" as a normal state.
        """
        path = Path(path)
        items: List[CQ] = []
        warnings: List[str] = []
        if not path.exists():
            warnings.append(f"cq catalog not found: {path}")
            return cls(items=items, warnings=warnings)
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(CQ.model_validate(json.loads(line)))
            except Exception as exc:  # noqa: BLE001 - tolerant loader
                warnings.append(f"line {lineno}: skipped invalid CQ ({exc})")
        return cls(items=items, warnings=warnings)

    # ----------------------------------------------------------------- query
    def get(self, cq_id: str) -> Optional[CQ]:
        for item in self.items:
            if item.cq_id == cq_id:
                return item
        return None

    def by_status(self, status: Union[str, CQStatus]) -> List[CQ]:
        key = status.value if isinstance(status, CQStatus) else str(status)
        return [item for item in self.items if item.status == key]

    def frozen(self) -> List[CQ]:
        """Only ``frozen`` CQs enter M2 validation / M3 gap detection / M5 gating."""
        return self.by_status(CQStatus.FROZEN)

    def active(self) -> List[CQ]:
        """Non-deprecated CQs (M4 抽取约束 uses these as the focus vocabulary:
        draft/reviewed questions still signal extraction interest)."""
        return [item for item in self.items if item.status != CQStatus.DEPRECATED.value]

    def focus_vocabulary(self) -> Dict[str, Any]:
        """Aggregate CQ 关注词表 (M4) from active CQs.

        Returns ``{"entities", "relations", "attributes", "sources"}`` where
        term lists are de-duplicated (case-insensitive, first-seen order) and
        ``sources`` maps each lowercased term kind ``"<kind>:<term>"`` to the
        list of originating cq_ids — the hint/coverage reports use it to
        trace a missing term back to the business question that needs it.
        """
        entities: List[str] = []
        relations: List[str] = []
        attributes: List[str] = []
        sources: Dict[str, List[str]] = {}

        def _add(term: str, bucket: List[str], kind: str, cq_id: str) -> None:
            key = f"{kind}:{term.strip().lower()}"
            if key not in sources:
                sources[key] = []
                bucket.append(term.strip())
            if cq_id not in sources[key]:
                sources[key].append(cq_id)

        for cq in self.active():
            for term in cq.entities:
                _add(term, entities, "entity", cq.cq_id)
            for term in cq.relations:
                _add(term, relations, "relation", cq.cq_id)
            for term in cq.attributes:
                _add(term, attributes, "attribute", cq.cq_id)
        return {
            "entities": entities,
            "relations": relations,
            "attributes": attributes,
            "sources": sources,
        }

    def __len__(self) -> int:
        return len(self.items)
