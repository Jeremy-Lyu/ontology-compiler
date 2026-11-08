"""Common utilities for pipeline stages."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

PIPELINE_VALIDATION_THRESHOLDS = {
    "relationship_min_confidence": 0.7,
    "required_evidence_fields": ("evidence_ref", "citation", "rule_no"),
    "dedup_similarity_threshold": 0.9,
}


def fragment_has_content(fragment_data: dict[str, Any]) -> bool:
    """True if a fragment carries nodes/instances/relationships (i.e. non-empty)."""
    return bool(
        fragment_data.get("nodes")
        or fragment_data.get("instances")
        or fragment_data.get("relationships")
    )


def resolve_stage_fragment(stage_results: Dict[str, Any]) -> str:
    """Return an on-disk fragment/master path from dedup/merge/extract results ("" if none).

    Mirrors the stage input wiring: extract exposes a fragment_paths list while
    merge/dedup expose master_path. Used by merge/relationship handlers so they
    can feed the real artifact to the downstream stage even when extraction was
    served from the document cache (whose files live outside the run root).
    """
    for stage_name in ["dedup", "merge", "extract"]:
        prev_stage = stage_results.get(stage_name)
        if not (prev_stage and prev_stage.data):
            continue
        master_path = prev_stage.data.get("master_path")
        if master_path and Path(master_path).is_file():
            return master_path
        fragment_paths = prev_stage.data.get("fragment_paths") or []
        for fragment_path in fragment_paths:
            if fragment_path and Path(fragment_path).is_file():
                return fragment_path
        fragment_path = prev_stage.data.get("fragment_path")
        if fragment_path and Path(fragment_path).is_file():
            return fragment_path
    return ""


def build_backup_metadata(source_path: str, backup_path: str, run_id: str = "", stage: str = "") -> dict[str, Any]:
    return {
        "source_path": source_path,
        "backup_path": backup_path,
        "run_id": run_id,
        "stage": stage,
        "created_at": datetime.now().isoformat(),
    }


@dataclass
class StageResult:
    stage: str
    success: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    artifact_paths: list[str] = field(default_factory=list)
    run_dir: str = ""
    fragment_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "success": self.success,
            "message": self.message,
            "data": self.data,
            "elapsed_seconds": self.elapsed_seconds,
            "artifact_paths": self.artifact_paths,
            "run_dir": self.run_dir,
            "fragment_path": self.fragment_path,
        }

    def __repr__(self) -> str:
        return f"StageResult({self.stage}, {self.success}, {self.message})"


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default or {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default or {}


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # P1-04: Handle permission errors with retry (OneDrive lock issue)
    import time as _time
    last_err = None
    for attempt in range(3):
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return
        except PermissionError as e:
            last_err = e
            _time.sleep(0.5 * (attempt + 1))
        except OSError as e:
            last_err = e
            _time.sleep(0.5 * (attempt + 1))
    # Final attempt: write to temp file then rename
    if last_err:
        try:
            import tempfile, os
            fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            if path.exists():
                path.unlink()
            os.rename(tmp_path, str(path))
            return
        except Exception:
            pass
        # If all retries fail, log and skip (don't crash pipeline)
        import logging
        logging.getLogger(__name__).warning(f"[EXPORT] Failed to write {path}: {last_err}, skipping")


def _latest_file(base: Path, pattern: str) -> Path | None:
    candidates = list(base.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _normalize_rel_type(rel: dict[str, Any]) -> str:
    return str(rel.get("type") or rel.get("relation_type") or rel.get("rel_type") or "").upper().strip()


def _entity_layer_value(n: dict[str, Any]) -> str:
    return n.get("layer") or n.get("layer_type") or n.get("canonical_layer") or "L1"


def _iter_master_entities(master: dict[str, Any], require_meta_id: bool = False) -> list[dict[str, Any]]:
    seen: set[str] = set()
    entities: list[dict[str, Any]] = []

    def _add(items):
        for item in items or []:
            if not isinstance(item, dict):
                continue
            eid = str(item.get("meta_id") or item.get("entity_id") or item.get("id") or "")
            if require_meta_id and not eid:
                continue
            if eid and eid in seen:
                continue
            if eid:
                seen.add(eid)
            entities.append(item)

    _add(master.get("nodes", []))

    for bucket in (master.get("entities") or {}).values():
        if isinstance(bucket, list):
            _add(bucket)

    for legacy_key in (
        "layer0_metadata",
        "layer1_concepts",
        "layer2_scenarios",
        "layer2_logic",
        "layer2_definitions",
        "layer25_mapping_rules",
        "instances",
    ):
        _add(master.get(legacy_key))

    return entities


def _load_master_entities(master: dict[str, Any]) -> tuple[list[str], list[str]]:
    nodes: list[str] = []
    instances: list[str] = []
    entities = _iter_master_entities(master)
    for item in entities:
        eid = str(item.get("meta_id") or item.get("entity_id") or item.get("id") or "")
        layer = _entity_layer_value(item)
        if layer == "L3":
            instances.append(eid)
        else:
            nodes.append(eid)
    return nodes, instances


def _normalize_fragment(fragment: dict[str, Any]) -> dict[str, Any]:
    return {
        "nodes": fragment.get("nodes", []),
        "instances": fragment.get("instances", []),
        "relationships": fragment.get("relationships") or fragment.get("edges") or [],
    }


def _layer_distribution(items: list[dict[str, Any]]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for item in items:
        layer = _entity_layer_value(item)
        dist[layer] = dist.get(layer, 0) + 1
    return dist


def _load_domain_keywords() -> dict[str, list[str]]:
    catalog_path = PROJECT_ROOT / "config" / "extraction" / "extractor_catalogs.yaml"
    if not catalog_path.exists():
        return {}
    try:
        import yaml as _yaml
        data = _yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
        return data.get("domain_keywords", {})
    except Exception:
        return {}


def infer_document_type(text: str) -> tuple[str, float, list[str], list[str]]:
    """Infer a coarse document type for routing.

    Returns (document_type, confidence, secondary_types, rationale).
    """
    text_l = (text or "").lower()
    scores = {
        "benchmark": 0,
        "mbom": 0,
        "process_governance": 0,
    }
    rationale: list[str] = []

    patterns = {
        "benchmark": ["benchmark", "kpi", "scope", "safety stock", "inventory control", "visualization", "reporting", "tool"],
        "mbom": ["mbom", "ebom", "bop", "delivery unit", "variant", "occurrence", "position", "plant mbom", "product mbom"],
        "process_governance": ["governance", "rule", "workflow", "gate", "audit", "approval", "exception", "owner"],
    }
    for dtype, kws in patterns.items():
        for kw in kws:
            if kw in text_l:
                scores[dtype] += 1
                rationale.append(f"{dtype}:{kw}")

    best = max(scores, key=scores.get) if scores else "unknown"
    best_score = scores.get(best, 0)
    if best_score == 0:
        return "unknown", 0.0, [], []

    sorted_types = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    secondary = [dtype for dtype, score in sorted_types[1:] if score > 0]
    confidence = round(best_score / max(sum(scores.values()), 1), 3)
    return best, confidence, secondary, rationale[:12]


def _infer_domain_from_files(source_path: Path, document_analysis: Optional[dict] = None) -> tuple[str, dict[str, int]]:
    """Infer domain from files with optional document analysis supplement.
    
    Args:
        source_path: Path to file or folder
        document_analysis: Optional analysis result from DocumentAnalyzer containing
                          document_type, key_topics, and language for enhanced domain detection
    
    Returns:
        (inferred_domain, scores) tuple
    """
    domain_keywords = _load_domain_keywords()
    if not domain_keywords:
        return "", {}

    all_text = ""
    if source_path.is_file():
        try:
            all_text += source_path.read_text(encoding="utf-8", errors="replace").lower()
        except Exception:
            pass
    else:
        for ext in [".md", ".txt", ".pdf", ".pptx", ".ppt", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".json"]:
            for file_path in source_path.rglob(f"*{ext}"):
                try:
                    all_text += file_path.read_text(encoding="utf-8", errors="replace").lower() + " "
                except Exception:
                    continue

    scores: dict[str, int] = {}
    for domain, keywords in domain_keywords.items():
        score = 0
        for keyword in keywords:
            score += all_text.count(keyword.lower())
        scores[domain] = score

    # [CRITICAL FIX T0-2]: Enhance domain detection with DocumentAnalyzer results
    if document_analysis:
        document_type = document_analysis.get("document_type", "")
        key_topics = document_analysis.get("key_topics", [])
        language = document_analysis.get("language", "")
        
        # Boost score based on document type
        type_bonus = {
            "technical": {"MANUFACTURING": 5, "OTA": 5, "SUPPLY_CHAIN": 3},
            "business": {"SUPPLY_CHAIN": 5, "FINANCE": 3, "SALES": 3},
            "contract": {"FINANCE": 5, "LEGAL": 3},
            "report": {"PERFORMANCE": 5, "FINANCE": 3},
        }
        if document_type in type_bonus:
            for domain, bonus in type_bonus[document_type].items():
                if domain in scores:
                    scores[domain] += bonus
        
        # Boost score based on key topics
        topic_bonus_map = {
            "diagnostic": "DIAGNOSTICS",
            "diagnostics": "DIAGNOSTICS",
            "fault": "DIAGNOSTICS",
            "failure": "DIAGNOSTICS",
            "education": "EDUCATION",
            "training": "EDUCATION",
            "learning": "EDUCATION",
            "academy": "EDUCATION",
            "course": "EDUCATION",
            "inventory": "SUPPLY_CHAIN",
            "supply": "SUPPLY_CHAIN",
            "logistics": "LOGISTICS",
            "performance": "PERFORMANCE",
            "kpi": "PERFORMANCE",
            "ota": "OTA",
            "software update": "OTA",
            "firmware": "OTA",
            "ecu": "OTA",
            "telematics": "OTA",
            "flash": "OTA",
            "calibration": "OTA",
            "manufacturing": "MANUFACTURING",
            "production": "MANUFACTURING",
            "plant floor": "MANUFACTURING",
            "assembly": "MANUFACTURING",
            "mes": "MANUFACTURING",
            "machine": "MANUFACTURING",
            "equipment": "MANUFACTURING",
        }
        for topic in key_topics:
            topic_lower = topic.lower()
            for topic_keyword, domain in topic_bonus_map.items():
                if topic_keyword in topic_lower and domain in scores:
                    scores[domain] += 3

    if not scores or max(scores.values()) == 0:
        return "", scores

    inferred_domain = max(scores, key=scores.get)
    return inferred_domain, scores


# ======================================================================
# Consolidated helpers moved from src/pipeline/attribute_projection.py
# and src/pipeline/entity_normalization.py (single source of truth).
# ======================================================================

def project_metadata_to_context(metadata: dict[str, Any] | None, context_metadata: dict[str, Any]) -> None:
    if not metadata:
        return
    context_metadata.setdefault("structured_metadata", {})
    structured = context_metadata["structured_metadata"]
    if isinstance(structured, dict):
        structured.update(metadata)


_ALIASES_PATH = PROJECT_ROOT / "config" / "entity_aliases.json"


def _load_aliases() -> list[dict[str, Any]]:
    if not _ALIASES_PATH.exists():
        return []
    try:
        data = json.loads(_ALIASES_PATH.read_text(encoding="utf-8"))
        return data.get("aliases", []) if isinstance(data, dict) else []
    except Exception:
        return []


def normalize_entity_name(name: str) -> tuple[str, str | None]:
    normalized = (name or "").strip()
    if not normalized:
        return normalized, None
    lower = normalized.lower()
    for item in _load_aliases():
        canonical = str(item.get("canonical", "")).strip()
        aliases = [str(a).strip().lower() for a in item.get("aliases", []) if str(a).strip()]
        if lower == canonical.lower():
            return canonical, None
        if lower in aliases:
            return canonical, normalized
    return normalized, None


def normalize_entity(entity: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(entity, dict):
        return entity
    entity = dict(entity)
    canonical_name, original_name = normalize_entity_name(str(entity.get("name", "")))
    entity["name"] = canonical_name
    if original_name and original_name != canonical_name:
        entity.setdefault("aliases", [])
        if original_name not in entity["aliases"]:
            entity["aliases"].append(original_name)
        entity["normalized_from"] = original_name
    if str(entity.get("name", "")).lower() == "concept" and str(entity.get("type", "")).lower() == "data":
        entity["type"] = "Concept"
    return entity
