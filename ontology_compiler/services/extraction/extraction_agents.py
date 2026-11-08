"""Extraction-side multi-agent skeleton (ENTITY_CLASSIFICATION_ROADMAP T3.1).

Six-role decomposition per docs/ENTITY_CLASSIFICATION_SPEC.md §9
and the refactor plan: classifier / entity / rule(L2.5) / relation / evidence /
reviewer. Design constraints follow the project's established patterns
(pipeline-first, explicit inputs, pure helpers, characterization locks):

- Roles are explicit extension points, NOT fake implementations. Entity +
  relation extraction delegate to the existing unified extractor behind an
  injectable callable; classification/evidence accept optional injectable
  providers and degrade to ``skipped`` runs when absent.
- The Reviewer is fully functional today: a pure SPEC-compliance checker over
  OntologyPacket-shaped dicts (required fields, enums, endpoint resolution,
  CQ provenance). It needs no LLM.
- The serial orchestrator composes roles in order and never raises: failures
  become per-role ``error`` entries so partial progress is observable.

DAG parallelization is deliberately deferred until prompts stabilize
(roadmap: 不在协议未稳前拆 agent).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from .extraction_packet_adapter import (
    adapt_extraction_json_to_packet,
    iter_packet_entities,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ROLE_CLASSIFIER",
    "ROLE_ENTITY",
    "ROLE_RULE_L25",
    "ROLE_RELATION",
    "ROLE_EVIDENCE",
    "ROLE_REVIEWER",
    "ALL_ROLES",
    "review_ontology_packet",
    "reviewer_recommendation",
    "dag_parallel_feasibility",
    "assess_cq_evidence_stability",
    "run_serial_extraction",
]

# --- role registry ----------------------------------------------------------

ROLE_CLASSIFIER = "classifier"
ROLE_ENTITY = "entity"          # covers entity+relation extraction today
ROLE_RULE_L25 = "rule_l25"      # L2.5 rules/mappings (skill-governed)
ROLE_RELATION = "relation"      # standalone narrow-prompt pass (future)
ROLE_EVIDENCE = "evidence"      # span binding / enrichment
ROLE_REVIEWER = "reviewer"      # pure SPEC compliance gate

ALL_ROLES = (
    ROLE_CLASSIFIER,
    ROLE_ENTITY,
    ROLE_RULE_L25,
    ROLE_RELATION,
    ROLE_EVIDENCE,
    ROLE_REVIEWER,
)

_VALID_LAYERS = {"L0", "L1", "L2", "L2.5", "L3"}
_VALID_STATUS = {"active", "deprecated", "proposed", "draft", "ghost", "archived"}
_VALID_PROVENANCE = {"llm", "skill", "rule", "inferred"}

# --- Reviewer: pure SPEC compliance checks ----------------------------------


def _issue(code: str, message: str, *, severity: str = "error", **extra: Any) -> Dict[str, Any]:
    out = {"code": code, "message": message, "severity": severity}
    out.update(extra)
    return out


def review_ontology_packet(packet: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Check a packet against SPEC §3/§4 field discipline. Pure function.

    Returns a list of issue dicts sorted by severity (error first). Checks:
    entity required fields (id/name/layer/type) as errors; missing layer_scope
    as a warning. Relation type and endpoints are errors; unresolved endpoints,
    missing layer_scope/evidence_ref/confidence are warnings. Unknown
    provenance is an error. CQ-dead relations are one info issue.
    """
    issues: List[Dict[str, Any]] = []

    entity_ids: set = set()
    for ent in iter_packet_entities(packet):
        eid = ent.get("id") or ""
        if eid:
            entity_ids.add(eid)
        path_prefix = f"entity[{eid or '?'}]"
        if not _first_of(ent, "id"):
            issues.append(_issue("ENTITY_MISSING_ID", f"{path_prefix}: id is required"))
        if not _first_of(ent, "name"):
            issues.append(_issue("ENTITY_MISSING_NAME", f"{path_prefix}: name is required"))
        layer = _first_of(ent, "layer")
        if not layer:
            issues.append(_issue("ENTITY_MISSING_LAYER", f"{path_prefix}: layer is required"))
        elif layer not in _VALID_LAYERS:
            issues.append(_issue("LAYER_INVALID", f"{path_prefix}: unknown layer {layer!r}"))
        if not _first_of(ent, "type"):
            issues.append(_issue("ENTITY_MISSING_TYPE", f"{path_prefix}: type (meta_class) is required"))
        if not _first_of(ent, "layer_scope"):
            issues.append(_issue(
                "ENTITY_MISSING_LAYER_SCOPE",
                f"{path_prefix}: layer_scope (spec|impl) is required",
                severity="warning",
            ))
        status = _first_of(ent, "status")
        if status and status not in _VALID_STATUS:
            issues.append(_issue("STATUS_INVALID", f"{path_prefix}: unknown status {status!r}"))
        conf = ent.get("confidence")
        if conf is not None and not (isinstance(conf, (int, float)) and 0.0 <= conf <= 1.0):
            issues.append(_issue("CONFIDENCE_OUT_OF_RANGE", f"{path_prefix}: confidence {conf!r} outside [0,1]"))

    dead = 0
    for idx, rel in enumerate(packet.get("relations") or []):
        path_prefix = f"relation[{idx}]"
        src = rel.get("source_id")
        tgt = rel.get("target_id")
        if not rel.get("relation_type"):
            issues.append(_issue("RELATION_MISSING_TYPE", f"{path_prefix}: relation_type is required"))
        if not src or not tgt:
            issues.append(_issue("RELATION_ENDPOINT_MISSING", f"{path_prefix}: endpoints are required"))
            continue
        if src not in entity_ids:
            issues.append(_issue(
                "ENDPOINT_UNRESOLVED",
                f"{path_prefix}: source endpoint {src!r} not among extracted entities",
                severity="warning", source_id=src,
            ))
        if tgt not in entity_ids:
            issues.append(_issue(
                "ENDPOINT_UNRESOLVED",
                f"{path_prefix}: target endpoint {tgt!r} not among extracted entities",
                severity="warning", target_id=tgt,
            ))
        prov = rel.get("provenance")
        if prov and prov not in _VALID_PROVENANCE:
            issues.append(_issue("PROVENANCE_INVALID", f"{path_prefix}: unknown provenance {prov!r}"))
        if not _first_of(rel, "layer_scope"):
            issues.append(_issue(
                "RELATION_MISSING_LAYER_SCOPE",
                f"{path_prefix}: layer_scope (spec|impl) is required",
                severity="warning",
            ))
        if not _first_of(rel, "evidence_ref"):
            issues.append(_issue(
                "RELATION_MISSING_EVIDENCE",
                f"{path_prefix}: evidence_ref is required",
                severity="warning",
            ))
        if rel.get("confidence") is None:
            issues.append(_issue(
                "RELATION_MISSING_CONFIDENCE",
                f"{path_prefix}: confidence is required",
                severity="warning",
            ))
        if not str(rel.get("source_cq") or "").strip():
            dead += 1

    total_rels = len(packet.get("relations") or [])
    if total_rels and dead:
        issues.append(_issue(
            "CQ_DEAD_RELATIONS",
            f"{dead}/{total_rels} relations carry no source_cq (CQ provenance registry)",
            severity="info",
            dead_relations=dead, total_relations=total_rels,
        ))

    order = {"error": 0, "warning": 1, "info": 2}
    issues.sort(key=lambda i: order.get(i["severity"], 3))
    return issues


def _first_of(obj: Dict[str, Any], key: str) -> str:
    val = obj.get(key)
    return str(val).strip() if isinstance(val, str) else ("" if val is None else str(val))


# --- serial orchestrator -----------------------------------------------------


def _run_entry(role: str, ok: bool, output: Any = None, error: str = "", skipped: bool = False) -> Dict[str, Any]:
    return {"role": role, "ok": ok, "skipped": skipped, "output": output, "error": error}


# --- P2-3: narrow-pass hooks (relation→entity, opt-in) --------------------


def _bind_evidence_to_entities(entities: List[Dict[str, Any]], raw_text: str) -> List[Dict[str, Any]]:
    """Attach short evidence snippets to entities lacking explicit evidence.

    Appends at most one sentence from ``raw_text`` containing the entity name
    as ``evidence`` / ``provenance`` so downstream Reviewer and the CQ
    evidence check have something to work with. Deterministic, no LLM.
    Prepend-like fix: the evidence binding was missing in the prototype and is
    now done here for the evidence+relation narrow passes.
    """
    if not raw_text or not raw_text.strip():
        return entities
    lines = [ln.strip() for ln in (raw_text or "").splitlines() if ln.strip()]
    out: List[Dict[str, Any]] = []
    for ent in entities:
        name = str(ent.get("name") or ent.get("meta_name") or "").strip()
        if not name or ent.get("evidence") or ent.get("provenance"):
            out.append(ent)
            continue
        snippet = ""
        for ln in lines:
            if name.lower() in ln.lower():
                snippet = ln[:400]
                break
        if snippet:
            ent = {**ent, "evidence": snippet, "provenance": {"origin": "deterministic_narrow_evidence"}}
        out.append(ent)
    return out


def _bind_evidence_to_relations(relations: List[Dict[str, Any]], raw_text: str) -> List[Dict[str, Any]]:
    """Attach evidence to relations lacking it.

    Uses the same deterministic sentence scan as entities; side-effect-free:
    only annotates items missing an ``evidence``/``provenance`` field.
    """
    if not raw_text or not raw_text.strip():
        return relations
    lines = [ln.strip() for ln in (raw_text or "").splitlines() if ln.strip()]
    out: List[Dict[str, Any]] = []
    for rel in relations:
        if rel.get("evidence") or rel.get("provenance"):
            out.append(rel)
            continue
        blob = " ".join(str(v) for v in rel.values() if isinstance(v, str)).lower()
        snippet = ""
        for ln in lines:
            low = ln.lower()
            if any(tok in low for tok in blob.split() if len(tok) >= 3):
                snippet = ln[:400]
                break
        if snippet:
            rel = {**rel, "evidence": snippet, "provenance": {"origin": "deterministic_narrow_evidence"}}
        out.append(rel)
    return out


NARROW_RELATION_ENABLED_ENV = "EXTRACTION_NARROW_RELATION"
NARROW_ENTITY_ENABLED_ENV = "EXTRACTION_NARROW_ENTITY"
DAG_PARALLEL_ENABLED_ENV = "EXTRACTION_DAG_PARALLEL"
# Reviewer controls (P2-3c): gating policy for the packet.
# - fail mode: issues with severity=="error" make the packet non-releasable
#   and expose a fallback (merged-stage packet) when available.
# - warn mode (default): issues are reported but do not downgrade the
#   packet — the gate still returns a recommendation the caller can honor.
REVIEWER_FAIL_ON_ERROR_ENV = "REVIEWER_FAIL_ON_ERROR"

# Stability thresholds for auto-enabling entity narrow / DAG.
_CQ_STABLE_MIN_RATIO = 0.80
_EVIDENCE_COMPLETE_MIN_RATIO = 0.80
_MIN_RELATIONS_FOR_STABILITY = 1


def _env_truthy(env_name: str, *, default: bool = False) -> bool:
    """Parse a boolean env flag. Unset → ``default`` (P2-3 relation defaults ON)."""
    import os
    raw = os.environ.get(env_name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _env_tri_state(env_name: str, *, default: str = "off") -> str:
    """Return ``on`` / ``off`` / ``auto`` for gated features."""
    import os
    raw = os.environ.get(env_name)
    if raw is None or not str(raw).strip():
        return default
    value = str(raw).strip().lower()
    if value in ("1", "true", "yes", "on"):
        return "on"
    if value in ("0", "false", "no", "off"):
        return "off"
    if value == "auto":
        return "auto"
    return default


def _collect_relations(payload: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    for key in ("relations", "relationships"):
        items = payload.get(key)
        if isinstance(items, list):
            return [r for r in items if isinstance(r, dict)]
    return []


def _collect_entities(payload: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    entities = payload.get("entities")
    if isinstance(entities, list):
        return [e for e in entities if isinstance(e, dict)]
    if isinstance(entities, dict):
        out: List[Dict[str, Any]] = []
        for items in entities.values():
            if isinstance(items, list):
                out.extend(e for e in items if isinstance(e, dict))
        return out
    out = []
    for key in (
        "nodes",
        "L0_entities",
        "L1_entities",
        "L2_entities",
        "L2_5_entities",
        "L25_entities",
        "L3_entities",
        "instances",
    ):
        items = payload.get(key)
        if isinstance(items, list):
            out.extend(e for e in items if isinstance(e, dict))
    return out


def assess_cq_evidence_stability(
    payload: Optional[Dict[str, Any]],
    *,
    cq_min_ratio: float = _CQ_STABLE_MIN_RATIO,
    evidence_min_ratio: float = _EVIDENCE_COMPLETE_MIN_RATIO,
    min_relations: int = _MIN_RELATIONS_FOR_STABILITY,
) -> Dict[str, Any]:
    """Grade P0 CQ provenance + evidence coverage on a legacy/packet dict.

    Used to auto-enable ``EXTRACTION_NARROW_ENTITY=auto`` and
    ``EXTRACTION_DAG_PARALLEL=auto`` once serial quality is stable.
    """
    relations = _collect_relations(payload)
    entities = _collect_entities(payload)
    rel_count = len(relations)
    if rel_count < min_relations:
        return {
            "p0_cq_stable": False,
            "evidence_complete": False,
            "relation_count": rel_count,
            "entity_count": len(entities),
            "cq_ratio": 0.0,
            "evidence_ratio": 0.0,
            "reason": f"need >= {min_relations} relations for stability assessment",
        }

    with_cq = sum(1 for r in relations if str(r.get("source_cq") or r.get("cq_id") or "").strip())
    with_ev = 0
    for r in relations:
        if (
            str(r.get("evidence_ref") or "").strip()
            or r.get("evidence")
            or r.get("provenance")
        ):
            with_ev += 1
    # Entity evidence also contributes (shared evidence plane).
    ent_with_ev = sum(
        1
        for e in entities
        if str(e.get("evidence_ref") or "").strip() or e.get("evidence") or e.get("provenance")
    )
    cq_ratio = with_cq / rel_count
    # Blend relation + entity evidence; prefer relation coverage.
    evidence_ratio = with_ev / rel_count
    if entities:
        evidence_ratio = max(evidence_ratio, ent_with_ev / max(len(entities), 1))

    p0_cq_stable = cq_ratio >= cq_min_ratio
    evidence_complete = evidence_ratio >= evidence_min_ratio
    return {
        "p0_cq_stable": p0_cq_stable,
        "evidence_complete": evidence_complete,
        "relation_count": rel_count,
        "entity_count": len(entities),
        "cq_ratio": round(cq_ratio, 3),
        "evidence_ratio": round(evidence_ratio, 3),
        "reason": (
            "stable"
            if (p0_cq_stable and evidence_complete)
            else "CQ or evidence coverage below threshold"
        ),
    }


def _resolve_gated_flag(mode: str, *, stable: bool) -> bool:
    if mode == "on":
        return True
    if mode == "off":
        return False
    return bool(stable)


def reviewer_recommendation(
    issues: List[Dict[str, Any]],
    *,
    fail_on_error: Optional[bool] = None,
) -> Dict[str, Any]:
    """Compute the Reviewer's releasability recommendation (P2-3c gate, pure).

    Precedence hierarchy (Reviewer > relation > entity):
    - ``answerable = False`` iff ``fail_on_error`` and there is an error-severity issue.
    - Otherwise the packet is answerable; issues are carried as warnings.

    The pipeline caller owns the fallback choice; this function only returns the
    gate signal so the decision is testable without the pipeline.
    """
    if fail_on_error is None:
        fail_on_error = _env_truthy(REVIEWER_FAIL_ON_ERROR_ENV)
    has_error = any(i.get("severity") == "error" for i in (issues or []))
    if fail_on_error and has_error:
        return {
            "answerable": False,
            "verdict": "degrade",
            "reason": "reviewer gate: error-severity issues present",
            "fallback": "merged_stage_packet",
            "precedence": "reviewer > relation > entity",
        }
    if has_error:
        return {
            "answerable": True,
            "verdict": "warn",
            "reason": "reviewer gate: issues present but not blocking",
            "fallback": None,
            "precedence": "reviewer > relation > entity",
        }
    return {
        "answerable": True,
        "verdict": "pass",
        "reason": "reviewer gate: no issues",
        "fallback": None,
        "precedence": "reviewer > relation > entity",
    }


def dag_parallel_feasibility(
    *,
    narrow_relation_enabled: Optional[bool] = None,
    narrow_entity_enabled: Optional[bool] = None,
    dag_enabled: Optional[bool] = None,
    p0_cq_stable: bool = False,
    evidence_complete: bool = False,
) -> Dict[str, Any]:
    """Evaluate whether DAG-parallel agent execution may be *allowed* (feasibility flag).

    Policy: ``dag_allowed=True`` only when BOTH conditions hold:
    1. serial narrow quality is stable (P0 CQ stable AND evidence complete)
    2. DAG toggle is on (env ``1``/``true`` or ``auto`` when stable)

    Important: this is a **permission / readiness flag only**.
    ``run_serial_extraction`` always executes the six roles **serially** today;
    ``dag_allowed`` does **not** start parallel workers or speed up extraction.
    True parallel DAG execution needs a separate design and must not be inferred
    from this flag. Pure function for testability.
    """
    serial_stable = bool(p0_cq_stable and evidence_complete)
    if narrow_relation_enabled is None:
        narrow_relation_enabled = _env_truthy(NARROW_RELATION_ENABLED_ENV, default=True)
    if narrow_entity_enabled is None:
        entity_mode = _env_tri_state(NARROW_ENTITY_ENABLED_ENV, default="auto")
        narrow_entity_enabled = _resolve_gated_flag(entity_mode, stable=serial_stable)
    if dag_enabled is None:
        dag_mode = _env_tri_state(DAG_PARALLEL_ENABLED_ENV, default="auto")
        dag_enabled = _resolve_gated_flag(dag_mode, stable=serial_stable)
    narrow_ready = bool(narrow_relation_enabled and narrow_entity_enabled)
    allowed = bool(serial_stable and narrow_ready and dag_enabled)
    reasons: List[str] = []
    if not narrow_ready:
        reasons.append("narrow entity/relation passes not both enabled")
    if not serial_stable:
        reasons.append("serial narrow quality not stable (P0 CQ + evidence)")
    if not dag_enabled:
        reasons.append("DAG toggle not enabled")
    return {
        "dag_allowed": allowed,
        "serial_stable": serial_stable,
        "narrow_ready": narrow_ready,
        "reasons": reasons,
        "note": (
            "dag_allowed is feasibility only — six roles still run serially; "
            "true parallel DAG is not implemented (2026-11-08)"
        ),
    }


def run_serial_extraction(
    legacy_extraction: Optional[Dict[str, Any]] = None,
    *,
    extractor: Optional[Callable[[str], Dict[str, Any]]] = None,
    document_classifier: Optional[Callable[[str], Dict[str, Any]]] = None,
    evidence_provider: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    skill_contracts: Optional[List[Dict[str, Any]]] = None,
    document_name: str = "",
    domain: str = "",
    document_text: str = "",
    # P2-3 narrow/DAG: relation default ON; entity/DAG default auto (stable → on)
    relation_narrow_provider: Optional[Callable[[Dict[str, Any]], List[Dict[str, Any]]]] = None,
    entity_narrow_provider: Optional[Callable[[str], Dict[str, Any]]] = None,
    p0_cq_stable: Optional[bool] = None,
    evidence_complete: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run the six roles serially over one document. Never raises.

    Inputs are injected callables so tests can stub every stage and live
    wiring stays in the caller (true_unified_extractor stays the default
    entity/relation engine):

    - ``extractor(legacy_view_source)`` → legacy task-oriented dict
      (required for the entity role; without it that role is skipped).
    - ``document_classifier(text)`` → {"domain": ..., "classes": ...} (optional)
    - ``evidence_provider(packet)`` → mutated/augmented packet (optional)

    Returns::

        {
          "packet": {...}|None,     # OntologyPacket shape (schema-conformant)
          "issues": [...],          # reviewer findings (error-first sorted)
          "role_runs": [...],       # one entry per executed/skipped role
        }
    """
    role_runs: List[Dict[str, Any]] = []
    packet: Optional[Dict[str, Any]] = None

    # Role 1: classifier (optional provider)
    if document_classifier is not None:
        try:
            cls_out = document_classifier(document_text)
            if domain == "" and isinstance(cls_out, dict):
                domain = str(cls_out.get("domain") or "")
            role_runs.append(_run_entry(ROLE_CLASSIFIER, True, output=cls_out))
        except Exception as exc:
            role_runs.append(_run_entry(ROLE_CLASSIFIER, False, error=str(exc)))
    else:
        role_runs.append(_run_entry(ROLE_CLASSIFIER, True, skipped=True))

    # Roles 2-4 (entity / rule_l25 / relation): today they share the unified
    # extraction pass; the legacy dict is the shared substrate. Narrow-prompt
    # splits land here later (roadmap Phase 3).
    legacy: Optional[Dict[str, Any]] = legacy_extraction
    if legacy is None and extractor is not None:
        try:
            legacy = extractor(document_text)
            role_runs.append(_run_entry(ROLE_ENTITY, True, output={"mode": "unified"}))
        except Exception as exc:
            role_runs.append(_run_entry(ROLE_ENTITY, False, error=str(exc)))
    elif legacy is not None:
        role_runs.append(_run_entry(ROLE_ENTITY, True, output={"mode": "precomputed"}))
    else:
        role_runs.append(_run_entry(ROLE_ENTITY, True, skipped=True))

    role_runs.append(_run_entry(
        ROLE_RULE_L25, True,
        output={"note": "governed inside unified extraction via l25_skill_loader"},
    ))

    contract_violations: List[Dict[str, Any]] = []
    if legacy is not None and skill_contracts:
        from .skill_contract import validate_extraction_against_contract
        for idx, contract in enumerate(skill_contracts):
            try:
                verdict = validate_extraction_against_contract(legacy, contract or {})
                skill_id = str((contract or {}).get("skill_id") or f"contract[{idx}]")
                for v in verdict.get("violations", []):
                    contract_violations.append({"skill_id": skill_id, **v})
                role_runs.append(_run_entry(
                    ROLE_RULE_L25, True,
                    output={
                        "kind": "contract_check", "skill_id": skill_id,
                        "ok": verdict["ok"],
                        "violation_count": len(verdict.get("violations", [])),
                    },
                ))
            except Exception as exc:  # defensive: contract layer never kills chain
                role_runs.append(_run_entry(ROLE_RULE_L25, False, error=str(exc)))

    # P2-3b: relation narrow pass (opt-in, standalone).
    # When a relation_narrow_provider is supplied (or the env toggle is on),
    # enrich legacy with relations that have per-relation evidence. This never
    # breaks extraction: on any failure the legacy is used as-is (degraded
    # fallback — the bundle still flows, evidence gate sees what it sees).
    narrow_rel = relation_narrow_provider
    if narrow_rel is None and _env_truthy(NARROW_RELATION_ENABLED_ENV, default=True) and legacy is not None:
        def _narrow_rel(packet_shape: Dict[str, Any]) -> List[Dict[str, Any]]:
            return _bind_evidence_to_relations(
                list((packet_shape or {}).get("relations") or []), document_text)
        narrow_rel = _narrow_rel
    if legacy is not None and narrow_rel is not None:
        try:
            narrow_relations = narrow_rel(dict(legacy))
            if isinstance(narrow_relations, list) and narrow_relations:
                legacy = {**legacy, "relations": _bind_evidence_to_relations(narrow_relations, document_text)}
        except Exception as exc:
            logger.debug("P2-3 relation narrow pass skipped (degraded): %s", exc)

    # Stability assessment on the current legacy view (pre-entity-narrow).
    cq_forced = p0_cq_stable is not None
    ev_forced = evidence_complete is not None
    stability = assess_cq_evidence_stability(legacy if isinstance(legacy, dict) else None)
    if not cq_forced:
        p0_cq_stable = bool(stability["p0_cq_stable"])
    if not ev_forced:
        evidence_complete = bool(stability["evidence_complete"])
    serial_stable = bool(p0_cq_stable and evidence_complete)

    entity_mode = _env_tri_state(NARROW_ENTITY_ENABLED_ENV, default="auto")
    entity_narrow_enabled = (
        entity_narrow_provider is not None
        or _resolve_gated_flag(entity_mode, stable=serial_stable)
    )

    # P2-3d: entity narrow — when auto/on, evidence-bind existing entities
    # (deterministic, no invent). Custom provider may supply entities.
    if entity_narrow_enabled:
        narrow_ent = entity_narrow_provider
        if narrow_ent is None:
            captured_legacy = dict(legacy) if isinstance(legacy, dict) else {}

            def _default_entity_narrow(text: str) -> Dict[str, Any]:
                return {
                    "entities": _bind_evidence_to_entities(
                        _collect_entities(captured_legacy), text
                    ),
                    "relations": [],
                }

            narrow_ent = _default_entity_narrow
        try:
            got = narrow_ent(document_text)
            if isinstance(got, dict) and got.get("entities"):
                legacy = {
                    **(legacy or {}),
                    "entities": _bind_evidence_to_entities(
                        list(got["entities"]), document_text
                    ),
                }
                if got.get("relations") and not (legacy or {}).get("relations"):
                    legacy = {
                        **legacy,
                        "relations": _bind_evidence_to_relations(
                            list(got["relations"]), document_text
                        ),
                    }
        except Exception as exc:
            logger.debug("P2-3 entity narrow pass skipped (degraded): %s", exc)

    if legacy is not None:
        try:
            packet = adapt_extraction_json_to_packet(
                legacy, document_name=document_name, domain=domain
            )
            role_runs.append(_run_entry(
                ROLE_RELATION, True,
                output={"relations": len((packet or {}).get("relations") or [])},
            ))
        except Exception as exc:
            role_runs.append(_run_entry(ROLE_RELATION, False, error=f"adapt failed: {exc}"))
    else:
        role_runs.append(_run_entry(ROLE_RELATION, True, skipped=True))

    # Role 5: evidence (optional provider)
    if packet is not None and evidence_provider is not None:
        try:
            augmented = evidence_provider(packet)
            if isinstance(augmented, dict):
                packet = augmented
            role_runs.append(_run_entry(ROLE_EVIDENCE, True, output={"enriched": True}))
        except Exception as exc:
            role_runs.append(_run_entry(ROLE_EVIDENCE, False, error=str(exc)))
    else:
        role_runs.append(_run_entry(ROLE_EVIDENCE, True, skipped=True))

    # Re-assess after packet/evidence enrichment so DAG sees final coverage.
    # If the adapter drops relations (whitelist / shape), keep the pre-adapt
    # stability rather than wiping a green CQ/evidence signal.
    if packet is not None:
        post = assess_cq_evidence_stability(packet)
        if post.get("relation_count", 0) > 0:
            stability = post
            if not cq_forced:
                p0_cq_stable = bool(post["p0_cq_stable"])
            if not ev_forced:
                evidence_complete = bool(post["evidence_complete"])
        serial_stable = bool(p0_cq_stable and evidence_complete)

    # Role 6: reviewer (always runs when a packet exists)
    issues: List[Dict[str, Any]] = []
    release_ok = True
    review_rec: Dict[str, Any] = {"answerable": True, "verdict": "pass", "fallback": None, "precedence": "reviewer > relation > entity"}
    if packet is not None:
        try:
            issues = review_ontology_packet(packet)
            role_runs.append(_run_entry(ROLE_REVIEWER, True, output={"issue_count": len(issues)}))
            review_rec = reviewer_recommendation(issues)
            release_ok = bool(review_rec["answerable"])
        except Exception as exc:  # defensive: reviewer must not kill the chain
            role_runs.append(_run_entry(ROLE_REVIEWER, False, error=str(exc)))
    else:
        role_runs.append(_run_entry(ROLE_REVIEWER, True, skipped=True))

    dag_ok = dag_parallel_feasibility(
        narrow_entity_enabled=entity_narrow_enabled,
        p0_cq_stable=bool(p0_cq_stable),
        evidence_complete=bool(evidence_complete),
    )

    return {
        "packet": packet,
        "issues": issues,
        "role_runs": role_runs,
        "contract_violations": contract_violations,
        "release_ok": release_ok,
        "review_recommendation": review_rec,
        "stability": stability,
        "dag_feasibility": dag_ok,
        "dag_allowed": dag_ok["dag_allowed"],
    }
