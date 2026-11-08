"""Skill governance validation extracted from ``TrueUnifiedExtractor``.

``apply_skill_governance`` depends only on the skill loader collaborator, so it
lives here as a plain module function (2026-08-11).
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def apply_skill_governance(
    text: str,
    domain: Optional[str],
    l1_entities: List[Dict],
    l2_entities: List[Dict],
    l25_entities: List[Dict],
    relationships: List[Dict],
    skill_loader,
) -> Dict[str, Any]:
    """根据已激活 skill 的 governance 字段对抽取结果做后置校验。

    Returns:
        {
            "activated_skills": [...],
            "mode": "strict" | "lenient" | "audit_only",
            "gating_level": "soft" | "hard" | "block",
            "blocked_relations": int,
            "entity_type_violations": int,
            "required_field_violations": int,
            "issues": [ {type, skill_id, detail} ],
        }
    """
    issues: List[Dict[str, Any]] = []
    activated: List[str] = []
    blocked_relations = 0
    entity_violations = 0
    field_violations = 0
    mode = "audit_only"
    gating = "soft"

    try:
        matching_skills = skill_loader.find_matching_skills(text) if skill_loader else []
    except Exception:
        matching_skills = []

    if domain:
        matching_skills = [
            s for s in matching_skills
            if not s.domain or domain.lower() in [d.lower() for d in s.domain]
        ]

    for skill in matching_skills:
        skill_id = getattr(skill, "skill_id", None) or getattr(skill, "skill_name", "unknown")
        activated.append(skill_id)
        gov = getattr(skill, "governance", None) or {}
        if not gov:
            continue

        mode = gov.get("validation_mode", mode)
        gating = gov.get("gating_level", gating)

        allowed_layers = set(gov.get("allowed_layers", []) or [])
        allowed_rels = set(gov.get("allowed_relation_types", []) or [])
        blocked_rels = set(gov.get("blocked_relation_types", []) or [])
        preferred_entities = set(gov.get("preferred_entity_types", []) or [])
        required_fields = gov.get("required_fields", []) or []
        entity_type_mapping = gov.get("entity_type_mapping", []) or []

        all_entities = []
        for layer_label, ent_list in (
            ("L1", l1_entities),
            ("L2", l2_entities),
            ("L2.5", l25_entities),
        ):
            for e in ent_list:
                if allowed_layers and layer_label not in allowed_layers and layer_label != "L1":
                    continue
                all_entities.append((layer_label, e))

        for layer_label, e in all_entities:
            ent_type = str(
                e.get("meta_class") or e.get("type") or e.get("scenario_type") or ""
            ).strip()
            if preferred_entities and ent_type and ent_type not in preferred_entities:
                suggested = None
                for mapping in entity_type_mapping:
                    if mapping.get("from_type") == ent_type:
                        suggested = mapping.get("suggest_type")
                        break
                issues.append({
                    "type": "ENTITY_TYPE_VIOLATION",
                    "skill_id": skill_id,
                    "entity_type": ent_type,
                    "layer": layer_label,
                    "preferred": list(preferred_entities),
                    "suggested_type": suggested,
                    "severity": "warn",
                })
                entity_violations += 1

            for rf in required_fields:
                if isinstance(rf, dict) and rf.get("entity_type") == ent_type:
                    for field_name in rf.get("fields", []):
                        if not str(e.get(field_name) or "").strip():
                            issues.append({
                                "type": "MISSING_REQUIRED_FIELD",
                                "skill_id": skill_id,
                                "entity_type": ent_type,
                                "layer": layer_label,
                                "field": field_name,
                                "severity": "warn",
                            })
                            field_violations += 1

        for rel in relationships:
            rel_type = str(
                rel.get("relation_type") or rel.get("relationship_type") or rel.get("type") or ""
            ).strip()
            if not rel_type:
                continue
            if rel_type in blocked_rels:
                issues.append({
                    "type": "BLOCKED_RELATION",
                    "skill_id": skill_id,
                    "relation_type": rel_type,
                    "severity": "error",
                })
                blocked_relations += 1
            elif allowed_rels and rel_type not in allowed_rels:
                issues.append({
                    "type": "RELATION_NOT_IN_WHITELIST",
                    "skill_id": skill_id,
                    "relation_type": rel_type,
                    "allowed": list(allowed_rels),
                    "severity": "warn",
                })
                blocked_relations += 1

    return {
        "activated_skills": activated,
        "mode": mode,
        "gating_level": gating,
        "blocked_relations": blocked_relations,
        "entity_type_violations": entity_violations,
        "required_field_violations": field_violations,
        "issues": issues,
    }
