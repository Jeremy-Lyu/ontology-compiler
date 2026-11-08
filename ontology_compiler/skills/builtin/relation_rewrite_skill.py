"""RelationRewriteSkill — deterministic RELATED_TO→semantic rewrite (C-1 experiment).

C-1 (2026-08-15): minimal, measurable skill integration experiment. The
``relation_rewrite`` skill.json has been metadata-only ("streaming" no-op that
reports success without doing anything). This skill is the *real, executable*
implementation:

- Pure + deterministic: no LLM cost, reuses ``relation_governance`` for
  canonical family rewrites and a text-blob keyword map (mirrors the pipeline's
  existing RELATED_TO rewrite).
- Measurable: output carries rewrite_count / unrewritten_count /
  related_to_share_before / related_to_share_after so the experiment can be
  judged with one real run.

Gating: the pipeline bridge only *registers* this skill; the pipeline router
only routes it when ``PIPELINE_SKILL_RELATION_REWRITE=1`` is set, so production
behavior is unchanged by default. See
``docs/PIPELINE_STABILITY_DIAGNOSIS_AND_REFACTOR.md`` Phase C.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from ..base import Skill, SkillContext, SkillPriority, SkillResult
from ontology_compiler.services.relation_governance import get_relation_governance_engine

_RELATED_TO_ALIASES = ("RELATED_TO", "RELATED", "ASSOCIATED")

# Text-blob keyword → concrete relation type (precedence by order).
_TEXT_REWRITE_RULES: List[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bdepends\s+on\b", re.I), "DEPENDS_ON"),
    (re.compile(r"\brequires?\b", re.I), "DEPENDS_ON"),
    (re.compile(r"\brelies\s+on\b", re.I), "DEPENDS_ON"),
    (re.compile(r"\bsupports?\b", re.I), "SUPPORTS"),
    (re.compile(r"\benables?\b", re.I), "SUPPORTS"),
    (re.compile(r"\bassists?\b", re.I), "SUPPORTS"),
    (re.compile(r"\bfacilitates?\b", re.I), "SUPPORTS"),
    (re.compile(r"\bderived\s+from\b", re.I), "DERIVED_FROM"),
    (re.compile(r"\boriginates?\s+from\b", re.I), "DERIVED_FROM"),
    (re.compile(r"\btransforms?\s+to\b", re.I), "TRANSFORMS_TO"),
    (re.compile(r"\bpart\s+of\b", re.I), "PART_OF"),
    (re.compile(r"\bcontained\s+in\b", re.I), "PART_OF"),
    (re.compile(r"\blocated\s+in\b", re.I), "LOCATED_IN"),
    (re.compile(r"\bat\s+each\s+plant\b", re.I), "LOCATED_IN"),
    (re.compile(r"\bflow(s|ing)?\s+(to|into|from)\b", re.I), "TRANSFORMS_TO"),
    (re.compile(r"\bowned\s+by\b", re.I), "OWNED_BY"),
    (re.compile(r"\bcontrols?\b", re.I), "CONTROLS"),
]


class RelationRewriteSkill(Skill):
    """Rewrite weak/ambiguous relations into concrete governed relation types.

    Applies when relationships carry a RELATED_TO-style type (or a text blob it
    can rewrite). Pure and deterministic; emits rewrite metrics for the
    experiment (rewrite_count / related-to share before/after).
    """

    @property
    def priority(self) -> SkillPriority:
        return SkillPriority.NORMAL

    @property
    def description(self) -> str:
        return "将 RELATED_TO 等弱关系重写为具体语义关系（确定性、零 LLM 成本）"

    async def can_apply(self, context: SkillContext) -> bool:
        relationships = self._relationships_from(context)
        return bool(relationships)

    async def execute(self, context: SkillContext) -> SkillResult:
        relationships = list(self._relationships_from(context))
        if not relationships:
            return SkillResult.skipped_result(self.name, "no relationships to rewrite")

        total = len(relationships)
        related_before = self._count_related(relationships)
        rewritten = []
        rewrites = 0
        unrewritten = 0

        engine = get_relation_governance_engine()

        for rel in relationships:
            if not isinstance(rel, dict):
                rewritten.append(rel)
                continue
            rel_type = str(
                rel.get("relation_type") or rel.get("relationship_type") or rel.get("type") or ""
            ).strip().upper()
            if rel_type in _RELATED_TO_ALIASES:
                new_rel, reason = self._rewrite_relation(rel, rel_type, engine)
                if new_rel is not None:
                    rewritten.append(new_rel)
                    rewrites += 1
                    continue
            # Attempt canonical family normalization for non-weak relations too
            normalized = engine.normalize(rel)
            if normalized.get("rewritten_relation_type"):
                rewrites += 1
            rewritten.append(normalized)
            unrewritten += 1

        related_after = self._count_related(rewritten)

        return SkillResult(
            success=True,
            skill_name=self.name,
            applied=rewrites > 0,
            output={
                "rewritten_relationships": rewritten,
                "rewrite_count": rewrites,
                "unrewritten_count": unrewritten,
                "total_count": total,
                "related_to_share_before": round(related_before / total, 4) if total else 0.0,
                "related_to_share_after": round(related_after / len(rewritten), 4) if rewritten else 0.0,
            },
        )

    # ---------- helpers ----------

    def _relationships_from(self, context: SkillContext) -> List[Dict[str, Any]]:
        """Extract the relationship list from the bridge's payload shape.

        The pipeline bridge passes ``{"stage": ..., "stage_data": {...}}`` in
        ``config``; the canonical registry path passes ``extraction_result``.
        Accept relationships directly, from ``config.relationships``, or from
        ``config["stage_data"]["relationships"]``.
        """
        if context.relationships:
            return context.relationships
        config = context.config or {}
        direct = config.get("relationships")
        if isinstance(direct, list):
            return direct
        stage_data = config.get("stage_data")
        if isinstance(stage_data, dict):
            for key in ("relationships", "relations"):
                value = stage_data.get(key)
                if isinstance(value, list):
                    return value
        extraction = context.extraction_result or {}
        for key in ("relationships", "relations"):
            value = extraction.get(key)
            if isinstance(value, list):
                return value
        return []

    def _count_related(self, relationships: List[Dict[str, Any]]) -> int:
        count = 0
        for rel in relationships:
            if not isinstance(rel, dict):
                continue
            rel_type = str(
                rel.get("relation_type") or rel.get("relationship_type") or rel.get("type") or ""
            ).strip().upper()
            if rel_type in _RELATED_TO_ALIASES:
                count += 1
        return count

    def _rewrite_relation(self, rel: Dict[str, Any], rel_type: str, engine: Any) -> tuple[Dict[str, Any] | None, str | None]:
        """Try to rewrite one RELATED_TO relation; returns (new_rel, reason) or (None, None)."""
        text_blob = " ".join(str(v) for v in rel.values() if isinstance(v, str)).lower()
        target: str | None = None
        reason: str | None = None
        for pattern, candidate in _TEXT_REWRITE_RULES:
            if pattern.search(text_blob):
                target = candidate
                reason = f"text keyword matched: {candidate}"
                break

        if target is None:
            # Fall back to canonical-family selection for the RELATED family.
            normalized = engine.normalize(rel)
            better = normalized.get("rewritten_relation_type")
            if better and better != rel_type:
                return normalized, "canonical family relation selected"
            return None, None

        new_rel = dict(rel)
        new_rel["type"] = target
        new_rel["relation_type"] = target
        new_rel.setdefault("rewrite_reason", reason)
        new_rel.setdefault("rewritten_from", rel_type)
        new_rel.setdefault("decision_trace", []).append(
            {"action": "relation_rewrite_skill", "reason": reason, "before": rel_type, "after": target}
        )
        return new_rel, reason