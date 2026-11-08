"""CrossFileAssociationSkill — deterministic cross-file entity association (P2-2 F9).

Wraps the rule-based half of
:mod:`ontology_compiler.services.cross_file_association_service`
(SemanticEquivalenceJudge pre-filtering + inverted-index similarity).
This skill makes the capability discoverable, versioned, and independently
testable. The LLM-driven judge is not invoked here — that remains in the
LLM-candidate path and is gated separately.

Boundaries (§6.2):
- Input:  file_entities dict (file → entity list), optional threshold/strategy
- Output: merged_entities + association_map + merge_count + duplicate_groups
- Trigger: merge stage, when file_count > 1
- Deterministic; no LLM calls.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..base import Skill, SkillContext, SkillPriority, SkillResult
from ontology_compiler.services.cross_file_association_service import (
    CrossFileAssociationService,
)


class CrossFileAssociationSkill(Skill):
    """Associate and merge entities that represent the same concept across files."""

    @property
    def priority(self) -> SkillPriority:
        return SkillPriority.HIGH

    @property
    def description(self) -> str:
        return "跨文件实体关联与合并（规则侧，确定性，无 LLM 调用）"

    async def can_apply(self, context: SkillContext) -> bool:
        return bool(self._file_entities_from(context))

    async def execute(self, context: SkillContext) -> SkillResult:
        file_entities = self._file_entities_from(context)
        if not file_entities or len(file_entities) < 2:
            return SkillResult.skipped_result(
                self.name, "need at least 2 files with entities")

        threshold = float(context.config.get("similarity_threshold", 0.85)
                          if context.config else 0.85)
        # merge stage uses a conservative strategy by default; this skill
        # mirrors that and exposes it as a config knob.
        strategy = str(context.config.get("merge_strategy", "conservative")
                       if context.config else "conservative")

        svc = CrossFileAssociationService()
        # Flatten for indexing; the service builds its own inverted index.
        all_entities: List[Dict[str, Any]] = []
        for ents in file_entities.values():
            all_entities.extend(ents)
        svc._build_entity_index(all_entities)

        # Use deterministic matching (rule side only); collect groups with
        # similarity >= threshold. Reuse the service's fast similarity.
        groups: List[List[str]] = []
        association_map: Dict[str, str] = {}
        seen: set[str] = set()

        for entity in all_entities:
            eid = entity.get("meta_id") or entity.get("id") or ""
            if not eid or eid in seen:
                continue
            group = [eid]
            seen.add(eid)
            name = svc._get_entity_name(entity) if hasattr(svc, "_get_entity_name") else ""
            candidates = svc._get_candidate_matches(
                name, threshold=threshold) if name else set()
            for cand in candidates:
                if cand in seen:
                    continue
                a = next((e for e in all_entities if (e.get("meta_id") or e.get("id")) == eid), None)
                b = next((e for e in all_entities if (e.get("meta_id") or e.get("id")) == cand), None)
                if not a or not b:
                    continue
                sim = svc._fast_name_similarity(
                    svc._normalized_name_cache.get(eid, ""),
                    svc._normalized_name_cache.get(cand, ""),
                )
                if sim >= threshold:
                    group.append(cand)
                    seen.add(cand)
            if len(group) > 1:
                groups.append(group)
                canonical = group[0]
                for mid in group:
                    association_map[mid] = canonical

        # merged_entities: deduplicate by canonical
        merged: List[Dict[str, Any]] = []
        for eid in association_map.values():
            if any(m.get("meta_id") == eid for m in merged):
                continue
            ent = next((e for e in all_entities if (e.get("meta_id") or e.get("id")) == eid), None)
            if ent is not None:
                merged.append(dict(ent))
        # also include singletons not in any group
        grouped_ids = set(association_map.keys())
        for ent in all_entities:
            eid = ent.get("meta_id") or ent.get("id") or ""
            if eid not in grouped_ids and eid not in association_map.values():
                merged.append(dict(ent))

        return SkillResult(
            success=True,
            skill_name=self.name,
            applied=len(groups) > 0,
            output={
                "merged_entities": merged,
                "association_map": association_map,
                "merge_count": len(groups),
                "unmerged_count": len(all_entities) - sum(len(g) for g in groups),
                "duplicate_groups": groups,
                "strategy": strategy,
                "threshold": threshold,
            },
        )

    def _file_entities_from(self, context: SkillContext) -> Dict[str, List[Dict[str, Any]]]:
        cfg = context.config or {}
        for key in ("file_entities", "file_groups", "stage_data"):
            v = cfg.get(key)
            if isinstance(v, dict) and len(v) >= 2:
                # accept either {file: [entities]} or stage_data containing it
                if key == "stage_data" and isinstance(v.get("file_entities"), dict):
                    return dict(v["file_entities"])
                # filter to file-shaped entries
                if all(isinstance(val, list) for val in v.values()):
                    return dict(v)
        # fallback: try extraction_result
        ext = context.extraction_result or {}
        fe = ext.get("file_entities")
        if isinstance(fe, dict):
            return dict(fe)
        return {}
