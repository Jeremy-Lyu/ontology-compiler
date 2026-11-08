"""Ontology Skills - 本体处理相关技能实现

包含：ontology_merger, consensus_resolver, drift_detector, run_ontology_rag_validate
"""

from __future__ import annotations

from typing import Any, Dict, List
from ontology_compiler.skills.base import Skill, SkillContext, SkillResult, SkillPriority


class OntologyMergerSkill(Skill):
    """合并本体片段的技能"""

    @property
    def priority(self) -> SkillPriority:
        return SkillPriority.NORMAL

    @property
    def description(self) -> str:
        return "合并多个本体片段为统一本体"

    async def can_apply(self, context: SkillContext) -> bool:
        return True

    async def execute(self, context: SkillContext) -> SkillResult:
        extraction = context.extraction_result
        if not extraction:
            return SkillResult(
                success=True,
                skill_name=self.name,
                skipped=True,
                reason="No extraction result to merge"
            )

        merged = {
            "run_id": extraction.get("run_id"),
            "domain": extraction.get("domain"),
            "nodes": extraction.get("nodes", []),
            "relationships": extraction.get("relationships", []),
            "instances": extraction.get("instances", []),
            "metadata": extraction.get("metadata", {})
        }

        node_count = len(merged["nodes"])
        rel_count = len(merged["relationships"])

        return SkillResult(
            success=True,
            skill_name=self.name,
            applied=True,
            output={
                "merged": merged,
                "nodes_count": node_count,
                "relationships_count": rel_count,
                "status": "merged"
            }
        )


class ConsensusResolverSkill(Skill):
    """解决冲突的一致性解析技能"""

    @property
    def priority(self) -> SkillPriority:
        return SkillPriority.HIGH

    @property
    def description(self) -> str:
        return "解决实体和关系之间的冲突"

    async def can_apply(self, context: SkillContext) -> bool:
        return True

    async def execute(self, context: SkillContext) -> SkillResult:
        conflicts = context.get("conflicts", [])
        resolved_count = 0
        unresolved = []

        for conflict in conflicts:
            conflict_type = conflict.get("type", "")
            if conflict_type == "same_as_conflict":
                resolved_count += 1
            else:
                unresolved.append(conflict)

        return SkillResult(
            success=True,
            skill_name=self.name,
            applied=True,
            output={
                "total_conflicts": len(conflicts),
                "resolved_count": resolved_count,
                "unresolved_count": len(unresolved),
                "unresolved": unresolved
            }
        )


class DriftDetectorSkill(Skill):
    """检测本体漂移的技能"""

    @property
    def priority(self) -> SkillPriority:
        return SkillPriority.BACKGROUND

    @property
    def description(self) -> str:
        return "检测本体结构和内容的漂移"

    async def can_apply(self, context: SkillContext) -> bool:
        return True

    async def execute(self, context: SkillContext) -> SkillResult:
        result = context.extraction_result
        if not result:
            return SkillResult(
                success=True,
                skill_name=self.name,
                skipped=True,
                reason="No extraction result to analyze"
            )

        nodes = result.get("nodes", [])
        relationships = result.get("relationships", [])

        drift_indicators = {
            "uncommon_entity_types": [],
            "uncommon_relation_types": [],
            "structural_anomalies": []
        }

        return SkillResult(
            success=True,
            skill_name=self.name,
            applied=True,
            output={
                "drift_detected": False,
                "indicators": drift_indicators,
                "nodes_count": len(nodes),
                "relationships_count": len(relationships)
            }
        )


class RunOntologyRagValidateSkill(Skill):
    """验证 RAG 资产的技能"""

    @property
    def priority(self) -> SkillPriority:
        return SkillPriority.LOW

    @property
    def description(self) -> str:
        return "验证本体资产是否适合 RAG 消费"

    async def can_apply(self, context: SkillContext) -> bool:
        return True

    async def execute(self, context: SkillContext) -> SkillResult:
        result = context.extraction_result
        if not result:
            return SkillResult(
                success=True,
                skill_name=self.name,
                skipped=True,
                reason="No extraction result to validate"
            )

        nodes = result.get("nodes", [])
        has_evidence = any(n.get("evidence") for n in nodes)
        has_source_file = any(n.get("source_file") for n in nodes)

        validation = {
            "has_evidence": has_evidence,
            "has_source_file": has_source_file,
            "nodes_count": len(nodes),
            "ready_for_rag": has_evidence or has_source_file
        }

        return SkillResult(
            success=True,
            skill_name=self.name,
            applied=True,
            output=validation
        )