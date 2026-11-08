"""Publisher Skills - 发布相关技能实现

包含：capsule_publisher
"""

from __future__ import annotations

from typing import Any, Dict, List
from ontology_compiler.skills.base import Skill, SkillContext, SkillResult, SkillPriority


class CapsulePublisherSkill(Skill):
    """发布本体胶囊的技能"""

    @property
    def priority(self) -> SkillPriority:
        return SkillPriority.BACKGROUND

    @property
    def description(self) -> str:
        return "将本体结果发布为胶囊格式"

    async def can_apply(self, context: SkillContext) -> bool:
        return True

    async def execute(self, context: SkillContext) -> SkillResult:
        result = context.extraction_result
        if not result:
            return SkillResult(
                success=True,
                skill_name=self.name,
                skipped=True,
                reason="No extraction result to publish"
            )

        capsule = {
            "version": "1.0",
            "type": "ontology_capsule",
            "content": {
                "domain": result.get("domain"),
                "nodes": result.get("nodes", []),
                "relationships": result.get("relationships", []),
                "instances": result.get("instances", []),
            },
            "metadata": {
                "generated_at": result.get("metadata", {}).get("generated_at"),
                "source": result.get("metadata", {}).get("source"),
                "total_files": result.get("metadata", {}).get("total_files"),
                "processed_files": result.get("metadata", {}).get("processed_files"),
            }
        }

        return SkillResult(
            success=True,
            skill_name=self.name,
            applied=True,
            output={
                "capsule": capsule,
                "status": "published",
                "nodes_count": len(capsule["content"]["nodes"]),
                "relationships_count": len(capsule["content"]["relationships"])
            }
        )