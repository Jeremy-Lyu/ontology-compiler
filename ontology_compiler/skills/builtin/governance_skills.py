"""Governance Skills - 治理相关技能实现

包含：check_config, validate_output_contracts, ontology_validator
"""

from __future__ import annotations

from typing import Any, Dict, List
from ontology_compiler.skills.base import Skill, SkillContext, SkillResult, SkillPriority


class CheckConfigSkill(Skill):
    """检查配置完整性的技能"""

    @property
    def priority(self) -> SkillPriority:
        return SkillPriority.CRITICAL

    @property
    def description(self) -> str:
        return "检查配置文件的完整性和有效性"

    async def can_apply(self, context: SkillContext) -> bool:
        return True

    async def execute(self, context: SkillContext) -> SkillResult:
        config = context.config
        checks = []
        errors = []

        if not config.get("domain"):
            errors.append("domain is required")
        else:
            checks.append(f"domain: {config['domain']}")

        if config.get("source_folder"):
            checks.append(f"source_folder: {config['source_folder']}")

        if config.get("output_folder"):
            checks.append(f"output_folder: {config['output_folder']}")

        if errors:
            return SkillResult(
                success=False,
                skill_name=self.name,
                reason=f"Configuration errors: {', '.join(errors)}",
                output={"checks": checks, "errors": errors}
            )

        return SkillResult(
            success=True,
            skill_name=self.name,
            applied=True,
            output={"checks": checks, "status": "valid"}
        )


class ValidateOutputContractsSkill(Skill):
    """验证输出契约的技能"""

    @property
    def priority(self) -> SkillPriority:
        return SkillPriority.HIGH

    @property
    def description(self) -> str:
        return "验证输出结果符合契约要求"

    async def can_apply(self, context: SkillContext) -> bool:
        return context.extraction_result is not None

    async def execute(self, context: SkillContext) -> SkillResult:
        result = context.extraction_result
        validation = {
            "nodes_valid": False,
            "relationships_valid": False,
            "evidence_valid": False,
            "issues": []
        }

        if result.get("nodes"):
            validation["nodes_valid"] = True
            node_count = len(result["nodes"])
            validation["nodes_count"] = node_count

            for i, node in enumerate(result["nodes"]):
                if not node.get("meta_id"):
                    validation["issues"].append(f"Node {i} missing meta_id")
                if not node.get("meta_class"):
                    validation["issues"].append(f"Node {i} missing meta_class")

        if result.get("relationships"):
            validation["relationships_valid"] = True
            validation["relationships_count"] = len(result["relationships"])

            for i, rel in enumerate(result["relationships"]):
                if not rel.get("source") or not rel.get("target"):
                    validation["issues"].append(f"Relationship {i} missing source/target")
                if not rel.get("type"):
                    validation["issues"].append(f"Relationship {i} missing type")

        validation["evidence_valid"] = True

        has_issues = len(validation["issues"]) > 0

        return SkillResult(
            success=not has_issues,
            skill_name=self.name,
            applied=True,
            output=validation
        )


class OntologyValidatorSkill(Skill):
    """验证本体一致性的技能"""

    @property
    def priority(self) -> SkillPriority:
        return SkillPriority.HIGH

    @property
    def description(self) -> str:
        return "验证本体结构的一致性和完整性"

    async def can_apply(self, context: SkillContext) -> bool:
        return context.extraction_result is not None

    async def execute(self, context: SkillContext) -> SkillResult:
        result = context.extraction_result
        issues: List[Dict[str, Any]] = []

        nodes = result.get("nodes", [])
        relationships = result.get("relationships", [])

        node_ids = {n.get("meta_id") for n in nodes if n.get("meta_id")}

        for rel in relationships:
            source = rel.get("source")
            target = rel.get("target")

            if source and source not in node_ids:
                issues.append({
                    "type": "dangling_source",
                    "source": source,
                    "message": f"Relationship source '{source}' not found in nodes"
                })

            if target and target not in node_ids:
                issues.append({
                    "type": "dangling_target",
                    "target": target,
                    "message": f"Relationship target '{target}' not found in nodes"
                })

        has_issues = len(issues) > 0

        return SkillResult(
            success=not has_issues,
            skill_name=self.name,
            applied=True,
            output={
                "total_nodes": len(nodes),
                "total_relationships": len(relationships),
                "issues": issues,
                "status": "valid" if not has_issues else "invalid"
            }
        )