"""QualityGateSkill — unified quality gate as a skill (P2-2 F8).

Wraps :class:`src.infrastructure.governance.quality_gate.QualityGate`
(the audit/guard/governance convergence layer). Output is the canonical
quality_report contract (aligned with schemas/quality_report.schema.yaml)
plus a bridge flag for the legacy gate payload.

Boundaries (§6.2):
- Input:  {entities, relationships, metadata?}
- Output: {result, grade, score, issues, quality_report, ...}
- Trigger: audit stage, after merge
- Deterministic; no LLM.
"""

from __future__ import annotations

from typing import Any, Dict

from ..base import Skill, SkillContext, SkillPriority, SkillResult
from src.infrastructure.governance.quality_gate import QualityGate


class QualityGateSkill(Skill):
    """Run the unified quality gate and emit a canonical quality report."""

    @property
    def priority(self) -> SkillPriority:
        return SkillPriority.HIGH

    @property
    def description(self) -> str:
        return "质量门禁（entity/relation/metadata/governance 四维检查，输出 canonical quality_report）"

    async def can_apply(self, context: SkillContext) -> bool:
        data = self._data_from(context)
        return bool(data.get("entities") or data.get("nodes"))

    async def execute(self, context: SkillContext) -> SkillResult:
        data = self._data_from(context)
        if not data.get("entities") and not data.get("nodes"):
            return SkillResult.skipped_result(self.name, "no entities to check")

        # Normalize nodes -> entities for the gate's expected shape
        if "entities" not in data and "nodes" in data:
            data = dict(data)
            data["entities"] = data["nodes"]

        gate = QualityGate()
        strict = bool(context.config.get("strict_mode")) if context.config else False
        report = gate.run(data)

        out = report.to_dict() if hasattr(report, "to_dict") else dict(report)
        # bridge: expose canonical quality_report alias
        out.setdefault("quality_report", out)
        if strict and out.get("result") == "warn":
            out["result"] = "fail"
            out["strict_mode_applied"] = True

        return SkillResult(
            success=True,
            skill_name=self.name,
            applied=True,
            output=out,
        )

    def _data_from(self, context: SkillContext) -> Dict[str, Any]:
        cfg = context.config or {}
        for key in ("data", "pipeline_output", "stage_data"):
            v = cfg.get(key)
            if isinstance(v, dict) and (v.get("entities") or v.get("nodes") or v.get("relationships") or v.get("relations")):
                return dict(v)
        ext = context.extraction_result or {}
        if ext.get("entities") or ext.get("nodes"):
            return dict(ext)
        if context.entities:
            return {"entities": list(context.entities),
                    "relationships": list(context.relationships or [])}
        return {}
