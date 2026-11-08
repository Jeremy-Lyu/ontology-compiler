"""Cost Aware Skip Skill — 成本意识智能跳过

R38-QA Skill #1: 当语义MD提取器覆盖率>=0.85时跳过LLM精化步骤。
"""

from typing import Any, Dict

from ..base import Skill, SkillContext, SkillResult, SkillPriority


class CostAwareSkipSkill(Skill):
    """Skip expensive LLM extraction when coverage is sufficient.
    
    Evidence: single_file_pipeline.py RIGORExtract.execute()
    Performance: Directly reduces LLM calls for well-covered documents.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.coverage_threshold = self.config.get("coverage_threshold", 0.85)
        self.skip_rigor = self.config.get("skip_rigor", True)
    
    @property
    def priority(self) -> SkillPriority:
        return SkillPriority.HIGH
    
    @property
    def description(self) -> str:
        return f"Skip LLM extraction when semantic coverage >= {self.coverage_threshold}"
    
    async def can_apply(self, context: SkillContext) -> bool:
        """Can apply if coverage data is available."""
        return context.coverage > 0 or "semantic_coverage" in context.metadata
    
    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute cost-aware skip logic."""
        # Get coverage from context
        coverage = context.coverage
        if coverage == 0:
            coverage = context.metadata.get("semantic_coverage", 0.0)
        
        # Check if we can skip
        if coverage >= self.coverage_threshold:
            # Calculate cost saved (approximate)
            estimated_cost = context.metadata.get("estimated_llm_cost", 0.015)
            
            # Mark as skipped
            context.set("skip_llm_extraction", True)
            context.set("skip_reason", f"Coverage {coverage:.2f} >= threshold {self.coverage_threshold}")
            
            return SkillResult(
                success=True,
                skill_name=self.name,
                applied=True,
                cost_saved=estimated_cost,
                output={
                    "coverage": coverage,
                    "threshold": self.coverage_threshold,
                    "skipped": True,
                }
            )
        
        # Cannot skip
        return SkillResult(
            success=True,
            skill_name=self.name,
            applied=False,
            output={
                "coverage": coverage,
                "threshold": self.coverage_threshold,
                "skipped": False,
                "reason": f"Coverage {coverage:.2f} below threshold",
            }
        )
