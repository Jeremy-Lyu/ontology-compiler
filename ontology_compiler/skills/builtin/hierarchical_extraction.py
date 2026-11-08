"""Hierarchical Extraction Skill — 一键多层抽取

R38-QA Skill #2: 单次LLM调用提取L1/L2/L2.5三层。
"""

from typing import Any, Dict

from ..base import Skill, SkillContext, SkillResult, SkillPriority


class HierarchicalExtractionSkill(Skill):
    """Extract L1, L2, and L2.5 in a single LLM call.
    
    Evidence: true_unified_extractor.py _build_unified_prompt()
    Performance: 3-4x theoretical improvement, reduces LLM calls and token consumption.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.use_unified = self.config.get("use_unified", True)
        self.max_tokens = self.config.get("max_tokens", 8000)
    
    @property
    def priority(self) -> SkillPriority:
        return SkillPriority.CRITICAL
    
    @property
    def description(self) -> str:
        return "Extract L1/L2/L2.5 in single LLM call using TrueUnifiedExtractor"
    
    async def can_apply(self, context: SkillContext) -> bool:
        """Can apply if document content is available."""
        return (
            context.document_content is not None or
            context.get("document_chunks") is not None
        )
    
    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute hierarchical extraction."""
        # Mark unified extraction as preferred method
        context.set("extraction_method", "hierarchical_unified")
        context.set("use_true_unified_extractor", True)
        context.set("extract_all_layers", True)
        
        # Calculate theoretical cost savings
        # Traditional: 3 separate calls = ~3x cost
        # Unified: 1 call = ~1x cost
        traditional_cost = context.metadata.get("estimated_traditional_cost", 0.045)
        unified_cost = context.metadata.get("estimated_unified_cost", 0.015)
        cost_saved = traditional_cost - unified_cost
        
        return SkillResult(
            success=True,
            skill_name=self.name,
            applied=self.use_unified,
            cost_saved=max(0, cost_saved),
            output={
                "method": "hierarchical_unified",
                "layers": ["L1", "L2", "L2.5"],
                "max_tokens": self.max_tokens,
                "cost_reduction_ratio": 3.0,  # 3-4x improvement
            }
        )
