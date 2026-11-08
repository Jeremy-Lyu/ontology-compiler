"""Skill Chain — 技能链执行

R38-QA: 配置动态组合技能链
"""

import logging
import time
from typing import Dict, List, Optional

from .base import Skill, SkillContext, SkillResult, SkillPriority
from .registry import SkillRegistry

logger = logging.getLogger(__name__)


class SkillChain:
    """Chain of skills for sequential execution.
    
    Skills are executed in priority order, with results from each skill
    feeding into the context for subsequent skills.
    """
    
    def __init__(
        self,
        skill_names: Optional[List[str]] = None,
        registry: Optional[SkillRegistry] = None,
        stop_on_error: bool = True
    ):
        """Initialize skill chain.
        
        Args:
            skill_names: List of skill names to execute (if None, uses all enabled)
            registry: Skill registry (if None, uses global)
            stop_on_error: Whether to stop chain on first error
        """
        self.skill_names = skill_names or []
        self.registry = registry or SkillRegistry()
        self.stop_on_error = stop_on_error
        self.results: List[SkillResult] = []
        
    def with_skills(self, *names: str) -> "SkillChain":
        """Add skills to chain.
        
        Returns:
            Self for chaining
        """
        self.skill_names.extend(names)
        return self
    
    def with_registry(self, registry: SkillRegistry) -> "SkillChain":
        """Set skill registry.
        
        Returns:
            Self for chaining
        """
        self.registry = registry
        return self
    
    async def execute(self, context: SkillContext) -> List[SkillResult]:
        """Execute skill chain.
        
        Args:
            context: Execution context (modified in place)
            
        Returns:
            List of skill execution results
        """
        self.results = []
        
        # Get skills to execute
        if self.skill_names:
            skills = []
            for name in self.skill_names:
                skill = self.registry.get(name)
                if skill:
                    skills.append(skill)
                else:
                    logger.warning(f"Skill '{name}' not found in registry")
                    self.results.append(SkillResult.error_result(name, "Skill not found"))
        else:
            # Use all enabled skills ordered by priority
            skills = self.registry.get_all_ordered()
        
        if not skills:
            logger.warning("No skills to execute")
            return self.results
        
        logger.info(f"Executing skill chain with {len(skills)} skills")
        
        # Execute each skill
        for skill in skills:
            if not skill.is_enabled():
                logger.debug(f"Skipping disabled skill: {skill.name}")
                self.results.append(SkillResult.skipped_result(skill.name, "Skill disabled"))
                continue
            
            start_time = time.time()
            
            try:
                # Check if skill can apply
                can_apply = await skill.can_apply(context)
                if not can_apply:
                    logger.debug(f"Skill '{skill.name}' cannot apply to current context")
                    self.results.append(SkillResult.skipped_result(skill.name, "Cannot apply"))
                    continue
                
                # Execute skill
                logger.info(f"Executing skill: {skill.name}")
                result = await skill.execute(context)
                
                # Record execution time
                result.execution_time_ms = int((time.time() - start_time) * 1000)
                self.results.append(result)
                
                # Update context with result
                if result.success and result.output:
                    context.metadata[f"{skill.name}_output"] = result.output
                
                # Handle errors
                if not result.success and self.stop_on_error:
                    logger.error(f"Skill '{skill.name}' failed, stopping chain")
                    break
                
            except Exception as e:
                logger.exception(f"Error executing skill '{skill.name}': {e}")
                result = SkillResult.error_result(skill.name, str(e))
                result.execution_time_ms = int((time.time() - start_time) * 1000)
                self.results.append(result)
                
                if self.stop_on_error:
                    break
        
        logger.info(f"Skill chain completed: {len([r for r in self.results if r.success])}/{len(self.results)} succeeded")
        return self.results
    
    def get_summary(self) -> Dict[str, any]:
        """Get execution summary.
        
        Returns:
            Summary dictionary
        """
        total = len(self.results)
        successful = len([r for r in self.results if r.success])
        skipped = len([r for r in self.results if r.skipped])
        failed = total - successful
        
        total_time = sum(r.execution_time_ms for r in self.results)
        total_cost_saved = sum(r.cost_saved for r in self.results)
        
        return {
            "total": total,
            "successful": successful,
            "skipped": skipped,
            "failed": failed,
            "total_time_ms": total_time,
            "total_cost_saved": total_cost_saved,
            "skills": [
                {
                    "name": r.skill_name,
                    "success": r.success,
                    "skipped": r.skipped,
                    "applied": r.applied,
                    "time_ms": r.execution_time_ms,
                }
                for r in self.results
            ]
        }


# Predefined skill chains for common workflows
class BuiltinChains:
    """Built-in skill chain configurations."""
    
    @staticmethod
    def optimization_chain() -> SkillChain:
        """Chain for cost optimization."""
        return SkillChain([
            "cost_aware_skip",
            "context_budgeting",
        ])
    
    @staticmethod
    def extraction_chain() -> SkillChain:
        """Chain for extraction with all optimizations."""
        return SkillChain([
            "trunk_analysis",
            "strategy_selection",
            "hierarchical_extraction",
        ])
    
    @staticmethod
    def post_processing_chain() -> SkillChain:
        """Chain for post-processing."""
        return SkillChain([
            "indexed_batch_association",
            "transactional_patch",
        ])
    
    @staticmethod
    def full_pipeline() -> SkillChain:
        """Complete pipeline with all skills."""
        return SkillChain([])  # Uses all enabled skills
