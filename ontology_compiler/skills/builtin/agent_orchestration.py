"""Agent Orchestration Skill — 分析策略编排

R38-QA Skill #6: 智能编排多 agent 协作，根据任务类型和上下文动态选择最优执行策略。
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from ..base import Skill, SkillContext, SkillResult, SkillPriority

logger = logging.getLogger(__name__)


# Task type definitions
class TaskType(str):
    EXTRACTION = "extraction"
    ANALYSIS = "analysis"
    VALIDATION = "validation"
    REPORT = "report"
    QUERY = "query"


# Strategy definitions
class Strategy(str):
    FAST = "fast"  # Prioritize speed, lower precision
    ACCURATE = "accurate"  # Prioritize accuracy, more thorough
    BALANCED = "balanced"  # Balance between speed and accuracy
    MINIMAL = "minimal"  # Minimal processing, basic output


@dataclass
class AgentStep:
    """Single agent execution step."""
    step_id: str
    agent_name: str
    inputs: Dict[str, Any]
    outputs: Optional[Dict[str, Any]] = None
    status: str = "pending"  # pending, running, completed, failed
    error: Optional[str] = None


@dataclass
class ExecutionPlan:
    """Complete execution plan for a task."""
    plan_id: str
    task_type: str
    strategy: str
    steps: List[AgentStep] = field(default_factory=list)
    estimated_steps: int = 0


class AgentOrchestrationSkill(Skill):
    """Orchestrate multi-agent collaboration with dynamic strategy selection.
    
    Features:
    1. Task classification and strategy selection
    2. Agent pipeline planning
    3. Step dependency management
    4. Result aggregation and validation
    
    Evidence: folder_analysis_service.py multi-agent coordination
    Performance: Optimizes agent usage based on task complexity.
    """
    
    # Agent registry
    AGENTS = {
        "planner": "PlannerAgent",
        "extractor": "ExtractorAgent",
        "deduper": "DeduperAgent",
        "linker": "LinkerAgent",
        "verifier": "VerifierAgent",
        "reporter": "ReporterAgent",
        "rag_answerer": "RAGAnswererAgent",
    }
    
    # Task to agent pipeline mapping
    TASK_PIPELINES = {
        TaskType.EXTRACTION: ["extractor", "deduper", "verifier"],
        TaskType.ANALYSIS: ["planner", "extractor", "linker", "verifier"],
        TaskType.VALIDATION: ["verifier"],
        TaskType.REPORT: ["planner", "reporter"],
        TaskType.QUERY: ["planner", "rag_answerer"],
    }
    
    # Strategy to step configuration
    STRATEGY_CONFIGS = {
        Strategy.FAST: {"max_retries": 1, "timeout": 30, "skip_optional": True},
        Strategy.ACCURATE: {"max_retries": 3, "timeout": 120, "skip_optional": False},
        Strategy.BALANCED: {"max_retries": 2, "timeout": 60, "skip_optional": False},
        Strategy.MINIMAL: {"max_retries": 1, "timeout": 15, "skip_optional": True},
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.default_strategy = self.config.get("default_strategy", Strategy.BALANCED)
        self.enable_adaptive = self.config.get("enable_adaptive", True)
        self.max_steps = self.config.get("max_steps", 10)
        self._execution_history: List[Dict[str, Any]] = []
    
    @property
    def priority(self) -> SkillPriority:
        return SkillPriority.CRITICAL
    
    @property
    def description(self) -> str:
        return "Orchestrate multi-agent collaboration with dynamic strategy"
    
    async def can_apply(self, context: SkillContext) -> bool:
        """Can apply if task config exists."""
        return context.config.get("task_type") is not None or context.get("task") is not None
    
    def _classify_task(self, context: SkillContext) -> str:
        """Classify task type based on context."""
        # Check explicit task type
        task_type = context.config.get("task_type")
        if task_type:
            return task_type
        
        task = context.get("task")
        if task:
            task_lower = task.lower()
            if "extract" in task_lower:
                return TaskType.EXTRACTION
            elif "analyze" in task_lower or "bom" in task_lower:
                return TaskType.ANALYSIS
            elif "validate" in task_lower or "quality" in task_lower:
                return TaskType.VALIDATION
            elif "report" in task_lower:
                return TaskType.REPORT
            elif "query" in task_lower or "question" in task_lower:
                return TaskType.QUERY
        
        # Infer from document content
        content = context.document_content or ""
        content_lower = content.lower()
        
        if "entity" in content_lower or "relation" in content_lower:
            return TaskType.EXTRACTION
        elif any(kw in content_lower for kw in ["bom", "change", "variant"]):
            return TaskType.ANALYSIS
        elif "quality" in content_lower or "conflict" in content_lower:
            return TaskType.VALIDATION
        elif "summary" in content_lower or "report" in content_lower:
            return TaskType.REPORT
        
        return TaskType.EXTRACTION  # Default
    
    def _select_strategy(self, context: SkillContext, task_type: str) -> str:
        """Select execution strategy based on context."""
        # Check explicit strategy
        strategy = context.config.get("strategy")
        if strategy:
            return strategy
        
        # Check domain-specific overrides
        domain = context.domain.upper()
        
        # High-precision domains
        if domain in ["MEDICAL", "FINANCIAL", "REGULATORY"]:
            return Strategy.ACCURATE
        
        # Low-complexity tasks
        if context.config.get("low_complexity"):
            return Strategy.FAST
        
        # Large documents benefit from minimal processing
        content_len = len(context.document_content or "")
        if content_len > 100000:
            return Strategy.BALANCED
        
        # Check historical performance
        if self.enable_adaptive and self._execution_history:
            recent = self._execution_history[-5:]
            avg_quality = sum(h.get("quality_score", 0) for h in recent) / len(recent)
            if avg_quality < 0.7:
                return Strategy.ACCURATE  # Boost accuracy if recent quality is low
            elif avg_quality > 0.9:
                return Strategy.FAST  # Can afford to be fast if quality is high
        
        return self.default_strategy
    
    def _plan_execution(self, task_type: str, strategy: str) -> ExecutionPlan:
        """Plan agent execution steps."""
        # Get base pipeline
        pipeline = self.TASK_PIPELINES.get(task_type, ["extractor"])
        
        # Get strategy config
        strategy_config = self.STRATEGY_CONFIGS.get(strategy, self.STRATEGY_CONFIGS[Strategy.BALANCED])
        
        # Build execution plan
        plan_id = f"plan-{task_type}-{strategy}"
        steps = []
        
        for i, agent_name in enumerate(pipeline):
            step = AgentStep(
                step_id=f"{plan_id}-step-{i}",
                agent_name=self.AGENTS.get(agent_name, agent_name),
                inputs={"agent_type": agent_name},
            )
            steps.append(step)
        
        plan = ExecutionPlan(
            plan_id=plan_id,
            task_type=task_type,
            strategy=strategy,
            steps=steps,
            estimated_steps=len(steps)
        )
        
        # Limit steps if needed
        if len(plan.steps) > self.max_steps:
            plan.steps = plan.steps[:self.max_steps]
            plan.estimated_steps = self.max_steps
        
        logger.info(f"Created execution plan: {plan_id} with {len(plan.steps)} steps")
        return plan
    
    async def _execute_step(self, step: AgentStep, context: SkillContext, config: Dict[str, Any]) -> bool:
        """Execute single agent step."""
        step.status = "running"
        
        try:
            # Simulate agent execution
            # In real implementation, this would call actual agent
            agent_type = step.inputs.get("agent_type")
            
            if agent_type == "extractor":
                # Extractor agent - entity extraction
                pass
            elif agent_type == "deduper":
                # Deduper agent - deduplication
                pass
            elif agent_type == "verifier":
                # Verifier agent - quality validation
                pass
            elif agent_type == "linker":
                # Linker agent - cross-file association
                pass
            elif agent_type == "reporter":
                # Reporter agent - generate report
                pass
            elif agent_type == "rag_answerer":
                # RAG answerer agent
                pass
            elif agent_type == "planner":
                # Planner agent - task planning
                pass
            
            step.status = "completed"
            step.outputs = {"status": "success"}
            return True
        
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            logger.error(f"Step {step.step_id} failed: {e}")
            return False
    
    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute orchestrated multi-agent workflow."""
        # Classify task
        task_type = self._classify_task(context)
        
        # Select strategy
        strategy = self._select_strategy(context, task_type)
        
        # Plan execution
        plan = self._plan_execution(task_type, strategy)
        strategy_config = self.STRATEGY_CONFIGS.get(strategy, {})
        
        # Execute plan
        completed_steps = 0
        failed_steps = 0
        aggregated_result: Dict[str, Any] = {}
        
        for step in plan.steps:
            success = await self._execute_step(step, context, strategy_config)
            if success:
                completed_steps += 1
                if step.outputs:
                    aggregated_result.update(step.outputs)
            else:
                failed_steps += 1
                if not strategy_config.get("skip_optional", False):
                    # Stop on critical failures
                    break
        
        # Record execution
        execution_record = {
            "task_type": task_type,
            "strategy": strategy,
            "plan_id": plan.plan_id,
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "quality_score": aggregated_result.get("quality_score", 0),
            "timestamp": context.metadata.get("timestamp")
        }
        self._execution_history.append(execution_record)
        
        # Keep history bounded
        if len(self._execution_history) > 100:
            self._execution_history = self._execution_history[-50:]
        
        # Update context with aggregated results
        context.set("execution_plan", plan)
        context.set("aggregated_result", aggregated_result)
        
        return SkillResult(
            success=failed_steps == 0,
            skill_name=self.name,
            applied=True,
            output={
                "plan_id": plan.plan_id,
                "task_type": task_type,
                "strategy": strategy,
                "completed_steps": completed_steps,
                "failed_steps": failed_steps,
                "total_steps": len(plan.steps),
                "execution_history_size": len(self._execution_history)
            }
        )
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution history for analysis."""
        return self._execution_history[-limit:]
    
    def get_agent_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities for each agent."""
        return {
            "planner": ["task_decomposition", "strategy_selection", "step_planning"],
            "extractor": ["entity_extraction", "relation_extraction", "attribute_extraction"],
            "deduper": ["entity_dedup", "relation_dedup", "canonical_merge"],
            "linker": ["cross_file_association", "semantic_linking", "entity_resolution"],
            "verifier": ["quality_check", "conflict_detection", "evidence_validation"],
            "reporter": ["summary_generation", "markdown_report", "structured_output"],
            "rag_answerer": ["question_answering", "knowledge_retrieval", "evidence_binding"],
        }
