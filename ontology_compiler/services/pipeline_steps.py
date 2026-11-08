"""Pipeline Step Interface - ABC for all pipeline steps (PIPE-07).

Defines the contract for pipeline steps enabling:
- Step composition and chaining
- Config-driven step selection
- Testing with mocks
- Monitoring and metrics collection
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar
from datetime import datetime


T = TypeVar('T')


@dataclass
class StepContext:
    """Execution context passed to each pipeline step."""
    run_id: str
    start_time: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    
    def record_metric(self, name: str, value: float) -> None:
        self.metrics[name] = value
    
    def get_metric(self, name: str, default: float = 0.0) -> float:
        return self.metrics.get(name, default)


@dataclass
class StepResult(Generic[T]):
    """Result wrapper for pipeline step execution."""
    success: bool
    data: T | None = None
    error: str | None = None
    execution_time_ms: float = 0.0
    context_updates: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def ok(cls, data: T, **updates) -> StepResult[T]:
        return cls(success=True, data=data, context_updates=updates)
    
    @classmethod
    def fail(cls, error: str, **updates) -> StepResult[T]:
        return cls(success=False, error=error, context_updates=updates)


class PipelineStep(ABC, Generic[T]):
    """Abstract base class for all pipeline steps.
    
    All pipeline steps must implement:
    - name: Step identifier
    - execute(): Core logic
    - can_handle(): Input validation
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique step name for logging and configuration."""
        pass
    
    @property
    def description(self) -> str:
        """Human-readable description."""
        return self.name
    
    @abstractmethod
    async def execute(
        self, 
        input_data: Any, 
        context: StepContext
    ) -> StepResult[T]:
        """Execute the step with given input and context.
        
        Args:
            input_data: Input from previous step or initial input
            context: Shared execution context
            
        Returns:
            StepResult with success/failure, data, and updates
        """
        pass
    
    def can_handle(self, input_data: Any) -> bool:
        """Check if this step can process the input.
        
        Override for input validation before execution.
        """
        return True
    
    async def cleanup(self, context: StepContext) -> None:
        """Cleanup resources after execution (optional).
        
        Called regardless of success/failure.
        """
        pass


class CompositeStep(PipelineStep[T]):
    """Combines multiple steps into a single composite step."""
    
    def __init__(self, name: str, steps: list[PipelineStep]):
        self._name = name
        self._steps = steps
    
    @property
    def name(self) -> str:
        return self._name
    
    async def execute(self, input_data: Any, context: StepContext) -> StepResult[T]:
        current = input_data
        
        for step in self._steps:
            if not step.can_handle(current):
                return StepResult.fail(
                    f"Step '{step.name}' cannot handle input: {type(current).__name__}"
                )
            
            result = await step.execute(current, context)
            
            # Update context with step results
            for key, value in result.context_updates.items():
                context.metadata[key] = value
            
            if not result.success:
                return StepResult.fail(
                    f"Step '{step.name}' failed: {result.error}"
                )
            
            current = result.data
        
        return StepResult.ok(current)


__all__ = [
    'PipelineStep',
    'CompositeStep',
    'StepContext',
    'StepResult',
]
