"""
ExecutionPlan - Structured execution workflow

Defines the complete execution path: diagnosis → decision → execution → validation
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum


class StepType(str, Enum):
    """Step type enumeration"""
    DIAGNOSIS = "diagnosis"
    DECISION = "decision"
    EXECUTION = "execution"
    VALIDATION = "validation"


@dataclass
class DiagnosisStep:
    """Diagnosis step - Analyze current state"""
    
    name: str
    action: str  # SQL query, command, etc.
    expected_output: str
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class DecisionRule:
    """Decision rule - Determine next action"""
    
    condition: str  # if condition
    action: str    # then action
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ExecutionStep:
    """Execution step - Perform action"""
    
    step: int
    name: str
    action: str
    rollback: str  # Rollback action if needed
    description: str = ""
    estimated_duration: str = ""  # e.g., "5 minutes"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ValidationMetric:
    """Validation metric - Verify result"""
    
    metric: str  # e.g., "p95_latency_ms"
    threshold: str  # e.g., "< 200"
    check_interval: str = ""  # e.g., "5 minutes"
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ExecutionPlan:
    """
    Complete execution plan with 4 phases.
    
    Phases:
    1. Diagnosis: Analyze current state
    2. Decision: Determine actions based on diagnosis
    3. Execution: Perform the actions
    4. Validation: Verify the results
    """
    
    # Diagnosis phase
    diagnosis: List[DiagnosisStep] = field(default_factory=list)
    
    # Decision phase
    decision: List[DecisionRule] = field(default_factory=list)
    
    # Execution phase
    execution: List[ExecutionStep] = field(default_factory=list)
    
    # Validation phase
    validation: List[ValidationMetric] = field(default_factory=list)
    
    # Metadata
    description: str = ""
    estimated_total_duration: str = ""  # e.g., "30 minutes"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'diagnosis': [step.to_dict() for step in self.diagnosis],
            'decision': [rule.to_dict() for rule in self.decision],
            'execution': [step.to_dict() for step in self.execution],
            'validation': [metric.to_dict() for metric in self.validation],
            'description': self.description,
            'estimated_total_duration': self.estimated_total_duration,
        }
    
    def add_diagnosis_step(self, step: DiagnosisStep) -> None:
        """Add diagnosis step"""
        self.diagnosis.append(step)
    
    def add_decision_rule(self, rule: DecisionRule) -> None:
        """Add decision rule"""
        self.decision.append(rule)
    
    def add_execution_step(self, step: ExecutionStep) -> None:
        """Add execution step"""
        self.execution.append(step)
    
    def add_validation_metric(self, metric: ValidationMetric) -> None:
        """Add validation metric"""
        self.validation.append(metric)
    
    def get_step_count(self) -> int:
        """Get total number of steps"""
        return len(self.diagnosis) + len(self.decision) + len(self.execution) + len(self.validation)
    
    def validate(self) -> bool:
        """Validate execution plan"""
        # Must have at least one step in each phase
        if not self.diagnosis:
            raise ValueError("ExecutionPlan must have at least one diagnosis step")
        if not self.execution:
            raise ValueError("ExecutionPlan must have at least one execution step")
        if not self.validation:
            raise ValueError("ExecutionPlan must have at least one validation metric")
        return True

