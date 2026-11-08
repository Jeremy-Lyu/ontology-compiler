"""
Capsule Layer - Experience Asset Management

Wraps ontology fragments as reusable capability capsules.
"""

from .capsule import Capsule, CapsuleMetadata
from .execution_plan import ExecutionPlan, DiagnosisStep, DecisionRule, ExecutionStep, ValidationMetric
from .glossary import Glossary, Term, TaskType, ProblemCategory
from .evolution import Evolution, Experiment, EvolutionChain, ExperimentStatus
from .manager import CapsuleManager

__all__ = [
    'Capsule',
    'CapsuleMetadata',
    'ExecutionPlan',
    'DiagnosisStep',
    'DecisionRule',
    'ExecutionStep',
    'ValidationMetric',
    'Glossary',
    'Term',
    'TaskType',
    'ProblemCategory',
    'Evolution',
    'Experiment',
    'EvolutionChain',
    'ExperimentStatus',
    'CapsuleManager',
]

