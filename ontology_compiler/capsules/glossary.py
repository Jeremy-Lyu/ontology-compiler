"""
Glossary - Unified terminology management

Defines terms, task types, and problem categories used across capsules.
"""

import copy
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum


@dataclass
class Term:
    """Glossary term"""
    
    name: str
    definition: str
    related: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    category: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class TaskType(str, Enum):
    """Task type enumeration"""
    TROUBLESHOOTING = "troubleshooting"
    REFACTOR = "refactor"
    PERFORMANCE_TUNING = "performance_tuning"
    SECURITY_AUDIT = "security_audit"
    MIGRATION = "migration"
    OPTIMIZATION = "optimization"
    INTEGRATION = "integration"
    DEPLOYMENT = "deployment"


class ProblemCategory(str, Enum):
    """Problem category enumeration"""
    INFRASTRUCTURE = "infrastructure"
    DATABASE = "database"
    FRONTEND = "frontend"
    BACKEND = "backend"
    SECURITY = "security"
    DEVOPS = "devops"
    PERFORMANCE = "performance"
    INTEGRATION = "integration"


@dataclass
class Glossary:
    """
    Unified glossary for terminology management.
    
    Provides:
    - Common terms and definitions
    - Task type classifications
    - Problem category classifications
    - Related terms and examples
    """
    
    # Terms
    terms: Dict[str, Term] = field(default_factory=dict)
    
    # Task types (predefined)
    task_types: Dict[str, str] = field(default_factory=lambda: {
        'troubleshooting': 'Problem diagnosis and resolution',
        'refactor': 'Code refactoring and optimization',
        'performance_tuning': 'Performance optimization',
        'security_audit': 'Security audit and hardening',
        'migration': 'System migration',
        'optimization': 'General optimization',
        'integration': 'System integration',
        'deployment': 'Deployment and release',
    })
    
    # Problem categories (predefined)
    problem_categories: Dict[str, str] = field(default_factory=lambda: {
        'infrastructure': 'Infrastructure issues',
        'database': 'Database issues',
        'frontend': 'Frontend issues',
        'backend': 'Backend issues',
        'security': 'Security issues',
        'devops': 'DevOps issues',
        'performance': 'Performance issues',
        'integration': 'Integration issues',
    })
    
    # Task phases
    task_phases: Dict[str, str] = field(default_factory=lambda: {
        'planning': 'Planning phase',
        'diagnosis': 'Diagnosis phase',
        'execution': 'Execution phase',
        'verification': 'Verification phase',
    })
    
    def add_term(self, term: Term) -> None:
        """Add term to glossary"""
        self.terms[term.name] = term
    
    def get_term(self, name: str) -> Optional[Term]:
        """Get term by name"""
        return self.terms.get(name)
    
    def remove_term(self, name: str) -> None:
        """Remove term from glossary"""
        if name in self.terms:
            del self.terms[name]
    
    def list_terms(self, category: Optional[str] = None) -> List[Term]:
        """List all terms, optionally filtered by category"""
        if category:
            return [t for t in self.terms.values() if t.category == category]
        return list(self.terms.values())
    
    def get_task_types(self) -> List[str]:
        """Get all task types"""
        return list(self.task_types.keys())
    
    def get_problem_categories(self) -> List[str]:
        """Get all problem categories"""
        return list(self.problem_categories.keys())
    
    def get_task_phases(self) -> List[str]:
        """Get all task phases"""
        return list(self.task_phases.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'terms': {name: term.to_dict() for name, term in self.terms.items()},
            'task_types': self.task_types,
            'problem_categories': self.problem_categories,
            'task_phases': self.task_phases,
        }
    
    def validate_task_type(self, task_type: str) -> bool:
        """Validate task type"""
        return task_type in self.task_types
    
    def validate_problem_category(self, category: str) -> bool:
        """Validate problem category"""
        return category in self.problem_categories
    
    def validate_task_phase(self, phase: str) -> bool:
        """Validate task phase"""
        return phase in self.task_phases


# Default glossary instance with common ontology terms
DEFAULT_GLOSSARY = Glossary()

# Register common ontology terms
DEFAULT_GLOSSARY.add_term(Term(
    name="ontology",
    definition="A formal representation of knowledge in a domain, including concepts, relationships, and rules",
    category="core",
    examples=["Domain Ontology", "Enterprise Ontology"]
))
DEFAULT_GLOSSARY.add_term(Term(
    name="entity",
    definition="A distinct object, concept, or thing in the domain that can be uniquely identified",
    category="core",
    examples=["Customer", "Order", "Product"]
))
DEFAULT_GLOSSARY.add_term(Term(
    name="relationship",
    definition="A connection or association between two entities in the ontology",
    category="core",
    examples=["Customer-Order", "Order-Product"]
))
DEFAULT_GLOSSARY.add_term(Term(
    name="attribute",
    definition="A property or characteristic of an entity",
    category="core",
    examples=["customer_name", "order_date", "product_price"]
))


def get_default_glossary() -> Glossary:
    """Get default glossary (returns a deep copy to prevent mutation)"""
    return copy.deepcopy(DEFAULT_GLOSSARY)


def register_term(name: str, definition: str, category: str = "", 
                   related: List[str] | None = None, examples: List[str] | None = None) -> None:
    """Register a new term to the default glossary"""
    term = Term(
        name=name,
        definition=definition,
        category=category,
        related=related or [],
        examples=examples or []
    )
    DEFAULT_GLOSSARY.add_term(term)

