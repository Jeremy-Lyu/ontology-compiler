"""
Capsule - Reusable Capability Asset

Wraps ontology fragments with execution plans, glossary, and evolution tracking.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


@dataclass
class CapsuleMetadata:
    """Capsule metadata"""
    
    id: str
    name: str
    version: str = "1.0.0"
    author: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Classification
    category: str = ""  # database, performance, security, etc.
    tags: List[str] = field(default_factory=list)
    contexts: List[str] = field(default_factory=list)  # Use cases
    
    # Quality metrics
    success_rate: float = 0.0
    avg_improvement: str = ""
    verified_in: List[str] = field(default_factory=list)  # Project names
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class Capsule:
    """
    Capability Capsule - Wraps ontology fragment with execution plan.
    
    A capsule represents a reusable piece of experience/knowledge that can be:
    - Queried and discovered
    - Executed with context
    - Validated and verified
    - Evolved and improved
    """
    
    # Metadata
    metadata: CapsuleMetadata
    
    # Content
    summary: str = ""
    description: str = ""
    
    # Ontology fragment
    ontology_fragment: Dict[str, Any] = field(default_factory=dict)
    
    # Ability boundaries
    env_constraints: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    costs_and_risks: List[str] = field(default_factory=list)
    
    # Execution plan (from execution_plan.py)
    execution_plan: Optional[Dict[str, Any]] = None
    
    # Glossary terms used in this capsule
    glossary_terms: List[str] = field(default_factory=list)
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    
    # Evolution chain
    evolution_chain: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate capsule"""
        if not self.metadata.id:
            raise ValueError("Capsule ID is required")
        if not self.metadata.name:
            raise ValueError("Capsule name is required")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'metadata': self.metadata.to_dict(),
            'summary': self.summary,
            'description': self.description,
            'ontology_fragment': self.ontology_fragment,
            'env_constraints': self.env_constraints,
            'preconditions': self.preconditions,
            'costs_and_risks': self.costs_and_risks,
            'execution_plan': self.execution_plan,
            'glossary_terms': self.glossary_terms,
            'dependencies': self.dependencies,
            'evolution_chain': self.evolution_chain,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Capsule':
        """Create capsule from dictionary with validation."""
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data).__name__}")

        if "metadata" not in data:
            raise ValueError("Missing required field: metadata")

        metadata_data = data["metadata"]
        if not isinstance(metadata_data, dict):
            raise ValueError(f"Expected metadata to be dict, got {type(metadata_data).__name__}")

        if not metadata_data.get("id"):
            raise ValueError("Missing required field: metadata.id")

        if not metadata_data.get("name"):
            raise ValueError("Missing required field: metadata.name")

        for list_field in ("tags", "contexts", "verified_in"):
            if list_field in metadata_data and not isinstance(metadata_data[list_field], list):
                raise ValueError(
                    f"metadata.{list_field} must be a list, got {type(metadata_data[list_field]).__name__}"
                )

        metadata = CapsuleMetadata(**metadata_data)

        remaining = {k: v for k, v in data.items() if k != "metadata"}
        known_fields = {
            "summary", "description", "ontology_fragment", "env_constraints",
            "preconditions", "costs_and_risks", "execution_plan",
            "glossary_terms", "dependencies", "evolution_chain",
        }
        unknown = set(remaining.keys()) - known_fields
        if unknown:
            raise ValueError(f"Unknown fields in capsule data: {sorted(unknown)}")

        return cls(metadata=metadata, **remaining)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Capsule':
        """Create capsule from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def add_experiment(self, experiment: Dict[str, Any]) -> None:
        """Add experiment to evolution chain"""
        self.evolution_chain.append(experiment)
        self.metadata.updated_at = datetime.now().isoformat()
    
    def update_success_rate(self, success_count: int, total_count: int) -> None:
        """Update success rate based on experiments"""
        if total_count > 0:
            self.metadata.success_rate = success_count / total_count
            self.metadata.updated_at = datetime.now().isoformat()

