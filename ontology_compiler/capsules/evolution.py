"""
Evolution - Capsule evolution tracking

Tracks experiments, versions, and improvements over time.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class ExperimentStatus(str, Enum):
    """Experiment status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class Experiment:
    """Single experiment record"""
    
    id: str
    capsule_id: str
    project_name: str
    status: ExperimentStatus = ExperimentStatus.PENDING
    
    # Results
    result_summary: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    duration_seconds: int = 0
    
    # Success tracking
    success: bool = False
    improvement_rate: float = 0.0  # e.g., 0.35 for 35% improvement
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'capsule_id': self.capsule_id,
            'project_name': self.project_name,
            'status': self.status.value,
            'result_summary': self.result_summary,
            'metrics': self.metrics,
            'notes': self.notes,
            'created_at': self.created_at,
            'completed_at': self.completed_at,
            'duration_seconds': self.duration_seconds,
            'success': self.success,
            'improvement_rate': self.improvement_rate,
        }
    
    def mark_completed(self, success: bool, improvement_rate: float = 0.0) -> None:
        """Mark experiment as completed"""
        self.completed_at = datetime.now().isoformat()
        self.success = success
        self.improvement_rate = improvement_rate
        self.status = ExperimentStatus.SUCCESS if success else ExperimentStatus.FAILED


@dataclass
class EvolutionChain:
    """Evolution chain - Track capsule evolution"""
    
    capsule_id: str
    
    # Version history
    versions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Experiments
    experiments: List[Experiment] = field(default_factory=list)
    
    # Statistics
    total_experiments: int = 0
    successful_experiments: int = 0
    failed_experiments: int = 0
    avg_improvement_rate: float = 0.0
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_version(self, version: str, changes: Dict[str, Any]) -> None:
        """Add version to history"""
        self.versions.append({
            'version': version,
            'changes': changes,
            'timestamp': datetime.now().isoformat(),
        })
        self.updated_at = datetime.now().isoformat()
    
    def add_experiment(self, experiment: Experiment) -> None:
        """Add experiment to chain"""
        self.experiments.append(experiment)
        self.total_experiments += 1
        
        if experiment.success:
            self.successful_experiments += 1
        else:
            self.failed_experiments += 1
        
        # Update average improvement rate
        self._update_stats()
        self.updated_at = datetime.now().isoformat()
    
    def _update_stats(self) -> None:
        """Update statistics"""
        if self.experiments:
            total_improvement = sum(e.improvement_rate for e in self.experiments)
            self.avg_improvement_rate = total_improvement / len(self.experiments)
    
    def get_success_rate(self) -> float:
        """Get success rate"""
        if self.total_experiments == 0:
            return 0.0
        return self.successful_experiments / self.total_experiments
    
    def get_latest_version(self) -> Optional[str]:
        """Get latest version"""
        if self.versions:
            return self.versions[-1]['version']
        return None
    
    def get_experiments_by_project(self, project_name: str) -> List[Experiment]:
        """Get experiments by project"""
        return [e for e in self.experiments if e.project_name == project_name]
    
    def get_successful_experiments(self) -> List[Experiment]:
        """Get successful experiments"""
        return [e for e in self.experiments if e.success]
    
    def get_failed_experiments(self) -> List[Experiment]:
        """Get failed experiments"""
        return [e for e in self.experiments if not e.success]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'capsule_id': self.capsule_id,
            'versions': self.versions,
            'experiments': [e.to_dict() for e in self.experiments],
            'total_experiments': self.total_experiments,
            'successful_experiments': self.successful_experiments,
            'failed_experiments': self.failed_experiments,
            'avg_improvement_rate': self.avg_improvement_rate,
            'success_rate': self.get_success_rate(),
            'created_at': self.created_at,
            'updated_at': self.updated_at,
        }


@dataclass
class Evolution:
    """
    Evolution manager - Manages capsule evolution chains.
    
    Tracks:
    - Version history
    - Experiments and results
    - Success rates and improvements
    - Project verification
    """
    
    # Evolution chains by capsule ID
    chains: Dict[str, EvolutionChain] = field(default_factory=dict)
    
    def create_chain(self, capsule_id: str) -> EvolutionChain:
        """Create new evolution chain"""
        if capsule_id not in self.chains:
            self.chains[capsule_id] = EvolutionChain(capsule_id=capsule_id)
        return self.chains[capsule_id]
    
    def get_chain(self, capsule_id: str) -> Optional[EvolutionChain]:
        """Get evolution chain"""
        return self.chains.get(capsule_id)
    
    def add_experiment(self, capsule_id: str, experiment: Experiment) -> None:
        """Add experiment to capsule's evolution chain"""
        chain = self.create_chain(capsule_id)
        chain.add_experiment(experiment)
    
    def add_version(self, capsule_id: str, version: str, changes: Dict[str, Any]) -> None:
        """Add version to capsule's evolution chain"""
        chain = self.create_chain(capsule_id)
        chain.add_version(version, changes)
    
    def get_capsule_stats(self, capsule_id: str) -> Dict[str, Any]:
        """Get capsule statistics"""
        chain = self.get_chain(capsule_id)
        if not chain:
            return {}
        
        return {
            'capsule_id': capsule_id,
            'total_experiments': chain.total_experiments,
            'successful_experiments': chain.successful_experiments,
            'failed_experiments': chain.failed_experiments,
            'success_rate': chain.get_success_rate(),
            'avg_improvement_rate': chain.avg_improvement_rate,
            'latest_version': chain.get_latest_version(),
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all capsules"""
        return {
            capsule_id: self.get_capsule_stats(capsule_id)
            for capsule_id in self.chains.keys()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'chains': {
                capsule_id: chain.to_dict()
                for capsule_id, chain in self.chains.items()
            }
        }

