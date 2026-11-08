"""Failure Memory — 失败样本记录与经验复用

P3-EXP-1: 记录抽取/处理过程中的失败样本，支持经验复用和策略优化。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FailureSample:
    """Single failure sample record."""
    
    # Identification
    sample_id: str
    timestamp: str
    
    # Context
    task_type: str  # extraction, validation, association, etc.
    document_type: str  # pdf, markdown, excel, etc.
    domain: str  # SUPPLY_CHAIN, INVENTORY, etc.
    
    # Failure details
    failure_type: str  # parsing_error, llm_timeout, validation_failed, etc.
    error_message: str
    error_code: Optional[str] = None
    
    # Input that caused failure
    input_content: str = ""  # Truncated input
    input_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Resolution
    resolution: Optional[str] = None  # How it was resolved
    resolution_strategy: Optional[str] = None  # skip, retry, fallback, etc.
    was_recovered: bool = False
    
    # Impact assessment
    severity: str = "medium"  # low, medium, high, critical
    retry_count: int = 0
    fallback_used: bool = False
    
    # Attribution
    skill_used: Optional[str] = None  # Which skill was active
    agent_name: Optional[str] = None  # Which agent was running
    layer: Optional[str] = None  # L0, L1, L2, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureSample":
        """Deserialize from dict."""
        return cls(**data)


@dataclass
class CorrectionSample:
    """Corrected sample with ground truth."""
    
    # Identification
    sample_id: str
    timestamp: str
    
    # Original vs Corrected
    original_output: Dict[str, Any]  # Before correction
    corrected_output: Dict[str, Any]  # After correction
    
    # Correction details
    correction_type: str  # entity_fix, relation_add, attribute_update, etc.
    correction_reason: str
    corrected_by: str = "human"  # human, automated, llm_review
    
    # Validation
    validated: bool = False
    validation_method: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CorrectionSample":
        """Deserialize from dict."""
        return cls(**data)


@dataclass
class ExperiencePattern:
    """Pattern discovered from multiple samples."""
    
    pattern_id: str
    pattern_type: str  # recurring_failure, successful_strategy, etc.
    
    # Pattern description
    description: str
    trigger_conditions: List[str]  # Conditions that trigger this pattern
    
    # Statistics
    occurrence_count: int = 0
    success_rate: float = 0.0
    avg_retry_count: float = 0.0
    
    # Recommended actions
    recommended_strategy: Optional[str] = None
    suggested_prompt_modification: Optional[str] = None
    skill_adjustments: Dict[str, Any] = field(default_factory=dict)
    
    # Evidence
    sample_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return asdict(self)


class FailureMemory:
    """Failure memory for storing and analyzing failure samples.
    
    Features:
    - Record failure samples with full context
    - Track correction samples
    - Discover patterns from historical failures
    - Generate recommendations for future tasks
    """
    
    def __init__(
        self,
        storage_path: Path | str | None = None,
        max_samples: int = 10000,
        pattern_threshold: int = 3
    ):
        self.storage_path = Path(storage_path) if storage_path else _default_storage()
        self.max_samples = max_samples
        self.pattern_threshold = pattern_threshold
        
        # In-memory cache
        self._failures: List[FailureSample] = []
        self._corrections: List[CorrectionSample] = []
        self._patterns: List[ExperiencePattern] = []
        
        # Statistics
        self._stats: Dict[str, Any] = {
            "total_failures": 0,
            "total_corrections": 0,
            "by_failure_type": {},
            "by_domain": {},
            "recovery_rate": 0.0
        }
        
        # Load existing data
        self._load()
    
    def _load(self) -> None:
        """Load existing samples from storage."""
        failures_file = self.storage_path / "failures.jsonl"
        corrections_file = self.storage_path / "corrections.jsonl"
        patterns_file = self.storage_path / "patterns.json"
        
        # Load failures
        if failures_file.exists():
            try:
                with open(failures_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            self._failures.append(FailureSample.from_dict(data))
                logger.info(f"Loaded {len(self._failures)} failure samples")
            except Exception as e:
                logger.error(f"Failed to load failures: {e}")
        
        # Load corrections
        if corrections_file.exists():
            try:
                with open(corrections_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            self._corrections.append(CorrectionSample.from_dict(data))
                logger.info(f"Loaded {len(self._corrections)} correction samples")
            except Exception as e:
                logger.error(f"Failed to load corrections: {e}")
        
        # Load patterns
        if patterns_file.exists():
            try:
                with open(patterns_file, 'r', encoding='utf-8') as f:
                    patterns_data = json.load(f)
                    self._patterns = [ExperiencePattern(**p) for p in patterns_data]
                logger.info(f"Loaded {len(self._patterns)} experience patterns")
            except Exception as e:
                logger.error(f"Failed to load patterns: {e}")
        
        self._update_stats()
    
    def _save(self) -> None:
        """Save samples to storage."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        failures_file = self.storage_path / "failures.jsonl"
        corrections_file = self.storage_path / "corrections.jsonl"
        patterns_file = self.storage_path / "patterns.json"
        
        # Save failures
        try:
            with open(failures_file, 'w', encoding='utf-8') as f:
                for sample in self._failures:
                    f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Failed to save failures: {e}")
        
        # Save corrections
        try:
            with open(corrections_file, 'w', encoding='utf-8') as f:
                for sample in self._corrections:
                    f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Failed to save corrections: {e}")
        
        # Save patterns
        try:
            with open(patterns_file, 'w', encoding='utf-8') as f:
                json.dump([p.to_dict() for p in self._patterns], f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save patterns: {e}")
    
    def _update_stats(self) -> None:
        """Update statistics."""
        self._stats["total_failures"] = len(self._failures)
        self._stats["total_corrections"] = len(self._corrections)
        
        # By failure type
        self._stats["by_failure_type"] = {}
        for sample in self._failures:
            ft = sample.failure_type
            self._stats["by_failure_type"][ft] = self._stats["by_failure_type"].get(ft, 0) + 1
        
        # By domain
        self._stats["by_domain"] = {}
        for sample in self._failures:
            domain = sample.domain
            self._stats["by_domain"][domain] = self._stats["by_domain"].get(domain, 0) + 1
        
        # Recovery rate
        recovered = sum(1 for s in self._failures if s.was_recovered)
        self._stats["recovery_rate"] = recovered / len(self._failures) if self._failures else 0.0
    
    def record_failure(
        self,
        task_type: str,
        document_type: str,
        domain: str,
        failure_type: str,
        error_message: str,
        input_content: str = "",
        input_metadata: Optional[Dict[str, Any]] = None,
        skill_used: Optional[str] = None,
        agent_name: Optional[str] = None,
        layer: Optional[str] = None,
        severity: str = "medium"
    ) -> str:
        """Record a failure sample."""
        sample_id = f"fail-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self._failures)}"
        
        sample = FailureSample(
            sample_id=sample_id,
            timestamp=datetime.now().isoformat(),
            task_type=task_type,
            document_type=document_type,
            domain=domain,
            failure_type=failure_type,
            error_message=error_message,
            input_content=input_content[:5000],  # Truncate
            input_metadata=input_metadata or {},
            skill_used=skill_used,
            agent_name=agent_name,
            layer=layer,
            severity=severity
        )
        
        self._failures.append(sample)
        
        # Trim if exceeding max
        if len(self._failures) > self.max_samples:
            self._failures = self._failures[-self.max_samples:]
        
        self._update_stats()
        self._save()
        
        logger.info(f"Recorded failure sample: {sample_id}")
        return sample_id
    
    def record_correction(
        self,
        original_output: Dict[str, Any],
        corrected_output: Dict[str, Any],
        correction_type: str,
        correction_reason: str,
        original_sample_id: Optional[str] = None,
        validated: bool = False
    ) -> str:
        """Record a correction sample."""
        sample_id = f"corr-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self._corrections)}"
        
        sample = CorrectionSample(
            sample_id=sample_id,
            timestamp=datetime.now().isoformat(),
            original_output=original_output,
            corrected_output=corrected_output,
            correction_type=correction_type,
            correction_reason=correction_reason,
            validated=validated
        )
        
        self._corrections.append(sample)
        
        # Update original failure if referenced
        if original_sample_id:
            for failure in self._failures:
                if failure.sample_id == original_sample_id:
                    failure.was_recovered = True
                    failure.resolution = correction_reason
                    break
        
        self._update_stats()
        self._save()
        
        logger.info(f"Recorded correction sample: {sample_id}")
        return sample_id
    
    def update_failure_resolution(
        self,
        sample_id: str,
        resolution: str,
        resolution_strategy: str,
        was_recovered: bool = True
    ) -> bool:
        """Update failure sample with resolution info."""
        for sample in self._failures:
            if sample.sample_id == sample_id:
                sample.resolution = resolution
                sample.resolution_strategy = resolution_strategy
                sample.was_recovered = was_recovered
                self._update_stats()
                self._save()
                return True
        return False
    
    def get_recommendations(
        self,
        task_type: str,
        domain: str,
        document_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recommendations based on historical failures."""
        recommendations = []
        
        # Find similar failures
        similar_failures = [
            f for f in self._failures
            if f.task_type == task_type and f.domain == domain
        ]
        
        if not similar_failures:
            return recommendations
        
        # Group by failure type
        failure_groups: Dict[str, List[FailureSample]] = {}
        for failure in similar_failures:
            ft = failure.failure_type
            if ft not in failure_groups:
                failure_groups[ft] = []
            failure_groups[ft].append(failure)
        
        # Generate recommendations. Unresolved repeats still count: the
        # prompt experience section needs a lesson before anyone records a fix.
        for ft, failures in failure_groups.items():
            count = len(failures)
            if count < 2:
                continue
            
            resolved = [f for f in failures if f.was_recovered]
            if resolved:
                best_resolution = max(resolved, key=lambda x: x.retry_count)
                strategy = best_resolution.resolution_strategy
            else:
                strategy = _DEFAULT_STRATEGY_BY_FAILURE.get(ft, "")
            recommendations.append({
                "failure_type": ft,
                "occurrence_count": count,
                "recovery_rate": len(resolved) / count,
                "recommended_strategy": strategy,
                "severity": _worst_severity(failures),
                "sample_ids": [f.sample_id for f in failures[:3]]
            })
        
        # Sort by occurrence and recovery rate
        recommendations.sort(key=lambda x: (x["occurrence_count"], x["recovery_rate"]), reverse=True)
        
        return recommendations[:5]  # Top 5
    
    def discover_patterns(self) -> List[ExperiencePattern]:
        """Discover patterns from failure samples."""
        patterns = []
        
        # Group failures by common characteristics
        failure_groups: Dict[str, List[FailureSample]] = {}
        
        for sample in self._failures:
            # Create key from failure characteristics
            key = f"{sample.task_type}:{sample.failure_type}:{sample.domain}"
            if key not in failure_groups:
                failure_groups[key] = []
            failure_groups[key].append(sample)
        
        # Generate patterns
        for key, samples in failure_groups.items():
            if len(samples) >= self.pattern_threshold:
                parts = key.split(":")
                pattern = ExperiencePattern(
                    pattern_id=f"pattern-{len(self._patterns)}",
                    pattern_type="recurring_failure",
                    description=f"Recurring {parts[1]} failures in {parts[0]} tasks for {parts[2]} domain",
                    trigger_conditions=[
                        f"task_type={parts[0]}",
                        f"failure_type={parts[1]}",
                        f"domain={parts[2]}"
                    ],
                    occurrence_count=len(samples),
                    success_rate=sum(1 for s in samples if s.was_recovered) / len(samples),
                    avg_retry_count=sum(s.retry_count for s in samples) / len(samples),
                    sample_ids=[s.sample_id for s in samples[:5]]
                )
                
                # Find successful resolutions
                resolved = [s for s in samples if s.was_recovered and s.resolution_strategy]
                if resolved:
                    # Most common resolution strategy
                    strategies = {}
                    for s in resolved:
                        st = s.resolution_strategy
                        strategies[st] = strategies.get(st, 0) + 1
                    if strategies:
                        pattern.recommended_strategy = max(strategies, key=strategies.get)
                
                patterns.append(pattern)
        
        self._patterns = patterns
        self._save()
        
        logger.info(f"Discovered {len(patterns)} new patterns")
        return patterns
    
    def get_stats(self) -> Dict[str, Any]:
        """Get failure memory statistics."""
        return self._stats.copy()
    
    def export_for_learning(
        self,
        output_path: Path | str,
        format: str = "jsonl"
    ) -> int:
        """Export samples for machine learning or analysis."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        count = 0
        
        if format == "jsonl":
            with open(output, 'w', encoding='utf-8') as f:
                for sample in self._failures:
                    f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')
                    count += 1
        elif format == "json":
            with open(output, 'w', encoding='utf-8') as f:
                json.dump({
                    "failures": [s.to_dict() for s in self._failures],
                    "corrections": [s.to_dict() for s in self._corrections],
                    "patterns": [p.to_dict() for p in self._patterns]
                }, f, ensure_ascii=False, indent=2)
                count = len(self._failures) + len(self._corrections)
        
        logger.info(f"Exported {count} samples to {output}")
        return count

    def _failures_for_document(self, document_key: str) -> List[FailureSample]:
        if not document_key:
            return []
        key = str(document_key)
        return [
            sample for sample in self._failures
            if str((sample.input_metadata or {}).get("document_key") or "") == key
        ]

    def check_should_adjust_strategy(self, document_key: str) -> Dict[str, Any]:
        """Return the shape expected by extraction_failure_memory.

        Two or more failures of the same type on this document arm an
        adjustment. Extraction, timeout, call, and JSON failures also request
        fast_mode (``fallback_mode``).
        """
        recent = self._failures_for_document(document_key)
        counts: Dict[str, int] = {}
        for sample in recent:
            counts[sample.failure_type] = counts.get(sample.failure_type, 0) + 1
        adjustments: List[Dict[str, Any]] = []
        for failure_type, count in counts.items():
            if count < 2:
                continue
            kind = "fallback_mode" if failure_type in _FALLBACK_FAILURE_TYPES else "prompt_strategy"
            adjustments.append({
                "type": kind,
                "failure_type": failure_type,
                "count": count,
            })
        return {
            "should_adjust": bool(adjustments),
            "recent_failure_count": len(recent),
            "recommended_adjustments": adjustments,
        }

    def get_adjusted_strategy(self, document_key: str, current_strategy: str = "general") -> str:
        """Map a repeating document failure onto a known prompt strategy name."""
        recent = self._failures_for_document(document_key)
        counts: Dict[str, int] = {}
        for sample in recent:
            counts[sample.failure_type] = counts.get(sample.failure_type, 0) + 1
        for failure_type, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
            if count < 2:
                continue
            target = _PROMPT_STRATEGY_BY_FAILURE.get(failure_type)
            if target and target != current_strategy:
                return target
        return current_strategy


_SEVERITY_RANK = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
_FALLBACK_FAILURE_TYPES = {
    "extraction_failed",
    "llm_timeout",
    "llm_call_failed",
    "json_parse_error",
}
_PROMPT_STRATEGY_BY_FAILURE = {
    "json_parse_error": "contract",
    "schema_violation": "contract",
    "validation_failed": "contract",
    "llm_timeout": "general",
    "llm_call_failed": "general",
    "extraction_failed": "general",
}
_DEFAULT_STRATEGY_BY_FAILURE = {
    "json_parse_error": "strict_json",
    "schema_violation": "explicit_schema",
    "validation_failed": "explicit_classes",
    "llm_timeout": "simplify_content",
    "llm_call_failed": "simplify_content",
    "extraction_failed": "fallback_mode",
}


def _worst_severity(failures: List[FailureSample]) -> str:
    return max(failures, key=lambda sample: _SEVERITY_RANK.get(sample.severity, 2)).severity


def _default_storage() -> Path:
    return Path(__file__).resolve().parents[2] / "outputs" / "failure_memory"


# Global failure memory instance
_failure_memory: Optional[FailureMemory] = None


def get_failure_memory() -> FailureMemory:
    """Get or create global failure memory instance."""
    global _failure_memory
    if _failure_memory is None:
        _failure_memory = FailureMemory(storage_path=_default_storage())
    return _failure_memory


def reset_failure_memory() -> None:
    """Drop the process-wide instance. Tests use this to isolate storage."""
    global _failure_memory
    _failure_memory = None


def bind_failure_memory(memory: FailureMemory) -> None:
    """Install *memory* as the process-wide instance."""
    global _failure_memory
    _failure_memory = memory


def _document_metadata(document_key: Optional[str]) -> Dict[str, Any]:
    if not document_key:
        return {}
    return {"document_key": str(document_key)}


def record_extraction_failure(
    error_message: str,
    document_type: str = "unknown",
    skill_used: Optional[str] = None,
    layer: Optional[str] = None,
    failure_type: str = "extraction_failed",
    severity: str = "medium",
    document_key: Optional[str] = None,
    domain: str = "SUPPLY_CHAIN",
) -> str:
    """Convenience function to record extraction failures.

    ``failure_type`` and ``severity`` are accepted because llm_call records
    LLM failures through this function.
    """
    memory = get_failure_memory()
    return memory.record_failure(
        task_type="extraction",
        document_type=document_type,
        domain=domain,
        failure_type=failure_type,
        error_message=error_message,
        skill_used=skill_used,
        layer=layer,
        severity=severity,
        input_metadata=_document_metadata(document_key),
    )


def record_validation_failure(
    error_message: str,
    document_type: str = "unknown",
    layer: Optional[str] = None,
    document_key: Optional[str] = None,
    domain: str = "SUPPLY_CHAIN",
) -> str:
    """Convenience function to record validation failures."""
    memory = get_failure_memory()
    return memory.record_failure(
        task_type="validation",
        document_type=document_type,
        domain=domain,
        failure_type="validation_failed",
        error_message=error_message,
        layer=layer,
        input_metadata=_document_metadata(document_key),
    )


def record_llm_timeout_failure(
    error_message: str,
    skill_used: Optional[str] = None,
    document_type: str = "unknown",
    layer: Optional[str] = None,
    document_key: Optional[str] = None,
    domain: str = "SUPPLY_CHAIN",
) -> str:
    """Record an LLM timeout. Called from llm_call."""
    memory = get_failure_memory()
    return memory.record_failure(
        task_type="extraction",
        document_type=document_type,
        domain=domain,
        failure_type="llm_timeout",
        error_message=error_message,
        skill_used=skill_used,
        layer=layer,
        severity="high",
        input_metadata=_document_metadata(document_key),
    )


def record_json_parse_failure(
    error_message: str,
    document_type: str = "llm_response",
    layer: Optional[str] = None,
    document_key: Optional[str] = None,
    domain: str = "SUPPLY_CHAIN",
) -> str:
    """Record a JSON parse failure. Called from llm_response_parser."""
    memory = get_failure_memory()
    return memory.record_failure(
        task_type="extraction",
        document_type=document_type,
        domain=domain,
        failure_type="json_parse_error",
        error_message=error_message,
        layer=layer,
        input_metadata=_document_metadata(document_key),
    )


def record_schema_violation(
    error_message: str,
    document_type: str = "unknown",
    layer: Optional[str] = None,
    document_key: Optional[str] = None,
    domain: str = "SUPPLY_CHAIN",
    severity: str = "high",
) -> str:
    """Record a schema violation. Imported by extraction_failure_memory."""
    memory = get_failure_memory()
    return memory.record_failure(
        task_type="validation",
        document_type=document_type,
        domain=domain,
        failure_type="schema_violation",
        error_message=error_message,
        layer=layer,
        severity=severity,
        input_metadata=_document_metadata(document_key),
    )
