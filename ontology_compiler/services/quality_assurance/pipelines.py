"""Pipeline orchestrators for conflict detection and quality gates.

This module consolidates pipeline logic from conflict_engine.py and quality_gate_rules.py.

[W5-QUALITY-GATE] Enhanced with configuration file support, regression testing,
and comprehensive quality reporting.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .base import (
    BaseConflictRule,
    Conflict,
    DetectionContext,
    QualityContext,
    QualityGateRule,
    with_conflict_stage,
)
from .conflict_dedup import ConflictDedupService, ConflictStats
from .conflict_rules import (
    AttributeTypeConflictRule,
    CircularRelationshipsRule,
    CrossFileNameDriftRule,
    DuplicateNodesRule,
    InstanceBindingConflictRule,
    LayerScopeConsistencyRule,
    MetaClassDriftRule,
    MetaNameDriftRule,
    RelationWhitelistValidator,
    RelationshipKeyConflictRule,
    SameAsConflictRule,
    SemanticConflictRule,
)
from .quality_rules import (
    CrossLayerConsistencyRule,
    DomainAlignmentRule,
    DuplicateDetectionRule,
    EvidenceCoverageRule,
    FieldCompletenessRule,
    LayerBalanceRule,
    LayerInstanceDisciplineRule,
    NamingConsistencyRule,
    NodeCompletenessRule,
    RelationshipValidityRule,
    SemanticQualityRule,
)

# [Phase 6-OBSV-2] Execution tracer integration
from ontology_compiler.services.execution_tracer import get_tracer, ExecutionContext

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "quality_policy.yaml"


# =============================================================================
# Conflict Detection Pipeline (from conflict_engine.py)
# =============================================================================


class ConflictDetectionPipeline:
    """Unified conflict detection pipeline.
    
    Consolidates SemanticOverlapValidator and IntegratorJudge conflict detection
    into a single pluggable engine with rule-based extensibility.
    
    [Conflict Dedup Integration]: Integrated with ConflictDedupService for
    unified conflict counting and deduplication.
    """

    def __init__(self, rules: Optional[List[BaseConflictRule]] = None, enable_dedup: bool = True):
        self.rules = rules or self._default_rules()
        self.logger = logging.getLogger("ConflictDetection.Pipeline")
        self.enable_dedup = enable_dedup
        self._dedup_service = ConflictDedupService()

    def _default_rules(self) -> List[BaseConflictRule]:
        """Get default conflict detection rules."""
        return [
            DuplicateNodesRule(),
            CircularRelationshipsRule(),
            SemanticConflictRule(),
            AttributeTypeConflictRule(),
            RelationWhitelistValidator(),
            SameAsConflictRule(),
            MetaNameDriftRule(),
            MetaClassDriftRule(),
            InstanceBindingConflictRule(),
            RelationshipKeyConflictRule(),
            CrossFileNameDriftRule(),
            LayerScopeConsistencyRule(),
        ]

    def detect(self, context: DetectionContext, stage: str = "") -> List[Conflict]:
        """Run all conflict detection rules with deduplication."""
        all_conflicts: List[Conflict] = []
        
        self.logger.info(f"Running conflict detection with {len(self.rules)} rules")
        
        if self.enable_dedup:
            self.reset_dedup()
        
        for rule in self.rules:
            try:
                conflicts = rule.detect(context)
                for conflict in conflicts:
                    all_conflicts.append(with_conflict_stage(conflict, stage))
                if conflicts:
                    self.logger.debug(
                        f"Rule {rule.name} detected {len(conflicts)} conflicts"
                    )
            except Exception as e:
                self.logger.error(f"Rule {rule.name} failed: {e}")
        
        if self.enable_dedup:
            all_conflicts = self._dedup_service.dedup_conflicts(all_conflicts, stage)
            stats = self._dedup_service.get_stats()
            self.logger.info(
                f"Conflict detection completed: "
                f"unique={stats.unique_conflict_count}, "
                f"total={stats.total_detected_count}, "
                f"duplicate_discoveries={stats.duplicate_discovery_count}"
            )
        else:
            self.logger.info(f"Total conflicts detected: {len(all_conflicts)}")
        
        return all_conflicts

    def merge_from_other_stage(self, conflicts: List[Conflict], stage: str = "") -> None:
        """Merge conflicts from another stage for cross-stage deduplication."""
        staged_conflicts = [with_conflict_stage(conflict, stage) for conflict in conflicts]
        self._dedup_service.merge_conflicts(staged_conflicts)

    def get_unique_conflicts(self) -> List[Conflict]:
        """Get all unique conflicts across all stages."""
        return self._dedup_service.get_unique_conflicts()

    def get_stats(self) -> ConflictStats:
        """Get unified conflict statistics."""
        return self._dedup_service.get_stats()

    def reset_dedup(self) -> None:
        """Reset deduplication state for a new pipeline run."""
        self._dedup_service.reset()

    def add_rule(self, rule: BaseConflictRule) -> None:
        """Add a custom conflict detection rule."""
        self.rules.append(rule)

    def remove_rule(self, rule_name: str) -> bool:
        """Remove a rule by name."""
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                self.rules.pop(i)
                return True
        return False


# =============================================================================
# Quality Gate Pipeline (from quality_gate_rules.py)
# =============================================================================


@dataclass
class RuleResult:
    """Result of a single quality gate rule."""
    rule_id: str
    score: float
    passed: bool
    weight: float
    issues: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0


@dataclass
class PipelineResult:
    """Result from quality gate pipeline execution.
    
    [W5-QUALITY-GATE] Enhanced with detailed rule results and regression support.
    """
    passed: bool
    confidence: float
    quality_level: str
    scores: Dict[str, float]
    rule_results: List[RuleResult] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    regression_status: str = "UNKNOWN"
    regression_comparison: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "passed": self.passed,
            "confidence": self.confidence,
            "quality_level": self.quality_level,
            "scores": self.scores,
            "rule_results": [
                {
                    "rule_id": rr.rule_id,
                    "score": rr.score,
                    "passed": rr.passed,
                    "weight": rr.weight,
                    "issues": rr.issues,
                    "execution_time_ms": rr.execution_time_ms,
                }
                for rr in self.rule_results
            ],
            "issues": self.issues,
            "warnings": self.warnings,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
            "regression_status": self.regression_status,
            "regression_comparison": self.regression_comparison,
        }


class QualityGatePipeline:
    """Pipeline for executing quality gate rules.
    
    Replaces monolithic RIGOR checks with composable rule chains.
    
    [Phase 6-OBSV-2]: Integrated with ExecutionTracer for quality gate tracking
    [W5-QUALITY-GATE]: Enhanced with configuration file support, regression testing,
    and comprehensive quality reporting.
    """
    
    def __init__(
        self,
        rules: Optional[List[QualityGateRule]] = None,
        tracer: Optional[ExecutionContext] = None,
        config_path: Optional[Path] = None,
    ):
        self.rules = rules or self._default_rules()
        self.logger = logging.getLogger("QualityGate.Pipeline")
        self.tracer = tracer or get_tracer()
        self.config = self._load_config(config_path or DEFAULT_CONFIG_PATH)
        self.thresholds = self.config.get("quality_thresholds", {})
    
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load quality policy configuration from YAML file."""
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}, using defaults")
        return {}
    
    def _get_rule_threshold(self, rule_id: str) -> Dict[str, float]:
        """Get threshold configuration for a specific rule."""
        default = {"min_score": 0.7, "warning_threshold": 0.5, "weight": 1.0}
        return self.thresholds.get(rule_id, default)
    
    def _default_rules(self) -> List[QualityGateRule]:
        """Get default rule set.
        
        Includes both original quality gate rules and rules from QualityReviewer
        for comprehensive quality assessment (M03 - 2026-06-10).
        """
        return [
            EvidenceCoverageRule(),
            
            NodeCompletenessRule(),
            FieldCompletenessRule(),
            LayerInstanceDisciplineRule(),
            
            RelationshipValidityRule(),
            
            SemanticQualityRule(),
            NamingConsistencyRule(),
            
            LayerBalanceRule(),
            CrossLayerConsistencyRule(),
            
            DomainAlignmentRule(),
            DuplicateDetectionRule(),
        ]
    
    async def execute(self, fragment: Dict[str, Any], domain: str = "SUPPLY_CHAIN", run_id: str = "") -> PipelineResult:
        """Execute all rules in the pipeline.
        
        Args:
            fragment: The ontology fragment to check
            domain: Domain context for alignment checks
            run_id: Optional run ID for tracing
            
        Returns:
            PipelineResult with aggregated scores and issues
        """
        start_time = time.time()
        
        if run_id:
            self.tracer.record_step_start(run_id, "quality_gate")
        
        context = QualityContext(
            fragment=fragment,
            domain=domain,
            metadata={"started_at": datetime.now().isoformat()}
        )
        
        rule_results: List[RuleResult] = []
        warnings: List[str] = []
        
        self.logger.info(f"Starting quality gate pipeline for {len(self.rules)} rules")
        
        for rule in self.rules:
            rule_start = time.time()
            try:
                score = await rule.check(context)
                context.add_score(rule.rule_id, score)
                
                threshold = self._get_rule_threshold(rule.rule_id)
                min_score = threshold.get("min_score", 0.7)
                warning_threshold = threshold.get("warning_threshold", 0.5)
                
                rule_passed = score >= min_score
                
                if score < warning_threshold:
                    warnings.append(f"Rule {rule.rule_id} below warning threshold: {score:.3f}")
                
                rule_exec_time = (time.time() - rule_start) * 1000
                
                rule_results.append(RuleResult(
                    rule_id=rule.rule_id,
                    score=score,
                    passed=rule_passed,
                    weight=threshold.get("weight", rule.weight),
                    execution_time_ms=rule_exec_time,
                ))
                
                self.logger.debug(f"Rule {rule.rule_id}: {score:.3f} (threshold={min_score})")
                
                if run_id:
                    self.tracer.record_quality_check(
                        run_id=run_id,
                        rule_id=rule.rule_id,
                        score=score,
                        passed=rule_passed,
                        details={
                            "threshold": min_score,
                            "weight": threshold.get("weight", rule.weight),
                        }
                    )
            except Exception as e:
                self.logger.error(f"Rule {rule.rule_id} failed: {e}")
                context.add_score(rule.rule_id, 0.0)
                context.add_issue(f"Rule {rule.rule_id} execution failed: {e}")
                
                rule_results.append(RuleResult(
                    rule_id=rule.rule_id,
                    score=0.0,
                    passed=False,
                    weight=rule.weight,
                    issues=[str(e)],
                    execution_time_ms=(time.time() - rule_start) * 1000,
                ))
                
                if run_id:
                    self.tracer.record_quality_check(
                        run_id=run_id,
                        rule_id=rule.rule_id,
                        score=0.0,
                        passed=False,
                        details={"error": str(e)}
                    )
        
        execution_time = (time.time() - start_time) * 1000
        
        total_weight = sum(rr.weight for rr in rule_results)
        weighted_score = sum(
            context.scores.get(rr.rule_id, 0.0) * rr.weight
            for rr in rule_results
        ) / total_weight if total_weight > 0 else 0.0
        
        governance = self.config.get("governance_rules", {})
        max_issues = governance.get("max_blocked_relations", 15)
        
        failed_rules = [rr for rr in rule_results if not rr.passed]
        major_failed_rules = [rr for rr in rule_results if not rr.passed and rr.score < 0.3]
        
        passed = weighted_score >= 0.6 and len(context.issues) < max_issues * 2 and len(major_failed_rules) < 2
        
        if weighted_score >= 0.85:
            quality_level = "GOLD"
        elif weighted_score >= 0.65:
            quality_level = "SILVER"
        elif weighted_score >= 0.45:
            quality_level = "BRONZE"
        else:
            quality_level = "FAILED"
        
        if run_id:
            self.tracer.record_step_end(
                run_id,
                "quality_gate",
                execution_time,
                details={
                    "quality_level": quality_level,
                    "confidence": weighted_score,
                    "issue_count": len(context.issues),
                    "rule_count": len(self.rules),
                    "failed_rules": [rr.rule_id for rr in failed_rules],
                }
            )
        
        self.logger.info(
            f"Pipeline completed: passed={passed}, "
            f"confidence={weighted_score:.3f}, level={quality_level}, "
            f"issues={len(context.issues)}, failed_rules={len(failed_rules)}"
        )
        
        return PipelineResult(
            passed=passed,
            confidence=weighted_score,
            quality_level=quality_level,
            scores=context.scores.copy(),
            rule_results=rule_results,
            issues=context.issues.copy(),
            warnings=warnings,
            execution_time_ms=execution_time,
            metadata={
                "rule_count": len(self.rules),
                "domain": domain,
                "failed_rules": [rr.rule_id for rr in failed_rules],
                "passed_rules": [rr.rule_id for rr in rule_results if rr.passed],
                **context.metadata
            }
        )
    
    def add_rule(self, rule: QualityGateRule) -> None:
        """Add a rule to the pipeline."""
        self.rules.append(rule)
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID."""
        for i, rule in enumerate(self.rules):
            if rule.rule_id == rule_id:
                self.rules.pop(i)
                return True
        return False
    
    def run_regression_test(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run regression tests against known test cases.
        
        Args:
            test_cases: List of test cases with expected results
            
        Returns:
            Regression report with pass/fail status for each test case
        """
        results = []
        all_passed = True
        
        for test_case in test_cases:
            fragment = test_case.get("fragment", {})
            expected_level = test_case.get("expected_level", "SILVER")
            expected_min_score = test_case.get("expected_min_score", 0.7)
            test_name = test_case.get("name", "Unnamed Test")
            
            result = self.execute(fragment)
            
            passed = (result.quality_level == expected_level and 
                      result.confidence >= expected_min_score)
            all_passed &= passed
            
            results.append({
                "name": test_name,
                "passed": passed,
                "expected_level": expected_level,
                "actual_level": result.quality_level,
                "expected_min_score": expected_min_score,
                "actual_score": result.confidence,
                "issues": result.issues,
            })
        
        return {
            "all_passed": all_passed,
            "test_count": len(test_cases),
            "passed_count": sum(1 for r in results if r["passed"]),
            "failed_count": sum(1 for r in results if not r["passed"]),
            "results": results,
        }


# =============================================================================
# Convenience Functions
# =============================================================================


async def run_quality_gate(
    fragment: Dict[str, Any],
    domain: str = "SUPPLY_CHAIN",
    custom_rules: Optional[List[QualityGateRule]] = None,
    config_path: Optional[Path] = None,
) -> PipelineResult:
    """Run quality gate with default or custom rules.
    
    This is the main entry point for RIGOR replacement.
    
    Args:
        fragment: The ontology fragment to check
        domain: Domain context for alignment checks
        custom_rules: Optional custom rules to use instead of defaults
        config_path: Optional path to quality policy configuration
    
    Returns:
        PipelineResult with aggregated scores and issues
    """
    pipeline = QualityGatePipeline(rules=custom_rules, config_path=config_path)
    return await pipeline.execute(fragment, domain)


async def run_regression_test(
    config_path: Optional[Path] = None,
    regression_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run regression tests against sample files in regression directory.
    
    Args:
        config_path: Optional path to quality policy configuration
        regression_dir: Optional path to regression sample directory
    
    Returns:
        Regression report with pass/fail status for each test case
    """
    if regression_dir is None:
        regression_dir = Path(__file__).parent.parent.parent / "tests" / "regression"
    
    test_cases = []
    
    if regression_dir.exists():
        for json_file in regression_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    test_case = json.load(f)
                    test_case.setdefault("name", json_file.stem)
                    test_cases.append(test_case)
            except Exception as e:
                logger.warning(f"Failed to load regression test {json_file}: {e}")
    
    if not test_cases:
        logger.warning("No regression test cases found")
        return {"all_passed": True, "test_count": 0, "passed_count": 0, "failed_count": 0, "results": []}
    
    pipeline = QualityGatePipeline(config_path=config_path)
    return pipeline.run_regression_test(test_cases)