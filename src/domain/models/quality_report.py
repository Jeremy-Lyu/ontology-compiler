from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional

VALID_GRADES = {"UNKNOWN", "F", "D", "C", "B", "A", "S", "PLATINUM"}

_QUALITY_LEVEL_MAP: ClassVar[dict[str, str]] = {
    "GOLD": "A",
    "SILVER": "B",
    "BRONZE": "C",
    "FAILED": "F",
}


@dataclass
class RuleScore:
    rule_id: str
    score: float
    weight: float
    passed: bool
    issues: List[str] = field(default_factory=list)


@dataclass
class QualityReport:
    grade: str = "UNKNOWN"
    score: int = 0
    issues: List[str] = field(default_factory=list)
    l1_score: Optional[float] = None
    l2_score: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    rule_scores: List[RuleScore] = field(default_factory=list)
    quality_level: str = "UNKNOWN"
    passed: bool = False
    execution_time_ms: float = 0.0
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Contract metadata (see schemas/quality_report.schema.yaml v1.0.0)
    contract_id: str = "quality_report"
    contract_version: str = "1.0.0"

    def __post_init__(self):
        if not 0 <= self.score <= 100:
            raise ValueError(f"score must be between 0 and 100, got {self.score}")
        if self.grade not in VALID_GRADES:
            raise ValueError(f"grade must be one of {VALID_GRADES}, got {self.grade}")
    
    @property
    def overall_score(self) -> float:
        if self.scores:
            return sum(self.scores.values()) / len(self.scores)
        return self.score / 100.0
    
    @property
    def failed_rules(self) -> List[str]:
        return [rs.rule_id for rs in self.rule_scores if not rs.passed]
    
    @property
    def passed_rules(self) -> List[str]:
        return [rs.rule_id for rs in self.rule_scores if rs.passed]

    @classmethod
    def from_pipeline_result(cls, result: Any) -> "QualityReport":
        if result is None:
            return cls()
        
        quality_level = getattr(result, "quality_level", "UNKNOWN")
        grade = _QUALITY_LEVEL_MAP.get(quality_level, quality_level)
        confidence = getattr(result, "confidence", 0.0)
        
        rule_scores = []
        for rr in getattr(result, "rule_results", []):
            rule_scores.append(RuleScore(
                rule_id=rr.rule_id,
                score=rr.score,
                weight=rr.weight,
                passed=rr.passed,
                issues=getattr(rr, "issues", []),
            ))
        
        return cls(
            grade=grade,
            score=int(confidence * 100),
            quality_level=quality_level,
            passed=getattr(result, "passed", False),
            issues=getattr(result, "issues", []),
            warnings=getattr(result, "warnings", []),
            scores=getattr(result, "scores", {}),
            rule_scores=rule_scores,
            execution_time_ms=getattr(result, "execution_time_ms", 0.0),
            metadata=getattr(result, "metadata", {}),
        )

    @classmethod
    def from_legacy_dict(cls, data: Optional[Dict[str, Any]]) -> "QualityReport":
        if not data:
            return cls()
        score = data.get("score", 0)
        grade = data.get("grade", "UNKNOWN")

        if isinstance(grade, str):
            grade = _QUALITY_LEVEL_MAP.get(grade.upper(), grade.upper())
        
        if isinstance(score, float) and 0 <= score <= 1:
            score = int(score * 100)
        
        return cls(
            grade=grade,
            score=score,
            issues=data.get("issues", []),
            l1_score=data.get("l1_score"),
            l2_score=data.get("l2_score"),
            warnings=data.get("warnings", []),
            scores=data.get("scores", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grade": self.grade,
            "score": self.score,
            "quality_level": self.quality_level,
            "passed": self.passed,
            "issues": self.issues,
            "warnings": self.warnings,
            "l1_score": self.l1_score,
            "l2_score": self.l2_score,
            "scores": self.scores,
            "rule_scores": [
                {
                    "rule_id": rs.rule_id,
                    "score": rs.score,
                    "weight": rs.weight,
                    "passed": rs.passed,
                    "issues": rs.issues,
                }
                for rs in self.rule_scores
            ],
            "failed_rules": self.failed_rules,
            "passed_rules": self.passed_rules,
            "execution_time_ms": self.execution_time_ms,
            "generated_at": self.generated_at,
            "metadata": self.metadata,
            "contract_id": self.contract_id,
            "contract_version": self.contract_version,
        }
