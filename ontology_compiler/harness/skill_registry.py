from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from .manifest import SkillManifest, load_skill_manifest
from src.domain.models.skill_contract import SkillDefinition as CanonicalSkillDefinition, SkillRuntimeContract, SkillSpec, SkillRuntimeType


@dataclass
class SkillExecutionRecord:
    """记录技能执行历史，用于评分和经验积累"""
    timestamp: str
    run_id: str
    success: bool
    duration_ms: float
    error_message: str = ""
    input_hash: str = ""
    output_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SkillPerformance:
    """技能性能统计"""
    total_executions: int = 0
    success_count: int = 0
    avg_duration_ms: float = 0.0
    last_execution: str = ""
    execution_records: List[SkillExecutionRecord] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_executions == 0:
            return 0.5  # 默认值，避免除零
        return self.success_count / self.total_executions

    @property
    def score(self) -> float:
        """综合评分（成功率 + 平均耗时）"""
        rate_score = self.success_rate * 0.7
        speed_score = min(1.0, 1000 / (self.avg_duration_ms + 1)) * 0.3
        return rate_score + speed_score

    def record_execution(self, record: SkillExecutionRecord) -> None:
        """记录执行"""
        self.total_executions += 1
        if record.success:
            self.success_count += 1
        self.last_execution = record.timestamp
        self.execution_records.append(record)
        
        # 更新平均耗时
        if self.total_executions == 1:
            self.avg_duration_ms = record.duration_ms
        else:
            self.avg_duration_ms = (
                self.avg_duration_ms * (self.total_executions - 1) + record.duration_ms
            ) / self.total_executions


@dataclass
class SkillDefinition:
    """Legacy harness skill definition.

    Keep the existing discovery/telemetry model, but expose conversion helpers
    to the canonical `src.domain.models.skill_contract` types.
    """

    name: str
    description: str
    command: str
    script_path: str
    category: str = "workflow"
    enabled: bool = True
    args_schema: dict[str, Any] | None = None
    stages: list[str] | None = None
    version: str = "1.0"
    domains: list[str] = field(default_factory=list)
    layers: list[str] = field(default_factory=list)
    supported_formats: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    performance: SkillPerformance = field(default_factory=SkillPerformance)

    allowed_layers: list[str] = field(default_factory=list)
    allowed_entity_types: list[str] = field(default_factory=list)
    allowed_relation_types: list[str] = field(default_factory=list)
    required_fields: list[str] = field(default_factory=list)
    # P1-2 (2026-10-16): 声明式输出契约（skill.json output_schema），
    # 供 ontology_compiler.skills.evaluator 做 schema 驱动评估。
    output_schema: dict[str, Any] | None = None
    validation_mode: str = "strict"
    gating_level: str = "medium"
    domain_scope: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["performance"] = self.performance.to_dict()
        return result

    def to_canonical_definition(self) -> CanonicalSkillDefinition:
        return CanonicalSkillDefinition(
            skill_id=self.name,
            name=self.name,
            version=self.version,
            description=self.description,
            status="stable" if self.enabled else "deprecated",
            owner=self.domain_scope or None,
            command=self.command or None,
            script_path=self.script_path or None,
            category=self.category or None,
            enabled=self.enabled,
            business_domain=list(self.domains),
            layer_scope=list(self.layers),
            capabilities=list(self.tags),
            metrics=[],
            quality_thresholds={"validation_mode": self.validation_mode, "gating_level": self.gating_level},
            metadata={
                "args_schema": self.args_schema or {},
                "supported_formats": list(self.supported_formats),
                "allowed_layers": list(self.allowed_layers),
                "allowed_entity_types": list(self.allowed_entity_types),
                "allowed_relation_types": list(self.allowed_relation_types),
                "required_fields": list(self.required_fields),
            },
        )

    @classmethod
    def from_manifest(cls, manifest: SkillManifest, *, project_root: Path) -> "SkillDefinition":
        script_path = manifest.script_path
        if script_path and not Path(script_path).is_absolute():
            script_path = str(project_root / script_path)

        governance = getattr(manifest, "governance", {})

        return cls(
            name=manifest.name,
            description=manifest.description,
            command=manifest.command,
            script_path=script_path,
            category=manifest.category,
            enabled=manifest.enabled,
            args_schema=manifest.args_schema,
            stages=manifest.stages,
            version=manifest.version,
            domains=manifest.domains,
            layers=manifest.layers,
            supported_formats=manifest.supported_formats,
            tags=manifest.skill_tags,
            allowed_layers=governance.get("allowed_layers", []),
            allowed_entity_types=governance.get("allowed_entity_types", []),
            allowed_relation_types=governance.get("allowed_relation_types", []),
            required_fields=governance.get("required_fields", []),
            validation_mode=governance.get("validation_mode", "strict"),
            gating_level=governance.get("gating_level", "medium"),
            domain_scope=governance.get("domain_scope", ""),
            output_schema=dict(manifest.output_schema) or None,
        )

    def is_applicable(self, domain: str = "", layer: str = "", file_format: str = "") -> bool:
        if not self.enabled:
            return False
        if domain and self.domains and domain not in self.domains:
            return False
        if layer and self.layers and layer not in self.layers:
            return False
        if file_format and self.supported_formats and file_format not in self.supported_formats:
            return False
        return True

    def validate_entity(self, entity: dict) -> tuple[bool, list[str]]:
        errors = []
        if self.allowed_layers:
            entity_layer = entity.get("layer", "")
            if entity_layer and entity_layer not in self.allowed_layers:
                errors.append(f"Layer '{entity_layer}' not allowed. Allowed: {self.allowed_layers}")
        if self.allowed_entity_types:
            entity_class = entity.get("meta_class", "") or entity.get("class", "")
            if entity_class and entity_class not in self.allowed_entity_types:
                errors.append(f"Entity type '{entity_class}' not allowed. Allowed: {self.allowed_entity_types}")
        if self.required_fields:
            missing_fields = [f for f in self.required_fields if not entity.get(f)]
            if missing_fields:
                errors.append(f"Missing required fields: {missing_fields}")
        return (len(errors) == 0, errors)


class SkillRegistry:
    """Enhanced skill registry with version management, scoring, and scenario applicability.

    This remains the canonical harness registry implementation. A compatibility
    facade can continue to import this class, but new code should use the
    canonical skill contract types and the harness registry API.
    """
    def __init__(self) -> None:
        self._skills: dict[str, SkillDefinition] = {}
        self._discovery_errors: list[dict[str, str]] = []
        self._stage_to_skill: dict[str, str] = {}
        self._domain_to_skills: dict[str, list[str]] = {}
        self._layer_to_skills: dict[str, list[str]] = {}
        self._format_to_skills: dict[str, list[str]] = {}
        self._storage_path: Path = Path("outputs/skill_registry")
        self._telemetry: list[dict[str, Any]] = []
        self._trigger_counts: dict[str, int] = {}
        self._source_coverage: dict[str, set[str]] = {}

    def register(self, skill: SkillDefinition) -> None:
        self._skills[skill.name] = skill
        
        # Stage mapping
        for stage in skill.stages or []:
            self._stage_to_skill[stage] = skill.name
        
        # Domain mapping
        for domain in skill.domains:
            if domain not in self._domain_to_skills:
                self._domain_to_skills[domain] = []
            if skill.name not in self._domain_to_skills[domain]:
                self._domain_to_skills[domain].append(skill.name)
        
        # Layer mapping
        for layer in skill.layers:
            if layer not in self._layer_to_skills:
                self._layer_to_skills[layer] = []
            if skill.name not in self._layer_to_skills[layer]:
                self._layer_to_skills[layer].append(skill.name)
        
        # Format mapping
        for fmt in skill.supported_formats:
            if fmt not in self._format_to_skills:
                self._format_to_skills[fmt] = []
            if skill.name not in self._format_to_skills[fmt]:
                self._format_to_skills[fmt].append(skill.name)

    def list_skills(self, *, enabled_only: bool = True) -> list[SkillDefinition]:
        skills = list(self._skills.values())
        if enabled_only:
            skills = [s for s in skills if s.enabled]
        return sorted(skills, key=lambda s: s.name)

    def get(self, name: str) -> SkillDefinition | None:
        return self._skills.get(name)

    def get_by_stage(self, stage: str) -> SkillDefinition | None:
        skill_name = self._stage_to_skill.get(stage)
        if not skill_name:
            return None
        return self.get(skill_name)

    def get_discovery_errors(self) -> list[dict[str, str]]:
        return list(self._discovery_errors)

    # -- P1-2 (2026-10-16): schema-driven output evaluation ---------------
    def evaluable_skill_names(self, *, enabled_only: bool = True) -> list[str]:
        """Names of skills whose metadata declares an ``output_schema``."""
        return [
            s.name
            for s in self.list_skills(enabled_only=enabled_only)
            if s.output_schema
        ]

    def evaluate_output(self, skill_name: str, output: Any) -> list[str]:
        """Validate *output* against a skill's declared ``output_schema``.

        Returns violation strings (``[]`` == conforms). Unknown skill or a
        skill without a declared schema returns ``[]`` (nothing to check) and
        never raises.
        """
        from ontology_compiler.skills.evaluator import evaluate_output as _eval

        skill = self._skills.get(skill_name)
        if skill is None:
            return []
        return _eval(skill.output_schema, output)

    def find_applicable(self, *, domain: str = "", layer: str = "", 
                       file_format: str = "", min_score: float = 0.0) -> list[SkillDefinition]:
        """查找适用于给定场景的技能
        
        Args:
            domain: 领域
            layer: 层级
            file_format: 文件格式
            min_score: 最低评分
        
        Returns:
            按评分排序的技能列表
        """
        applicable = []
        for skill in self._skills.values():
            if skill.is_applicable(domain=domain, layer=layer, file_format=file_format):
                if skill.performance.score >= min_score:
                    applicable.append(skill)
        
        return sorted(applicable, key=lambda s: s.performance.score, reverse=True)

    def find_by_domain(self, domain: str) -> list[SkillDefinition]:
        """按领域查找技能"""
        skill_names = self._domain_to_skills.get(domain, [])
        return [self._skills[name] for name in skill_names if self._skills[name].enabled]

    def find_by_layer(self, layer: str) -> list[SkillDefinition]:
        """按层级查找技能"""
        skill_names = self._layer_to_skills.get(layer, [])
        return [self._skills[name] for name in skill_names if self._skills[name].enabled]

    def find_by_format(self, file_format: str) -> list[SkillDefinition]:
        """按文件格式查找技能"""
        skill_names = self._format_to_skills.get(file_format, [])
        return [self._skills[name] for name in skill_names if self._skills[name].enabled]

    def record_execution(self, skill_name: str, run_id: str, success: bool, 
                        duration_ms: float, error_message: str = "",
                        input_hash: str = "", output_summary: str = "") -> None:
        """记录技能执行结果，用于评分"""
        skill = self._skills.get(skill_name)
        if skill:
            record = SkillExecutionRecord(
                timestamp=datetime.now().isoformat(),
                run_id=run_id,
                success=success,
                duration_ms=duration_ms,
                error_message=error_message,
                input_hash=input_hash,
                output_summary=output_summary,
            )
            skill.performance.record_execution(record)
            self._save_performance()

    def record_trigger(
        self,
        skill_id: str,
        source_file: str,
        result_stats: Optional[dict[str, Any]] = None,
    ) -> None:
        """记录一次 skill 触发的 telemetry。

        供流水线 / extractor 在 skill 命中后调用。
        """
        result_stats = result_stats or {}
        entry: dict[str, Any] = {
            "skill_id": skill_id,
            "version": None,
            "source_file": source_file,
            "entity_count": result_stats.get("entity_count", 0),
            "relation_count": result_stats.get("relation_count", 0),
            "validation_issues": list(result_stats.get("validation_issues", [])) if result_stats.get("validation_issues") else [],
            "domain": result_stats.get("domain"),
            "timestamp": datetime.utcnow().isoformat(),
        }
        skill = self._skills.get(skill_id)
        if skill:
            entry["version"] = skill.version
            entry["domain"] = entry["domain"] or (skill.domains[0] if skill.domains else None)

        self._telemetry.append(entry)
        self._trigger_counts[skill_id] = self._trigger_counts.get(skill_id, 0) + 1
        self._source_coverage.setdefault(skill_id, set()).add(source_file)

    def get_coverage_report(self) -> dict[str, Any]:
        """返回 skill 覆盖率 + 使用率报告。"""
        total_telemetry = len(self._telemetry)
        total_skills = len(self._skills)
        triggered_skills = set(self._trigger_counts.keys())

        skill_breakdown: list[dict[str, Any]] = []
        for skill in sorted(self._skills.values(), key=lambda s: s.name):
            trigger_count = self._trigger_counts.get(skill.name, 0)
            covered_sources = self._source_coverage.get(skill.name, set())
            skill_breakdown.append({
                "skill_id": skill.name,
                "version": skill.version,
                "enabled": skill.enabled,
                "triggers": trigger_count,
                "covered_sources": len(covered_sources),
                "domains": skill.domains,
                "layers": skill.layers,
            })

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "total_skills": total_skills,
            "skills_triggered": len(triggered_skills),
            "skills_never_triggered": [s.name for s in self._skills.values() if s.name not in triggered_skills],
            "total_triggers": total_telemetry,
            "skill_breakdown": skill_breakdown,
        }

    def save_telemetry(self, path: Optional[Path] = None) -> Path:
        """把当前 telemetry 保存为 JSON 文件。"""
        path = Path(path or self._storage_path / "telemetry.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._telemetry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def get_top_skills(self, n: int = 5) -> list[SkillDefinition]:
        """获取评分最高的技能"""
        skills = [s for s in self._skills.values() if s.enabled]
        return sorted(skills, key=lambda s: s.performance.score, reverse=True)[:n]

    def get_skill_stats(self, skill_name: str) -> dict[str, Any] | None:
        """获取技能统计信息"""
        skill = self._skills.get(skill_name)
        if not skill:
            return None
        return {
            "name": skill.name,
            "version": skill.version,
            "category": skill.category,
            "domains": skill.domains,
            "layers": skill.layers,
            "enabled": skill.enabled,
            "performance": {
                "total_executions": skill.performance.total_executions,
                "success_rate": skill.performance.success_rate,
                "avg_duration_ms": skill.performance.avg_duration_ms,
                "score": skill.performance.score,
            },
        }

    def _record_discovery_error(self, manifest_path: Path, exc: Exception) -> None:
        self._discovery_errors.append(
            {
                "kind": "skill_manifest",
                "path": str(manifest_path),
                "error": str(exc),
            }
        )

    def _load_performance(self) -> None:
        """加载性能统计"""
        perf_file = self._storage_path / "performance.json"
        if perf_file.exists():
            try:
                with open(perf_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for skill_name, perf_data in data.items():
                        if skill_name in self._skills:
                            perf = SkillPerformance(
                                total_executions=perf_data.get("total_executions", 0),
                                success_count=perf_data.get("success_count", 0),
                                avg_duration_ms=perf_data.get("avg_duration_ms", 0.0),
                                last_execution=perf_data.get("last_execution", ""),
                            )
                            self._skills[skill_name].performance = perf
            except Exception as e:
                pass

    def _save_performance(self) -> None:
        """保存性能统计"""
        self._storage_path.mkdir(parents=True, exist_ok=True)
        perf_file = self._storage_path / "performance.json"
        data = {
            name: {
                "total_executions": skill.performance.total_executions,
                "success_count": skill.performance.success_count,
                "avg_duration_ms": skill.performance.avg_duration_ms,
                "last_execution": skill.performance.last_execution,
            }
            for name, skill in self._skills.items()
        }
        with open(perf_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def discover(self, skills_root: Path, *, project_root: Path) -> list[SkillDefinition]:
        self._discovery_errors = []
        if not skills_root.exists():
            return self.list_skills(enabled_only=False)

        for skill_dir in sorted(skills_root.rglob(".ontology-skill")):
            if not skill_dir.is_dir():
                continue
            manifest_path = skill_dir / "skill.json"
            if not manifest_path.exists():
                continue
            try:
                manifest = load_skill_manifest(manifest_path)
                self.register(SkillDefinition.from_manifest(manifest, project_root=project_root))
                sub_skills = getattr(manifest, "sub_skills", [])
                for sub_skill in sub_skills:
                    sub_script_path = sub_skill.get("script_path", "")
                    if sub_script_path and not Path(sub_script_path).is_absolute():
                        sub_script_path = str(project_root / sub_script_path)
                    sub_def = SkillDefinition(
                        name=sub_skill.get("name", ""),
                        description=sub_skill.get("description", ""),
                        command="",
                        script_path=sub_script_path,
                        category="pipeline_stage",
                        enabled=True,
                        args_schema={},
                        stages=[sub_skill.get("stage", "")],
                        version=manifest.version,
                        domains=[],
                        layers=[],
                        supported_formats=[],
                        tags=[],
                    )
                    self.register(sub_def)
            except Exception as exc:
                self._record_discovery_error(manifest_path, exc)

        self._load_performance()
        return self.list_skills(enabled_only=False)


class SkillVersionManager:
    """Manage skill versions with release, rollback, and tracking capabilities."""

    @dataclass
    class VersionRecord:
        version: str
        skill_id: str
        skill_name: str
        released_at: str
        released_by: str
        description: str
        rollback_version: Optional[str] = None
        is_active: bool = True
        metrics: dict[str, Any] = field(default_factory=dict)

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = Path(storage_path) if storage_path else Path("outputs/skill_registry")
        self.versions: dict[str, list[SkillVersionManager.VersionRecord]] = {}
        self._load_versions()

    def release_version(
        self,
        skill_id: str,
        skill_name: str,
        version: str,
        released_by: str,
        description: str,
        metrics: Optional[dict] = None,
    ) -> VersionRecord:
        current_version = self.get_current_version(skill_id)
        
        record = self.VersionRecord(
            version=version,
            skill_id=skill_id,
            skill_name=skill_name,
            released_at=datetime.now().isoformat(),
            released_by=released_by,
            description=description,
            rollback_version=current_version.version if current_version else None,
            is_active=True,
            metrics=metrics or {},
        )

        if skill_id not in self.versions:
            self.versions[skill_id] = []
        
        for prev in self.versions[skill_id]:
            prev.is_active = False
        
        self.versions[skill_id].append(record)
        self._save_versions()
        
        return record

    def rollback(self, skill_id: str, target_version: str) -> bool:
        if skill_id not in self.versions:
            return False
        
        records = self.versions[skill_id]
        target_record = None
        
        for record in records:
            if record.version == target_version:
                target_record = record
                break
        
        if not target_record:
            return False
        
        for record in records:
            record.is_active = False
        
        target_record.is_active = True
        target_record.rollback_version = None
        
        self._save_versions()
        return True

    def get_current_version(self, skill_id: str) -> Optional[VersionRecord]:
        if skill_id not in self.versions:
            return None
        
        for record in reversed(self.versions[skill_id]):
            if record.is_active:
                return record
        
        return None

    def get_version_history(self, skill_id: str) -> list[VersionRecord]:
        return self.versions.get(skill_id, [])

    def compare_versions(self, skill_id: str, version1: str, version2: str) -> Optional[dict]:
        if skill_id not in self.versions:
            return None
        
        records = self.versions[skill_id]
        v1_record = None
        v2_record = None
        
        for record in records:
            if record.version == version1:
                v1_record = record
            if record.version == version2:
                v2_record = record
        
        if not v1_record or not v2_record:
            return None
        
        return {
            "skill_id": skill_id,
            "version1": v1_record.version,
            "version2": v2_record.version,
            "v1_released_at": v1_record.released_at,
            "v2_released_at": v2_record.released_at,
            "v1_metrics": v1_record.metrics,
            "v2_metrics": v2_record.metrics,
            "description_diff": {
                "v1": v1_record.description,
                "v2": v2_record.description,
            },
        }

    def _save_versions(self) -> None:
        self.storage_path.mkdir(parents=True, exist_ok=True)
        data = {}
        for skill_id, records in self.versions.items():
            data[skill_id] = [
                {
                    "version": r.version,
                    "skill_id": r.skill_id,
                    "skill_name": r.skill_name,
                    "released_at": r.released_at,
                    "released_by": r.released_by,
                    "description": r.description,
                    "rollback_version": r.rollback_version,
                    "is_active": r.is_active,
                    "metrics": r.metrics,
                }
                for r in records
            ]
        
        with open(self.storage_path / "versions.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_versions(self) -> None:
        versions_file = self.storage_path / "versions.json"
        if not versions_file.exists():
            return
        
        try:
            with open(versions_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for skill_id, records_data in data.items():
                self.versions[skill_id] = []
                for record_data in records_data:
                    record = self.VersionRecord(
                        version=record_data["version"],
                        skill_id=record_data["skill_id"],
                        skill_name=record_data["skill_name"],
                        released_at=record_data["released_at"],
                        released_by=record_data["released_by"],
                        description=record_data["description"],
                        rollback_version=record_data.get("rollback_version"),
                        is_active=record_data.get("is_active", True),
                        metrics=record_data.get("metrics", {}),
                    )
                    self.versions[skill_id].append(record)
        except Exception:
            pass

    def get_all_versions(self) -> dict[str, list[VersionRecord]]:
        return dict(self.versions)


def build_default_skill_registry(project_root: Path | None = None) -> SkillRegistry:
    """
    Build skill registry from file discovery only.

    Skills are discovered from `skills/<skill-name>/.ontology-skill/skill.json`.
    """
    root = project_root or Path(__file__).resolve().parents[2]
    reg = SkillRegistry()
    reg.discover(root / "skills", project_root=root)
    return reg
