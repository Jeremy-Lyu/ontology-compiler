"""Unified Contract Registry - 统一契约注册表

根据 CONTRACT_GOVERNANCE_SPEC.md 和 PROJECT_ARCHITECTURE_AUDIT_REPORT_20260708.md 的要求：
- 建立统一索引，登记所有主契约
- 记录类型、主文件、版本、状态、依赖、适用范围
- 提供契约校验链入口

注册表包含：
1. SkillRegistry - 能力契约注册表
2. SchemaRegistry - 结构契约注册表
3. ConfigRegistry - 行为契约注册表
4. RuleRegistry - 规则注册表
5. PipelineStepRegistry - Pipeline步骤注册表

设计原则：
- 单一真源：每类契约只有一个权威主文件
- 版本可追踪：记录版本号和变更信息
- 执行可验证：提供校验接口
- 职责可分离：skill管能力，schema管结构，config管策略，code管执行
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

logger = logging.getLogger(__name__)


class ContractStatus(Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    RETIRED = "retired"
    EXPERIMENTAL = "experimental"


class ContractType(Enum):
    SKILL = "skill"
    SCHEMA = "schema"
    CONFIG = "config"
    RULE = "rule"
    PIPELINE_STEP = "pipeline_step"


class ContractCategory(Enum):
    BUSINESS = "business"
    EXECUTION = "execution"
    GOVERNANCE = "governance"


class VersionChangeType(Enum):
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"
    DEPRECATION = "deprecation"
    RETIREMENT = "retirement"


@dataclass
class ContractVersion:
    version: str
    contract_id: str
    change_type: VersionChangeType
    description: str
    author: str = ""
    date: str = ""
    backward_compatible: bool = True
    migration_notes: str = ""
    affected_contracts: List[str] = field(default_factory=list)


class ContractVersionManager:
    """契约版本管理器 - 管理版本历史、变更审查和迁移策略
    
    根据 CONTRACT_GOVERNANCE_SPEC.md 第5节和 PROJECT_ARCHITECTURE_AUDIT_REPORT_20260708.md 第5.2节：
    - 版本策略：major/minor/patch
    - 版本升级要求：向后兼容、影响分析、测试更新、文档同步
    - 退役机制：退役状态、替代契约、迁移说明、生效截止时间
    """

    def __init__(self):
        self._version_history: Dict[str, List[ContractVersion]] = {}

    def record_version(
        self,
        contract_id: str,
        version: str,
        change_type: VersionChangeType,
        description: str,
        author: str = "",
        backward_compatible: bool = True,
        migration_notes: str = "",
        affected_contracts: List[str] = None,
    ):
        """记录版本变更"""
        if contract_id not in self._version_history:
            self._version_history[contract_id] = []

        version_entry = ContractVersion(
            version=version,
            contract_id=contract_id,
            change_type=change_type,
            description=description,
            author=author,
            date=self._current_timestamp(),
            backward_compatible=backward_compatible,
            migration_notes=migration_notes,
            affected_contracts=affected_contracts or [],
        )

        self._version_history[contract_id].append(version_entry)
        logger.info(f"Version recorded: {contract_id} v{version} ({change_type.value})")

    def get_version_history(self, contract_id: str) -> List[ContractVersion]:
        """获取契约的版本历史"""
        return self._version_history.get(contract_id, [])

    def get_latest_version(self, contract_id: str) -> Optional[str]:
        """获取契约的最新版本"""
        history = self._version_history.get(contract_id, [])
        if history:
            return history[-1].version
        return None

    def compare_versions(self, contract_id: str, version1: str, version2: str) -> Optional[VersionChangeType]:
        """比较两个版本的变更类型"""
        history = self._version_history.get(contract_id, [])
        versions = {v.version: v for v in history}
        
        v1 = versions.get(version1)
        v2 = versions.get(version2)
        
        if not v1 or not v2:
            return None

        v1_idx = next((i for i, v in enumerate(history) if v.version == version1), -1)
        v2_idx = next((i for i, v in enumerate(history) if v.version == version2), -1)

        if v2_idx > v1_idx:
            return v2.change_type
        elif v1_idx > v2_idx:
            return v1.change_type
        return None

    def validate_version_change(self, contract_id: str, from_version: str, to_version: str) -> Tuple[bool, List[str]]:
        """验证版本变更是否安全"""
        issues: List[str] = []
        
        history = self._version_history.get(contract_id, [])
        if not history:
            issues.append(f"No version history found for {contract_id}")
            return False, issues

        versions = {v.version: v for v in history}
        from_v = versions.get(from_version)
        to_v = versions.get(to_version)

        if not from_v:
            issues.append(f"Source version {from_version} not found")
        if not to_v:
            issues.append(f"Target version {to_version} not found")

        if to_v and not to_v.backward_compatible:
            issues.append(f"Version {to_version} is not backward compatible")

        if to_v and to_v.affected_contracts:
            issues.append(f"Version {to_version} affects: {', '.join(to_v.affected_contracts)}")

        return len(issues) == 0, issues

    def get_migration_plan(self, contract_id: str, target_version: str) -> Dict[str, Any]:
        """获取迁移计划"""
        history = self._version_history.get(contract_id, [])
        versions = {v.version: v for v in history}
        
        target = versions.get(target_version)
        if not target:
            return {"error": f"Target version {target_version} not found"}

        plan = {
            "contract_id": contract_id,
            "target_version": target_version,
            "change_type": target.change_type.value,
            "description": target.description,
            "backward_compatible": target.backward_compatible,
            "migration_notes": target.migration_notes,
            "affected_contracts": target.affected_contracts,
            "date": target.date,
            "author": target.author,
        }

        if not target.backward_compatible:
            plan["warnings"] = ["This change is not backward compatible", "Review all dependent contracts"]

        return plan

    def get_all_version_stats(self) -> Dict[str, Any]:
        """获取所有版本统计"""
        stats = {}
        for contract_id, history in self._version_history.items():
            stats[contract_id] = {
                "total_versions": len(history),
                "latest_version": history[-1].version if history else "N/A",
                "changes": [v.change_type.value for v in history],
            }
        return stats

    def _current_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()


class ContractChangeReview:
    """契约变更审查 - 管理变更审查流程"""

    def __init__(self):
        self._review_queue: List[Dict[str, Any]] = []

    def submit_for_review(
        self,
        contract_id: str,
        change_description: str,
        change_type: VersionChangeType,
        author: str = "",
        affected_contracts: List[str] = None,
    ) -> str:
        """提交变更审查"""
        review_id = f"REVIEW-{contract_id}-{self._current_timestamp()}"
        
        review = {
            "review_id": review_id,
            "contract_id": contract_id,
            "change_description": change_description,
            "change_type": change_type.value,
            "author": author,
            "status": "pending",
            "affected_contracts": affected_contracts or [],
            "submitted_at": self._current_timestamp(),
            "reviewer": "",
            "review_comments": [],
            "approved_at": "",
        }

        self._review_queue.append(review)
        logger.info(f"Change review submitted: {review_id}")
        return review_id

    def approve_review(self, review_id: str, reviewer: str, comments: str = "") -> bool:
        """批准变更审查"""
        for review in self._review_queue:
            if review["review_id"] == review_id:
                review["status"] = "approved"
                review["reviewer"] = reviewer
                review["review_comments"].append(comments)
                review["approved_at"] = self._current_timestamp()
                logger.info(f"Change review approved: {review_id}")
                return True
        return False

    def reject_review(self, review_id: str, reviewer: str, comments: str = "") -> bool:
        """拒绝变更审查"""
        for review in self._review_queue:
            if review["review_id"] == review_id:
                review["status"] = "rejected"
                review["reviewer"] = reviewer
                review["review_comments"].append(comments)
                logger.info(f"Change review rejected: {review_id}")
                return True
        return False

    def get_review_status(self, review_id: str) -> Optional[Dict[str, Any]]:
        """获取审查状态"""
        for review in self._review_queue:
            if review["review_id"] == review_id:
                return review
        return None

    def get_pending_reviews(self) -> List[Dict[str, Any]]:
        """获取待审查的变更"""
        return [r for r in self._review_queue if r["status"] == "pending"]

    def _current_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d-%H%M%S")


@dataclass
class ContractEntry:
    contract_id: str
    contract_type: ContractType
    name: str
    version: str
    status: ContractStatus
    primary_file: str
    description: str = ""
    purpose: str = ""
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    applicable_layers: List[str] = field(default_factory=list)
    quality_expectations: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    owner: str = ""
    created_at: str = ""
    last_updated: str = ""
    notes: str = ""
    category: ContractCategory = ContractCategory.BUSINESS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "contract_type": self.contract_type.value,
            "category": self.category.value,
            "name": self.name,
            "version": self.version,
            "status": self.status.value,
            "primary_file": self.primary_file,
            "description": self.description,
            "purpose": self.purpose,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "dependencies": self.dependencies,
            "applicable_layers": self.applicable_layers,
            "quality_expectations": self.quality_expectations,
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "owner": self.owner,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "notes": self.notes,
        }


class SkillRegistry:
    """能力契约注册表 - 登记所有 skill 定义"""

    def __init__(self):
        self._entries: Dict[str, ContractEntry] = {}

    def register(
        self,
        skill_id: str,
        name: str,
        version: str,
        primary_file: str,
        status: ContractStatus = ContractStatus.ACTIVE,
        description: str = "",
        purpose: str = "",
        inputs: List[str] = None,
        outputs: List[str] = None,
        dependencies: List[str] = None,
        applicable_layers: List[str] = None,
        quality_expectations: Dict[str, Any] = None,
        preconditions: List[str] = None,
        postconditions: List[str] = None,
        owner: str = "",
        **kwargs,
    ):
        entry = ContractEntry(
            contract_id=skill_id,
            contract_type=ContractType.SKILL,
            name=name,
            version=version,
            status=status,
            primary_file=primary_file,
            description=description,
            purpose=purpose,
            inputs=inputs or [],
            outputs=outputs or [],
            dependencies=dependencies or [],
            applicable_layers=applicable_layers or [],
            quality_expectations=quality_expectations or {},
            preconditions=preconditions or [],
            postconditions=postconditions or [],
            owner=owner,
            **kwargs,
        )
        self._entries[skill_id] = entry
        logger.debug(f"Registered skill: {skill_id}")

    def get(self, skill_id: str) -> Optional[ContractEntry]:
        return self._entries.get(skill_id)

    def list(self, status: Optional[ContractStatus] = None, category: Optional[ContractCategory] = None) -> List[ContractEntry]:
        entries = list(self._entries.values())
        if status:
            entries = [e for e in entries if e.status == status]
        if category:
            entries = [e for e in entries if e.category == category]
        return entries

    def list_by_layer(self, layer: str) -> List[ContractEntry]:
        return [e for e in self._entries.values() if layer in e.applicable_layers]

    def list_by_category(self, category: ContractCategory) -> List[ContractEntry]:
        return [e for e in self._entries.values() if e.category == category]

    def validate(self, skill_id: str) -> Tuple[bool, List[str]]:
        entry = self.get(skill_id)
        if not entry:
            return False, [f"Skill '{skill_id}' not found"]

        errors = []

        if not entry.name:
            errors.append("Skill name is empty")
        if not entry.description:
            errors.append("Skill description is empty")
        if not entry.primary_file:
            errors.append("Skill primary_file is empty")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        return {skill_id: entry.to_dict() for skill_id, entry in self._entries.items()}


class SchemaRegistry:
    """结构契约注册表 - 登记所有 schema 定义
    
    集成 core/schema_registry.py 的本体 schema 验证能力，实现单一真源原则
    """

    def __init__(self):
        self._entries: Dict[str, ContractEntry] = {}
        self._schema_cache: Dict[str, Dict[str, Any]] = {}
        self._ontology_schema_registry = None

    @property
    def ontology_schema_registry(self):
        if self._ontology_schema_registry is None:
            try:
                from core.schema_registry import get_schema_registry
                self._ontology_schema_registry = get_schema_registry()
            except ImportError:
                logger.warning("core.schema_registry not available, ontology schema validation disabled")
        return self._ontology_schema_registry

    def register(
        self,
        schema_id: str,
        name: str,
        version: str,
        primary_file: str,
        status: ContractStatus = ContractStatus.ACTIVE,
        description: str = "",
        purpose: str = "",
        inputs: List[str] = None,
        outputs: List[str] = None,
        dependencies: List[str] = None,
        **kwargs,
    ):
        entry = ContractEntry(
            contract_id=schema_id,
            contract_type=ContractType.SCHEMA,
            name=name,
            version=version,
            status=status,
            primary_file=primary_file,
            description=description,
            purpose=purpose,
            inputs=inputs or [],
            outputs=outputs or [],
            dependencies=dependencies or [],
            **kwargs,
        )
        self._entries[schema_id] = entry
        logger.debug(f"Registered schema: {schema_id}")

    def get(self, schema_id: str) -> Optional[ContractEntry]:
        return self._entries.get(schema_id)

    def list(self, status: Optional[ContractStatus] = None, category: Optional[ContractCategory] = None) -> List[ContractEntry]:
        entries = list(self._entries.values())
        if status:
            entries = [e for e in entries if e.status == status]
        if category:
            entries = [e for e in entries if e.category == category]
        return entries

    def list_by_category(self, category: ContractCategory) -> List[ContractEntry]:
        return [e for e in self._entries.values() if e.category == category]

    def load_schema(self, schema_id: str) -> Optional[Dict[str, Any]]:
        if schema_id in self._schema_cache:
            return self._schema_cache[schema_id]

        entry = self.get(schema_id)
        if not entry:
            return None

        try:
            schema_path = Path(entry.primary_file)
            if not schema_path.exists():
                return None

            with open(schema_path, "r", encoding="utf-8") as f:
                if schema_path.suffix == ".json":
                    schema = json.load(f)
                else:
                    schema = yaml.safe_load(f)

            self._schema_cache[schema_id] = schema
            return schema
        except Exception as e:
            logger.error(f"Failed to load schema {schema_id}: {e}")
            return None

    def validate(self, schema_id: str) -> Tuple[bool, List[str]]:
        entry = self.get(schema_id)
        if not entry:
            return False, [f"Schema '{schema_id}' not found"]

        errors = []

        if not entry.name:
            errors.append("Schema name is empty")
        if not entry.version:
            errors.append("Schema version is empty")
        if not entry.primary_file:
            errors.append("Schema primary_file is empty")

        schema = self.load_schema(schema_id)
        if schema:
            if not isinstance(schema, dict):
                errors.append("Schema is not a valid JSON object")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        return {schema_id: entry.to_dict() for schema_id, entry in self._entries.items()}


class ConfigRegistry:
    """行为契约注册表 - 登记所有 config 定义"""

    def __init__(self):
        self._entries: Dict[str, ContractEntry] = {}
        self._config_cache: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        config_id: str,
        name: str,
        version: str,
        primary_file: str,
        status: ContractStatus = ContractStatus.ACTIVE,
        description: str = "",
        purpose: str = "",
        **kwargs,
    ):
        entry = ContractEntry(
            contract_id=config_id,
            contract_type=ContractType.CONFIG,
            name=name,
            version=version,
            status=status,
            primary_file=primary_file,
            description=description,
            purpose=purpose,
            **kwargs,
        )
        self._entries[config_id] = entry
        logger.debug(f"Registered config: {config_id}")

    def get(self, config_id: str) -> Optional[ContractEntry]:
        return self._entries.get(config_id)

    def list(self, status: Optional[ContractStatus] = None, category: Optional[ContractCategory] = None) -> List[ContractEntry]:
        entries = list(self._entries.values())
        if status:
            entries = [e for e in entries if e.status == status]
        if category:
            entries = [e for e in entries if e.category == category]
        return entries

    def list_by_category(self, category: ContractCategory) -> List[ContractEntry]:
        return [e for e in self._entries.values() if e.category == category]

    def load_config(self, config_id: str) -> Optional[Dict[str, Any]]:
        if config_id in self._config_cache:
            return self._config_cache[config_id]

        entry = self.get(config_id)
        if not entry:
            return None

        try:
            config_path = Path(entry.primary_file)
            if not config_path.exists():
                return None

            with open(config_path, "r", encoding="utf-8") as f:
                if config_path.suffix == ".json":
                    config = json.load(f)
                else:
                    config = yaml.safe_load(f)

            self._config_cache[config_id] = config
            return config
        except Exception as e:
            logger.error(f"Failed to load config {config_id}: {e}")
            return None

    def validate(self, config_id: str) -> Tuple[bool, List[str]]:
        entry = self.get(config_id)
        if not entry:
            return False, [f"Config '{config_id}' not found"]

        errors = []

        if not entry.name:
            errors.append("Config name is empty")
        if not entry.version:
            errors.append("Config version is empty")
        if not entry.primary_file:
            errors.append("Config primary_file is empty")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        return {config_id: entry.to_dict() for config_id, entry in self._entries.items()}


class RuleRegistry:
    """规则注册表 - 登记所有规则定义"""

    def __init__(self):
        self._entries: Dict[str, ContractEntry] = {}
        self._rule_cache: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        rule_id: str,
        name: str,
        version: str,
        primary_file: str,
        status: ContractStatus = ContractStatus.ACTIVE,
        description: str = "",
        purpose: str = "",
        dependencies: List[str] = None,
        **kwargs,
    ):
        entry = ContractEntry(
            contract_id=rule_id,
            contract_type=ContractType.RULE,
            name=name,
            version=version,
            status=status,
            primary_file=primary_file,
            description=description,
            purpose=purpose,
            dependencies=dependencies or [],
            **kwargs,
        )
        self._entries[rule_id] = entry
        logger.debug(f"Registered rule: {rule_id}")

    def get(self, rule_id: str) -> Optional[ContractEntry]:
        return self._entries.get(rule_id)

    def list(self, status: Optional[ContractStatus] = None, category: Optional[ContractCategory] = None) -> List[ContractEntry]:
        entries = list(self._entries.values())
        if status:
            entries = [e for e in entries if e.status == status]
        if category:
            entries = [e for e in entries if e.category == category]
        return entries

    def list_by_category(self, category: ContractCategory) -> List[ContractEntry]:
        return [e for e in self._entries.values() if e.category == category]

    def load_rule(self, rule_id: str) -> Optional[Dict[str, Any]]:
        if rule_id in self._rule_cache:
            return self._rule_cache[rule_id]

        entry = self.get(rule_id)
        if not entry:
            return None

        try:
            rule_path = Path(entry.primary_file)
            if not rule_path.exists():
                return None

            with open(rule_path, "r", encoding="utf-8") as f:
                if rule_path.suffix == ".json":
                    rule = json.load(f)
                else:
                    rule = yaml.safe_load(f)

            self._rule_cache[rule_id] = rule
            return rule
        except Exception as e:
            logger.error(f"Failed to load rule {rule_id}: {e}")
            return None

    def validate(self, rule_id: str) -> Tuple[bool, List[str]]:
        entry = self.get(rule_id)
        if not entry:
            return False, [f"Rule '{rule_id}' not found"]

        errors = []

        if not entry.name:
            errors.append("Rule name is empty")
        if not entry.version:
            errors.append("Rule version is empty")
        if not entry.primary_file:
            errors.append("Rule primary_file is empty")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        return {rule_id: entry.to_dict() for rule_id, entry in self._entries.items()}


class PipelineStepRegistry:
    """Pipeline步骤注册表 - 登记所有 pipeline step 定义"""

    def __init__(self):
        self._entries: Dict[str, ContractEntry] = {}

    def register(
        self,
        step_id: str,
        name: str,
        version: str,
        primary_file: str,
        status: ContractStatus = ContractStatus.ACTIVE,
        description: str = "",
        purpose: str = "",
        inputs: List[str] = None,
        outputs: List[str] = None,
        dependencies: List[str] = None,
        **kwargs,
    ):
        entry = ContractEntry(
            contract_id=step_id,
            contract_type=ContractType.PIPELINE_STEP,
            name=name,
            version=version,
            status=status,
            primary_file=primary_file,
            description=description,
            purpose=purpose,
            inputs=inputs or [],
            outputs=outputs or [],
            dependencies=dependencies or [],
            **kwargs,
        )
        self._entries[step_id] = entry
        logger.debug(f"Registered pipeline step: {step_id}")

    def get(self, step_id: str) -> Optional[ContractEntry]:
        return self._entries.get(step_id)

    def list(self, status: Optional[ContractStatus] = None, category: Optional[ContractCategory] = None) -> List[ContractEntry]:
        entries = list(self._entries.values())
        if status:
            entries = [e for e in entries if e.status == status]
        if category:
            entries = [e for e in entries if e.category == category]
        return entries

    def list_by_category(self, category: ContractCategory) -> List[ContractEntry]:
        return [e for e in self._entries.values() if e.category == category]

    def validate(self, step_id: str) -> Tuple[bool, List[str]]:
        entry = self.get(step_id)
        if not entry:
            return False, [f"Pipeline step '{step_id}' not found"]

        errors = []

        if not entry.name:
            errors.append("Pipeline step name is empty")
        if not entry.version:
            errors.append("Pipeline step version is empty")
        if not entry.primary_file:
            errors.append("Pipeline step primary_file is empty")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        return {step_id: entry.to_dict() for step_id, entry in self._entries.items()}


class UnifiedContractRegistry:
    """统一契约注册表 - 整合所有子注册表"""

    def __init__(self):
        self.skill_registry = SkillRegistry()
        self.schema_registry = SchemaRegistry()
        self.config_registry = ConfigRegistry()
        self.rule_registry = RuleRegistry()
        self.pipeline_step_registry = PipelineStepRegistry()

    def load_from_config(self, config_dir: Optional[Path] = None):
        """从配置目录加载所有契约注册表"""
        config_dir = config_dir or Path("config")
        logger.info("Loading unified contract registry from config...")

        self._load_skill_registry(config_dir)
        self._load_schema_registry(config_dir)
        self._load_config_registry(config_dir)
        self._load_rule_registry(config_dir)
        self._load_pipeline_step_registry(config_dir)

        logger.info("Unified contract registry loaded successfully")

    def _load_skill_registry(self, config_dir: Path):
        skills_dir = config_dir.parent / "skills"
        if not skills_dir.exists():
            logger.warning(f"Skills directory not found: {skills_dir}")
            return

        for skill_dir in skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue

            if skill_dir.name.startswith("_") or skill_dir.name == "__pycache__":
                continue

            self._load_skill_from_directory(skill_dir)

            for sub_dir in skill_dir.iterdir():
                if sub_dir.is_dir() and not sub_dir.name.startswith("_"):
                    self._load_skill_from_directory(sub_dir)

    def _load_skill_from_directory(self, skill_dir: Path):
        skill_json = skill_dir / "skill.json"
        ontology_skill_json = skill_dir / ".ontology-skill" / "skill.json"
        skill_yaml_files = list(skill_dir.glob("*_skill.yaml"))

        if ontology_skill_json.exists():
            self._load_skill_from_json(ontology_skill_json)
        elif skill_json.exists():
            self._load_skill_from_json(skill_json)
        elif skill_yaml_files:
            for yaml_file in skill_yaml_files:
                self._load_skill_from_yaml(yaml_file)

    def _load_skill_from_json(self, skill_file: Path):
        try:
            with open(skill_file, "r", encoding="utf-8") as f:
                skill_data = json.load(f)

            skill_id = skill_data.get("id", skill_file.parent.name)
            category = self._infer_skill_category(skill_id, skill_file)

            self.skill_registry.register(
                skill_id=skill_id,
                name=skill_data.get("name", skill_file.parent.name),
                version=skill_data.get("version", "1.0"),
                primary_file=str(skill_file),
                description=skill_data.get("description", ""),
                purpose=skill_data.get("purpose", ""),
                inputs=skill_data.get("inputs", []),
                outputs=skill_data.get("outputs", []),
                dependencies=skill_data.get("dependencies", []),
                applicable_layers=skill_data.get("supported_layers", []),
                quality_expectations=skill_data.get("quality_expectations", {}),
                preconditions=skill_data.get("preconditions", []),
                postconditions=skill_data.get("postconditions", []),
                owner=skill_data.get("owner", ""),
                category=category,
            )
            logger.debug(f"Loaded skill from JSON: {skill_id}")
        except Exception as e:
            logger.error(f"Failed to load skill from {skill_file}: {e}")

    def _load_skill_from_yaml(self, skill_file: Path):
        try:
            with open(skill_file, "r", encoding="utf-8") as f:
                skill_data = yaml.safe_load(f)

            skill_id = skill_data.get("id", skill_file.stem.replace("_skill", ""))
            category = self._infer_skill_category(skill_id, skill_file)

            self.skill_registry.register(
                skill_id=skill_id,
                name=skill_data.get("name", skill_file.stem.replace("_skill", "")),
                version=skill_data.get("version", "1.0"),
                primary_file=str(skill_file),
                description=skill_data.get("description", ""),
                purpose=skill_data.get("purpose", ""),
                inputs=skill_data.get("inputs", []),
                outputs=skill_data.get("outputs", []),
                dependencies=skill_data.get("dependencies", []),
                applicable_layers=skill_data.get("supported_layers", []),
                quality_expectations=skill_data.get("quality_expectations", {}),
                preconditions=skill_data.get("preconditions", []),
                postconditions=skill_data.get("postconditions", []),
                owner=skill_data.get("owner", ""),
                category=category,
            )
            logger.debug(f"Loaded skill from YAML: {skill_id}")
        except Exception as e:
            logger.error(f"Failed to load skill from {skill_file}: {e}")

    def _infer_skill_category(self, skill_id: str, skill_file: Path) -> ContractCategory:
        path_parts = skill_file.parts
        if "system" in path_parts or "pipeline" in path_parts:
            return ContractCategory.EXECUTION
        if "ontology" in path_parts or "validate" in skill_id.lower() or "governance" in skill_id.lower():
            return ContractCategory.GOVERNANCE
        return ContractCategory.BUSINESS

    def _load_schema_registry(self, config_dir: Path):
        schemas_dir = config_dir.parent / "schemas"
        if not schemas_dir.exists():
            logger.warning(f"Schemas directory not found: {schemas_dir}")
            return

        for schema_file in schemas_dir.rglob("*.json"):
            try:
                with open(schema_file, "r", encoding="utf-8") as f:
                    schema_data = json.load(f)

                # Normalise double-extension ids: `ontology_packet.schema.json`
                # registers as `ontology_packet` (not `ontology_packet.schema`)
                # so callers using the schema-id convention can find it.
                raw_id = schema_file.stem
                schema_id = raw_id[:-len(".schema")] if raw_id.endswith(".schema") else raw_id

                self.schema_registry.register(
                    schema_id=schema_id,
                    name=schema_data.get("title", schema_id),
                    version=schema_data.get("version", "1.0"),
                    primary_file=str(schema_file),
                    description=schema_data.get("description", ""),
                )
            except Exception as e:
                logger.error(f"Failed to load schema {schema_file}: {e}")

    def _load_config_registry(self, config_dir: Path):
        config_files = [
            ("quality_config", "quality/quality_config.yaml"),
            ("pipeline_steps", "pipeline/pipeline_steps.yaml"),
            ("pipeline_policy", "pipeline/pipeline_policy.json"),
            ("entity_classification", "ontology/entity_classification_unified.yaml"),
            ("extraction_policies", "extraction/extraction_policies.yaml"),
            ("extractor_catalogs", "extractor_catalogs.yaml"),
            ("domain_registry", "domain_registry.yaml"),
            ("report_templates", "report_templates.yaml"),
        ]

        for config_id, relative_path in config_files:
            config_path = config_dir / relative_path
            if config_path.exists():
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        if config_path.suffix == ".json":
                            config_data = json.load(f)
                        else:
                            config_data = yaml.safe_load(f) or {}

                    self.config_registry.register(
                        config_id=config_id,
                        name=config_data.get("name", config_id.replace("_", " ").title()),
                        version=config_data.get("version", "1.0"),
                        primary_file=str(config_path),
                        description=config_data.get("description", ""),
                    )
                except Exception as e:
                    logger.error(f"Failed to load config {config_path}: {e}")

    def _load_rule_registry(self, config_dir: Path):
        rule_files = [
            ("relationship_rules", "ontology/relationship_rules.yaml", ContractCategory.GOVERNANCE),
            ("entity_classification_rules", "ontology/entity_classification_unified.yaml", ContractCategory.BUSINESS),
            ("conflict_rules", "quality/conflict_rules.yaml", ContractCategory.GOVERNANCE),
            ("rigor_thresholds", "quality/rigor_thresholds.yaml", ContractCategory.EXECUTION),
        ]

        for rule_id, relative_path, category in rule_files:
            rule_path = config_dir / relative_path
            if rule_path.exists():
                try:
                    with open(rule_path, "r", encoding="utf-8") as f:
                        if rule_path.suffix == ".json":
                            rule_data = json.load(f)
                        else:
                            rule_data = yaml.safe_load(f) or {}

                    self.rule_registry.register(
                        rule_id=rule_id,
                        name=rule_data.get("name", rule_id.replace("_", " ").title()),
                        version=rule_data.get("version", "1.0"),
                        primary_file=str(rule_path),
                        description=rule_data.get("description", ""),
                        category=category,
                    )
                except Exception as e:
                    logger.error(f"Failed to load rule {rule_path}: {e}")

    def _load_pipeline_step_registry(self, config_dir: Path):
        step_files = [
            ("preflight", "pipeline/pipeline_steps.yaml"),
            ("extract", "pipeline/pipeline_steps.yaml"),
            ("merge", "pipeline/pipeline_steps.yaml"),
            ("dedup", "pipeline/pipeline_steps.yaml"),
            ("relationship", "pipeline/pipeline_steps.yaml"),
            ("audit", "pipeline/pipeline_steps.yaml"),
            ("guard", "pipeline/pipeline_steps.yaml"),
            ("export", "pipeline/pipeline_steps.yaml"),
            ("summarize", "pipeline/pipeline_steps.yaml"),
        ]

        for step_id, relative_path in step_files:
            step_path = config_dir / relative_path
            if step_path.exists():
                step_descriptions = {
                    "preflight": "预检查阶段 - 检查输入文件和环境",
                    "extract": "抽取阶段 - 从文档中抽取实体和关系",
                    "merge": "合并阶段 - 合并多个文档的本体",
                    "dedup": "去重阶段 - 去除重复实体",
                    "relationship": "关系构建阶段 - 构建实体间关系",
                    "audit": "审计阶段 - 审计本体质量",
                    "guard": "质量门禁阶段 - 质量门控",
                    "export": "导出阶段 - 导出最终产物",
                    "summarize": "汇总阶段 - 生成执行摘要",
                }

                self.pipeline_step_registry.register(
                    step_id=step_id,
                    name=step_id.replace("_", " ").title(),
                    version="1.0",
                    primary_file=str(step_path),
                    description=step_descriptions.get(step_id, ""),
                )

    def validate_all(self) -> Tuple[bool, List[str]]:
        """验证所有契约"""
        all_errors: List[str] = []

        registries = [
            ("skill", self.skill_registry),
            ("schema", self.schema_registry),
            ("config", self.config_registry),
            ("rule", self.rule_registry),
            ("pipeline_step", self.pipeline_step_registry),
        ]

        for name, registry in registries:
            for entry in registry.list():
                valid, errors = registry.validate(entry.contract_id)
                if not valid:
                    all_errors.extend([f"{name}/{entry.contract_id}: {e}" for e in errors])

        return len(all_errors) == 0, all_errors

    def get_stats(self) -> Dict[str, Any]:
        """获取注册表统计信息"""
        return {
            "skill_count": len(self.skill_registry.list()),
            "schema_count": len(self.schema_registry.list()),
            "config_count": len(self.config_registry.list()),
            "rule_count": len(self.rule_registry.list()),
            "pipeline_step_count": len(self.pipeline_step_registry.list()),
            "total_contracts": (
                len(self.skill_registry.list())
                + len(self.schema_registry.list())
                + len(self.config_registry.list())
                + len(self.rule_registry.list())
                + len(self.pipeline_step_registry.list())
            ),
            "by_category": {
                "business": len(self.skill_registry.list_by_category(ContractCategory.BUSINESS))
                + len(self.schema_registry.list_by_category(ContractCategory.BUSINESS))
                + len(self.config_registry.list_by_category(ContractCategory.BUSINESS))
                + len(self.rule_registry.list_by_category(ContractCategory.BUSINESS))
                + len(self.pipeline_step_registry.list_by_category(ContractCategory.BUSINESS)),
                "execution": len(self.skill_registry.list_by_category(ContractCategory.EXECUTION))
                + len(self.schema_registry.list_by_category(ContractCategory.EXECUTION))
                + len(self.config_registry.list_by_category(ContractCategory.EXECUTION))
                + len(self.rule_registry.list_by_category(ContractCategory.EXECUTION))
                + len(self.pipeline_step_registry.list_by_category(ContractCategory.EXECUTION)),
                "governance": len(self.skill_registry.list_by_category(ContractCategory.GOVERNANCE))
                + len(self.schema_registry.list_by_category(ContractCategory.GOVERNANCE))
                + len(self.config_registry.list_by_category(ContractCategory.GOVERNANCE))
                + len(self.rule_registry.list_by_category(ContractCategory.GOVERNANCE))
                + len(self.pipeline_step_registry.list_by_category(ContractCategory.GOVERNANCE)),
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "skill_registry": self.skill_registry.to_dict(),
            "schema_registry": self.schema_registry.to_dict(),
            "config_registry": self.config_registry.to_dict(),
            "rule_registry": self.rule_registry.to_dict(),
            "pipeline_step_registry": self.pipeline_step_registry.to_dict(),
            "stats": self.get_stats(),
        }


_contract_registry: Optional[UnifiedContractRegistry] = None


def get_contract_registry(
    config_dir: Optional[Path] = None,
    force_reload: bool = False,
) -> UnifiedContractRegistry:
    """获取统一契约注册表单例"""
    global _contract_registry

    if _contract_registry is None or force_reload:
        _contract_registry = UnifiedContractRegistry()
        if config_dir:
            _contract_registry.load_from_config(config_dir)
        else:
            try:
                from src.utils.runtime_paths import get_project_root
                _contract_registry.load_from_config(get_project_root() / "config")
            except ImportError:
                _contract_registry.load_from_config()

    return _contract_registry


def reload_contract_registry(config_dir: Optional[Path] = None) -> UnifiedContractRegistry:
    """重新加载统一契约注册表"""
    return get_contract_registry(config_dir, force_reload=True)