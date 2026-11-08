"""Unified Configuration Manager (P2-2 Enhancement).

集中化配置管理 - 统一加载所有 YAML/JSON 配置文件。

Usage:
    from src.infrastructure.config.unified_config import get_unified_config

    config = get_unified_config()

    # Access all configurations through unified interface
    rag_config = config.rag
    pipeline_config = config.pipeline
    report_templates = config.report_templates
    domain_registry = config.domain_registry
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .rag_config import RAGPipelineConfig
from .pipeline_config import PipelineConfig
from src.pipeline.stages.common import PROJECT_ROOT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UnifiedConfig:
    """Unified configuration container.

    P2-2 Enhancement (2026-05-30):
    Centralizes all YAML/JSON configurations into a single access point.

    P3-04/P3-07 Enhancement (2026-07-05):
    Added entity classification configurations loaded from config/ontology/.

    P3-08 Enhancement (2026-07-08):
    Unified entity classification into single config file entity_classification_unified.yaml.
    Backward-compatible with old field names (entity_classification, classification_rules, entity_type_registry).
    """

    # RAG Pipeline configuration
    rag: RAGPipelineConfig = field(default_factory=RAGPipelineConfig)

    # Pipeline step configuration
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    # Report templates (loaded from YAML)
    report_templates: Dict[str, Any] = field(default_factory=dict)

    # Domain registry (loaded from YAML)
    domain_registry: Dict[str, Any] = field(default_factory=dict)

    # Extractor catalogs (loaded from YAML)
    extractor_catalogs: Dict[str, Any] = field(default_factory=dict)

    # Quality configuration (loaded from YAML)
    quality_config: Dict[str, Any] = field(default_factory=dict)

    # Unified entity classification configuration (single source of truth)
    entity_classification_unified: Dict[str, Any] = field(default_factory=dict)

    # Entity alias dictionary (loaded from config/)
    entity_aliases: Dict[str, Any] = field(default_factory=dict)

    # L0 classification rules (loaded from config/ontology/)
    l0_classification_rules: Dict[str, Any] = field(default_factory=dict)

    # Skill layer mapping (loaded from config/ontology/)
    skill_layer_mapping: Dict[str, Any] = field(default_factory=dict)

    # Entity classification prompt (loaded from config/ontology/)
    entity_classification_prompt: Dict[str, Any] = field(default_factory=dict)

    # Relationship rules (loaded from config/ontology/)
    relationship_rules: Dict[str, Any] = field(default_factory=dict)

    # Relation type registry (loaded from config/ontology/)
    relation_type_registry: Dict[str, Any] = field(default_factory=dict)

    # Pipeline skill routing (loaded from config/)
    pipeline_skill_routing: Dict[str, Any] = field(default_factory=dict)

    # Extraction policies (loaded from config/extraction/)
    extraction_policies: Dict[str, Any] = field(default_factory=dict)

    # Quality policy (loaded from config/)
    quality_policy: Dict[str, Any] = field(default_factory=dict)

    # Conflict rules (loaded from config/quality/)
    conflict_rules: Dict[str, Any] = field(default_factory=dict)

    # Rigor thresholds (loaded from config/quality/)
    rigor_thresholds: Dict[str, Any] = field(default_factory=dict)

    # Domain sources (loaded from config/domains/)
    domain_sources: Dict[str, Any] = field(default_factory=dict)

    # Pipeline policy (loaded from config/pipeline/)
    pipeline_policy: Dict[str, Any] = field(default_factory=dict)

    # Prompts (loaded from config/prompts/)
    extraction_prompts: Dict[str, Any] = field(default_factory=dict)

    # Canary config (loaded from config/ops/)
    canary_config: Dict[str, Any] = field(default_factory=dict)

    # CI policy (loaded from config/ops/)
    ci_policy: Dict[str, Any] = field(default_factory=dict)

    # Identity registry (loaded from config/ops/)
    identity_registry: Dict[str, Any] = field(default_factory=dict)

    # Output settings (loaded from config/output/)
    output_settings: Dict[str, Any] = field(default_factory=dict)

    # Neo4j config (loaded from config/connections/)
    neo4j_config: Dict[str, Any] = field(default_factory=dict)

    # Diagnostics / health report
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # Config metadata
    version: str = "1.1.0"
    loaded_at: str = field(default_factory=lambda: __import__('datetime').datetime.now().isoformat())

    @property
    def entity_classification(self) -> Dict[str, Any]:
        """Backward-compatible access to entity classification config."""
        unified = self.entity_classification_unified
        return {
            "version": unified.get("version", "1.0"),
            "domain": unified.get("domain", "automotive_ontology"),
            "layers": unified.get("layers", {}),
            "classification_rules": unified.get("classification_rules", []),
            "conflict_resolution": unified.get("conflict_resolution", {}),
            "validation": unified.get("validation", {}),
            "output_fields": unified.get("output_fields", {}),
            "relation_family_defaults": unified.get("relation_family_defaults", {}),
            "valid_meta_classes": unified.get("valid_meta_classes", []),
            "field_mappings": unified.get("field_mappings", {}),
            "node_optimizations": unified.get("node_optimizations", {}),
        }

    @property
    def classification_rules(self) -> Dict[str, Any]:
        """Backward-compatible access to classification rules."""
        unified = self.entity_classification_unified
        return {
            "version": unified.get("version", "1.0"),
            "rules": unified.get("classification_rules", []),
            "conflict_resolution": unified.get("conflict_resolution", {}),
            "validation": unified.get("validation", {}),
        }

    @property
    def entity_type_registry(self) -> Dict[str, Any]:
        """Backward-compatible access to entity type registry."""
        unified = self.entity_classification_unified
        return {
            "version": unified.get("version", "1.0"),
            "types": unified.get("types", {}),
            "layer_rules": unified.get("layer_rules", {}),
            "relation_defaults": unified.get("relation_defaults", {}),
            "field_defaults": unified.get("field_defaults", {}),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "version": self.version,
            "loaded_at": self.loaded_at,
            "rag": self._dataclass_to_dict(self.rag),
            "pipeline": self._dataclass_to_dict(self.pipeline),
            "report_templates": self.report_templates,
            "domain_registry": self.domain_registry,
            "extractor_catalogs": self.extractor_catalogs,
            "quality_config": self.quality_config,
            "entity_classification": self.entity_classification,
            "classification_rules": self.classification_rules,
            "entity_type_registry": self.entity_type_registry,
            "entity_aliases": self.entity_aliases,
            "l0_classification_rules": self.l0_classification_rules,
            "skill_layer_mapping": self.skill_layer_mapping,
            "entity_classification_prompt": self.entity_classification_prompt,
            "relationship_rules": self.relationship_rules,
            "relation_type_registry": self.relation_type_registry,
            "pipeline_skill_routing": self.pipeline_skill_routing,
            "extraction_policies": self.extraction_policies,
            "quality_policy": self.quality_policy,
            "conflict_rules": self.conflict_rules,
            "rigor_thresholds": self.rigor_thresholds,
            "domain_sources": self.domain_sources,
            "pipeline_policy": self.pipeline_policy,
            "extraction_prompts": self.extraction_prompts,
            "canary_config": self.canary_config,
            "ci_policy": self.ci_policy,
            "identity_registry": self.identity_registry,
            "output_settings": self.output_settings,
            "neo4j_config": self.neo4j_config,
            "diagnostics": self.diagnostics,
        }

    @staticmethod
    def _dataclass_to_dict(obj: Any) -> Any:
        """Recursively convert dataclass to dictionary."""
        if hasattr(obj, '__dataclass_fields__'):
            return {
                k: UnifiedConfig._dataclass_to_dict(v)
                for k, v in obj.__dict__.items()
            }
        elif isinstance(obj, (list, tuple)):
            return [UnifiedConfig._dataclass_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {
                k: UnifiedConfig._dataclass_to_dict(v)
                for k, v in obj.items()
            }
        return obj


class UnifiedConfigLoader:
    """Loader for unified configuration.

    P2-2 Enhancement (2026-05-30):
    Loads all configuration files from config/ directory.
    """

    REQUIRED_ENTITY_CLASSIFICATION_KEYS = (
        "version",
        "layers",
        "types",
        "classification_rules",
        "conflict_resolution",
        "validation",
        "output_fields",
    )

    REQUIRED_RELATIONSHIP_RULES_KEYS = (
        "relation_family_map",
        "relation_rules",
        "layer_pair_rules",
        "evidence_requirements",
        "rewrite_priorities",
    )

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or (PROJECT_ROOT / "config")
        self._config: Optional[UnifiedConfig] = None

    def load(self, force_reload: bool = False) -> UnifiedConfig:
        """Load all configurations.

        Args:
            force_reload: Force reload even if already loaded

        Returns:
            UnifiedConfig with all configurations loaded
        """
        if self._config is not None and not force_reload:
            return self._config

        logger.info("Loading unified configuration...")

        # Load RAG configuration
        rag_config = self._load_rag_config()

        # Load pipeline configuration
        pipeline_config = self._load_pipeline_config()

        # Load report templates
        report_templates = self._load_yaml_config("output/report_templates.yaml")

        # Load domain registry
        domain_registry = self._load_yaml_config("domains/domain_registry.yaml")

        # Load extractor catalogs
        extractor_catalogs = self._load_yaml_config("extraction/extractor_catalogs.yaml")

        # Load quality configuration
        quality_config = self._load_yaml_config("quality/quality_config.yaml")

        # Load unified entity classification configuration (single source of truth)
        entity_classification_unified = self._load_yaml_config("ontology/entity_classification_unified.yaml")

        # Load entity aliases and L0 classification rules
        entity_aliases = self._load_yaml_config("entity_aliases.yaml")
        l0_classification_rules = self._load_yaml_config("ontology/l0_classification_rules.yaml")

        # Load remaining ontology configurations
        skill_layer_mapping = self._load_yaml_config("ontology/skill_layer_mapping.yaml")
        entity_classification_prompt = self._load_yaml_config("ontology/entity_classification_prompt.yaml")
        relationship_rules = self._load_yaml_config("ontology/relationship_rules.yaml")
        relation_type_registry = self._load_yaml_config("ontology/relation_type_registry.yaml")

        # Load pipeline skill routing
        pipeline_skill_routing = self._load_json_config("pipeline_skill_routing.json")

        # Load extraction configurations
        extraction_policies = self._load_yaml_config("extraction/extraction_policies.yaml")
        extraction_prompts = self._load_json_config("prompts/extraction_prompts.json")

        # Load quality configurations
        quality_policy = self._load_yaml_config("quality_policy.yaml")
        conflict_rules = self._load_yaml_config("quality/conflict_rules.yaml")
        rigor_thresholds = self._load_yaml_config("quality/rigor_thresholds.yaml")

        # Load domain configurations
        domain_sources = self._load_json_config("domains/domain_sources.json")

        # Load pipeline configurations
        pipeline_policy = self._load_json_config("pipeline/pipeline_policy.json")

        # Load ops configurations
        canary_config = self._load_yaml_config("ops/canary_config.yaml")
        ci_policy = self._load_json_config("ops/ci_policy.json")
        identity_registry = self._load_json_config("ops/identity_registry.json")

        # Load output configurations
        output_settings = self._load_json_config("output/settings.json")

        # Load connection configurations
        neo4j_config = self._load_yaml_config("connections/neo4j_config.yaml")

        validation_report = self._build_validation_report(
            entity_classification_unified=entity_classification_unified,
            relationship_rules=relationship_rules,
        )
        if validation_report.get("issues"):
            message = "Configuration validation failed:\n- " + "\n- ".join(validation_report["issues"])
            logger.error(message)
            raise ValueError(message)

        diagnostics = self._build_diagnostics(
            entity_classification_unified=entity_classification_unified,
            entity_aliases=entity_aliases,
            l0_classification_rules=l0_classification_rules,
            relationship_rules=relationship_rules,
            validation_report=validation_report,
        )
        diagnostics["status"] = "ok"
        diagnostics["validation"] = {"status": "passed", "issues": []}
        diagnostics["loaded_sections"] = {
            "entity_classification_unified": bool(entity_classification_unified),
            "entity_aliases": bool(entity_aliases),
            "l0_classification_rules": bool(l0_classification_rules),
            "relationship_rules": bool(relationship_rules),
            "pipeline": bool(getattr(pipeline_config, "steps", None)),
        }

        self._config = UnifiedConfig(
            rag=rag_config,
            pipeline=pipeline_config,
            report_templates=report_templates,
            domain_registry=domain_registry,
            extractor_catalogs=extractor_catalogs,
            quality_config=quality_config,
            entity_classification_unified=entity_classification_unified,
            entity_aliases=entity_aliases,
            l0_classification_rules=l0_classification_rules,
            skill_layer_mapping=skill_layer_mapping,
            entity_classification_prompt=entity_classification_prompt,
            relationship_rules=relationship_rules,
            relation_type_registry=relation_type_registry,
            pipeline_skill_routing=pipeline_skill_routing,
            extraction_policies=extraction_policies,
            quality_policy=quality_policy,
            conflict_rules=conflict_rules,
            rigor_thresholds=rigor_thresholds,
            domain_sources=domain_sources,
            pipeline_policy=pipeline_policy,
            extraction_prompts=extraction_prompts,
            canary_config=canary_config,
            ci_policy=ci_policy,
            identity_registry=identity_registry,
            output_settings=output_settings,
            neo4j_config=neo4j_config,
            diagnostics=diagnostics,
        )
        self._validate_configuration(self._config)

        logger.info("Unified configuration loaded successfully")
        return self._config

    def _load_rag_config(self) -> RAGPipelineConfig:
        """Load RAG pipeline configuration."""
        try:
            from .rag_config import get_rag_config
            return get_rag_config()
        except Exception as e:
            logger.warning(f"Failed to load RAG config: {e}, using defaults")
            return RAGPipelineConfig()

    def _load_pipeline_config(self) -> PipelineConfig:
        """Load pipeline configuration."""
        try:
            # Canonical location is `config/pipeline/pipeline_steps.yaml`
            # (registered by the contract registry); fall back to the legacy
            # `config/pipeline_steps.yaml` path for robustness.
            config_path = self.config_dir / "pipeline" / "pipeline_steps.yaml"
            if not config_path.exists():
                config_path = self.config_dir / "pipeline_steps.yaml"
            return PipelineConfig.load(config_path)
        except Exception as e:
            logger.warning(f"Failed to load pipeline config: {e}, using defaults")
            return PipelineConfig()

    def _load_yaml_config(self, filename: str) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        config_path = self.config_dir / filename
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return {}

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
                logger.debug(f"Loaded config from {config_path}")
                return content or {}
        except Exception as e:
            logger.error(f"Failed to load {config_path}: {e}")
            return {}

    def _load_json_config(self, filename: str) -> Dict[str, Any]:
        """Load a JSON configuration file."""
        config_path = self.config_dir / filename
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return {}

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                content = json.load(f)
                logger.debug(f"Loaded config from {config_path}")
                return content or {}
        except Exception as e:
            logger.error(f"Failed to load {config_path}: {e}")
            return {}

    def _validate_configuration(self, config: UnifiedConfig) -> None:
        """Validate configuration consistency after loading."""
        validation_report = self._build_validation_report(
            entity_classification_unified=config.entity_classification_unified or {},
            relationship_rules=config.relationship_rules or {},
        )
        if validation_report.get("issues"):
            message = "Configuration validation failed:\n- " + "\n- ".join(validation_report["issues"])
            logger.error(message)
            raise ValueError(message)

    def _build_validation_report(
        self,
        entity_classification_unified: Dict[str, Any],
        relationship_rules: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build validation issues and summary for diagnostics."""
        issues: list[str] = []

        issues.extend(self._validate_required_keys(
            "ontology/entity_classification_unified.yaml",
            entity_classification_unified,
            self.REQUIRED_ENTITY_CLASSIFICATION_KEYS,
        ))
        issues.extend(self._validate_mapping_shapes(
            "ontology/entity_classification_unified.yaml",
            entity_classification_unified.get("layers"),
            expected_value_type=dict,
            required_name="layers",
        ))
        issues.extend(self._validate_mapping_shapes(
            "ontology/entity_classification_unified.yaml",
            entity_classification_unified.get("types"),
            expected_value_type=dict,
            required_name="types",
        ))
        issues.extend(self._validate_node_optimization_shapes(
            "ontology/entity_classification_unified.yaml",
            entity_classification_unified.get("node_optimizations"),
        ))

        issues.extend(self._validate_required_keys(
            "ontology/relationship_rules.yaml",
            relationship_rules,
            self.REQUIRED_RELATIONSHIP_RULES_KEYS,
        ))
        issues.extend(self._validate_mapping_shapes(
            "ontology/relationship_rules.yaml",
            relationship_rules.get("relation_family_map"),
            expected_value_type=list,
            required_name="relation_family_map",
        ))
        issues.extend(self._validate_mapping_shapes(
            "ontology/relationship_rules.yaml",
            relationship_rules.get("relation_rules"),
            expected_value_type=dict,
            required_name="relation_rules",
        ))
        issues.extend(self._validate_mapping_shapes(
            "ontology/relationship_rules.yaml",
            relationship_rules.get("layer_pair_rules"),
            expected_value_type=dict,
            required_name="layer_pair_rules",
        ))
        issues.extend(self._validate_relationship_family_entries("ontology/relationship_rules.yaml", relationship_rules))

        return {
            "status": "passed" if not issues else "failed",
            "issues": issues,
            "entity_classification": {
                "version": entity_classification_unified.get("version", "unknown"),
                "layer_count": len(entity_classification_unified.get("layers", {}) or {}),
                "type_count": len(entity_classification_unified.get("types", {}) or {}),
                "rule_count": len(entity_classification_unified.get("classification_rules", []) or []),
                "node_optimization_keys": sorted((entity_classification_unified.get("node_optimizations", {}) or {}).keys()),
            },
            "relationship_rules": {
                "relation_families": len((relationship_rules or {}).get("relation_family_map", {}) or {}),
                "relation_rules": len((relationship_rules or {}).get("relation_rules", {}) or {}),
                "layer_pairs": len((relationship_rules or {}).get("layer_pair_rules", {}) or {}),
            },
        }

    @staticmethod
    def _validate_required_keys(name: str, payload: Dict[str, Any], required_keys: tuple[str, ...]) -> list[str]:
        missing = [key for key in required_keys if key not in payload]
        return [f"{name}: missing required key '{key}'" for key in missing]

    @staticmethod
    def _validate_mapping_shapes(
        name: str,
        payload: Any,
        expected_value_type: type,
        required_name: str,
    ) -> list[str]:
        if payload is None:
            return [f"{name}: missing required mapping '{required_name}'"]
        if not isinstance(payload, dict):
            return [f"{name}: '{required_name}' must be a mapping"]
        errors: list[str] = []
        for key, value in payload.items():
            if not isinstance(value, expected_value_type):
                errors.append(
                    f"{name}: '{required_name}.{key}' must be {expected_value_type.__name__}, got {type(value).__name__}"
                )
        return errors

    @staticmethod
    def _validate_node_optimization_shapes(name: str, payload: Any) -> list[str]:
        if payload is None:
            return [f"{name}: missing required mapping 'node_optimizations'"]
        if not isinstance(payload, dict):
            return [f"{name}: 'node_optimizations' must be a mapping"]
        errors: list[str] = []
        for key in ("layer_promotions", "layer_downgrades", "name_rewrites", "attr_overrides", "merge_into_parent"):
            value = payload.get(key)
            if value is None:
                errors.append(f"{name}: node_optimizations missing '{key}'")
            elif not isinstance(value, dict):
                errors.append(f"{name}: node_optimizations.{key} must be a mapping")
        return errors

    @staticmethod
    def _validate_relationship_family_entries(name: str, payload: Dict[str, Any]) -> list[str]:
        family_map = payload.get("relation_family_map") or {}
        relation_rules = payload.get("relation_rules") or {}
        errors: list[str] = []
        family_names = set(family_map.keys())
        for relation, meta in relation_rules.items():
            if not isinstance(meta, dict):
                errors.append(f"{name}: relation_rules.{relation} must be a mapping")
                continue
            family = meta.get("family")
            if family and family not in family_names:
                errors.append(f"{name}: relation_rules.{relation}.family='{family}' not found in relation_family_map")
            severity = meta.get("severity")
            if severity and severity not in {"info", "warn", "error"}:
                errors.append(f"{name}: relation_rules.{relation}.severity='{severity}' is invalid")
        return errors

    def _build_diagnostics(
        self,
        entity_classification_unified: Dict[str, Any],
        entity_aliases: Dict[str, Any],
        l0_classification_rules: Dict[str, Any],
        relationship_rules: Dict[str, Any],
        validation_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a compact configuration health report."""
        issues = validation_report.get("issues") or []
        return {
            "entity_classification": {
                "version": entity_classification_unified.get("version", "unknown"),
                "layer_count": len(entity_classification_unified.get("layers", {}) or {}),
                "type_count": len(entity_classification_unified.get("types", {}) or {}),
                "rule_count": len(entity_classification_unified.get("classification_rules", []) or []),
                "node_optimization_keys": sorted((entity_classification_unified.get("node_optimizations", {}) or {}).keys()),
            },
            "entity_aliases": {
                "alias_groups": len((entity_aliases or {}).get("aliases", []) or []),
            },
            "l0_classification_rules": {
                "valid_l0_classes": len((l0_classification_rules or {}).get("valid_l0_classes", []) or []),
                "prohibited_examples": len((l0_classification_rules or {}).get("l0_prohibited_examples", []) or []),
            },
            "relationship_rules": {
                "relation_families": len((relationship_rules or {}).get("relation_family_map", {}) or {}),
                "relation_rules": len((relationship_rules or {}).get("relation_rules", {}) or {}),
                "layer_pairs": len((relationship_rules or {}).get("layer_pair_rules", {}) or {}),
            },
            "validation": {
                "status": validation_report.get("status", "unknown"),
                "issue_count": len(issues),
                "issues": issues[:20],
            },
        }


# Singleton instance
_unified_config_loader: Optional[UnifiedConfigLoader] = None


def get_unified_config(
    config_dir: Optional[Path] = None,
    force_reload: bool = False,
) -> UnifiedConfig:
    """Get unified configuration singleton.

    P2-2 Enhancement (2026-05-30):
    Centralized access to all application configurations.

    Args:
        config_dir: Configuration directory (default: project_root/config)
        force_reload: Force reload from files

    Returns:
        UnifiedConfig with all configurations loaded

    Usage:
        config = get_unified_config()
        print(config.rag.rigor.max_iterations)
        print(config.report_templates.get("sections"))
    """
    global _unified_config_loader

    if _unified_config_loader is None or config_dir is not None:
        _unified_config_loader = UnifiedConfigLoader(config_dir)

    return _unified_config_loader.load(force_reload)


def reload_unified_config(config_dir: Optional[Path] = None) -> UnifiedConfig:
    """Reload unified configuration from files.

    Args:
        config_dir: Configuration directory (default: project_root/config)

    Returns:
        Reloaded UnifiedConfig
    """
    return get_unified_config(config_dir, force_reload=True)


def get_unified_config_diagnostics(
    config_dir: Optional[Path] = None,
    force_reload: bool = False,
) -> Dict[str, Any]:
    """Return only unified config diagnostics for quick health checks."""
    return get_unified_config(config_dir=config_dir, force_reload=force_reload).diagnostics
