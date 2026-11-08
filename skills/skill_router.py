"""Skill Router - Routes entities to applicable skills based on type and domain.

P5 Optimization (2026-06-09): Implements automatic skill routing based on
entity characteristics (meta_class, domain, category).

2026-06-27: business-domain structure (mfg/ontology/pipeline/rag/system/).
2026-11-08: discovery via ``ontology_compiler.harness`` only — no
``skills.skill_registry`` dependency.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Protocol

from ontology_compiler.harness import build_default_skill_registry
from ontology_compiler.harness.skill_registry import SkillDefinition, SkillRegistry

from .skill_interface import BaseSkill

logger = logging.getLogger(__name__)


class _SkillLookup(Protocol):
    def get(self, name: str) -> Any: ...


class SkillsRouter:
    """
    Routes entities to applicable skills based on their characteristics.

    Supports routing by:
    - meta_class (System, Capability, Process, Data, Actor, Asset)
    - domain (MANUFACTURING, SUPPLY_CHAIN, LOGISTICS, etc.)
    - category (BOM_MANAGEMENT, DOCUMENT_MGMT, PROCESS_PLANNING, etc.)
    - business_domain (mfg, ontology, pipeline, rag, system)
    """

    BUSINESS_DOMAINS: Dict[str, Dict[str, Any]] = {
        "mfg": {
            "name": "Manufacturing",
            "description": "制造域 - BOM分析、库存管理、场景分析",
            "skills": ["bom_timeseries_analysis", "build_inventory_field_catalog", "scene_analysis"],
        },
        "ontology": {
            "name": "Ontology Compilation",
            "description": "本体编译域 - 验证、合并、漂移检测",
            "skills": [
                "consensus_resolver",
                "drift_detector",
                "generate_ontology_patch",
                "ontology_merger",
                "ontology_validator",
            ],
        },
        "pipeline": {
            "name": "Pipeline Orchestration",
            "description": "管道编排域 - 阶段运行器",
            "skills": ["stage_runner"],
        },
        "rag": {
            "name": "RAG Retrieval",
            "description": "RAG检索域 - 复杂编排、多Agent分析",
            "skills": [
                "complex_rag_orchestrator",
                "multi_agent_analysis",
                "run_ontology_rag_eval",
                "run_ontology_rag_validate",
            ],
        },
        "system": {
            "name": "System Tools",
            "description": "系统工具域 - 配置检查、清理、发布",
            "skills": [
                "capsule_publisher",
                "check_config",
                "cleanup_comp_runs",
                "switch_llm_provider",
                "validate_output_contracts",
            ],
        },
    }

    _ROUTE_RULES: Dict[str, Dict[str, List[str]]] = {
        "meta_class": {
            "System": ["ontology_validator", "drift_detector"],
            "Capability": ["consensus_resolver", "scene_analysis"],
            "Process": ["scene_analysis", "ontology_validator"],
            "Data": ["ontology_merger", "drift_detector"],
            "Actor": ["consensus_resolver"],
            "Asset": ["ontology_validator"],
            "BusinessScenario": ["scene_analysis", "consensus_resolver"],
        },
        "domain": {
            "MANUFACTURING": ["complex_rag_orchestrator", "multi_agent_analysis", "bom_timeseries_analysis"],
            "SUPPLY_CHAIN": ["complex_rag_orchestrator", "consensus_resolver"],
            "LOGISTICS": ["drift_detector", "scene_analysis"],
            "PRODUCT_ENGINEERING": ["ontology_merger", "ontology_validator"],
            "ENTERPRISE": ["validate_output_contracts", "check_config"],
            "QUALITY": ["ontology_validator", "run_ontology_rag_validate"],
            "CHANGE_MANAGEMENT": ["consensus_resolver", "generate_ontology_patch"],
            "COMPLIANCE": ["validate_output_contracts"],
            "BOM_ANALYSIS": ["bom_timeseries_analysis"],
            "ONTOLOGY_COMPILATION": ["ontology_validator", "ontology_merger"],
            "PIPELINE": ["stage_runner"],
        },
        "category": {
            "BOM_MANAGEMENT": ["bom_timeseries_analysis", "multi_agent_analysis"],
            "DOCUMENT_MGMT": ["drift_detector", "scene_analysis"],
            "PROCESS_PLANNING": ["scene_analysis", "ontology_validator"],
            "WARRANTY": ["ontology_merger", "consensus_resolver"],
            "PART_MANAGEMENT": ["ontology_validator", "drift_detector"],
            "SYSTEM_INTEGRATION": ["complex_rag_orchestrator"],
            "CHANGE_MGMT": ["consensus_resolver", "generate_ontology_patch"],
            "PIPELINE_STAGE": ["stage_runner"],
            "VALIDATION": ["validate_output_contracts"],
            "MAINTENANCE": ["check_config", "cleanup_comp_runs"],
        },
    }

    def __init__(
        self,
        skill_registry: Optional[Any] = None,
        *,
        harness_registry: Optional[SkillRegistry] = None,
    ):
        """Initialize skills router.

        ``harness_registry`` is preferred. ``skill_registry`` is accepted for
        call-site compatibility (SkillPipelineBridge) but ignored for discovery;
        harness is always the source of truth for skill existence.
        """
        self._harness: SkillRegistry = harness_registry or build_default_skill_registry()
        # Optional runtime instance lookup (ontology_compiler.skills.registry).
        self._runtime: Optional[_SkillLookup] = None
        if skill_registry is not None and not isinstance(skill_registry, SkillRegistry):
            self._runtime = skill_registry
        self.logger = logging.getLogger("SkillsRouter")

    @property
    def skill_registry(self) -> SkillRegistry:
        """Canonical harness registry (discovery)."""
        return self._harness

    def route(self, entity: Dict[str, Any]) -> List[str]:
        skills: List[str] = []

        meta_class = entity.get("meta_class")
        domain = entity.get("domain") or entity.get("sandbox_domain")
        category = entity.get("category")

        if meta_class:
            skills.extend(self._ROUTE_RULES["meta_class"].get(meta_class, []))

        if domain:
            skills.extend(self._ROUTE_RULES["domain"].get(str(domain).upper(), []))

        if category:
            skills.extend(self._ROUTE_RULES["category"].get(str(category).upper(), []))

        unique_skills = list(dict.fromkeys(skills))

        if unique_skills:
            self.logger.debug(
                "Routed entity %s to skills: %s",
                entity.get("meta_name"),
                unique_skills,
            )

        return unique_skills

    def route_batch(self, entities: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        results: Dict[str, List[str]] = {}
        for entity in entities:
            meta_id = entity.get("meta_id") or entity.get("scenario_id") or str(id(entity))
            results[str(meta_id)] = self.route(entity)
        return results

    def _resolve_definition(self, skill_name: str) -> Optional[SkillDefinition]:
        return self._harness.get(skill_name)

    def _resolve_runtime_instance(self, skill_name: str) -> Optional[BaseSkill]:
        if self._runtime is None:
            return None
        getter = getattr(self._runtime, "get_instance", None) or getattr(self._runtime, "get", None)
        if getter is None:
            return None
        try:
            obj = getter(skill_name)
        except Exception:
            return None
        if isinstance(obj, BaseSkill):
            return obj
        return None

    async def apply_skills(
        self,
        entity: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        skill_names = self.route(entity)
        results: Dict[str, Any] = {}

        for skill_name in skill_names:
            definition = self._resolve_definition(skill_name)
            if definition is None:
                self.logger.warning("Skill %s not found in harness registry", skill_name)
                continue
            try:
                instance = self._resolve_runtime_instance(skill_name)
                if instance is not None:
                    results[skill_name] = await self._execute_skill(instance, context)
                else:
                    results[skill_name] = {
                        "discovered": True,
                        "executed": False,
                        "name": definition.name,
                        "description": definition.description,
                        "script_path": definition.script_path,
                        "note": "harness definition only; no runtime BaseSkill instance",
                    }
                self.logger.debug("Applied skill %s to %s", skill_name, entity.get("meta_name"))
            except Exception as e:
                self.logger.error("Error applying skill %s: %s", skill_name, e, exc_info=True)
                results[skill_name] = {"error": str(e)}

        return results

    async def _execute_skill(self, skill: BaseSkill, context: Dict[str, Any]) -> Any:
        import asyncio

        if asyncio.iscoroutinefunction(skill.execute):
            return await skill.execute(context)
        return skill.execute(context)

    def add_route_rule(
        self,
        rule_type: str,
        key: str,
        skill_names: List[str],
    ) -> None:
        if rule_type not in self._ROUTE_RULES:
            self._ROUTE_RULES[rule_type] = {}

        self._ROUTE_RULES[rule_type][key] = skill_names
        self.logger.info("Added route rule: %s.%s -> %s", rule_type, key, skill_names)

    def get_all_rules(self) -> Dict[str, Dict[str, List[str]]]:
        return self._ROUTE_RULES

    def suggest_skills_for_domain(self, domain: str) -> List[str]:
        return self._ROUTE_RULES["domain"].get(domain.upper(), [])

    def route_by_business_domain(self, business_domain: str) -> List[str]:
        domain_info = self.BUSINESS_DOMAINS.get(business_domain.lower())
        if domain_info:
            return list(domain_info.get("skills", []))
        return []

    def get_business_domain_info(self, business_domain: str) -> Optional[Dict[str, Any]]:
        return self.BUSINESS_DOMAINS.get(business_domain.lower())

    def get_all_business_domains(self) -> Dict[str, Dict[str, Any]]:
        return self.BUSINESS_DOMAINS

    def list_skills_by_business_domain(self) -> Dict[str, List[str]]:
        return {domain: list(info["skills"]) for domain, info in self.BUSINESS_DOMAINS.items()}
