"""Build Layer2/Layer3 grouped ontology outputs.

R11 Migration (2026-05-29): Dedup logic migrated to EntityDeduplicationService.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List
import yaml

# R11: Migrate deduplication to unified service
from ontology_compiler.services.merge.entity_dedup_service import EntityDeduplicationService


class DomainRegistry:
    """Load and manage domain configuration from YAML file."""

    def __init__(self, config_path: str | Path | None = None):
        """Initialize domain registry from configuration file.
        
        Args:
            config_path: Path to domain_registry.yaml. If None, uses default path.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "domain_registry.yaml"
        else:
            config_path = Path(config_path)

        self.config_path = config_path
        self._config = self._load_config()
        self.domains = self._config.get("domains", {})
        self.default_domain = self._config.get("default_domain", "SUPPLY_CHAIN")

    def _load_config(self) -> Dict:
        """Load domain configuration from YAML file."""
        if not self.config_path.exists():
            # Fallback to hardcoded domains if config file doesn't exist
            return self._get_fallback_config()
        
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            # Fallback to hardcoded domains if loading fails
            return self._get_fallback_config()

    def _get_fallback_config(self) -> Dict:
        """Fallback configuration for backward compatibility."""
        return {
            "domains": {
                "SUPPLY_CHAIN": {
                    "description": "供应链与库存管理域",
                    "rules": [
                        {
                            "rule_id": "RULE_CS_001",
                            "rule_name": "Cycle Stock计算规则",
                            "formula": "cycle_stock = avg_order_qty_60days /2",
                            "limiting_factors": [
                                "ShippingFrequency",
                                "ShippingQty",
                                "PalletizedUnitLoad",
                                "UnitLoad",
                            ],
                        },
                    ],
                },
                "LOGISTICS": {
                    "description": "工厂内物流与拉动系统域",
                    "rules": [
                        {
                            "rule_id": "RULE_LOG_001",
                            "rule_name": "Line Back 拉动规则",
                            "formula": "line_side_stock = takt_time * lane_factor",
                            "limiting_factors": ["TaktTime", "LaneCapacity"],
                        },
                    ],
                },
                "PERFORMANCE": {
                    "description": "绩效管理与指标治理域",
                    "rules": [
                        {
                            "rule_id": "RULE_PERF_001",
                            "rule_name": "Performance KPI跟踪",
                            "formula": "performance_index = weighted(kpi_set)",
                            "limiting_factors": ["FTT", "DPV", "Output"],
                        },
                    ],
                },
            },
            "default_domain": "SUPPLY_CHAIN",
        }

    def get_domain_config(self, domain: str) -> Dict:
        """Get configuration for a specific domain."""
        return self.domains.get(domain, {"description": domain, "rules": []})

    def get_domain_description(self, domain: str) -> str:
        """Get description for a specific domain."""
        config = self.get_domain_config(domain)
        return config.get("description", domain)

    def get_domain_rules(self, domain: str) -> List[Dict]:
        """Get rules for a specific domain."""
        config = self.get_domain_config(domain)
        return config.get("rules", [])

    def get_all_domains(self) -> List[str]:
        """Get list of all registered domains."""
        return list(self.domains.keys())


class LayeredOntologyBuilder:
    """Aggregate extracted nodes/instances into layered domain outputs."""

    def __init__(self, domain_registry: DomainRegistry | None = None):
        """Initialize builder with optional custom domain registry.
        
        Args:
            domain_registry: Custom domain registry. If None, creates default.
        """
        self.domain_registry = domain_registry or DomainRegistry()

    def build(self, fragments: List[Dict]) -> Dict:
        domains = {}
        for domain_name in self.domain_registry.get_all_domains():
            domains[domain_name] = self._new_domain_bucket(domain_name)

        for fragment in fragments:
            self._merge_fragment(domains, fragment)

        self._dedupe_domains(domains)

        return {
            "schema_version": "v1.0",
            "description": "Layer2运行实例层",
            "generated_at": datetime.now().isoformat(),
            "domains": domains,
        }

    def _merge_fragment(self, domains: Dict[str, Dict], fragment: Dict) -> None:
        fragment_domain = fragment.get("domain", self.domain_registry.default_domain)
        # R18 Phase 6: 兼容 nodes/classes 和 edges/relationships 两种字段名
        nodes = fragment.get("nodes", fragment.get("classes", []))
        instances = fragment.get("instances", [])
        edges = fragment.get("relationships", fragment.get("edges", []))

        target_domain = fragment_domain if fragment_domain in domains else self.domain_registry.default_domain
        if target_domain not in domains:
            # Dynamically add unknown domain
            domains[target_domain] = self._new_domain_bucket(target_domain)
        
        domain_bucket = domains[target_domain]

        domain_bucket["nodes"].extend(nodes)
        domain_bucket["instances"].extend(instances)
        domain_bucket["relationships"].extend(edges)

    def _new_domain_bucket(self, domain: str) -> Dict:
        return {
            "sandbox_description": self.domain_registry.get_domain_description(domain),
            "sandbox_rules": self.domain_registry.get_domain_rules(domain),
            "nodes": [],
            "instances": [],
            "relationships": [],
        }

    def _dedupe_domains(self, domains: Dict[str, Dict]) -> None:
        """R11: Use EntityDeduplicationService for consistent deduplication."""
        dedup_service = EntityDeduplicationService(strategy="first")
        for domain_bucket in domains.values():
            domain_bucket["nodes"] = dedup_service.deduplicate_nodes(
                domain_bucket["nodes"], by="meta_id"
            )
            domain_bucket["instances"] = dedup_service.deduplicate_nodes(
                domain_bucket["instances"], by="meta_id"
            )
            domain_bucket["relationships"] = dedup_service.deduplicate_relationships(
                domain_bucket["relationships"],
                source_key="source",
                predicate_key="predicate",
                target_key="target"
            )
