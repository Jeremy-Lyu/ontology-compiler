"""关系治理引擎 — 从 config/ontology/relationship_rules.yaml 加载规则。

替代 scripts/run_pipeline_v2.py 中的 if/elif 脚本，提供：

- 关系类型 → 族(family) 分类（STRUCTURAL / REALIZATION / GOVERNANCE / ...）
- 层间对(layer pair) 是否允许、需要哪些证据
- 必填字段校验 + 风险等级
- 统一入口，供 run_audit() / run_guard() / merge 使用

单一真值来源: config/ontology/relationship_rules.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "ontology" / "relationship_rules.yaml"


class RelationGovernanceEngine:
    def __init__(self, config_path: str | Path | None = None):
        path = Path(config_path) if config_path else _DEFAULT_CONFIG
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        self._family_map = raw.get("relation_family_map", {}) or {}
        self._per_relation = raw.get("relation_rules", {}) or {}
        self._layer_pairs = raw.get("layer_pair_rules", {}) or {}
        self._evidence_tiers = raw.get("evidence_requirements", {}) or {}
        self._low_confidence_threshold = float(raw.get("low_confidence_threshold", 0.5) or 0.5)
        self._domain_whitelist: dict[str, list[str]] = raw.get("domain_whitelist", {}) or {}
        self._rewrite_priorities: list[str] = raw.get("rewrite_priorities", [
            "STRUCTURAL", "REALIZATION", "GOVERNANCE", "DEPENDENCY_TRACE", "CONFIG_EFFECTIVITY", "CHANGE_FLOW", "GENERAL"
        ]) or []

        self._rel_to_family: dict[str, str] = {}
        self._family_to_canonical_relation: dict[str, str] = {}
        for family, rels in self._family_map.items():
            rel_list = [str(rel).upper() for rel in (rels or []) if str(rel).strip()]
            for rel in rel_list:
                self._rel_to_family[rel] = family
            if rel_list:
                self._family_to_canonical_relation[family] = rel_list[0]

    # ---------- family ----------

    def get_family(self, relation_type: str) -> str:
        rel = str(relation_type or "").strip().upper()
        if not rel:
            return "GENERAL"
        if rel in self._rel_to_family:
            return self._rel_to_family[rel]
        meta = self._per_relation.get(rel)
        if meta and meta.get("family"):
            return meta["family"]
        return "GENERAL"

    def family_of(self, relation: dict[str, Any]) -> str:
        return self.get_family(self._read_relation_type(relation))

    # ---------- layer-pair ----------

    def layer_pair_rule(self, source_layer: str, target_layer: str) -> dict[str, Any]:
        key = f"{source_layer}-{target_layer}"
        if key in self._layer_pairs:
            return self._layer_pairs[key]
        return {
            "allowed": True,
            "allowed_families": [],
            "requires_evidence": False,
            "notes": f"Layer pair {key} not in rules — default allowed",
        }

    def is_layer_pair_allowed(self, source_layer: str, target_layer: str) -> bool:
        return bool(self.layer_pair_rule(source_layer, target_layer).get("allowed", True))

    def layer_pair_requires_evidence(self, source_layer: str, target_layer: str) -> bool:
        return bool(self.layer_pair_rule(source_layer, target_layer).get("requires_evidence", False))

    def family_allowed_for_layer_pair(
        self, family: str, source_layer: str, target_layer: str
    ) -> bool:
        rule = self.layer_pair_rule(source_layer, target_layer)
        allowed = rule.get("allowed_families") or []
        if not allowed:
            return True
        return family in allowed

    # ---------- per-relation meta ----------

    def relation_meta(self, relation_type: str) -> Optional[dict[str, Any]]:
        return self._per_relation.get(str(relation_type or "").upper())

    def relation_severity(self, relation_type: str) -> str:
        meta = self.relation_meta(relation_type) or {}
        return meta.get("severity", "info")

    def is_relation_allowed_for_domain(self, domain: str, relation_type: str) -> bool:
        domain_key = str(domain or "").upper()
        allowed = self._domain_whitelist.get(domain_key)
        if not allowed:
            return True
        rel = str(relation_type or "").upper()
        return rel in {r.upper() for r in allowed}

    def low_confidence_threshold(self) -> float:
        return self._low_confidence_threshold

    def rewrite_priority(self, family: str) -> int:
        try:
            return self._rewrite_priorities.index(family)
        except ValueError:
            return len(self._rewrite_priorities)

    def rewrite_rank(self, family: str) -> int:
        return self.rewrite_priority(family)

    def rewrite_relation_type(self, relation: dict[str, Any]) -> tuple[str, str, str | None, int | None]:
        rel_type = self._read_relation_type(relation)
        family = self.get_family(rel_type)
        canonical, rank = self._choose_canonical_relation([family])
        if canonical and canonical != rel_type.upper():
            return canonical, family, canonical, rank
        return rel_type.upper(), family, None, rank

    def normalize(self, relation: dict[str, Any]) -> dict[str, Any]:
        rewritten_type, family, rewrite_target, rank = self.rewrite_relation_type(relation)
        normalized = dict(relation or {})
        if rewrite_target:
            normalized["relation_type"] = rewritten_type
            normalized.setdefault("rewritten_relation_type", rewritten_type)
            normalized.setdefault("rewrite_reason", f"rewritten to canonical family relation {rewrite_target}")
            normalized.setdefault("rewrite_rank", rank)
            normalized.setdefault("decision_trace", []).append({
                "action": "rewrite_relationship_type",
                "reason": f"canonical family relation selected for {family}",
                "before": self._read_relation_type(relation),
                "after": rewritten_type,
                "rewrite_rank": rank,
                "rule_source": "relation_governance.normalize",
            })
        normalized["relation_family"] = family
        if rank is not None:
            normalized.setdefault("rewrite_rank", rank)
        return normalized


    def validate(
        self,
        relation: dict[str, Any],
        source_layer: Optional[str] = None,
        target_layer: Optional[str] = None,
    ) -> dict[str, Any]:
        rel_type = self._read_relation_type(relation)
        family = self.get_family(rel_type)
        issues: list[dict[str, str]] = []
        result_class = "valid"
        rewrite_target: Optional[str] = None
        rewrite_rank: Optional[int] = None
        fallback_reason: Optional[str] = None

        meta = self.relation_meta(rel_type)
        required_fields = meta.get("required_fields", []) if meta else []
        for field in required_fields:
            if not str(relation.get(field) or "").strip():
                severity = meta.get("severity", "info") if meta else "info"
                issues.append({
                    "type": "MISSING_REQUIRED_FIELD",
                    "reason": f"Relation {rel_type}: missing required field '{field}'",
                    "severity": severity,
                })
                if severity == "error":
                    result_class = "fallback"
                    fallback_reason = fallback_reason or f"missing required field '{field}'"
                elif result_class == "valid":
                    result_class = "rewrite"
                    rewrite_target = self._choose_canonical_relation([family])
                    rewrite_rank = self.rewrite_priority(family)

        if not rel_type:
            issues.append({"type": "MISSING_RELATION_TYPE", "reason": "Relation has no relation_type", "severity": "error"})
            result_class = "fallback"
            fallback_reason = fallback_reason or "missing relation_type"

        if source_layer and target_layer:
            pair_rule = self.layer_pair_rule(source_layer, target_layer)
            if pair_rule.get("allowed") is False:
                issues.append({
                    "type": "LAYER_PAIR_NOT_ALLOWED",
                    "reason": f"Layer pair {source_layer}->{target_layer} is not allowed (family={family})",
                    "severity": "error",
                })
                result_class = "fallback"
                fallback_reason = fallback_reason or f"layer pair {source_layer}->{target_layer} not allowed"
            allowed_families = pair_rule.get("allowed_families") or []
            if allowed_families and family not in allowed_families:
                issues.append({
                    "type": "RELATION_FAMILY_NOT_ALLOWED",
                    "reason": (
                        f"Family '{family}' is not allowed for {source_layer}->{target_layer}; "
                        f"allowed: {allowed_families}"
                    ),
                    "severity": "warn",
                })
                if result_class == "valid":
                    result_class = "rewrite"
                canonical, rank = self._choose_canonical_relation(allowed_families)
                rewrite_target = rewrite_target or canonical or self._suggest_family_rewrite(family, allowed_families)
                rewrite_rank = rewrite_rank if rewrite_rank is not None else rank
            if pair_rule.get("requires_evidence") and not self._has_evidence(relation):
                issues.append({
                    "type": "LAYER_PAIR_EVIDENCE_REQUIRED",
                    "reason": (
                        f"Layer pair {source_layer}->{target_layer} requires evidence "
                        f"(evidence_ref/citation/source_file)"
                    ),
                    "severity": "warn",
                })
                if result_class == "valid":
                    result_class = "fallback"
                fallback_reason = fallback_reason or f"evidence required for {source_layer}->{target_layer}"

        if meta and meta.get("severity") == "warn" and result_class == "valid":
            result_class = "rewrite"
            canonical, rank = self._choose_canonical_relation([family])
            rewrite_target = rewrite_target or canonical or self._suggest_family_rewrite(family, self._rewrite_priorities)
            rewrite_rank = rewrite_rank if rewrite_rank is not None else rank

        return {
            "valid": not issues,
            "issues": issues,
            "family": family,
            "relation_type": rel_type,
            "severity": self.relation_severity(rel_type),
            "result_class": result_class,
            "rewrite_target": rewrite_target,
            "rewrite_rank": rewrite_rank,
            "fallback_reason": fallback_reason,
        }

    # ---------- helpers ----------

    @staticmethod
    def _read_relation_type(rel: dict[str, Any]) -> str:
        return str(
            rel.get("relation_type")
            or rel.get("relationship_type")
            or rel.get("type")
            or ""
        ).strip()

    @staticmethod
    def _has_evidence(rel: dict[str, Any]) -> bool:
        return any(
            str(rel.get(f) or "").strip()
            for f in ("evidence_ref", "citation", "source_file", "rule_no")
        )

    def _choose_canonical_relation(self, families: list[str]) -> tuple[str | None, int | None]:
        ranked = sorted(
            [str(f).upper() for f in families if str(f).strip()],
            key=self.rewrite_priority,
        )
        for family in ranked:
            canonical = self._family_to_canonical_relation.get(family)
            if canonical:
                return canonical, self.rewrite_rank(family)
        return None, None

    @staticmethod
    def _suggest_family_rewrite(family: str, candidates: list[str]) -> str | None:
        if not candidates:
            return None
        family_upper = str(family or "").upper()
        if family_upper in candidates:
            return family_upper
        return str(candidates[0]).upper()

    def get_validation_summary(self) -> dict[str, Any]:
        return {
            "relation_families": sorted(self._family_map.keys()),
            "layer_pairs": sorted(self._layer_pairs.keys()),
            "domain_whitelist": {k: len(v or []) for k, v in self._domain_whitelist.items()},
            "low_confidence_threshold": self._low_confidence_threshold,
        }


_ENGINE_SINGLETON: Optional[RelationGovernanceEngine] = None


def get_relation_governance_engine(config_path: str | Path | None = None) -> RelationGovernanceEngine:
    global _ENGINE_SINGLETON
    if _ENGINE_SINGLETON is None or config_path is not None:
        _ENGINE_SINGLETON = RelationGovernanceEngine(config_path)
    return _ENGINE_SINGLETON
