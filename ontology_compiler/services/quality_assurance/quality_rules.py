"""Quality gate rules for ontology extraction validation.

This module contains all quality gate rules from the original quality_gate_rules.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

from core.utils.relationship_helpers import (
    collect_entity_ids,
    relationship_endpoints,
)

from .base import BaseQualityRule, QualityContext

_INSTANCE_CODE_RE = re.compile(
    r"(?i)^(?:pn[-_]?|sku[-_]?|vin[-_]?|sn[-_]?)[a-z0-9._-]{3,}$"
    r"|^(?:[A-Z]{1,5}[-_]?\d{4,}|\d{6,})[A-Z0-9._-]*$"
)
_ABSTRACT_LAYERS = frozenset({"L0", "L1", "l0", "l1"})


class NodeCompletenessRule(BaseQualityRule):
    """Check if nodes have required fields."""
    
    rule_id = "node_completeness"
    description = "Check node field completeness"
    weight = 1.5
    
    REQUIRED_FIELDS = ["meta_id", "meta_name", "meta_class"]
    OPTIONAL_FIELDS = ["description", "domain"]
    
    async def check(self, context: QualityContext) -> float:
        # Include both nodes and instances for completeness check
        nodes = self._get_nodes(context.fragment)
        instances = context.fragment.get('instances', []) or []
        all_entities = list(nodes) + list(instances)
        
        if not all_entities:
            context.add_issue("No nodes found in fragment")
            return 0.0
        
        scores = []
        for entity in all_entities:
            required_score = sum(1 for f in self.REQUIRED_FIELDS if entity.get(f)) / len(self.REQUIRED_FIELDS)
            optional_score = sum(0.5 for f in self.OPTIONAL_FIELDS if entity.get(f)) / len(self.OPTIONAL_FIELDS)
            scores.append(min(required_score + optional_score, 1.0))
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        if avg_score < 0.8:
            context.add_issue(f"Low node completeness: {avg_score:.2f}")
        
        return avg_score


class RelationshipValidityRule(BaseQualityRule):
    """Check if relationships reference valid nodes."""
    
    rule_id = "relationship_validity"
    description = "Check relationship references are valid"
    weight = 1.8
    
    def _get_instances(self, fragment: dict) -> list[dict]:
        """Extract instances (L2 scenarios) from fragment."""
        instances = fragment.get("instances", []) or []
        normalized = []
        for item in instances:
            if isinstance(item, dict):
                normalized.append(item)
            elif hasattr(item, "model_dump"):
                normalized.append(item.model_dump())
            elif hasattr(item, "to_dict"):
                normalized.append(item.to_dict())
        return normalized

    async def check(self, context: QualityContext) -> float:
        nodes = self._get_nodes(context.fragment)
        instances = self._get_instances(context.fragment)
        relationships = self._get_relationships(context.fragment)
        
        if not relationships:
            return 1.0  # No relationships is valid
        
        all_entities = nodes + instances
        
        valid_node_ids = collect_entity_ids(all_entities)
        
        name_to_id_map = {}
        for entity in all_entities:
            entity_id = entity.get('meta_id') or entity.get('id', '')
            entity_name = entity.get('meta_name') or entity.get('name', '')
            if entity_id:
                valid_node_ids.add(str(entity_id))
                if entity_name:
                    name_to_id_map[str(entity_name).lower()] = str(entity_id)
        
        valid_count = 0
        for rel in relationships:
            source, target = relationship_endpoints(rel)
            
            source_resolved = source
            target_resolved = target
            
            if source and source not in valid_node_ids:
                source_resolved = name_to_id_map.get(str(source).lower(), '')
            if target and target not in valid_node_ids:
                target_resolved = name_to_id_map.get(str(target).lower(), '')
            
            if source_resolved in valid_node_ids and target_resolved in valid_node_ids:
                valid_count += 1
        
        score = valid_count / len(relationships) if relationships else 1.0
        
        if score < 0.8:
            context.add_issue(f"Many relationships reference invalid nodes: {score:.2f}")
        
        return score


class SemanticQualityRule(BaseQualityRule):
    """Check semantic quality of descriptions and names."""
    
    rule_id = "semantic_quality"
    description = "Check description quality and naming"
    weight = 1.2
    
    async def check(self, context: QualityContext) -> float:
        # Include both nodes and instances for semantic check
        nodes = self._get_nodes(context.fragment)
        instances = context.fragment.get('instances', []) or []
        all_entities = list(nodes) + list(instances)
        
        if not all_entities:
            return 0.0
        
        scores = []
        for entity in all_entities:
            score = 0.0
            
            # Check description quality
            desc = entity.get('description', '')
            if desc:
                desc_len = len(desc)
                if 30 <= desc_len <= 200:
                    score += 0.5
                elif desc_len > 10:
                    score += 0.25
            
            # Check name quality
            name = entity.get('meta_name', '')
            if name and len(name) >= 3:
                score += 0.3
                # Bonus for descriptive names (not generic)
                generic = ['item', 'object', 'thing', 'entity', 'record']
                if not any(g in name.lower() for g in generic):
                    score += 0.2
            
            scores.append(min(score, 1.0))
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        if avg_score < 0.5:
            context.add_issue(f"Low semantic quality: {avg_score:.2f}")
        
        return avg_score


class DomainAlignmentRule(BaseQualityRule):
    """Check if entities align with specified domain."""
    
    rule_id = "domain_alignment"
    description = "Check domain consistency"
    weight = 0.8
    
    VALID_DOMAINS = {
        "SUPPLY_CHAIN", "LOGISTICS", "MANUFACTURING", "BOM", "SWDL",
        "PRODUCT_ENGINEERING", "QUALITY", "DATA_FLOW", "PROCUREMENT",
        "AUTO", "AUTOMOTIVE", "VEHICLE", "PERFORMANCE"
    }
    
    RELATED_DOMAINS = {
        "SUPPLY_CHAIN": ["LOGISTICS", "PROCUREMENT", "MANUFACTURING", "BOM"],
        "MANUFACTURING": ["BOM", "PRODUCT_ENGINEERING", "QUALITY"],
        "QUALITY": ["MANUFACTURING", "BOM", "SWDL"],
        "SWDL": ["VEHICLE", "AUTOMOTIVE", "SOFTWARE"],
        "BOM": ["MANUFACTURING", "PRODUCT_ENGINEERING"],
        "LOGISTICS": ["SUPPLY_CHAIN", "PROCUREMENT"],
        "PRODUCT_ENGINEERING": ["BOM", "MANUFACTURING", "VEHICLE"],
        "AUTOMOTIVE": ["VEHICLE", "SWDL", "PRODUCT_ENGINEERING"],
        "VEHICLE": ["AUTOMOTIVE", "SWDL", "PRODUCT_ENGINEERING"],
        "DATA_FLOW": ["SUPPLY_CHAIN", "MANUFACTURING"],
        "PROCUREMENT": ["SUPPLY_CHAIN", "LOGISTICS"],
        "AUTO": ["AUTOMOTIVE", "VEHICLE"],
        "PERFORMANCE": ["MANUFACTURING", "QUALITY", "SUPPLY_CHAIN"],
    }
    
    async def check(self, context: QualityContext) -> float:
        nodes = self._get_nodes(context.fragment)
        instances = context.fragment.get('instances', []) or []
        all_entities = list(nodes) + list(instances)
        
        if not all_entities:
            return 0.0
        
        target_domain = context.domain.upper().replace(" ", "_")
        total_score = 0.0
        
        for entity in all_entities:
            entity_domain_raw = entity.get('domain', '')
            entity_domain = entity_domain_raw.upper().replace(" ", "_")
            
            if entity_domain == target_domain:
                total_score += 1.0
            elif entity_domain in self.VALID_DOMAINS:
                if entity_domain in self.RELATED_DOMAINS.get(target_domain, []):
                    total_score += 0.8
                else:
                    total_score += 0.6
            else:
                total_score += 0.3
        
        score = total_score / len(all_entities) if all_entities else 0.0
        
        if score < 0.6:
            context.add_issue(f"Low domain alignment: {score:.2f}")
        
        return score


class CrossLayerConsistencyRule(BaseQualityRule):
    """Check consistency between L1, L2, and L2.5 layers."""
    
    rule_id = "cross_layer_consistency"
    description = "Check cross-layer references"
    weight = 1.0
    
    def _get_entity_layer(self, entity: dict) -> str:
        """Determine layer from entity metadata."""
        layer = entity.get('layer', '')
        if layer:
            return str(layer).upper()
        
        meta_id = entity.get('meta_id', '')
        if meta_id:
            prefix = str(meta_id).upper()[:3]
            if prefix.startswith("L1"):
                return "L1"
            elif prefix == "L25":
                return "L2.5"
            elif prefix.startswith("L2"):
                return "L2"
        
        return "UNKNOWN"
    
    async def check(self, context: QualityContext) -> float:
        nodes = self._get_nodes(context.fragment)
        instances = context.fragment.get('instances', []) or []
        all_entities = list(nodes) + list(instances)
        relationships = self._get_relationships(context.fragment)
        
        if not all_entities or not relationships:
            return 1.0
        
        entity_layers = {}
        for entity in all_entities:
            entity_id = entity.get('meta_id') or entity.get('id', '')
            if entity_id:
                entity_layers[entity_id] = self._get_entity_layer(entity)
        
        cross_layer_count = 0
        valid_relationships = 0
        
        for rel in relationships:
            source = rel.get('from') or rel.get('source', '')
            target = rel.get('to') or rel.get('target', '')
            
            if source and target:
                valid_relationships += 1
                source_layer = entity_layers.get(source, "UNKNOWN")
                target_layer = entity_layers.get(target, "UNKNOWN")
                
                if source_layer != target_layer and source_layer != "UNKNOWN" and target_layer != "UNKNOWN":
                    cross_layer_count += 1
        
        if valid_relationships > 0:
            ratio = cross_layer_count / valid_relationships
            if 0.2 <= ratio <= 0.5:
                return 1.0
            elif ratio > 0.5:
                return 0.9
            elif ratio > 0:
                return 0.6 + ratio * 0.8
            else:
                return 0.5
        else:
            return 1.0


class DuplicateDetectionRule(BaseQualityRule):
    """Check for duplicate entities within fragment."""
    
    rule_id = "duplicate_detection"
    description = "Detect duplicate entities"
    weight = 0.5
    
    def _get_entity_layer(self, entity: dict) -> str:
        """Determine layer from entity metadata."""
        layer = entity.get('layer', '')
        if layer:
            return str(layer).upper()
        
        meta_id = entity.get('meta_id', '')
        if meta_id:
            prefix = str(meta_id).upper()[:3]
            if prefix.startswith("L1"):
                return "L1"
            elif prefix == "L25":
                return "L2.5"
            elif prefix.startswith("L2"):
                return "L2"
            elif prefix.startswith("L3"):
                return "L3"
        
        return "UNKNOWN"
    
    def _smart_tokenize(self, text: str) -> set[str]:
        if not text:
            return set()
        words = text.split()
        tokens = set()
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "being", "have", "has", "had", "do", "does", "did", "will",
                     "would", "could", "should", "may", "might", "must", "shall",
                     "can", "need", "dare", "ought", "used", "to", "of", "in",
                     "for", "on", "with", "at", "by", "from", "as", "into",
                     "through", "during", "before", "after", "above", "below",
                     "between", "under", "and", "but", "or", "yet", "so", "if",
                     "because", "although", "though", "while", "where", "when",
                     "that", "which", "who", "whom", "whose", "what", "this",
                     "these", "those", "i", "you", "he", "she", "it", "we", "they"}
        for word in words:
            cleaned = word.strip(".,;:!?()[]{}\"'").lower()
            if cleaned and len(cleaned) > 2 and cleaned not in stopwords:
                for suffix in ["ing", "ed", "s", "es", "tion", "ment", "ness"]:
                    if cleaned.endswith(suffix) and len(cleaned) > len(suffix) + 3:
                        cleaned = cleaned[:-len(suffix)]
                        break
                tokens.add(cleaned)
        return tokens
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0
        tokens1 = self._smart_tokenize(text1)
        tokens2 = self._smart_tokenize(text2)
        if not tokens1 or not tokens2:
            return 0.0
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        return len(intersection) / len(union) if union else 0.0
    
    async def check(self, context: QualityContext) -> float:
        nodes = self._get_nodes(context.fragment)
        instances = context.fragment.get('instances', []) or []
        all_entities = list(nodes) + list(instances)
        
        if not all_entities:
            return 0.0
        
        seen_ids = set()
        exact_duplicates = 0
        semantic_duplicates = 0
        entity_info = {}
        
        for entity in all_entities:
            entity_id = entity.get('meta_id', '')
            name = entity.get('meta_name', '').lower()
            layer = self._get_entity_layer(entity)
            meta_class = str(entity.get('meta_class', '')).lower()
            description = entity.get('description', '') or entity.get('meta_description', '')
            entity_text = f"{name} {description}".strip()
            
            if entity_id and entity_id in seen_ids:
                exact_duplicates += 1
            
            seen_ids.add(entity_id)
            
            name_key = f"{name}||{layer}||{meta_class}"
            if name_key in entity_info:
                exact_duplicates += 1
            else:
                entity_info[name_key] = {
                    'id': entity_id,
                    'text': entity_text,
                    'layer': layer
                }
        
        for name_key, info in entity_info.items():
            for other_key, other_info in entity_info.items():
                if name_key >= other_key:
                    continue
                if info['layer'] == other_info['layer']:
                    similarity = self._calculate_similarity(info['text'], other_info['text'])
                    if similarity >= 0.90:
                        semantic_duplicates += 1
                        break

        total_duplicates = exact_duplicates + semantic_duplicates

        if total_duplicates > 0:
            # Use sqrt-like capped penalty to avoid over-penalizing large entity sets
            penalty = min(0.3, total_duplicates / max(len(all_entities), 1))
            score = max(0.0, 1.0 - penalty)
            context.add_issue(f"Found {total_duplicates} duplicate entities ({exact_duplicates} exact, {semantic_duplicates} semantic)")
            return score

        return 1.0


class LayerBalanceRule(BaseQualityRule):
    """Check balance between L1, L2, L2.5, and L3 layers."""
    
    rule_id = "layer_balance"
    description = "Check layer balance"
    weight = 0.8

    def _infer_layer_from_meta_id(self, meta_id: str) -> str:
        """Infer layer from meta_id prefix if layer attribute is missing."""
        if not meta_id:
            return "unknown"
        prefix = str(meta_id).upper()[:3]
        if prefix.startswith("L1"):
            return "L1"
        elif prefix.startswith("L2"):
            if prefix == "L25" or prefix == "L2.":
                return "L2.5"
            return "L2"
        elif prefix.startswith("L3"):
            return "L3"
        return "unknown"
    
    async def check(self, context: QualityContext) -> float:
        nodes = self._get_nodes(context.fragment)
        instances = context.fragment.get('instances', []) or []
        
        all_nodes = list(nodes)
        for inst in instances:
            if isinstance(inst, dict):
                all_nodes.append(inst)
            elif hasattr(inst, 'to_dict'):
                all_nodes.append(inst.to_dict())
        
        if not all_nodes:
            return 0.0
        
        layer_counts = {}
        for node in all_nodes:
            layer = node.get('layer', '')
            if not layer or layer.lower() == 'unknown':
                layer = self._infer_layer_from_meta_id(node.get('meta_id', ''))
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        
        total = len(all_nodes)
        if total == 0:
            return 0.0
        
        l1_count = layer_counts.get('L1', 0) + layer_counts.get('l1', 0)
        l2_count = layer_counts.get('L2', 0) + layer_counts.get('l2', 0)
        l25_count = layer_counts.get('L2.5', 0) + layer_counts.get('l2.5', 0)
        l3_count = layer_counts.get('L3', 0) + layer_counts.get('l3', 0)
        
        l1_ratio = l1_count / total
        l2_ratio = l2_count / total
        l25_ratio = l25_count / total
        l3_ratio = l3_count / total
        
        score = 0.0
        
        if 0.10 <= l1_ratio <= 0.80:
            score += 0.5
        elif 0.05 <= l1_ratio <= 0.90:
            score += 0.3
        elif l1_ratio > 0:
            score += 0.1
        
        if 0.15 <= l2_ratio <= 0.80:
            score += 0.5
        elif 0.05 <= l2_ratio <= 0.90:
            score += 0.3
        elif l2_ratio > 0:
            score += 0.1
        
        if l25_ratio >= 0.05:
            score += 0.3
        elif l25_ratio > 0:
            score += 0.15
        
        if l3_ratio >= 0.02:
            score += 0.15
        elif l3_ratio > 0:
            score += 0.08
        
        return min(score, 1.0)


class FieldCompletenessRule(BaseQualityRule):
    """Check completeness of required fields across all entities.

    Follows docs/ENTITY_CLASSIFICATION_SPEC.md §3/§9. Canonical names are
    id/name/layer/type. Extractors still emit meta_id/meta_name/meta_class;
    either name satisfies the same field. Type-specific extras are not in
    the spec and are not scored.
    """
    
    rule_id = "field_completeness"
    description = "Check overall field completeness"
    weight = 1.3
    
    # (spec name, accepted keys). First key is the name reported when missing.
    REQUIRED_FIELD_GROUPS = (
        ("id", ("id", "meta_id")),
        ("name", ("name", "meta_name")),
        ("layer", ("layer",)),
        ("type", ("type", "meta_class")),
    )
    
    OPTIONAL_FIELDS = ["description", "domain", "parent_id"]
    
    async def check(self, context: QualityContext) -> float:
        nodes = self._get_nodes(context.fragment)
        instances = context.fragment.get('instances', []) or []
        all_entities = list(nodes) + list(instances)
        
        if not all_entities:
            return 0.0
        
        total_score = 0.0
        entity_count = 0
        
        for entity in all_entities:
            entity_score = self._score_entity_fields(entity, context)
            total_score += entity_score
            entity_count += 1
        
        avg_score = total_score / entity_count if entity_count > 0 else 0.0
        
        if avg_score < 0.7:
            context.add_issue(f"Low field completeness: {avg_score:.2f}")
        
        return avg_score
    
    def _score_entity_fields(self, entity: dict, context: QualityContext) -> float:
        required_score = self._check_required_fields(entity, context)
        optional_score = self._check_optional_fields(entity)
        weighted_score = (required_score * 0.7) + (optional_score * 0.3)
        return min(weighted_score, 1.0)

    @staticmethod
    def _present(entity: dict, keys: tuple[str, ...]) -> bool:
        for key in keys:
            value = entity.get(key)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return True
        return False
    
    def _check_required_fields(self, entity: dict, context: QualityContext) -> float:
        missing_fields = [
            name for name, keys in self.REQUIRED_FIELD_GROUPS
            if not self._present(entity, keys)
        ]
        
        if missing_fields:
            entity_id = entity.get("id") or entity.get("meta_id") or "unknown"
            entity_name = entity.get("name") or entity.get("meta_name") or ""
            context.add_issue(
                f"Entity '{entity_name or entity_id}' missing required fields: {', '.join(missing_fields)}"
            )
        
        present = len(self.REQUIRED_FIELD_GROUPS) - len(missing_fields)
        return present / len(self.REQUIRED_FIELD_GROUPS)
    
    def _check_optional_fields(self, entity: dict) -> float:
        if not self.OPTIONAL_FIELDS:
            return 1.0
        
        found_count = sum(1 for f in self.OPTIONAL_FIELDS if entity.get(f))
        return found_count / len(self.OPTIONAL_FIELDS)


class LayerInstanceDisciplineRule(BaseQualityRule):
    """Part / serial-like identifiers belong on L3, not L0/L1.

    Aligns with ENTITY_CLASSIFICATION_SPEC_20260627 §9 (concrete parts and
    vehicle instances are L3). Abstract layers keep category meaning only.
    """

    rule_id = "layer_instance_discipline"
    description = "Reject instance-like codes on abstract layers"
    weight = 1.2

    async def check(self, context: QualityContext) -> float:
        nodes = self._get_nodes(context.fragment)
        instances = context.fragment.get("instances", []) or []
        all_entities = list(nodes) + list(instances)
        if not all_entities:
            return 1.0

        offenders = []
        for entity in all_entities:
            layer = str(entity.get("layer") or "")
            if layer not in _ABSTRACT_LAYERS:
                continue
            code = str(entity.get("code") or "").strip()
            name = str(entity.get("name") or entity.get("meta_name") or "").strip()
            if (code and _INSTANCE_CODE_RE.match(code)) or (
                name and _INSTANCE_CODE_RE.match(name)
            ):
                offenders.append(entity.get("id") or entity.get("meta_id") or name)

        if not offenders:
            return 1.0

        context.add_issue(
            f"{len(offenders)} instance-like identifiers on L0/L1 "
            f"(expected L3): {', '.join(str(x) for x in offenders[:8])}"
        )
        # Score decays with offender ratio but stays above 0 for soft gating.
        ratio = len(offenders) / max(1, len(all_entities))
        return max(0.0, 1.0 - min(1.0, ratio * 4.0))


class NamingConsistencyRule(BaseQualityRule):
    """Check naming consistency across entities."""
    
    rule_id = "naming_consistency"
    description = "Check naming conventions"
    weight = 0.8
    
    def _is_valid_name(self, name: str, layer: str = "") -> bool:
        if not name or len(name.strip()) == 0:
            return False
        
        cleaned = name.strip()
        
        if len(cleaned) < 2 or len(cleaned) > 100:
            return False
        
        invalid_chars = set("!@#$%^&*()+=<>?/\\|[]{}~`")
        if any(c in invalid_chars for c in cleaned):
            return False
        
        if cleaned[0].isupper():
            words = [w for w in cleaned.split() if w]
            if words:
                for word in words:
                    if word and not word[0].isupper():
                        return False
            return True
        
        if cleaned.islower():
            return True
        
        return False
    
    async def check(self, context: QualityContext) -> float:
        nodes = self._get_nodes(context.fragment)
        instances = context.fragment.get('instances', []) or []
        all_entities = list(nodes) + list(instances)
        
        if not all_entities:
            return 0.0
        
        consistent_count = 0
        for entity in all_entities:
            name = entity.get('meta_name', '')
            layer = entity.get('layer', '')
            if name and self._is_valid_name(name, layer):
                consistent_count += 1
        
        score = consistent_count / len(all_entities) if all_entities else 0.0
        
        if score < 0.8:
            context.add_issue(f"Naming inconsistency detected: {score:.2f}")
        
        return score


class EvidenceCoverageRule(BaseQualityRule):
    """Check evidence coverage for entities - critical quality indicator."""
    
    rule_id = "evidence_coverage"
    description = "Check if entities have source evidence/location"
    weight = 1.5

    def _has_evidence(self, entity: dict) -> bool:
        """Check if entity has any evidence information."""
        if entity.get('source_file'):
            return True
        
        evidence = entity.get('evidence')
        if evidence:
            if isinstance(evidence, dict):
                if evidence.get('source_file') or evidence.get('line_range') or evidence.get('char_offset'):
                    return True
            else:
                return True
        
        if entity.get('location') or entity.get('source_location'):
            return True
        
        if entity.get('citation') or entity.get('evidence_ref'):
            return True
        
        return False
    
    def _get_evidence_score(self, entity: dict) -> float:
        """Get evidence score with partial credit for partial evidence."""
        score = 0.0
        
        if entity.get('source_file'):
            score += 0.5
        
        evidence = entity.get('evidence')
        if evidence:
            if isinstance(evidence, dict):
                if evidence.get('source_file') or evidence.get('line_range') or evidence.get('char_offset'):
                    score += 0.3
                if evidence.get('text') or evidence.get('quote') or evidence.get('content'):
                    score += 0.2
            else:
                score += 0.5
        
        if entity.get('location') or entity.get('source_location'):
            score += 0.2
        
        if entity.get('citation') or entity.get('evidence_ref'):
            score += 0.2
        
        return min(score, 1.0)
    
    async def check(self, context: QualityContext) -> float:
        nodes = self._get_nodes(context.fragment)
        instances = context.fragment.get('instances', []) or []
        all_entities = list(nodes) + list(instances)
        
        if not all_entities:
            return 0.0
        
        l1_scores = []
        l2_scores = []
        l25_scores = []
        l3_scores = []
        
        for entity in all_entities:
            layer = entity.get('layer', '')
            if not layer:
                meta_id = entity.get('meta_id', '')
                if meta_id and str(meta_id).upper().startswith('L1'):
                    layer = 'L1'
                elif meta_id and str(meta_id).upper().startswith('L25'):
                    layer = 'L2.5'
                elif meta_id and str(meta_id).upper().startswith('L2'):
                    layer = 'L2'
                elif meta_id and str(meta_id).upper().startswith('L3'):
                    layer = 'L3'
            
            evidence_score = self._get_evidence_score(entity)
            
            if layer in ('L1', 'l1'):
                l1_scores.append(evidence_score)
            elif layer in ('L2', 'l2'):
                l2_scores.append(evidence_score)
            elif layer in ('L2.5', 'l2.5', 'L25', 'l25'):
                l25_scores.append(evidence_score)
            elif layer in ('L3', 'l3'):
                l3_scores.append(evidence_score)
        
        weights = {
            'L1': 0.25,
            'L2': 0.30,
            'L2.5': 0.20,
            'L3': 0.25,
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        if l1_scores:
            l1_score = sum(l1_scores) / len(l1_scores)
            total_score += l1_score * weights['L1']
            total_weight += weights['L1']
            if l1_score < 0.4:
                context.add_issue(f"L1 evidence coverage low: {l1_score:.2f} ({len(l1_scores)} entities)")
        
        if l2_scores:
            l2_score = sum(l2_scores) / len(l2_scores)
            total_score += l2_score * weights['L2']
            total_weight += weights['L2']
            if l2_score < 0.35:
                context.add_issue(f"L2 evidence coverage low: {l2_score:.2f} ({len(l2_scores)} entities)")
        
        if l25_scores:
            l25_score = sum(l25_scores) / len(l25_scores)
            total_score += l25_score * weights['L2.5']
            total_weight += weights['L2.5']
            if l25_score < 0.35:
                context.add_issue(f"L2.5 evidence coverage low: {l25_score:.2f} ({len(l25_scores)} entities)")
        
        if l3_scores:
            l3_score = sum(l3_scores) / len(l3_scores)
            total_score += l3_score * weights['L3']
            total_weight += weights['L3']
        
        if total_weight > 0:
            avg_score = total_score / total_weight
        elif all_entities:
            avg_score = 0.4
        else:
            avg_score = 0.0
        
        if avg_score < 0.45:
            context.add_issue(f"Overall evidence coverage low: {avg_score:.2f}")
        
        return avg_score


# =============================================================================
# Legacy Compatibility Functions (from quality_gate_rules.py)
# =============================================================================


def clamp(v: float) -> float:
    """Clamp value to [0.0, 1.0] range."""
    return max(0.0, min(1.0, v))


def stars(v: float) -> str:
    """Convert score [0.0, 1.0] to star rating string."""
    n = round(v * 5)
    return "★" * n + "☆" * (5 - n)


def score_grade(v: float) -> str:
    """Convert score [0.0, 1.0] to letter grade."""
    if v >= 0.85:
        return "A"
    if v >= 0.70:
        return "B"
    if v >= 0.50:
        return "C"
    return "D"


DEFAULT_WEIGHTS: Dict[str, float] = {
    "coverage": 0.20,
    "scale": 0.25,
    "depth": 0.25,
    "diversity": 0.15,
    "l1_score": 0.15,
}

DOMAIN_WEIGHTS: Dict[str, Dict[str, float]] = {
    "MANUFACTURING": {
        "coverage": 0.15,
        "scale": 0.30,
        "depth": 0.25,
        "diversity": 0.15,
        "l1_score": 0.15,
    },
    "SUPPLY_CHAIN": {
        "coverage": 0.25,
        "scale": 0.20,
        "depth": 0.20,
        "diversity": 0.20,
        "l1_score": 0.15,
    },
    "PRODUCT_ENGINEERING": {
        "coverage": 0.20,
        "scale": 0.20,
        "depth": 0.30,
        "diversity": 0.15,
        "l1_score": 0.15,
    },
    "QUALITY": {
        "coverage": 0.30,
        "scale": 0.15,
        "depth": 0.25,
        "diversity": 0.15,
        "l1_score": 0.15,
    },
}


@dataclass(frozen=True)
class QualityDimensions:
    """Quality dimension scores for ontology analysis."""

    coverage: float
    scale: float
    depth: float
    diversity: float
    l1_score: float
    weights: Dict[str, float] = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())

    def __post_init__(self):
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.001:
            normalized = {k: v / total for k, v in self.weights.items()}
            object.__setattr__(self, "weights", normalized)

    @property
    def total_score(self) -> float:
        """Calculate weighted total score."""
        return (
            self.coverage * self.weights["coverage"] +
            self.scale * self.weights["scale"] +
            self.depth * self.weights["depth"] +
            self.diversity * self.weights["diversity"] +
            self.l1_score * self.weights["l1_score"]
        )

    @property
    def grade(self) -> str:
        """Get letter grade for total score."""
        return score_grade(self.total_score)

    @property
    def star_rating(self) -> str:
        """Get star rating string for total score."""
        return stars(self.total_score)

    @property
    def breakdown(self) -> Dict[str, float]:
        """Get score breakdown by dimension."""
        return {
            "coverage": self.coverage * self.weights["coverage"],
            "scale": self.scale * self.weights["scale"],
            "depth": self.depth * self.weights["depth"],
            "diversity": self.diversity * self.weights["diversity"],
            "l1_score": self.l1_score * self.weights["l1_score"],
        }


def get_domain_weights(domain: str) -> Dict[str, float]:
    """Get quality dimension weights for a specific domain."""
    return DOMAIN_WEIGHTS.get(domain.upper(), DEFAULT_WEIGHTS).copy()


def calculate_dimensions(
    evidence_coverage: float,
    node_count: int,
    rel_count: int,
    rigor_iters: int,
    src_ratio: float,
    l1_score: float,
    domain: str = "",
    custom_weights: Optional[Dict[str, float]] = None,
) -> QualityDimensions:
    """Calculate all quality dimension scores with configurable weights."""
    weights = custom_weights or get_domain_weights(domain)
    
    return QualityDimensions(
        coverage=clamp(evidence_coverage),
        scale=clamp(node_count / 20.0) * 0.5 + clamp(rel_count / 30.0) * 0.5,
        depth=clamp(rigor_iters / 3.0),
        diversity=clamp(src_ratio),
        l1_score=clamp(float(l1_score or 0.0)),
        weights=weights,
    )


def calculate_coverage_score(
    nodes: list[dict],
    edges: list[dict],
    source_weight: float = 0.6,
    desc_weight: float = 0.2,
    edge_weight: float = 0.2,
) -> float:
    """Calculate evidence coverage score for RIGOR skip decision."""
    if not nodes:
        return 0.0

    nodes_with_source = sum(1 for n in nodes if n.get("source_file"))
    source_coverage = nodes_with_source / len(nodes)

    nodes_with_desc = sum(1 for n in nodes if n.get("description"))
    desc_coverage = nodes_with_desc / len(nodes)

    edges_with_source = sum(1 for e in edges if e.get("source_file"))
    edge_coverage = edges_with_source / len(edges) if edges else 1.0

    total = (
        source_coverage * source_weight +
        desc_coverage * desc_weight +
        edge_coverage * edge_weight
    )
    return min(total, 1.0)


def build_briefing_lines(
    node_count: int,
    rel_count: int,
    rigor_iters: int,
    src_ratio: float,
    q_warnings: list[str],
) -> list[str]:
    """Build quality briefing sentences."""
    lines = []

    if node_count >= 20:
        lines.append(
            f"本次从 {node_count} 个领域概念节点（含 {rel_count} 条关系）中成功提取制造本体，"
            f"覆盖 BOM/BOP/BOE 等核心文档类型。"
        )
    elif node_count >= 5:
        lines.append(
            f"提取到 {node_count} 个领域概念节点和 {rel_count} 条关系，语义层次初步建立。"
        )
    else:
        lines.append(
            f"提取节点数量偏少（{node_count}），建议检查文件内容或扩大样本。"
        )

    if rigor_iters >= 2:
        lines.append(f"RIGOR LLM 精化执行了 {rigor_iters} 次迭代，概念质量得到有效提升。")
    elif rigor_iters == 1:
        lines.append("RIGOR LLM 精化执行了 1 次迭代，可进一步提高迭代上限以改善质量。")
    else:
        lines.append("RIGOR LLM 迭代未执行，建议检查 LLM 连通性或降低 rigor 阈值。")

    if src_ratio >= 0.5:
        lines.append(f"{int(src_ratio*100)}% 的节点有来源文件追溯，证据链完整。")
    else:
        lines.append(f"仅 {int(src_ratio*100)}% 的节点含来源证据，建议增强文档关联。")

    if len(q_warnings) == 0:
        lines.append("无警告，管道运行稳定。")
    else:
        lines.append(f"存在 {len(q_warnings)} 项警告，请关注详情。")

    return lines


async def calculate_unified_quality_score(
    l1_entities: List[Dict],
    l2_entities: List[Dict],
    l25_entities: List[Dict],
    relationships: List[Dict],
    domain: str = "SUPPLY_CHAIN"
) -> float:
    """Calculate unified quality score using QualityGatePipeline."""
    from .pipelines import QualityGatePipeline
    
    fragment = {
        "layer1_concepts": l1_entities,
        "layer2_scenarios": l2_entities,
        "layer25_rules": l25_entities,
        "relationships": relationships,
    }
    
    pipeline = QualityGatePipeline()
    result = await pipeline.execute(fragment, domain)
    
    return result.confidence * 100