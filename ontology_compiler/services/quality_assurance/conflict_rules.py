"""Conflict detection rules for ontology compilation.

This module contains all conflict detection rules from the original conflict_engine.py.
"""

from __future__ import annotations

from collections import defaultdict

from .base import BaseConflictRule, Conflict, DetectionContext
from .relation_type_registry import RelationTypeRegistry
from core.utils.relationship_helpers import entity_to_dict, relationship_endpoints

_relation_registry = RelationTypeRegistry()


# =============================================================================
# Fragment-level rules (from IntegratorJudge)
# =============================================================================


class DuplicateNodesRule(BaseConflictRule):
    """Detect duplicate node IDs within a fragment."""

    name = "duplicate_nodes"
    severity = "blocking"
    description = "Detect duplicate node IDs within a fragment"

    def detect(self, context: DetectionContext) -> list[Conflict]:
        if not context.fragment:
            return []

        conflicts: list[Conflict] = []
        nodes = context.fragment.get("nodes", [])

        node_ids = [n.get("node_id") for n in nodes if isinstance(n, dict)]
        seen: set[str] = set()
        duplicates: set[str] = set()

        for nid in node_ids:
            if nid in seen:
                duplicates.add(nid)
            seen.add(nid)

        for dup in duplicates:
            variants = [n for n in nodes if isinstance(n, dict) and n.get("node_id") == dup]
            conflicts.append(
                Conflict(
                    severity=self.severity,
                    conflict_type=self.name,
                    entity_id=str(dup),
                    message=f"Duplicate node ID: {dup}",
                    variants=variants,
                )
            )

        return conflicts


class CircularRelationshipsRule(BaseConflictRule):
    """Detect circular relationships in a fragment."""

    name = "circular_relationships"
    severity = "blocking"
    description = "Detect circular relationship chains"

    def detect(self, context: DetectionContext) -> list[Conflict]:
        if not context.fragment:
            return []

        conflicts: list[Conflict] = []
        relationships = context.fragment.get("relationships", [])

        # Build adjacency list
        graph: dict[str, set[str]] = defaultdict(set)
        rel_map: dict[tuple[str, str], list[dict]] = defaultdict(list)

        for rel in relationships:
            if not isinstance(rel, dict):
                rel = entity_to_dict(rel)
            src, tgt = relationship_endpoints(rel)
            if src and tgt:
                graph[src].add(tgt)
                rel_map[(src, tgt)].append(rel)

        # Find cycles using DFS
        def find_cycles(start: str) -> list[list[tuple[str, str]]]:
            cycles: list[list[tuple[str, str]]] = []
            visited: set[str] = set()
            path: list[str] = []

            def dfs(node: str) -> None:
                if node in path:
                    # Found cycle
                    cycle_start = path.index(node)
                    cycle_path = path[cycle_start:] + [node]
                    edges: list[tuple[str, str]] = []
                    for i in range(len(cycle_path) - 1):
                        edges.append((cycle_path[i], cycle_path[i + 1]))
                    cycles.append(edges)
                    return

                if node in visited:
                    return

                visited.add(node)
                path.append(node)

                for neighbor in graph.get(node, []):
                    dfs(neighbor)

                path.pop()

            for node in list(graph.keys()):
                if node not in visited:
                    dfs(node)

            return cycles

        # Check all nodes for cycles
        all_cycles: list[list[tuple[str, str]]] = []
        checked: set[str] = set()
        for node in graph:
            if node not in checked:
                cycles = find_cycles(node)
                all_cycles.extend(cycles)
                checked.update(node)

        # Create conflict for each unique cycle
        seen_cycles: set[str] = set()
        for cycle in all_cycles:
            cycle_key = "->".join([f"{src}-{tgt}" for src, tgt in cycle])
            if cycle_key not in seen_cycles:
                seen_cycles.add(cycle_key)
                variants = []
                for src, tgt in cycle:
                    variants.extend(rel_map.get((src, tgt), []))

                conflicts.append(
                    Conflict(
                        severity=self.severity,
                        conflict_type=self.name,
                        entity_id=cycle[0][0] if cycle else "unknown",
                        message=f"Circular relationship: {' -> '.join([src for src, _ in cycle] + [cycle[-1][1]])}",
                        variants=variants,
                        details={"cycle_edges": cycle},
                    )
                )

        return conflicts


class SemanticConflictRule(BaseConflictRule):
    """Detect semantic conflicts (same name, different class)."""

    name = "semantic_conflict"
    severity = "warning"
    description = "Detect nodes with same name but different classes"

    def detect(self, context: DetectionContext) -> list[Conflict]:
        if not context.fragment:
            return []

        conflicts: list[Conflict] = []
        nodes = context.fragment.get("nodes", [])

        # Group by name
        by_name: dict[str, list[dict]] = defaultdict(list)
        for node in nodes:
            if isinstance(node, dict):
                name = node.get("name") or node.get("meta_name")
                if name:
                    by_name[name].append(node)

        # Check for class conflicts
        for name, node_list in by_name.items():
            if len(node_list) < 2:
                continue

            classes: set[str] = set()
            for node in node_list:
                cls = node.get("meta_class") or node.get("class")
                if cls:
                    classes.add(str(cls))

            if len(classes) > 1:
                conflicts.append(
                    Conflict(
                        severity=self.severity,
                        conflict_type=self.name,
                        entity_id=name,
                        message=f"Same name '{name}' with different classes: {sorted(classes)}",
                        variants=node_list,
                        details={"conflicting_classes": sorted(classes)},
                    )
                )

        return conflicts


# =============================================================================
# Index-level rules (from SemanticOverlapValidator)
# =============================================================================


class MetaNameDriftRule(BaseConflictRule):
    """Detect meta_name drift across runs."""

    name = "meta_name_drift"
    severity = "blocking"
    description = "Detect same meta_id with different names across runs"

    def detect(self, context: DetectionContext) -> list[Conflict]:
        conflicts: list[Conflict] = []

        for meta_id, variants in context.by_meta_id.items():
            names = {
                str(v.get("meta_name") or "").strip()
                for v in variants
                if v.get("meta_name")
            }
            if len(names) > 1:
                conflicts.append(
                    Conflict(
                        severity=self.severity,
                        conflict_type=self.name,
                        entity_id=meta_id,
                        message=f"Meta name drift for {meta_id}: {sorted(names)}",
                        variants=variants,
                        details={"drifted_names": sorted(names)},
                    )
                )

        return conflicts


class MetaClassDriftRule(BaseConflictRule):
    """Detect meta_class drift across runs."""

    name = "meta_class_drift"
    severity = "blocking"
    description = "Detect same meta_id with different classes across runs"

    def detect(self, context: DetectionContext) -> list[Conflict]:
        conflicts: list[Conflict] = []

        for meta_id, variants in context.by_meta_id.items():
            classes = {
                str(v.get("meta_class") or "").strip()
                for v in variants
                if v.get("meta_class")
            }
            if len(classes) > 1:
                conflicts.append(
                    Conflict(
                        severity=self.severity,
                        conflict_type=self.name,
                        entity_id=meta_id,
                        message=f"Meta class drift for {meta_id}: {sorted(classes)}",
                        variants=variants,
                        details={"drifted_classes": sorted(classes)},
                    )
                )

        return conflicts


class InstanceBindingConflictRule(BaseConflictRule):
    """Detect conflicts in instance bindings."""

    name = "instance_binding_conflict"
    severity = "warning"
    description = "Detect conflicting instance bindings across runs"

    def detect(self, context: DetectionContext) -> list[Conflict]:
        conflicts: list[Conflict] = []

        for meta_id, variants in context.by_instance_binding.items():
            if len(variants) < 2:
                continue

            # Check for conflicting bindings
            seen_bindings: set[str] = set()
            conflicting = False
            for v in variants:
                binding = str(v.get("binding", ""))
                if binding and binding in seen_bindings:
                    conflicting = True
                    break
                seen_bindings.add(binding)

            if conflicting:
                conflicts.append(
                    Conflict(
                        severity=self.severity,
                        conflict_type=self.name,
                        entity_id=meta_id,
                        message=f"Conflicting instance bindings for {meta_id}",
                        variants=variants,
                    )
                )

        return conflicts


class RelationshipKeyConflictRule(BaseConflictRule):
    """Detect conflicting relationship definitions."""

    name = "relationship_key_conflict"
    severity = "warning"
    description = "Detect same relationship key with different definitions"

    def detect(self, context: DetectionContext) -> list[Conflict]:
        conflicts: list[Conflict] = []

        for key, variants in context.by_relationship_key.items():
            if len(variants) < 2:
                continue

            # Check if relationships have different properties
            seen_types: set[str] = set()
            for v in variants:
                rel_type = v.get("type", "")
                if rel_type:
                    seen_types.add(str(rel_type))

            if len(seen_types) > 1:
                conflicts.append(
                    Conflict(
                        severity=self.severity,
                        conflict_type=self.name,
                        entity_id=key,
                        message=f"Relationship '{key}' has conflicting types: {sorted(seen_types)}",
                        variants=variants,
                        details={"conflicting_types": sorted(seen_types)},
                    )
                )

        return conflicts


class AttributeTypeConflictRule(BaseConflictRule):
    """Detect attribute type conflicts within nodes (e.g., same attribute with different types)."""

    name = "attribute_type_conflict"
    severity = "warning"
    description = "Detect nodes with same attribute but different value types"

    def detect(self, context: DetectionContext) -> list[Conflict]:
        if not context.fragment:
            return []

        conflicts: list[Conflict] = []
        nodes = context.fragment.get("nodes", [])

        for node in nodes:
            if not isinstance(node, dict):
                continue

            node_id = node.get("node_id") or node.get("meta_id") or "unknown"
            attrs = node.get("attributes") or node.get("properties") or {}

            if not isinstance(attrs, dict):
                continue

            type_conflicts: dict[str, set[str]] = {}
            for attr_name, attr_value in attrs.items():
                if attr_value is None:
                    continue
                value_type = type(attr_value).__name__

                if attr_name not in type_conflicts:
                    type_conflicts[attr_name] = set()
                type_conflicts[attr_name].add(value_type)

            for attr_name, types in type_conflicts.items():
                if len(types) > 1:
                    conflicts.append(
                        Conflict(
                            severity=self.severity,
                            conflict_type=self.name,
                            entity_id=str(node_id),
                            message=f"Attribute '{attr_name}' has conflicting types: {sorted(types)}",
                            variants=[node],
                            details={
                                "attribute_name": attr_name,
                                "conflicting_types": sorted(types),
                                "node_id": node_id,
                            },
                        )
                    )

        return conflicts


class LayerScopeConsistencyRule(BaseConflictRule):
    """Detect layer_scope consistency conflicts for same meta_id."""

    name = "layer_scope_consistency"
    severity = "warning"
    description = "Detect same meta_id with different layer_scope values"

    def detect(self, context: DetectionContext) -> list[Conflict]:
        conflicts: list[Conflict] = []

        for meta_id, variants in context.by_meta_id.items():
            if len(variants) < 2:
                continue

            scopes = set()
            for v in variants:
                scope = v.get("layer_scope", "").strip()
                if scope:
                    scopes.add(scope)

            if len(scopes) > 1:
                conflicts.append(
                    Conflict(
                        severity=self.severity,
                        conflict_type=self.name,
                        entity_id=meta_id,
                        message=f"Layer_scope inconsistency for {meta_id}: different scopes {sorted(scopes)}",
                        variants=variants,
                        details={"conflicting_scopes": sorted(scopes)},
                    )
                )

        return conflicts


class CrossFileNameDriftRule(BaseConflictRule):
    """Detect cross-file name drift for same meta_id."""

    name = "cross_file_name_drift"
    severity = "warning"
    description = "Detect same meta_id with different names across source files"

    def detect(self, context: DetectionContext) -> list[Conflict]:
        conflicts: list[Conflict] = []

        for meta_id, variants in context.by_meta_id.items():
            if len(variants) < 2:
                continue

            names_by_file: dict[str, set[str]] = {}
            for v in variants:
                source_file = v.get("source_file") or v.get("evidence", {}).get("source_file") or "unknown"
                meta_name = str(v.get("meta_name") or v.get("name") or "").strip()
                if meta_name:
                    if source_file not in names_by_file:
                        names_by_file[source_file] = set()
                    names_by_file[source_file].add(meta_name)

            if len(names_by_file) > 1:
                all_names: set[str] = set()
                for names in names_by_file.values():
                    all_names.update(names)

                if len(all_names) > 1:
                    conflicts.append(
                        Conflict(
                            severity=self.severity,
                            conflict_type=self.name,
                            entity_id=meta_id,
                            message=f"Cross-file name drift for {meta_id}: different names across {len(names_by_file)} files",
                            variants=variants,
                            details={
                                "names_by_file": {k: sorted(v) for k, v in names_by_file.items()},
                                "all_names": sorted(all_names),
                                "file_count": len(names_by_file),
                            },
                        )
                    )

        return conflicts


class RelationWhitelistValidator(BaseConflictRule):
    """Validate that relationship types are in the allowed whitelist from relation_type_registry.yaml."""

    name = "relation_whitelist_violation"
    severity = "warning"
    description = "Detect relationships with types not in the allowed whitelist"

    def detect(self, context: DetectionContext) -> list[Conflict]:
        if not context.fragment:
            return []

        conflicts: list[Conflict] = []
        relationships = context.fragment.get("relationships", [])
        allowed_types = _relation_registry.get_allowed_types()

        for rel in relationships:
            if not isinstance(rel, dict):
                rel = entity_to_dict(rel)

            rel_type = rel.get("type") or rel.get("relation_type") or rel.get("predicate") or ""
            if not rel_type:
                continue

            normalized_type = _relation_registry.normalize_type(rel_type)

            if normalized_type not in allowed_types:
                source_id, target_id = relationship_endpoints(rel)
                entity_id = f"{source_id}->{target_id}" if source_id and target_id else "unknown"

                conflicts.append(
                    Conflict(
                        severity=self.severity,
                        conflict_type=self.name,
                        entity_id=entity_id,
                        message=f"Relationship type '{rel_type}' is not in the allowed whitelist. Allowed types: {sorted(allowed_types)}",
                        variants=[rel],
                        details={
                            "invalid_type": rel_type,
                            "suggested_types": self._suggest_related_types(rel_type, allowed_types),
                            "allowed_types": sorted(allowed_types),
                        },
                    )
                )

        return conflicts

    def _suggest_related_types(self, invalid_type: str, allowed_types: set) -> list[str]:
        """Suggest related valid relation types based on similarity."""
        normalized = invalid_type.strip().lower()
        suggestions = []

        for allowed in allowed_types:
            allowed_lower = allowed.lower()
            if normalized in allowed_lower or allowed_lower in normalized:
                suggestions.append(allowed)
            elif len(set(normalized) & set(allowed_lower)) / len(set(normalized)) > 0.5:
                suggestions.append(allowed)

        return sorted(suggestions)[:5]


class SameAsConflictRule(BaseConflictRule):
    """Detect conflicts in same_as relationships."""

    name = "same_as_conflict"
    rule_id: str = "same_as_conflict"
    description: str = "Detect invalid same_as relationships"
    severity: str = "warning"

    DEFAULT_THRESHOLD: float = 0.4
    THRESHOLDS_BY_TYPE: dict[str, float] = {
        "BusinessScenario": 0.35,
        "Concept": 0.45,
        "Process": 0.4,
        "DataEntity": 0.5,
        "Actor": 0.35,
        "System": 0.4,
        "Data": 0.45,
        "Asset": 0.4,
        "Capability": 0.4,
        "BusinessRule": 0.35,
        "MappingRule": 0.35,
    }

    STOPWORDS: set[str] = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "is", "are", "was", "were", "be", "been",
        "this", "that", "these", "those", "it", "its", "from", "as",
        "system", "process", "data", "used", "using", "based", "via"
    }

    def detect(self, context: DetectionContext) -> list[Conflict]:
        conflicts = []

        if context.fragment:
            relationships = context.fragment.get("relationships", [])
            nodes = context.fragment.get("nodes", [])
            instances = context.fragment.get("instances", [])
            nodes_dict = {
                entity_to_dict(n).get("meta_id"): entity_to_dict(n)
                for n in nodes + instances
                if entity_to_dict(n).get("meta_id")
            }

            for rel in relationships:
                rel_dict = entity_to_dict(rel)
                rel_type = rel_dict.get("predicate") or rel_dict.get("type") or ""
                if rel_type != "same_as":
                    continue

                source_id, target_id = relationship_endpoints(rel_dict)
                if source_id not in nodes_dict or target_id not in nodes_dict:
                    continue

                source = nodes_dict[source_id]
                target = nodes_dict[target_id]

                conflicts.extend(self._check_attribute_conflicts(source, target, rel_dict))

        return conflicts

    def _get_threshold_for_entity(self, entity: dict) -> float:
        meta_class = entity.get("meta_class", "")
        if isinstance(meta_class, str):
            return self.THRESHOLDS_BY_TYPE.get(meta_class, self.DEFAULT_THRESHOLD)
        return self.DEFAULT_THRESHOLD

    def _check_attribute_conflicts(
        self, source: dict, target: dict, relationship: dict
    ) -> list[Conflict]:
        conflicts = []

        source_desc = source.get("description", "").lower()
        target_desc = target.get("description", "").lower()

        if source_desc and target_desc:
            similarity = self._calculate_semantic_similarity(source_desc, target_desc)

            threshold = max(
                self._get_threshold_for_entity(source),
                self._get_threshold_for_entity(target)
            )

            if similarity < threshold:
                source_id = source.get("meta_id")
                target_id = target.get("meta_id")
                conflicts.append(
                    Conflict(
                        severity=self.severity,
                        conflict_type=self.rule_id,
                        entity_id=f"{source_id}--{target_id}",
                        message=f"Same-as relationship with low semantic similarity ({similarity:.2f} < {threshold:.2f})",
                        details={
                            "relationship": relationship,
                            "source_description": source_desc[:100],
                            "target_description": target_desc[:100],
                            "similarity": similarity,
                            "threshold": threshold,
                            "entity_pair": [source_id, target_id],
                        },
                    )
                )

        return conflicts

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0

        tokens1 = self._smart_tokenize(text1)
        tokens2 = self._smart_tokenize(text2)

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        jaccard = len(intersection) / len(union) if union else 0.0

        business_terms = {
            "bom", "ebom", "mbom", "manufacturing", "production", "supply",
            "logistics", "quality", "control", "planning", "inventory",
            "procurement", "supplier", "delivery", "assembly", "part"
        }
        shared_business = intersection.intersection(business_terms)
        business_bonus = min(len(shared_business) * 0.05, 0.15)

        return min(jaccard + business_bonus, 1.0)

    def _smart_tokenize(self, text: str) -> set[str]:
        words = text.split()
        tokens = set()

        for word in words:
            cleaned = word.strip(".,;:!?()[]{}\"'").lower()

            if cleaned and len(cleaned) > 2 and cleaned not in self.STOPWORDS:
                cleaned = self._basic_stem(cleaned)
                tokens.add(cleaned)

        return tokens

    def _basic_stem(self, word: str) -> str:
        suffixes = ["ing", "ed", "s", "es", "tion", "ment", "ness"]
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 3:
                return word[:-len(suffix)]
        return word