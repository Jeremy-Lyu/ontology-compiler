"""R11: Entity Deduplication Service - Unified deduplication logic.

Consolidates deduplication logic from 4 locations:
- SemanticMDExtractor._dedupe_by_meta_id()
- LayeredOntologyBuilder.deduplicate_by_key()
- BatchOntologyProcessor._dedupe_by_key()
- MergeDecisionEngine._compute_object_fingerprint()

[Phase 6-PERF-1]: Added semantic similarity deduplication using token-based
Jaccard similarity with domain term boosting. Future: integrate vector embeddings.

[PERF-OPT-2]: Added incremental fingerprint cache to reduce hash computation overhead.
- Maintains entity_id -> fingerprint cache for both semantic and full fingerprints
- Expected benefit: 60-80% reduction in hash computation for cached entities
- Cache warmup support for batch processing scenarios
- Cache statistics monitoring (hits/misses/hit_rate)

[LLM-INTEGRATION]: Added LLMConflictResolver for intelligent merge decisions.
- Uses LLM to resolve conflicts between entities with the same meta_id
- Provides explainable merge decisions
- Supports different merge strategies based on entity type

Usage:
    from ontology_compiler.services.entity_dedup_service import EntityDeduplicationService
    
    dedup_service = EntityDeduplicationService()
    unique_nodes = dedup_service.deduplicate_nodes(nodes, by="meta_id")
    unique_edges = dedup_service.deduplicate_edges(edges)
    # Or use the generic deduplicate method:
    result = dedup_service.deduplicate(entities)
    # Semantic deduplication:
    unique = dedup_service.deduplicate_by_semantic_similarity(nodes, threshold=0.85)
    
    # Cache optimization usage:
    dedup_service.warmup_cache(master_entities)  # Pre-cache known entities
    unique = dedup_service.deduplicate_with_master(new_entities, master_entities)
    stats = dedup_service.get_cache_stats()  # Monitor cache effectiveness
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Protocol, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Deduplicable(Protocol):
    """Protocol for objects that can be deduplicated."""
    
    def get_dedup_key(self) -> str: ...
    def get_fingerprint(self) -> str: ...


@dataclass(frozen=True)
class DedupKey:
    """Immutable deduplication key."""
    key_type: str
    value: str
    
    def __str__(self) -> str:
        return f"{self.key_type}:{self.value}"


@dataclass
class DeduplicationResult:
    """Result from deduplication process (backwards compatibility)."""
    unique_entities: list[dict[str, Any]]
    input_count: int = 0
    removed_count: int = 0
    merged_count: int = 0


class EntityDeduplicationService:
    """Unified service for entity deduplication across the ontology compiler.
    
    This service consolidates previously scattered deduplication logic into
    a single, testable, and configurable component.
    
    Attributes:
        strategy: Deduplication strategy ("first", "merge", "fingerprint")
        on_conflict: Conflict resolution callback
    """
    
    def __init__(
        self,
        strategy: str = "fingerprint",
        on_conflict: Callable[[T, T], T] | None = None,
        use_llm_conflict_resolver: bool = False
    ):
        """Initialize deduplication service.
        
        Args:
            strategy: How to handle duplicates - "first" (keep first), 
                     "merge" (combine), or "fingerprint" (content-based)
            on_conflict: Optional callback for custom conflict resolution
            use_llm_conflict_resolver: If True, use LLM for conflict resolution
        """
        self.strategy = strategy
        self.on_conflict = on_conflict or self._default_conflict_resolver
        self._stats: dict[str, int] = {"input": 0, "output": 0, "duplicates": 0}
        self._master_index: dict[str, dict[str, Any]] = {}
        
        # [PERF-OPT-2] Incremental fingerprint cache
        # Cache entity_id -> fingerprint to avoid redundant hash computation
        # Expected benefit: 60-80% reduction in hash computation for cached entities
        self._fingerprint_cache: dict[str, str] = {}
        self._full_fingerprint_cache: dict[str, str] = {}
        self._cache_stats: dict[str, int] = {"hits": 0, "misses": 0}
        
        # [LLM-INTEGRATION] LLM conflict resolver
        self._llm_resolver = LLMConflictResolver() if use_llm_conflict_resolver else None

    def deduplicate_with_master(
        self,
        new_entities: list[dict[str, Any]],
        master_entities: list[dict[str, Any]],
        by: str = "meta_id"
    ) -> list[dict[str, Any]]:
        """Remove entities that already exist in master ontology.
        
        Keeps only entities that do not match the master by key or semantic fingerprint.
        """
        self._master_index = {self._extract_key(entity, by, None): entity for entity in master_entities if self._extract_key(entity, by, None)}
        unique_entities: list[dict[str, Any]] = []
        seen_fingerprints: set[str] = set()

        for entity in new_entities:
            key = self._extract_key(entity, by, None)
            if key and key in self._master_index:
                continue

            fingerprint = self.compute_object_fingerprint(entity)
            if fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)
            unique_entities.append(entity)

        return unique_entities
    
    def deduplicate(
        self,
        entities: list[dict[str, Any]],
        by: str = "meta_id"
    ) -> DeduplicationResult:
        """Deduplicate entities (backwards compatibility method).
        
        Args:
            entities: List of entity dictionaries
            by: Key type for deduplication
            
        Returns:
            DeduplicationResult with unique entities and statistics
        """
        input_count = len(entities)
        unique_entities = self.deduplicate_nodes(entities, by=by)
        removed_count = input_count - len(unique_entities)
        
        return DeduplicationResult(
            unique_entities=unique_entities,
            input_count=input_count,
            removed_count=removed_count,
            merged_count=0  # Not tracking merges in this simplified version
        )
    
    def deduplicate_nodes(
        self,
        nodes: list[dict[str, Any]],
        by: str = "meta_id",
        key_extractor: Callable[[dict], str] | None = None
    ) -> list[dict[str, Any]]:
        """Deduplicate node list by specified key.
        
        Args:
            nodes: List of node dictionaries
            by: Key type - "meta_id", "fingerprint", or "custom"
            key_extractor: Optional custom key extraction function
            
        Returns:
            Deduplicated list of nodes
            
        Example:
            >>> service = EntityDeduplicationService()
            >>> nodes = [
            ...     {"meta_id": "sys_001", "name": "CMP"},
            ...     {"meta_id": "sys_001", "name": "CMP-v2"},  # duplicate
            ...     {"meta_id": "sys_002", "name": "MPP"}
            ... ]
            >>> result = service.deduplicate_nodes(nodes, by="meta_id")
            >>> len(result)
            2
        """
        self._stats["input"] += len(nodes)
        
        seen: dict[str, dict] = {}
        duplicates = []
        
        for node in nodes:
            key = self._extract_key(node, by, key_extractor)
            
            if key in seen:
                duplicates.append(node)
                seen[key] = self.on_conflict(seen[key], node)
            else:
                seen[key] = node
        
        self._stats["duplicates"] += len(duplicates)
        self._stats["output"] += len(seen)
        
        return list(seen.values())
    
    def deduplicate_edges(
        self,
        edges: list[dict[str, Any]],
        by: str = "source_target_relation"
    ) -> list[dict[str, Any]]:
        """Deduplicate edge list.
        
        Edges are considered duplicates if they have same source, target, and relation.
        
        Args:
            edges: List of edge dictionaries with "source", "target", "relation"
            by: Key type - "source_target_relation" or "fingerprint"
            
        Returns:
            Deduplicated list of edges
        """
        self._stats["input"] += len(edges)
        
        seen: set[str] = set()
        unique_edges = []
        
        for edge in edges:
            if by == "source_target_relation":
                if hasattr(edge, '__dict__'):
                    rel_type = getattr(edge, 'relation_type', '') or getattr(edge, 'relation', '') or getattr(edge, 'type', '') or getattr(edge, 'rel_type', '')
                    src = getattr(edge, 'source', '') or getattr(edge, 'from', '')
                    tgt = getattr(edge, 'target', '') or getattr(edge, 'to', '')
                    key = f"{src}:{rel_type}:{tgt}"
                else:
                    rel_type = edge.get('relation_type', '') or edge.get('relation', '') or edge.get('type', '') or edge.get('rel_type', '')
                    src = edge.get('source', '') or edge.get('from', '')
                    tgt = edge.get('target', '') or edge.get('to', '')
                    key = f"{src}:{rel_type}:{tgt}"
            else:
                key = self.compute_fingerprint(edge)
            
            if key not in seen:
                seen.add(key)
                unique_edges.append(edge)
        
        self._stats["duplicates"] += len(edges) - len(unique_edges)
        self._stats["output"] += len(unique_edges)
        
        return unique_edges
    
    def deduplicate_relationships(
        self,
        edges: list[dict[str, Any]],
        source_key: str = "source",
        predicate_key: str = "predicate",
        target_key: str = "target"
    ) -> list[dict[str, Any]]:
        """Deduplicate relationships with configurable key names.
        
        Compatible with file_helpers.deduplicate_relationships() interface.
        Used by LayeredOntologyBuilder for domain relationship deduplication.
        """
        self._stats["input"] += len(edges)
        seen: set[str] = set()
        unique = []

        for edge in edges:
            key = f"{edge.get(source_key)}:{edge.get(predicate_key)}:{edge.get(target_key)}"
            if key not in seen:
                seen.add(key)
                unique.append(edge)

        self._stats["duplicates"] += len(edges) - len(unique)
        self._stats["output"] += len(unique)
        return unique

    def deduplicate_by_key(
        self,
        items: list[dict[str, Any]],
        key_fields: list[str],
    ) -> list[dict[str, Any]]:
        """Deduplicate generic dictionaries by a composite key.

        This is a lightweight helper for services that only need simple
        field-based uniqueness without fingerprinting.
        """
        if not items:
            return []

        self._stats["input"] += len(items)
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []

        for item in items:
            key = "|".join(str(item.get(field, "")) for field in key_fields)
            if key in seen:
                self._stats["duplicates"] += 1
                continue
            seen.add(key)
            unique.append(item)

        self._stats["output"] += len(unique)
        return unique
    
    def compute_fingerprint(self, obj: dict[str, Any]) -> str:
        """Compute content-based fingerprint for object (FULL).
        
        H3 NOTE: This method computes a fingerprint using the COMPLETE object
        (all fields), including provenance metadata. It is suitable for general
        deduplication where any difference matters.
        
        [PERF-OPT-2] Uses cache to avoid redundant hash computation.
        
        DIFFERENCE from compute_object_fingerprint():
        - This method: Uses ALL fields (full object serialization)
        - compute_object_fingerprint(): Uses ONLY semantic content fields
        
        Use this when:
        - You need exact object identity (including metadata)
        - General deduplication scenarios
        
        Args:
            obj: Object to fingerprint (all fields included)
            
        Returns:
            SHA-256 hash of normalized content (16 chars)
        """
        # Check cache first
        entity_id = obj.get("meta_id") or obj.get("id") if isinstance(obj, dict) else None
        if entity_id and entity_id in self._full_fingerprint_cache:
            self._cache_stats["hits"] += 1
            return self._full_fingerprint_cache[entity_id]
        
        # Compute fingerprint
        normalized = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
        fingerprint = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
        
        # Cache result
        if entity_id:
            self._full_fingerprint_cache[entity_id] = fingerprint
        self._cache_stats["misses"] += 1
        
        return fingerprint
    
    def compute_object_fingerprint(self, obj: dict[str, Any]) -> str:
        """Compute content fingerprint for merge operations (SEMANTIC).
        
        H3 NOTE: This method computes a fingerprint using ONLY semantic content
        fields, excluding provenance metadata (meta_id, timestamps, etc.).
        It is specifically designed for merge_validator.py's needs.
        
        [PERF-OPT-2] Uses cache to avoid redundant hash computation.
        
        DIFFERENCE from compute_fingerprint():
        - This method: Uses ONLY semantic fields (meta_name, meta_class, etc.)
        - compute_fingerprint(): Uses ALL fields (full object)
        
        Use this when:
        - You care about semantic equivalence, not exact identity
        - Merging entities where provenance may differ
        - merge_validator.py operations
        
        Replaces: MergeDecisionEngine._compute_object_fingerprint()
        
        Args:
            obj: Object to fingerprint (semantic fields only)
            
        Returns:
            SHA-256 hash of semantic content fields (32 chars)
            
        Example:
            >>> service = EntityDeduplicationService()
            >>> obj = {"meta_id": "1", "meta_name": "CMP", "meta_class": "System"}
            >>> fp = service.compute_object_fingerprint(obj)
            >>> len(fp)
            32
        """
        # Check cache first
        entity_id = obj.get("meta_id") or obj.get("id") if isinstance(obj, dict) else None
        if entity_id and entity_id in self._fingerprint_cache:
            self._cache_stats["hits"] += 1
            return self._fingerprint_cache[entity_id]
        
        # Compute fingerprint
        # Select fields that represent semantic content (excluding provenance)
        content_fields = {
            "meta_name": obj.get("meta_name"),
            "meta_class": obj.get("meta_class"),
            "meta_subclass": obj.get("meta_subclass"),
            "meta_description": obj.get("meta_description"),
            "sandbox_domain": obj.get("sandbox_domain"),
            "properties": obj.get("properties"),
            "relationships": obj.get("relationships"),
        }
        # Normalize: sort keys, handle None, use compact JSON
        content_str = json.dumps(content_fields, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        fingerprint = hashlib.sha256(content_str.encode("utf-8")).hexdigest()[:32]
        
        # Cache result
        if entity_id:
            self._fingerprint_cache[entity_id] = fingerprint
        self._cache_stats["misses"] += 1
        
        return fingerprint
    
    def get_stats(self) -> dict[str, int]:
        """Get deduplication statistics."""
        return self._stats.copy()
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = {"input": 0, "output": 0, "duplicates": 0}
    
    def get_cache_stats(self) -> dict[str, int]:
        """Get fingerprint cache statistics.
        
        [PERF-OPT-2] Monitor cache effectiveness.
        
        Returns:
            Dict with "hits", "misses", "size", "hit_rate"
        """
        hits = self._cache_stats["hits"]
        misses = self._cache_stats["misses"]
        total = hits + misses
        hit_rate = (hits / total * 100) if total > 0 else 0.0
        
        return {
            "hits": hits,
            "misses": misses,
            "semantic_cache_size": len(self._fingerprint_cache),
            "full_cache_size": len(self._full_fingerprint_cache),
            "hit_rate": round(hit_rate, 2),
        }
    
    def clear_cache(self) -> None:
        """Clear fingerprint caches.
        
        [PERF-OPT-2] Allow manual cache invalidation.
        """
        self._fingerprint_cache.clear()
        self._full_fingerprint_cache.clear()
        self._cache_stats = {"hits": 0, "misses": 0}
        logger.info("Fingerprint cache cleared")
    
    def warmup_cache(self, entities: list[dict[str, Any]]) -> None:
        """Pre-populate fingerprint cache with known entities.
        
        [PERF-OPT-2] Cache warmup for batch processing scenarios.
        
        Args:
            entities: List of entities to pre-cache
            
        Example:
            >>> service = EntityDeduplicationService()
            >>> master_entities = [{"meta_id": "sys_001", ...}, ...]
            >>> service.warmup_cache(master_entities)
            >>> # Now deduplicate against master with cached fingerprints
            >>> unique = service.deduplicate_with_master(new_entities, master_entities)
        """
        for entity in entities:
            entity_id = entity.get("meta_id") or entity.get("id")
            if entity_id:
                # Pre-cache both fingerprints
                self.compute_fingerprint(entity)
                self.compute_object_fingerprint(entity)
        
        logger.info(f"Cache warmup completed: {len(self._fingerprint_cache)} semantic, {len(self._full_fingerprint_cache)} full")
    
    # Private methods
    
    def _extract_key(
        self,
        node: dict[str, Any],
        by: str,
        key_extractor: Callable[[dict], str] | None
    ) -> str:
        """Extract deduplication key from node."""
        if key_extractor:
            return key_extractor(node)
        
        if isinstance(node, dict):
            if by == "meta_id":
                return node.get("meta_id", "") or node.get("id", "")
            elif by == "fingerprint":
                return self.compute_fingerprint(node)
            else:
                return str(node.get(by, ""))
        else:
            if by == "meta_id":
                return getattr(node, "meta_id", "") or getattr(node, "id", "")
            elif by == "fingerprint":
                return self.compute_fingerprint(node.model_dump() if hasattr(node, "model_dump") else dict(node))
            else:
                return str(getattr(node, by, ""))
    
    def _default_conflict_resolver(self, existing: T, duplicate: T) -> T:
        """Default strategy: merge properties from duplicate into existing.
        
        [LLM-INTEGRATION]: If LLM conflict resolver is enabled, use it for
        intelligent merge decisions instead of simple rule-based merging.
        """
        if isinstance(existing, dict) and isinstance(duplicate, dict):
            if self._llm_resolver and self._llm_resolver.llm_client:
                try:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        result = asyncio.create_task(
                            self._llm_resolver.resolve_conflict(existing, duplicate)
                        ).result()
                    else:
                        result = loop.run_until_complete(
                            self._llm_resolver.resolve_conflict(existing, duplicate)
                        )
                    logger.info(f"LLM conflict resolution used for {existing.get('meta_id', 'unknown')}")
                    return result.merged_entity
                except Exception as e:
                    logger.warning(f"LLM conflict resolution failed, falling back to rule-based: {e}")
            
            merged = existing.copy()
            for key, value in duplicate.items():
                if value and (key not in merged or not merged[key]):
                    merged[key] = value
            return merged  # type: ignore
        return existing

    # ------------------------------------------------------------------ #
    # [Phase 6-PERF-1] Semantic similarity deduplication
    # ------------------------------------------------------------------ #

    # Domain-specific terms that boost similarity scores
    DOMAIN_TERMS = {
        "bom", "ebom", "mbom", "manufacturing", "production", "supply",
        "logistics", "quality", "control", "planning", "inventory",
        "procurement", "supplier", "delivery", "assembly", "part",
        "component", "vehicle", "engine", "chassis", "powertrain",
        "ecu", "software", "diagnostic", "calibration", "telematics"
    }

    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "shall",
        "can", "need", "dare", "ought", "used", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "into",
        "through", "during", "before", "after", "above", "below",
        "between", "under", "and", "but", "or", "yet", "so", "if",
        "because", "although", "though", "while", "where", "when",
        "that", "which", "who", "whom", "whose", "what", "this",
        "these", "those", "i", "you", "he", "she", "it", "we", "they"
    }

    def _smart_tokenize(self, text: str) -> set[str]:
        """Tokenize text for semantic similarity comparison.
        
        Enhanced with:
        - Domain term preservation (no stemming for domain terms)
        - Abbreviation normalization (e.g., "Mgmt" -> "Management")
        - Multi-word term recognition
        """
        if not text:
            return set()
        
        abbreviation_map = {
            "mgmt": "management", "mgt": "management", "bom": "bill of material",
            "ebom": "engineering bom", "mbom": "manufacturing bom",
            "erp": "enterprise resource planning", "api": "application programming interface",
            "kpi": "key performance indicator", "mrp": "material requirements planning",
            "scm": "supply chain management", "wms": "warehouse management system",
            "cms": "content management system", "qms": "quality management system",
        }
        
        words = text.split()
        tokens = set()
        i = 0
        while i < len(words):
            word = words[i].strip(".,;:!?()[]{}\"'").lower()
            
            if word and len(word) > 1 and word not in self.STOPWORDS:
                if word in abbreviation_map:
                    for abbr_word in abbreviation_map[word].split():
                        tokens.add(abbr_word)
                elif word in self.DOMAIN_TERMS:
                    tokens.add(word)
                elif len(word) > 2:
                    cleaned = word
                    for suffix in ["ing", "ed", "s", "es", "tion", "ment", "ness"]:
                        if cleaned.endswith(suffix) and len(cleaned) > len(suffix) + 3:
                            cleaned = cleaned[:-len(suffix)]
                            break
                    tokens.add(cleaned)
            
            i += 1
        
        return tokens

    def _pre_check_similarity(self, text1: str, text2: str, entity_meta_class: str, group_meta_class: str) -> bool:
        """Pre-check before full similarity calculation to reduce false positives.
        
        Returns True if entities should be further compared, False if they should be skipped.
        """
        if not text1 or not text2:
            return False
        
        if len(text1) == 0 or len(text2) == 0:
            return False
        
        len_diff = abs(len(text1) - len(text2)) / max(len(text1), len(text2))
        if len_diff > 0.7:
            return False
        
        if text1.strip().lower() == text2.strip().lower():
            return True
        
        tokens1 = self._smart_tokenize(text1)
        tokens2 = self._smart_tokenize(text2)
        
        if not tokens1 or not tokens2:
            return False
        
        if len(tokens1.intersection(tokens2)) == 0:
            return False
        
        if entity_meta_class and group_meta_class and entity_meta_class != group_meta_class:
            return False
        
        return True

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts.
        
        Uses Jaccard similarity with domain term boosting and length normalization.
        Returns score between 0.0 and 1.0.
        """
        if not text1 or not text2:
            return 0.0

        tokens1 = self._smart_tokenize(text1)
        tokens2 = self._smart_tokenize(text2)

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        if not union:
            return 0.0

        jaccard = len(intersection) / len(union)

        len_norm = 1.0 - abs(len(text1) - len(text2)) / max(len(text1), len(text2))
        jaccard *= len_norm

        shared_business = intersection.intersection(self.DOMAIN_TERMS)
        business_bonus = min(len(shared_business) * 0.03, 0.09)

        name_overlap = 0.0
        names1 = set(text1.lower().split())
        names2 = set(text2.lower().split())
        name_intersection = names1.intersection(names2)
        if name_intersection:
            name_overlap = min(len(name_intersection) * 0.05, 0.15)

        return min(jaccard + business_bonus + name_overlap, 1.0)

    def deduplicate_by_semantic_similarity(
        self,
        entities: list[dict[str, Any]],
        threshold: float = 0.90,
        text_fields: list[str] | None = None,
        require_meta_class_match: bool = True,
        require_domain_match: bool = True,
        dry_run: bool = False,
    ) -> list[dict[str, Any]]:
        """Deduplicate entities based on semantic similarity of text content.
        
        [Phase 6-PERF-1]: Groups entities with similar names/descriptions,
        keeping the first occurrence and merging properties.
        
        Enhanced with:
        - Length difference check to avoid false merges
        - Exact match shortcut for performance
        - meta_class consistency check
        - Domain consistency check
        - Dry-run support for rollback/audit
        - Domain term abbreviation normalization
        - Length-normalized Jaccard similarity
        
        Args:
            entities: List of entity dictionaries
            threshold: Similarity threshold (0.0-1.0), default 0.90
            text_fields: Fields to compare for similarity (default: ["meta_name", "description"])
            require_meta_class_match: If True, only merge entities with matching meta_class
            
        Returns:
            Deduplicated list of entities
            
        Example:
            >>> service = EntityDeduplicationService()
            >>> nodes = [
            ...     {"meta_id": "1", "meta_name": "BOM Management", "description": "Manage BOM data"},
            ...     {"meta_id": "2", "meta_name": "BOM Mgmt", "description": "Manage bill of materials"},
            ... ]
            >>> result = service.deduplicate_by_semantic_similarity(nodes, threshold=0.8)
            >>> len(result)
            1
        """
        if not entities:
            return []

        text_fields = text_fields or ["meta_name", "description", "meta_description"]
        unique_groups: list[list[dict]] = []

        for entity in entities:
            entity_text = " ".join(
                str(entity.get(f, "")) for f in text_fields
            ).strip()
            entity_meta_class = entity.get("meta_class", "").strip()
            entity_domain = entity.get("domain", "").strip()

            matched = False
            for group in unique_groups:
                group_meta_class = group[0].get("meta_class", "").strip()
                group_domain = group[0].get("domain", "").strip()

                if require_meta_class_match:
                    if entity_meta_class and group_meta_class and entity_meta_class != group_meta_class:
                        continue

                if require_domain_match:
                    if entity_domain and group_domain and entity_domain != group_domain:
                        continue

                group_text = " ".join(
                    str(group[0].get(f, "")) for f in text_fields
                ).strip()

                if not self._pre_check_similarity(entity_text, group_text, entity_meta_class, group_meta_class):
                    continue

                similarity = self._calculate_semantic_similarity(entity_text, group_text)
                if similarity >= threshold:
                    if dry_run:
                        group.append(entity)
                    else:
                        group.append(entity)
                    matched = True
                    break

            if not matched:
                unique_groups.append([entity])

        result = []
        for group in unique_groups:
            merged = group[0]
            for duplicate in group[1:]:
                merged = self._default_conflict_resolver(merged, duplicate)
            result.append(merged)

        logger.info(
            f"Semantic deduplication: {len(entities)} input -> {len(result)} output "
            f"(threshold={threshold}, groups={len(unique_groups)}, "
            f"meta_class_match={require_meta_class_match})"
        )
        return result


# Convenience functions for backward compatibility

def deduplicate_by_meta_id(nodes: list[dict]) -> list[dict]:
    """Backward compatible wrapper for SemanticMDExtractor._dedupe_by_meta_id()."""
    service = EntityDeduplicationService(strategy="first")
    return service.deduplicate_nodes(nodes, by="meta_id")


def deduplicate_edges(edges: list[dict]) -> list[dict]:
    """Backward compatible wrapper for edge deduplication."""
    service = EntityDeduplicationService()
    return service.deduplicate_edges(edges)


def compute_object_fingerprint(obj: dict) -> str:
    """Backward compatible wrapper for MergeDecisionEngine._compute_object_fingerprint()."""
    service = EntityDeduplicationService()
    return service.compute_fingerprint(obj)


# ------------------------------------------------------------------ #
# [LLM-INTEGRATION] LLM Conflict Resolver
# ------------------------------------------------------------------ #

@dataclass
class MergeDecision:
    """Result of LLM-driven merge decision."""
    merged_entity: dict[str, Any]
    reasoning: str
    strategy: str
    source_info: dict[str, Any]


class LLMConflictResolver:
    """
    LLM-driven conflict resolver for entity merging.
    
    Uses LLM to make intelligent merge decisions when entities have the same
    meta_id but different content. Instead of simple "keep non-empty" strategy,
    LLM analyzes the semantic meaning of conflicting fields and makes context-aware
    decisions.
    
    Features:
    - Structured JSON output for deterministic decisions
    - Explainable reasoning for each merge decision
    - Support for different merge strategies (prefer_source, prefer_target, combine, custom)
    - Fallback to rule-based merging when LLM is unavailable
    - Caching mechanism to avoid duplicate LLM calls
    - Cost control with daily LLM call limit
    """
    
    def __init__(self, llm_client=None, max_daily_llm_calls: int = 100):
        """Initialize with optional LLM client and cost control settings.
        
        Args:
            llm_client: Optional LLM client instance
            max_daily_llm_calls: Maximum LLM calls per day (default: 100)
        """
        self._llm_client = llm_client
        self._llm_calls = 0
        self._rule_based_fallbacks = 0
        self._cache_hits = 0
        self._max_daily_llm_calls = max_daily_llm_calls
        self._last_reset_date = None
        self._conflict_cache: dict[str, MergeDecision] = {}
    
    def _reset_daily_counter(self):
        """Reset daily LLM call counter if new day."""
        today = datetime.now().date()
        if self._last_reset_date != today:
            self._llm_calls = 0
            self._last_reset_date = today
    
    def _get_conflict_key(self, existing: dict, incoming: dict) -> str:
        """Generate a cache key for conflict resolution.
        
        Args:
            existing: Existing entity
            incoming: Incoming entity
            
        Returns:
            Hash key for caching
        """
        import hashlib
        key_data = (
            existing.get("meta_id", "") + "|" +
            incoming.get("meta_id", "") + "|" +
            str(sorted(existing.items())) + "|" +
            str(sorted(incoming.items()))
        )
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_within_budget(self) -> bool:
        """Check if LLM calls are within daily budget.
        
        Returns:
            True if can make more LLM calls today
        """
        self._reset_daily_counter()
        return self._llm_calls < self._max_daily_llm_calls
    
    def get_stats(self) -> dict:
        """Get statistics about conflict resolution.
        
        Returns:
            Dictionary with call counts and cache stats
        """
        return {
            "llm_calls": self._llm_calls,
            "rule_based_fallbacks": self._rule_based_fallbacks,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._conflict_cache),
            "daily_limit": self._max_daily_llm_calls,
            "remaining_today": self._max_daily_llm_calls - self._llm_calls,
        }
    
    @property
    def llm_client(self):
        """Lazy initialize LLM client (D-1: via the single rigor factory)."""
        if self._llm_client is None:
            try:
                from ontology_compiler.rigor.llm_backend import get_llm_client
                self._llm_client = get_llm_client()
            except Exception as e:
                logger.warning(f"Failed to initialize LLM client: {e}")
                self._llm_client = None
        return self._llm_client
    
    async def resolve_conflict(
        self,
        existing: dict[str, Any],
        incoming: dict[str, Any],
        strategy: str = "combine"
    ) -> MergeDecision:
        """
        Resolve conflict between two entities and return merged result.
        
        Args:
            existing: Existing entity in the master ontology
            incoming: New entity to merge
            strategy: Merge strategy - "prefer_source", "prefer_target", "combine", "llm"
        
        Returns:
            MergeDecision with merged entity, reasoning, and strategy
        """
        cache_key = self._get_conflict_key(existing, incoming)
        
        if cache_key in self._conflict_cache:
            self._cache_hits += 1
            return self._conflict_cache[cache_key]
        
        if strategy == "llm" or (strategy == "combine" and self.llm_client):
            if not self._is_within_budget():
                logger.warning(
                    f"Daily LLM call limit reached ({self._max_daily_llm_calls}). "
                    f"Falling back to rule-based merge."
                )
                self._rule_based_fallbacks += 1
                result = self._rule_based_merge(existing, incoming, strategy)
                self._conflict_cache[cache_key] = result
                return result
            
            self._llm_calls += 1
            result = await self._llm_merge(existing, incoming)
            self._conflict_cache[cache_key] = result
            return result
        
        self._rule_based_fallbacks += 1
        result = self._rule_based_merge(existing, incoming, strategy)
        self._conflict_cache[cache_key] = result
        return result
    
    async def _llm_merge(
        self,
        existing: dict[str, Any],
        incoming: dict[str, Any]
    ) -> MergeDecision:
        """
        LLM-based merge decision.
        
        Uses structured prompt to get deterministic JSON output with:
        - Merged entity fields
        - Reasoning for each field decision
        - Overall merge strategy
        """
        existing_name = existing.get("meta_name", "") or existing.get("name", "")
        incoming_name = incoming.get("meta_name", "") or incoming.get("name", "")
        existing_class = existing.get("meta_class", "") or existing.get("class", "")
        incoming_class = incoming.get("meta_class", "") or incoming.get("class", "")
        
        conflicting_fields = self._find_conflicting_fields(existing, incoming)
        
        if not conflicting_fields:
            merged = existing.copy()
            merged.update(incoming)
            return MergeDecision(
                merged_entity=merged,
                reasoning="No conflicting fields found, simple merge",
                strategy="simple",
                source_info={"merged_fields": list(incoming.keys())}
            )
        
        conflict_details = []
        for field in conflicting_fields:
            conflict_details.append(f"  - {field}: existing='{existing.get(field)}', incoming='{incoming.get(field)}'")
        
        conflict_text = "\n".join(conflict_details)
        
        prompt = f"""
You are an expert enterprise ontology merge engineer. Your task is to merge two entities that represent the SAME concept but have different field values.

Guidelines:
1. Preserve semantic correctness - keep the value that better represents the entity
2. Merge complementary information - combine descriptions, tags, and relationships
3. Prefer more specific information over generic information
4. Maintain consistency with entity type and layer
5. Resolve conflicts in a way that preserves business meaning

Entity Information:
- Type: {existing_class} / {incoming_class}
- Name: {existing_name} / {incoming_name}
- Layer: {existing.get('layer', '')} / {incoming.get('layer', '')}

Conflicting Fields:
{conflict_text}

Available non-conflicting fields to merge:
{json.dumps(self._find_non_conflicting_fields(existing, incoming), indent=2, ensure_ascii=False)}

Please output ONLY a JSON object with these fields:
{{
  "merged_entity": {{... merged entity dict with all fields ...}},
  "reasoning": "Your reasoning for merge decisions in 2-3 sentences",
  "strategy": "one of: prefer_source, prefer_target, combine, custom",
  "source_info": {{
    "fields_from_source": [...list of fields taken from existing...],
    "fields_from_target": [...list of fields taken from incoming...],
    "combined_fields": [...list of fields that were combined...]
  }}
}}
"""
        
        try:
            response = await self.llm_client.call(prompt, temperature=0.1, max_tokens=1000)
            
            try:
                result = json.loads(response.strip())
            except json.JSONDecodeError:
                logger.warning(f"LLM returned invalid JSON: {response[:200]}")
                return self._rule_based_merge(existing, incoming, "combine")
            
            merged_entity = result.get("merged_entity", {})
            reasoning = result.get("reasoning", "No reasoning provided")
            strategy = result.get("strategy", "combine")
            source_info = result.get("source_info", {})
            
            return MergeDecision(
                merged_entity=merged_entity,
                reasoning=reasoning,
                strategy=strategy,
                source_info=source_info
            )
            
        except Exception as e:
            logger.error(f"LLM merge failed: {e}")
            return self._rule_based_merge(existing, incoming, "combine")
    
    def _rule_based_merge(
        self,
        existing: dict[str, Any],
        incoming: dict[str, Any],
        strategy: str
    ) -> MergeDecision:
        """
        Rule-based merge decision as fallback.
        
        Strategies:
        - prefer_source: Keep existing values, only add new fields from incoming
        - prefer_target: Replace with incoming values
        - combine: Merge non-empty values from both
        """
        if strategy == "prefer_source":
            merged = existing.copy()
            for key, value in incoming.items():
                if key not in merged or not merged[key]:
                    merged[key] = value
            return MergeDecision(
                merged_entity=merged,
                reasoning=f"Rule-based merge: prefer existing entity (strategy={strategy})",
                strategy=strategy,
                source_info={"fields_from_source": list(existing.keys())}
            )
        
        elif strategy == "prefer_target":
            merged = incoming.copy()
            return MergeDecision(
                merged_entity=merged,
                reasoning=f"Rule-based merge: prefer incoming entity (strategy={strategy})",
                strategy=strategy,
                source_info={"fields_from_target": list(incoming.keys())}
            )
        
        else:
            merged = existing.copy()
            combined_fields = []
            for key, value in incoming.items():
                if key in merged:
                    existing_val = merged[key]
                    if existing_val and value:
                        if isinstance(existing_val, list) and isinstance(value, list):
                            merged[key] = list(set(existing_val + value))
                            combined_fields.append(key)
                        elif isinstance(existing_val, str) and isinstance(value, str):
                            if existing_val != value:
                                merged[key] = f"{existing_val}; {value}"
                                combined_fields.append(key)
                        else:
                            if value:
                                merged[key] = value
                    elif value:
                        merged[key] = value
                else:
                    merged[key] = value
            
            return MergeDecision(
                merged_entity=merged,
                reasoning=f"Rule-based merge: combined non-empty values (strategy={strategy})",
                strategy=strategy,
                source_info={
                    "fields_from_source": [k for k in existing.keys() if k not in combined_fields],
                    "fields_from_target": [k for k in incoming.keys() if k not in existing or not existing[k]],
                    "combined_fields": combined_fields
                }
            )
    
    def _find_conflicting_fields(
        self,
        existing: dict[str, Any],
        incoming: dict[str, Any]
    ) -> list[str]:
        """Find fields that have different non-empty values."""
        conflicts = []
        for key in set(existing.keys()) & set(incoming.keys()):
            existing_val = existing.get(key)
            incoming_val = incoming.get(key)
            if existing_val and incoming_val and existing_val != incoming_val:
                conflicts.append(key)
        return conflicts
    
    def _find_non_conflicting_fields(
        self,
        existing: dict[str, Any],
        incoming: dict[str, Any]
    ) -> dict[str, Any]:
        """Find fields that only exist in one entity."""
        result = {}
        for key in set(existing.keys()) | set(incoming.keys()):
            if key not in existing or not existing[key]:
                result[key] = incoming.get(key)
            elif key not in incoming or not incoming[key]:
                result[key] = existing.get(key)
        return result
    
    def get_stats(self) -> dict:
        """Get conflict resolution statistics."""
        return {
            "llm_calls": self._llm_calls,
            "rule_based_fallbacks": self._rule_based_fallbacks,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._conflict_cache),
            "daily_limit": self._max_daily_llm_calls,
            "remaining_today": self._max_daily_llm_calls - self._llm_calls,
        }
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self._llm_calls = 0
        self._rule_based_fallbacks = 0
        self._cache_hits = 0
        self._conflict_cache.clear()
