"""Indexed Batch Association Skill — 批量关联优化

R38-QA Skill #4: 使用索引批量关联实体，减少重复查询，提升跨文件关联效率。
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from ..base import Skill, SkillContext, SkillResult, SkillPriority

logger = logging.getLogger(__name__)


class IndexedBatchAssociationSkill(Skill):
    """Optimize cross-file entity association using indexed lookups.
    
    Strategy:
    1. Build index from entities extracted in current batch
    2. Group association requests by type
    3. Execute batch lookups against index
    4. Cache results for subsequent operations
    
    Evidence: cross_file_association_service.py batch processing
    Performance: Reduces N+1 query problem, improves throughput.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.batch_size = self.config.get("batch_size", 50)
        self.enable_caching = self.config.get("enable_caching", True)
        self.cache_ttl = self.config.get("cache_ttl", 3600)  # seconds
        self._index: Dict[str, Any] = {}
        self._cache: Dict[str, Tuple[Any, float]] = {}  # (result, timestamp)
    
    @property
    def priority(self) -> SkillPriority:
        return SkillPriority.NORMAL
    
    @property
    def description(self) -> str:
        return "Batch associate entities using indexed lookups"
    
    async def can_apply(self, context: SkillContext) -> bool:
        """Can apply if entities exist."""
        return len(context.entities) > 0
    
    def _build_index(self, entities: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Build lookup index from entities.
        
        Index structure:
        - name -> entities (normalized)
        - type -> entities
        - id -> entity (exact match)
        """
        index: Dict[str, List[Dict[str, Any]]] = {
            "by_name": {},
            "by_type": {},
            "by_id": {},
        }
        
        for entity in entities:
            entity_id = entity.get("entity_id") or entity.get("id", "")
            name = entity.get("name", "").lower().strip()
            entity_type = entity.get("entity_type", "unknown")
            
            # Index by ID
            if entity_id:
                index["by_id"][entity_id] = entity
            
            # Index by name
            if name:
                if name not in index["by_name"]:
                    index["by_name"][name] = []
                index["by_name"][name].append(entity)
                
                # Also index by normalized variations
                variations = self._generate_name_variations(name)
                for variant in variations:
                    if variant not in index["by_name"]:
                        index["by_name"][variant] = []
                    if entity not in index["by_name"][variant]:
                        index["by_name"][variant].append(entity)
            
            # Index by type
            if entity_type not in index["by_type"]:
                index["by_type"][entity_type] = []
            index["by_type"][entity_type].append(entity)
        
        self._index = index
        logger.info(f"Built index: {len(index['by_id'])} by_id, {len(index['by_name'])} by_name, {len(index['by_type'])} by_type")
        return index
    
    def _generate_name_variations(self, name: str) -> List[str]:
        """Generate name variations for fuzzy matching."""
        variations = []
        
        # Remove special characters
        cleaned = "".join(c for c in name if c.isalnum() or c.isspace())
        variations.append(cleaned.lower().strip())
        
        # Remove spaces
        no_space = cleaned.replace(" ", "")
        variations.append(no_space.lower())
        
        # Common abbreviations
        if "bom" in name:
            variations.append("bill of materials")
        if "qty" in name:
            variations.append("quantity")
        
        return list(set(variations))
    
    def _lookup(self, key: str, index_type: str = "by_name") -> List[Dict[str, Any]]:
        """Lookup entities by key in index."""
        key_normalized = key.lower().strip()
        
        # Check cache first
        cache_key = f"{index_type}:{key_normalized}"
        if self.enable_caching and cache_key in self._cache:
            result, timestamp = self._cache[cache_key]
            return result
        
        # Query index
        index = getattr(self._index, index_type, self._index.get(index_type, {}))
        
        if index_type == "by_name":
            result = index.get(key_normalized, [])
            # Also check variations
            variations = self._generate_name_variations(key_normalized)
            for variant in variations:
                if variant in index:
                    for entity in index[variant]:
                        if entity not in result:
                            result.append(entity)
        else:
            result = index.get(key_normalized, [])
        
        # Cache result
        if self.enable_caching:
            self._cache[cache_key] = (result, 0)
        
        return result
    
    def _batch_lookup(self, keys: List[str], index_type: str = "by_name") -> Dict[str, List[Dict[str, Any]]]:
        """Batch lookup multiple keys."""
        results = {}
        for key in keys:
            results[key] = self._lookup(key, index_type)
        return results
    
    def _associate_entities(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Associate entities and expand relationships.
        
        Returns:
            (updated_entities, new_relationships)
        """
        updated_entities = []
        new_relationships = []
        seen_relations: Set[str] = set()
        
        # Build entity lookup map
        entity_map: Dict[str, Dict[str, Any]] = {}
        for entity in entities:
            entity_id = entity.get("entity_id") or entity.get("id", "")
            if entity_id:
                entity_map[entity_id] = entity
            name = entity.get("name", "").lower()
            if name:
                entity_map[name] = entity
        
        # Process existing relationships
        for rel in relationships:
            rel_id = rel.get("relation_id") or f"{rel.get('source')}-{rel.get('target')}"
            if rel_id in seen_relations:
                continue
            seen_relations.add(rel_id)
            
            # Expand source/target references
            source = rel.get("source", "")
            target = rel.get("target", "")
            
            # Resolve source entity
            if source in entity_map:
                source_entity = entity_map[source]
                if "linked_entities" not in source_entity:
                    source_entity["linked_entities"] = []
                if target not in source_entity["linked_entities"]:
                    source_entity["linked_entities"].append(target)
            
            # Resolve target entity
            if target in entity_map:
                target_entity = entity_map[target]
                if "linked_by" not in target_entity:
                    target_entity["linked_by"] = []
                if source not in target_entity["linked_by"]:
                    target_entity["linked_by"].append(source)
            
            updated_entities.extend([entity_map.get(source, {}), entity_map.get(target, {})])
        
        # Batch find associations
        for entity in entities:
            name = entity.get("name", "").lower()
            entity_type = entity.get("entity_type", "")
            
            # Find related by name similarity
            candidates = self._lookup(name, "by_name")
            for candidate in candidates[:5]:  # Limit to top 5
                if candidate.get("entity_id") != entity.get("entity_id"):
                    rel = {
                        "relation_id": f"auto-{entity.get('entity_id')}-{candidate.get('entity_id')}",
                        "source": entity.get("entity_id"),
                        "target": candidate.get("entity_id"),
                        "relation_type": "similar_name",
                        "confidence": 0.7
                    }
                    rel_id = rel["relation_id"]
                    if rel_id not in seen_relations:
                        seen_relations.add(rel_id)
                        new_relationships.append(rel)
        
        # Deduplicate
        updated_entities = list({e.get("entity_id"): e for e in updated_entities if e.get("entity_id")}.values())
        
        return list(updated_entities), new_relationships
    
    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute batch association optimization."""
        # Build index from current entities
        self._build_index(context.entities)
        
        # Track original counts
        original_entity_count = len(context.entities)
        original_rel_count = len(context.relationships)
        
        # Perform batch associations
        updated_entities, new_relationships = self._associate_entities(
            context.entities,
            context.relationships
        )
        
        # Update context
        context.entities = updated_entities
        context.relationships.extend(new_relationships)
        
        # Update index with new entities
        if new_relationships:
            new_entities = [e for e in updated_entities if e.get("entity_id") not in [orig.get("entity_id") for orig in context.entities]]
            for entity in new_entities:
                entity_id = entity.get("entity_id") or entity.get("id", "")
                name = entity.get("name", "").lower()
                if entity_id:
                    self._index["by_id"][entity_id] = entity
                if name:
                    if name not in self._index["by_name"]:
                        self._index["by_name"][name] = []
                    self._index["by_name"][name].append(entity)
        
        new_rel_count = len(new_relationships)
        total_entities = len(updated_entities)
        
        return SkillResult(
            success=True,
            skill_name=self.name,
            applied=new_rel_count > 0,
            output={
                "original_entities": original_entity_count,
                "total_entities": total_entities,
                "original_relationships": original_rel_count,
                "new_relationships": new_rel_count,
                "index_size": len(self._index["by_id"]),
                "cache_size": len(self._cache),
                "action": "batch_associated" if new_rel_count > 0 else "none"
            }
        )
    
    def clear_cache(self) -> None:
        """Clear lookup cache."""
        self._cache.clear()
        logger.info("Cleared batch association cache")
    
    def clear_index(self) -> None:
        """Clear lookup index."""
        self._index = {"by_name": {}, "by_type": {}, "by_id": {}}
        self.clear_cache()
        logger.info("Cleared batch association index")
