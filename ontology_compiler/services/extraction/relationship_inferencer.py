from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RelationshipInferencer:
    """
    Infer relationships between entities based on their classification, domain,
    and textual context.
    """
    
    RELATIONSHIP_PATTERNS = [
        {"source_class": "Capability", "target_class": "System", "relationship": "IMPLEMENTS"},
        {"source_class": "System", "target_class": "Capability", "relationship": "IMPLEMENTS"},
        {"source_class": "Process", "target_class": "Capability", "relationship": "IMPLEMENTS"},
        {"source_class": "System", "target_class": "Data", "relationship": "PRODUCES"},
        {"source_class": "Process", "target_class": "Data", "relationship": "PRODUCES"},
        {"source_class": "System", "target_class": "Data", "relationship": "CONSUMES"},
        {"source_class": "Process", "target_class": "Data", "relationship": "CONSUMES"},
        {"source_class": "Actor", "target_class": "System", "relationship": "USES"},
        {"source_class": "Actor", "target_class": "Asset", "relationship": "MANAGES"},
        {"source_class": "Actor", "target_class": "Asset", "relationship": "OWNS"},
        {"source_class": "Capability", "target_class": "Process", "relationship": "CONTAINS"},
        {"source_class": "System", "target_class": "System", "relationship": "DEPENDS_ON"},
        {"source_class": "System", "target_class": "Asset", "relationship": "USES"},
        {"source_class": "Actor", "target_class": "Process", "relationship": "TRIGGERS"},
        {"source_class": "System", "target_class": "Asset", "relationship": "STORES"},
        {"source_class": "Process", "target_class": "Data", "relationship": "TRANSFORMS"},
        {"source_class": "Actor", "target_class": "Capability", "relationship": "GOVERNS"},
        {"source_class": "Process", "target_class": "Data", "relationship": "PROCESSES"},
    ]
    
    CORE_CLASS_MAP = {
        "Capability": "Capability",
        "System": "System",
        "Actor": "Actor",
        "Asset": "Asset",
        "Data": "Data",
        "Process": "Process",
        "Enterprise_Platform": "System",
        "Application_System": "System",
        "Database_System": "System",
        "API_Gateway": "System",
        "Data_Warehouse": "System",
        "Data_Lake": "System",
        "API_Endpoint": "System",
        "Business_Process": "Process",
        "Workflow": "Process",
        "Business_Rule": "Process",
        "Business_Capability": "Capability",
        "IT_Capability": "Capability",
        "Organization": "Actor",
        "Person": "Actor",
        "Role": "Actor",
        "Equipment": "Asset",
        "Facility": "Asset",
        "Document": "Data",
        "Dataset": "Data",
        "Message": "Data",
        "Event": "Data",
    }
    
    def __init__(self):
        self.relation_id_counter = 1
    
    def _get_core_class(self, meta_class: str) -> str:
        """Map a meta_class to its core class."""
        return self.CORE_CLASS_MAP.get(meta_class, meta_class.split('_')[0] if '_' in meta_class else meta_class)
    
    def _extract_entity_info(self, entity: Dict) -> Dict:
        """Extract key information from an entity."""
        return {
            "meta_id": entity.get("meta_id", ""),
            "name": entity.get("name", entity.get("meta_name", "")),
            "meta_class": entity.get("meta_class", ""),
            "core_class": self._get_core_class(entity.get("meta_class", "")),
            "domain": entity.get("domain", ""),
            "parent_id": entity.get("parent_id", ""),
            "description": entity.get("description", ""),
        }
    
    def infer_relationships(self, entities: List[Dict], existing_relationships: List[Dict] = None) -> List[Dict]:
        """
        Infer new relationships based on entity classification and domain.
        
        Args:
            entities: List of extracted entities
            existing_relationships: Existing relationships to avoid duplicates
            
        Returns:
            List of inferred relationships
        """
        if existing_relationships is None:
            existing_relationships = []
        
        entity_info = [self._extract_entity_info(e) for e in entities if isinstance(e, dict)]
        inferred_relationships = []
        
        seen_pairs = set()
        for rel in existing_relationships:
            from_id = rel.get("from", rel.get("source", ""))
            to_id = rel.get("to", rel.get("target", ""))
            rel_type = rel.get("type", "")
            seen_pairs.add((str(from_id), str(to_id), str(rel_type)))
        
        for i, src_entity in enumerate(entity_info):
            for j, tgt_entity in enumerate(entity_info):
                if i == j:
                    continue
                
                src_core = src_entity["core_class"]
                tgt_core = tgt_entity["core_class"]
                src_id = src_entity["meta_id"]
                tgt_id = tgt_entity["meta_id"]
                
                if not src_id or not tgt_id:
                    continue
                
                same_domain = src_entity["domain"] and src_entity["domain"] == tgt_entity["domain"]
                
                for pattern in self.RELATIONSHIP_PATTERNS:
                    if (pattern["source_class"] == src_core and pattern["target_class"] == tgt_core) or \
                       (same_domain and pattern["source_class"] == tgt_core and pattern["target_class"] == src_core):
                        
                        rel_type = pattern["relationship"]
                        if (src_id, tgt_id, rel_type) in seen_pairs:
                            continue
                        
                        inferred_rel = {
                            "from": src_id,
                            "to": tgt_id,
                            "type": rel_type,
                            "description": f"{src_entity['name']} {rel_type} {tgt_entity['name']}",
                            "evidence_ref": "Inferred from entity classification",
                            "source_file": "relationship_inferencer",
                            "citation": "Classification-based inference",
                            "confidence": 0.70 if same_domain else 0.50,
                        }
                        
                        inferred_relationships.append(inferred_rel)
                        seen_pairs.add((src_id, tgt_id, rel_type))
        
        for entity in entity_info:
            parent_id = entity.get("parent_id", "")
            if parent_id and entity["meta_id"]:
                if (entity["meta_id"], parent_id, "CONTAINS") not in seen_pairs:
                    inferred_rel = {
                        "from": parent_id,
                        "to": entity["meta_id"],
                        "type": "CONTAINS",
                        "description": f"Parent contains child entity",
                        "evidence_ref": "Inferred from parent_id field",
                        "source_file": "relationship_inferencer",
                        "citation": "Parent-child relationship",
                        "confidence": 0.95,
                    }
                    inferred_relationships.append(inferred_rel)
        
        logger.info(f"Inferred {len(inferred_relationships)} new relationships from {len(entities)} entities")
        return inferred_relationships
    
    def enrich_with_context(self, entities: List[Dict], text: str, existing_relationships: List[Dict] = None) -> List[Dict]:
        """
        Enrich relationships by analyzing textual context for co-occurring entities.
        
        Args:
            entities: List of extracted entities
            text: Original document text
            existing_relationships: Existing relationships
            
        Returns:
            List of enriched relationships
        """
        if existing_relationships is None:
            existing_relationships = []
        
        entity_info = [self._extract_entity_info(e) for e in entities if isinstance(e, dict)]
        enriched_relationships = []
        
        seen_pairs = set()
        for rel in existing_relationships:
            from_id = rel.get("from", rel.get("source", ""))
            to_id = rel.get("to", rel.get("target", ""))
            rel_type = rel.get("type", "")
            seen_pairs.add((str(from_id), str(to_id), str(rel_type)))
        
        for i, src_entity in enumerate(entity_info):
            for j, tgt_entity in enumerate(entity_info):
                if i == j:
                    continue
                
                src_name = src_entity["name"].lower()
                tgt_name = tgt_entity["name"].lower()
                
                if src_name and tgt_name and src_name in text.lower() and tgt_name in text.lower():
                    paragraphs = text.split('\n\n')
                    for para in paragraphs:
                        if src_name in para.lower() and tgt_name in para.lower():
                            rel_type = self._infer_relation_from_text(para, src_name, tgt_name)
                            src_id = src_entity["meta_id"]
                            tgt_id = tgt_entity["meta_id"]
                            
                            if rel_type and (src_id, tgt_id, rel_type) not in seen_pairs:
                                enriched_relationships.append({
                                    "from": src_id,
                                    "to": tgt_id,
                                    "type": rel_type,
                                    "description": f"{src_entity['name']} {rel_type} {tgt_entity['name']}",
                                    "evidence_ref": para.strip()[:200],
                                    "source_file": "relationship_inferencer",
                                    "citation": "Contextual co-occurrence",
                                    "confidence": 0.85,
                                })
                                seen_pairs.add((src_id, tgt_id, rel_type))
        
        logger.info(f"Enriched {len(enriched_relationships)} relationships from context analysis")
        return enriched_relationships
    
    def _infer_relation_from_text(self, text: str, name1: str, name2: str) -> Optional[str]:
        """Infer relationship type from text between two entities."""
        text_lower = text.lower()
        
        if "produce" in text_lower or "generate" in text_lower:
            if name1 in text_lower and name2 in text_lower:
                return "PRODUCES"
        if "consume" in text_lower or "use" in text_lower or "utilize" in text_lower:
            if name1 in text_lower and name2 in text_lower:
                return "CONSUMES"
        if "implement" in text_lower or "realize" in text_lower:
            return "IMPLEMENTS"
        if "trigger" in text_lower or "activate" in text_lower or "initiate" in text_lower:
            return "TRIGGERS"
        if "depend" in text_lower or "require" in text_lower:
            return "DEPENDS_ON"
        if "contain" in text_lower or "include" in text_lower:
            return "CONTAINS"
        if "manage" in text_lower or "operate" in text_lower:
            return "MANAGES"
        if "own" in text_lower or "possess" in text_lower:
            return "OWNS"
        if "interact" in text_lower or "communicate" in text_lower:
            return "INTERACTS_WITH"
        if "store" in text_lower or "persist" in text_lower:
            return "STORES"
        if "transform" in text_lower or "convert" in text_lower:
            return "TRANSFORMS"
        if "govern" in text_lower or "control" in text_lower:
            return "GOVERNS"
        if "process" in text_lower or "handle" in text_lower:
            return "PROCESSES"
        
        return None
