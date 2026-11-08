"""
Ontology Context Loader Service - 统一上下文加载器

提供两类上下文加载能力：
1. MasterOntologyContext - 加载结构化本体数据（L1/L2/L2.5 实体），用于编译流程
2. AnalysisContextLoader - 加载非结构化分析文档块（md/json/fragments），用于数据分析

通过 ContextFactory 提供统一入口，避免命名冲突。

[Phase 6-PERF-1] Performance Enhancements:
- File modification time tracking for cache invalidation
- Memory cache with TTL (time-to-live) support
- Batch loading optimization for large ontology files

[Phase 3 - Vectorization] Entity vectorization and similarity search:
- TF-IDF based vectorization for local embedding
- Vector similarity search for efficient entity matching
- Reduced token consumption in LLM prompts
"""

import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class LoadedOntologyContext:
    """Container for loaded ontology context."""
    l1_entities: List[Dict[str, Any]] = field(default_factory=list)
    l2_entities: List[Dict[str, Any]] = field(default_factory=list)
    l25_entities: List[Dict[str, Any]] = field(default_factory=list)
    all_entities: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    entity_index: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # meta_id -> entity
    name_index: Dict[str, List[Dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    domain_entities: Dict[str, List[Dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    source_file: Optional[Path] = None
    total_count: int = 0
    
    entity_vectors: Dict[str, List[float]] = field(default_factory=dict)  # meta_id -> vector
    vector_index: List[Tuple[str, List[float]]] = field(default_factory=list)  # (meta_id, vector)
    tfidf_vocab: Dict[str, int] = field(default_factory=dict)
    idf_scores: Dict[str, float] = field(default_factory=dict)


class MasterOntologyContext:
    """
    Service for loading and indexing existing master ontology files.
    
    This service enables:
    1. Loading existing L1/L2/L2.5 entities from master ontology files
    2. Building entity indices for fast lookup
    3. Providing context for new extractions to avoid duplication
    4. Supporting cross-file entity relationship discovery
    """
    
    DEFAULT_MASTER_PATHS = [
        "outputs/runs/ontology/master_ontology",
        "outputs/runs/ontology/master_ontology_l2.json",
        "outputs/runs/ontology/master_ontology_v2.json",
        "outputs/runs/ontology/master_ontology_unified.json",
        "outputs/runs/ontology/runs/*/master_ontology_l2.json",
    ]
    
    # [Phase 6-PERF-1] Class-level cache shared across instances
    _global_cache: Optional[LoadedOntologyContext] = None
    _global_cache_path: Optional[Path] = None
    _global_cache_mtime: float = 0.0
    _global_cache_ttl: float = 300.0  # 5 minutes TTL
    _global_cache_timestamp: float = 0.0
    
    # [MONITOR-1] Cache hit rate statistics (class-level)
    _cache_stats: Dict[str, int] = {
        "instance_hits": 0,
        "instance_misses": 0,
        "global_hits": 0,
        "global_misses": 0,
        "total_loads": 0,
    }
    
    def __init__(self, custom_paths: Optional[List[str]] = None, cache_ttl: float = 300.0,
                enable_vectorization: bool = True, vector_dimension: int = 384):
        """
        Initialize the context loader.
        
        Args:
            custom_paths: Optional list of custom paths to master ontology files
            cache_ttl: Cache time-to-live in seconds (default: 300s = 5min)
            enable_vectorization: Enable entity vectorization for similarity search
            vector_dimension: Dimension of entity vectors (default: 384)
        """
        self.custom_paths = custom_paths or []
        self._context_cache: Optional[LoadedOntologyContext] = None
        self._last_loaded_path: Optional[Path] = None
        self._cache_ttl = cache_ttl
        self._cache_timestamp: float = 0.0
        self._cache_mtime: float = 0.0
        self._instance_stats: Dict[str, int] = {
            "hits": 0,
            "misses": 0,
        }
        self.enable_vectorization = enable_vectorization
        self.vector_dimension = vector_dimension
    
    def load_master_ontology(
        self, 
        preferred_path: Optional[Path] = None
    ) -> LoadedOntologyContext:
        """
        Load master ontology file and build indices.
        
        Args:
            preferred_path: Optional specific path to load
            
        Returns:
            LoadedOntologyContext with all entities and indices
        """
        # Try preferred path first
        if preferred_path and preferred_path.exists():
            return self._load_from_file(preferred_path)
        
        # Try custom paths
        for path_str in self.custom_paths:
            path = Path(path_str)
            if path.exists():
                return self._load_from_file(path)
        
        # Try default paths
        for path_pattern in self.DEFAULT_MASTER_PATHS:
            if "*" in path_pattern:
                matches = list(Path().glob(path_pattern))
                if matches:
                    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    return self._load_from_file(matches[0])
            else:
                path = Path(path_pattern)
                if path.exists():
                    return self._load_from_file(path)
        
        # Return empty context if nothing found
        logger.warning("No master ontology file found, returning empty context")
        return LoadedOntologyContext()
    
    def _load_from_file(self, path: Path) -> LoadedOntologyContext:
        """Load and parse a master ontology file or directory with caching."""
        current_time = time.time()
        
        is_directory = path.is_dir()
        cache_key_path = path
        
        try:
            if is_directory:
                latest_mtime = max(p.stat().st_mtime for p in path.iterdir())
                current_mtime = latest_mtime
            else:
                current_mtime = path.stat().st_mtime
        except Exception:
            current_mtime = 0.0
        
        # Check instance cache
        if (self._context_cache is not None and 
            self._last_loaded_path == cache_key_path and
            self._cache_mtime == current_mtime and
            (current_time - self._cache_timestamp) < self._cache_ttl):
            self._instance_stats["hits"] += 1
            MasterOntologyContext._cache_stats["instance_hits"] += 1
            logger.debug(f"Using instance cache for {path} (age: {current_time - self._cache_timestamp:.1f}s)")
            return self._context_cache
        
        # Check global cache
        if (MasterOntologyContext._global_cache is not None and
            MasterOntologyContext._global_cache_path == cache_key_path and
            MasterOntologyContext._global_cache_mtime == current_mtime and
            (current_time - MasterOntologyContext._global_cache_timestamp) < MasterOntologyContext._global_cache_ttl):
            self._instance_stats["misses"] += 1
            MasterOntologyContext._cache_stats["global_hits"] += 1
            logger.debug(f"Using global cache for {path} (age: {current_time - MasterOntologyContext._global_cache_timestamp:.1f}s)")
            self._context_cache = MasterOntologyContext._global_cache
            self._last_loaded_path = cache_key_path
            self._cache_mtime = current_mtime
            self._cache_timestamp = current_time
            return self._context_cache
        
        self._instance_stats["misses"] += 1
        MasterOntologyContext._cache_stats["global_misses"] += 1
        MasterOntologyContext._cache_stats["total_loads"] += 1
        logger.info(f"Loading master ontology from: {path}")
        
        try:
            if is_directory:
                data = self._load_from_directory(path)
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load master ontology: {e}")
            return LoadedOntologyContext()
        
        context = LoadedOntologyContext(source_file=path)
        
        # Extract entities from different possible structures
        if isinstance(data, dict):
            # Structure 1: {"L1": [...], "L2": [...], "L2.5": [...]}
            context.l1_entities = data.get("L1") or data.get("l1") or data.get("L1_entities") or []
            context.l2_entities = data.get("L2") or data.get("l2") or data.get("L2_entities") or data.get("business_scenarios") or []
            context.l25_entities = data.get("L2.5") or data.get("l25") or data.get("L2_5_entities") or data.get("mapping_rules") or []
            
            # Structure 2: {"entities": {"L1": [...], "L2": [...]}}
            if "entities" in data and isinstance(data["entities"], dict):
                entities = data["entities"]
                context.l1_entities = entities.get("L1") or context.l1_entities
                context.l2_entities = entities.get("L2") or context.l2_entities
                context.l25_entities = entities.get("L2.5") or context.l25_entities
            
            # Structure 3: {"master_ontology": {...}}
            if "master_ontology" in data:
                mo = data["master_ontology"]
                context.l1_entities = mo.get("L1") or mo.get("l1_entities") or context.l1_entities
                context.l2_entities = mo.get("L2") or mo.get("l2_entities") or mo.get("business_scenarios") or context.l2_entities
                context.l25_entities = mo.get("L2.5") or mo.get("l25_entities") or context.l25_entities
        elif isinstance(data, list):
            # Assume it's a list of entities, categorize by meta_class
            for entity in data:
                meta_class = entity.get("meta_class", "").upper()
                if "L1" in meta_class or meta_class in ["CONCEPT", "TERM", "DEFINITION"]:
                    context.l1_entities.append(entity)
                elif "L2" in meta_class or meta_class in ["BUSINESS_SCENARIO", "SCENARIO", "PROCESS"]:
                    context.l2_entities.append(entity)
                elif "L2.5" in meta_class or meta_class in ["MAPPING", "RULE", "CONSTRAINT"]:
                    context.l25_entities.append(entity)
                else:
                    # Default to L2
                    context.l2_entities.append(entity)
        
        # Build unified index
        context.all_entities = {
            "L1": context.l1_entities,
            "L2": context.l2_entities,
            "L2.5": context.l25_entities,
        }
        
        # Build entity indices
        for layer, entities in context.all_entities.items():
            for entity in entities:
                self._index_entity(entity, context)
        
        context.total_count = len(context.l1_entities) + len(context.l2_entities) + len(context.l25_entities)
        
        if self.enable_vectorization and context.total_count > 0:
            self._build_vector_index(context)
        
        logger.info(
            f"Loaded {context.total_count} entities: "
            f"L1={len(context.l1_entities)}, "
            f"L2={len(context.l2_entities)}, "
            f"L2.5={len(context.l25_entities)}"
            f"{' (vectorized)' if self.enable_vectorization else ''}"
        )
        
        # [Phase 6-PERF-1] Update caches
        current_time = time.time()
        self._context_cache = context
        self._last_loaded_path = path
        self._cache_mtime = current_mtime
        self._cache_timestamp = current_time
        
        # Update global cache
        MasterOntologyContext._global_cache = context
        MasterOntologyContext._global_cache_path = path
        MasterOntologyContext._global_cache_mtime = current_mtime
        MasterOntologyContext._global_cache_timestamp = current_time
        
        return context
    
    def _index_entity(self, entity: Dict[str, Any], context: LoadedOntologyContext):
        """Index a single entity for fast lookup."""
        meta_id = entity.get("meta_id")
        if meta_id:
            context.entity_index[meta_id] = entity
        
        # Index by name
        names = []
        if "name" in entity:
            names.append(entity["name"])
        if "scenario_name" in entity:
            names.append(entity["scenario_name"])
        if "meta_name" in entity:
            names.append(entity["meta_name"])
        
        for name in names:
            if name:
                context.name_index[name.lower()].append(entity)
        
        # Index by domain
        domain = entity.get("domain", entity.get("meta_domain", "UNKNOWN"))
        context.domain_entities[domain].append(entity)
    
    def _generate_entity_text(self, entity: Dict[str, Any]) -> str:
        """Generate text representation of an entity for vectorization."""
        parts = []
        
        for key in ["name", "scenario_name", "meta_name", "description"]:
            value = entity.get(key)
            if value and isinstance(value, str):
                parts.append(value)
        
        for key in ["meta_class", "domain", "layer"]:
            value = entity.get(key)
            if value and isinstance(value, str):
                parts.append(value)
        
        attributes = entity.get("attributes", entity.get("properties", {}))
        if isinstance(attributes, dict):
            for k, v in attributes.items():
                if isinstance(v, str) and v:
                    parts.append(f"{k}: {v}")
        
        return " ".join(parts)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff\s]', ' ', text)
        tokens = text.split()
        stop_words = {"the", "and", "or", "is", "are", "be", "of", "in", "for", "on", "with", "at", "to", "from"}
        return [t for t in tokens if t not in stop_words and len(t) > 2]
    
    def _build_vector_index(self, context: LoadedOntologyContext):
        """
        Build TF-IDF based vector index for all entities.
        
        Creates:
        - tfidf_vocab: word -> index mapping
        - idf_scores: word -> IDF score
        - entity_vectors: meta_id -> TF-IDF vector
        - vector_index: list of (meta_id, vector) tuples
        """
        all_texts: Dict[str, List[str]] = {}
        all_tokens: Set[str] = set()
        
        for layer, entities in context.all_entities.items():
            for entity in entities:
                meta_id = entity.get("meta_id")
                if meta_id:
                    text = self._generate_entity_text(entity)
                    tokens = self._tokenize(text)
                    if tokens:
                        all_texts[meta_id] = tokens
                        all_tokens.update(tokens)
        
        if not all_texts:
            return
        
        vocab = sorted(all_tokens)
        vocab_size = min(len(vocab), self.vector_dimension)
        context.tfidf_vocab = {word: idx for idx, word in enumerate(vocab[:vocab_size])}
        
        num_docs = len(all_texts)
        doc_freq: Dict[str, int] = defaultdict(int)
        
        for meta_id, tokens in all_texts.items():
            unique_tokens = set(tokens)
            for token in unique_tokens:
                if token in context.tfidf_vocab:
                    doc_freq[token] += 1
        
        for token, freq in doc_freq.items():
            context.idf_scores[token] = math.log((num_docs + 1) / (freq + 1)) + 1
        
        for meta_id, tokens in all_texts.items():
            vector = [0.0] * vocab_size
            token_counts: Dict[str, int] = defaultdict(int)
            for token in tokens:
                token_counts[token] += 1
            
            max_count = max(token_counts.values()) if token_counts else 1
            for token, count in token_counts.items():
                if token in context.tfidf_vocab:
                    idx = context.tfidf_vocab[token]
                    tf = 0.5 + 0.5 * (count / max_count)
                    idf = context.idf_scores.get(token, 1.0)
                    vector[idx] = tf * idf
            
            norm = math.sqrt(sum(v * v for v in vector))
            if norm > 0:
                vector = [v / norm for v in vector]
            
            context.entity_vectors[meta_id] = vector
            context.vector_index.append((meta_id, vector))
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def _text_to_vector(self, text: str, context: LoadedOntologyContext) -> List[float]:
        """Convert text to TF-IDF vector using existing vocabulary."""
        tokens = self._tokenize(text)
        vector = [0.0] * len(context.tfidf_vocab)
        
        if not tokens or not context.tfidf_vocab:
            return vector
        
        token_counts: Dict[str, int] = defaultdict(int)
        for token in tokens:
            token_counts[token] += 1
        
        max_count = max(token_counts.values()) if token_counts else 1
        for token, count in token_counts.items():
            if token in context.tfidf_vocab:
                idx = context.tfidf_vocab[token]
                tf = 0.5 + 0.5 * (count / max_count)
                idf = context.idf_scores.get(token, 1.0)
                vector[idx] = tf * idf
        
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        
        return vector
    
    def find_similar_entities(
        self, 
        query: str, 
        top_k: int = 10, 
        threshold: float = 0.3,
        layer: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find similar entities by text similarity using vector search.
        
        Args:
            query: Query text to search for similar entities
            top_k: Maximum number of results to return
            threshold: Minimum similarity threshold
            layer: Optional layer filter ("L1", "L2", "L2.5")
            
        Returns:
            List of similar entities with similarity scores
        """
        context = self._get_context()
        
        if not context.entity_vectors or not context.tfidf_vocab:
            return []
        
        query_vector = self._text_to_vector(query, context)
        
        results: List[Tuple[float, str]] = []
        for meta_id, vector in context.vector_index:
            if layer:
                entity = context.entity_index.get(meta_id)
                if not entity or entity.get("layer") != layer:
                    continue
            
            similarity = self._cosine_similarity(query_vector, vector)
            if similarity >= threshold:
                results.append((similarity, meta_id))
        
        results.sort(key=lambda x: -x[0])
        
        similar_entities = []
        for similarity, meta_id in results[:top_k]:
            entity = context.entity_index.get(meta_id)
            if entity:
                similar_entities.append({
                    "meta_id": meta_id,
                    "name": entity.get("name", entity.get("scenario_name", entity.get("meta_name", "Unknown"))),
                    "meta_class": entity.get("meta_class"),
                    "domain": entity.get("domain", entity.get("meta_domain")),
                    "layer": entity.get("layer"),
                    "similarity": round(similarity, 4),
                })
        
        return similar_entities
    
    def find_related_entities(
        self, 
        entity_id: str, 
        top_k: int = 10, 
        threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Find related entities by vector similarity to a given entity.
        
        Args:
            entity_id: Entity meta_id to find related entities for
            top_k: Maximum number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of related entities with similarity scores
        """
        context = self._get_context()
        
        if not context.entity_vectors or entity_id not in context.entity_vectors:
            return []
        
        entity_vector = context.entity_vectors[entity_id]
        
        results: List[Tuple[float, str]] = []
        for meta_id, vector in context.vector_index:
            if meta_id == entity_id:
                continue
            
            similarity = self._cosine_similarity(entity_vector, vector)
            if similarity >= threshold:
                results.append((similarity, meta_id))
        
        results.sort(key=lambda x: -x[0])
        
        related_entities = []
        for similarity, meta_id in results[:top_k]:
            entity = context.entity_index.get(meta_id)
            if entity:
                related_entities.append({
                    "meta_id": meta_id,
                    "name": entity.get("name", entity.get("scenario_name", entity.get("meta_name", "Unknown"))),
                    "meta_class": entity.get("meta_class"),
                    "domain": entity.get("domain", entity.get("meta_domain")),
                    "layer": entity.get("layer"),
                    "similarity": round(similarity, 4),
                })
        
        return related_entities
    
    def get_vectorized_context_for_prompt(self, max_entities: int = 20) -> str:
        """
        Generate vector-aware context for LLM prompt, prioritizing relevant entities.
        
        Args:
            max_entities: Maximum number of entities to include
            
        Returns:
            Formatted string for prompt context
        """
        context = self._get_context()
        
        if context.total_count == 0:
            return "No existing ontology context available."
        
        lines = [f"## Existing Ontology Context ({context.total_count} total entities)"]
        lines.append("")
        
        entities_to_include = []
        for layer_name, entities in [
            ("L1 (Core Concepts)", context.l1_entities),
            ("L2 (Business Scenarios)", context.l2_entities),
            ("L2.5 (Mapping Rules)", context.l25_entities),
        ]:
            if entities:
                sample_size = max_entities // 3
                entities_to_include.extend(entities[:sample_size])
        
        for entity in entities_to_include:
            name = entity.get("name", entity.get("scenario_name", entity.get("meta_name", "Unknown")))
            meta_id = entity.get("meta_id", "N/A")
            domain = entity.get("domain", entity.get("meta_domain", "Unknown"))
            meta_class = entity.get("meta_class", "Unknown")
            lines.append(f"  - [{meta_id}] {name} ({meta_class}, {domain})")
        
        lines.append("")
        lines.append("### Instructions:")
        lines.append("- DO NOT extract entities that already exist above")
        lines.append("- DO identify relationships to existing entities")
        lines.append("- DO extract NEW business scenarios not listed here")
        
        return "\n".join(lines)
    
    def _load_from_directory(self, dir_path: Path) -> Dict[str, Any]:
        """Load master ontology from directory format."""
        data: Dict[str, Any] = {}
        
        layer_files = {
            "nodes.json": "nodes",
            "instances.json": "instances",
            "relationships.json": "relationships",
            "layer0_metadata.json": "layer0_metadata",
            "layer1_concepts.json": "layer1_concepts",
            "layer2_scenarios.json": "layer2_scenarios",
            "layer25_mapping_rules.json": "layer25_mapping_rules",
            "metadata.json": "metadata",
        }
        
        for file_name, field_name in layer_files.items():
            file_path = dir_path / file_name
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data[field_name] = json.load(f)
                except Exception:
                    data[field_name] = [] if 'layer' in field_name or field_name in ['nodes', 'instances', 'relationships'] else {}
            else:
                data[field_name] = [] if 'layer' in field_name or field_name in ['nodes', 'instances', 'relationships'] else {}
        
        nodes = data.get("nodes", [])
        data["entities"] = {
            "L0": data.get("layer0_metadata", []),
            "L1": data.get("layer1_concepts", []),
            "L2": data.get("layer2_scenarios", []),
            "L2.5": data.get("layer25_mapping_rules", []),
            "L3": data.get("instances", []),
        }
        
        return data
    
    def get_existing_entities(
        self, 
        domain: Optional[str] = None,
        layer: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get existing entities, optionally filtered by domain and layer.
        
        Args:
            domain: Optional domain filter (e.g., "SWDL", "BOM")
            layer: Optional layer filter ("L1", "L2", "L2.5")
            
        Returns:
            List of matching entities
        """
        context = self._get_context()
        
        if domain and layer:
            # Filter by both
            entities = context.all_entities.get(layer, [])
            return [e for e in entities if e.get("domain", e.get("meta_domain")) == domain]
        elif domain:
            # Filter by domain only
            return context.domain_entities.get(domain, [])
        elif layer:
            # Filter by layer only
            return context.all_entities.get(layer, [])
        else:
            # Return all
            all_entities = []
            for layer_entities in context.all_entities.values():
                all_entities.extend(layer_entities)
            return all_entities
    
    def find_by_name(self, name: str) -> List[Dict[str, Any]]:
        """
        Find entities by name (case-insensitive).
        
        Args:
            name: Entity name to search for
            
        Returns:
            List of matching entities
        """
        context = self._get_context()
        return context.name_index.get(name.lower(), [])
    
    def find_by_meta_id(self, meta_id: str) -> Optional[Dict[str, Any]]:
        """
        Find entity by meta_id.
        
        Args:
            meta_id: The meta_id to look up
            
        Returns:
            Entity dict or None
        """
        context = self._get_context()
        return context.entity_index.get(meta_id)
    
    def check_exists(self, entity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check if an entity already exists in master ontology.
        
        Args:
            entity: Entity dict with at least meta_id or name
            
        Returns:
            Existing entity if found, None otherwise
        """
        context = self._get_context()
        
        # Check by meta_id first
        meta_id = entity.get("meta_id")
        if meta_id and meta_id in context.entity_index:
            return context.entity_index[meta_id]
        
        # Check by name
        names = []
        if "name" in entity:
            names.append(entity["name"])
        if "scenario_name" in entity:
            names.append(entity["scenario_name"])
        
        for name in names:
            existing = context.name_index.get(name.lower(), [])
            if existing:
                return existing[0]  # Return first match
        
        return None
    
    def get_domain_summary(self) -> Dict[str, int]:
        """
        Get summary of entities by domain.
        
        Returns:
            Dict mapping domain to entity count
        """
        context = self._get_context()
        return {domain: len(entities) for domain, entities in context.domain_entities.items()}
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about loaded ontology.
        
        Returns:
            Statistics dict
        """
        context = self._get_context()
        
        return {
            "total_entities": context.total_count,
            "l1_count": len(context.l1_entities),
            "l2_count": len(context.l2_entities),
            "l25_count": len(context.l25_entities),
            "source_file": str(context.source_file) if context.source_file else None,
            "domain_breakdown": self.get_domain_summary(),
            "unique_names": len(context.name_index),
            "unique_meta_ids": len(context.entity_index),
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        [MONITOR-1] Get cache hit rate statistics for monitoring.
        
        Returns:
            Cache statistics including hit rates
        """
        stats = MasterOntologyContext._cache_stats
        total_requests = stats["instance_hits"] + stats["global_hits"] + stats["global_misses"]
        instance_total = stats["instance_hits"] + stats["instance_misses"]
        
        return {
            "instance_hits": stats["instance_hits"],
            "instance_misses": stats["instance_misses"],
            "global_hits": stats["global_hits"],
            "global_misses": stats["global_misses"],
            "total_loads": stats["total_loads"],
            "instance_hit_rate": round(stats["instance_hits"] / instance_total * 100, 2) if instance_total > 0 else 0.0,
            "global_hit_rate": round((stats["instance_hits"] + stats["global_hits"]) / total_requests * 100, 2) if total_requests > 0 else 0.0,
            "overall_hit_rate": round((stats["instance_hits"] + stats["global_hits"]) / total_requests * 100, 2) if total_requests > 0 else 0.0,
            "instance_stats": self._instance_stats,
        }
    
    @classmethod
    def reset_cache_stats(cls) -> None:
        """
        [MONITOR-1] Reset cache statistics (for testing or periodic reporting).
        """
        cls._cache_stats = {
            "instance_hits": 0,
            "instance_misses": 0,
            "global_hits": 0,
            "global_misses": 0,
            "total_loads": 0,
        }
    
    def _get_context(self) -> LoadedOntologyContext:
        """Get cached context or load fresh."""
        if self._context_cache is None:
            return self.load_master_ontology()
        return self._context_cache
    
    def reload(self) -> LoadedOntologyContext:
        """Force reload of master ontology."""
        self._context_cache = None
        return self.load_master_ontology(self._last_loaded_path)
    
    def format_for_prompt(self, max_entities: int = 30) -> str:
        """
        Format existing entities for inclusion in LLM prompt.
        
        [OPTIMIZATION] Compressed format to minimize prompt size.
        
        Args:
            max_entities: Maximum number of entities to include
            
        Returns:
            Formatted string for prompt context
        """
        context = self._get_context()
        
        if context.total_count == 0:
            return "# Existing context: None\n"
        
        lines = [f"# Existing entities ({context.total_count} total)"]
        
        layer_config = [
            ("L1", context.l1_entities),
            ("L2", context.l2_entities),
            ("L2.5", context.l25_entities),
        ]
        
        entities_per_layer = max(1, max_entities // 3)
        
        for layer, entities in layer_config:
            if entities:
                layer_entities = entities[:entities_per_layer]
                entity_strs = []
                for entity in layer_entities:
                    name = entity.get("name", entity.get("scenario_name", entity.get("meta_name", "")))
                    meta_id = entity.get("meta_id", "")
                    entity_strs.append(f"{meta_id}:{name}")
                lines.append(f"{layer}: {', '.join(entity_strs)}")
        
        lines.append("Note: Avoid duplicates; link to existing entities.")
        
        return "\n".join(lines) + "\n"

    def summarize_for_deduplication(self, max_entities: int = 50) -> Dict[str, List[Dict[str, Any]]]:
        """Return compact context for deduplication and association services."""
        context = self._get_context()
        return {
            "L1": context.l1_entities[:max_entities],
            "L2": context.l2_entities[:max_entities],
            "L2.5": context.l25_entities[:max_entities],
        }


# Convenience function for quick loading
def load_ontology_context(path: Optional[Path] = None) -> LoadedOntologyContext:
    """Quick load ontology context."""
    loader = MasterOntologyContext()
    return loader.load_master_ontology(path)


# =============================================================================
# 数据分析上下文加载器（整合自 data_analysis/ontology_context.py）
# =============================================================================

def _extract_key_terms_from_text(text: str) -> List[str]:
    """从文本中提取关键词（实体名、技术术语）"""
    import re
    patterns = [
        r"[A-Z][a-zA-Z0-9]*(?:_[A-Z][a-zA-Z0-9]*)+",  # CONST_NAME
        r"[a-z]+_[a-z_]+",  # snake_case
        r"[A-Z][a-z]+[A-Z][a-zA-Z0-9]*",  # CamelCase
        r"`[^`]+`",  # `code`
        r"[0-9]{4,}[A-Z]{2,}",  # PML 编码
    ]
    terms = set()
    for pat in patterns:
        for m in re.finditer(pat, text):
            terms.add(m.group(0).strip("`"))
    domain_words = re.findall(r"[A-Za-z]{4,}", text)
    for w in domain_words:
        if w.isupper() or len(w) > 6:
            terms.add(w)
    return sorted(terms)[:50]


def _extract_key_terms_from_json(data: dict) -> List[str]:
    """从 JSON 数据中提取关键词"""
    text = json.dumps(data, ensure_ascii=False)
    return _extract_key_terms_from_text(text)


class AnalysisContextLoader:
    """
    数据分析上下文加载器。

    从 ontology run 输出目录加载所有相关文档块，
    并封装为 OntologyBlock 列表供分析使用。
    """

    MAX_FRAGMENTS = 30
    MAX_CHARS_CORE = 10000
    MAX_CHARS_DOMAIN = 8000
    MAX_CHARS_JSON = 6000
    MAX_CHARS_SCENARIO = 3000
    MAX_CHARS_MANIFEST = 2000

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)

    def load(self) -> List[Dict[str, Any]]:
        """
        加载所有 ontology 上下文块。

        Returns:
            List[Dict]: 按优先级排序的文档块列表
        """
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Ontology run 目录不存在: {self.run_dir}")

        blocks: List[Dict[str, Any]] = []

        def _add_md(path: Path, block_type: str, max_chars: int):
            if path.exists():
                text = path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
                key_terms = _extract_key_terms_from_text(text)
                blocks.append({
                    "source": str(path.name),
                    "block_type": block_type,
                    "content": text,
                    "confidence": 1.0,
                    "key_terms": key_terms,
                })

        def _add_json_summary(path: Path, block_type: str, max_chars: int):
            if path.exists():
                try:
                    raw = path.read_text(encoding="utf-8", errors="ignore")
                    text = raw[:max_chars]
                    if len(raw) > max_chars:
                        text += "\n... (截断)"
                    key_terms = _extract_key_terms_from_text(text)
                    blocks.append({
                        "source": str(path.name),
                        "block_type": block_type,
                        "content": text,
                        "confidence": 1.0,
                        "key_terms": key_terms,
                    })
                except Exception as _e:
                    logger.debug("跳过损坏的 JSON 文件 %s: %s", path.name, _e)

        def _add_fragment(frag_path: Path):
            try:
                data = json.loads(frag_path.read_text(encoding="utf-8", errors="ignore"))
                confidence = data.get("confidence", 0.5)
                summary = {
                    "source_file": data.get("source_file", ""),
                    "domain": data.get("domain", ""),
                    "confidence": confidence,
                    "nodes": data.get("nodes", [])[:5],
                    "relationships": data.get("relationships", [])[:5],
                    "suggestions": data.get("suggestions", [])[:3],
                    "domain_specific_concepts": data.get("domain_specific_concepts", [])[:3],
                }
                text = json.dumps(summary, ensure_ascii=False, indent=2)
                key_terms = _extract_key_terms_from_json(data)
                blocks.append({
                    "source": str(frag_path.relative_to(self.run_dir)),
                    "block_type": "fragment",
                    "content": text,
                    "confidence": confidence,
                    "key_terms": key_terms,
                })
            except Exception as _e:
                logger.debug("跳过无法解析的 fragment %s: %s", frag_path.name, _e)

        _add_md(self.run_dir / "CORE_RESULTS.md", "core_results", self.MAX_CHARS_CORE)
        _add_md(self.run_dir / "DOMAIN_ANALYSIS_REPORT.md", "domain_report", self.MAX_CHARS_DOMAIN)
        _add_json_summary(self.run_dir / "L1_DOMAIN_SANDBOX.json", "l1_sandbox", self.MAX_CHARS_JSON)
        _add_json_summary(self.run_dir / "L2_OPERATIONAL_INSTANCES.json", "l2_instances", self.MAX_CHARS_JSON)
        _add_json_summary(self.run_dir / "L2_BUSINESS_SCENARIOS.md", "l2_scenarios", self.MAX_CHARS_SCENARIO)
        _add_json_summary(self.run_dir / "analysis_manifest.json", "manifest", self.MAX_CHARS_MANIFEST)

        frag_dir = self.run_dir / "compilations"
        if frag_dir.exists():
            frag_files = sorted(
                frag_dir.rglob("layer1_fragment_*.json"),
                key=lambda p: p.stat().st_mtime,
            )
            for fp in frag_files[:self.MAX_FRAGMENTS]:
                _add_fragment(fp)

        return blocks

    @classmethod
    def from_run_id(cls, run_id: str, base_dir: str | Path = "outputs/ontology/runs") -> "AnalysisContextLoader":
        """通过 run_id 自动构建路径。"""
        run_dir = Path(base_dir) / run_id
        return cls(run_dir)

    @staticmethod
    def load_from_master_dir(master_dir: str | Path, topic: str = "") -> List[Dict[str, Any]]:
        """
        从 master 级别的 outputs/ontology 目录加载上下文块。

        Args:
            master_dir: outputs/ontology 目录路径
            topic: 分析主题关键词，用于过滤 RAG chunks

        Returns:
            List[Dict]: 文档块列表
        """
        master_dir = Path(master_dir)
        if not master_dir.exists():
            raise FileNotFoundError(f"Master ontology 目录不存在: {master_dir}")

        blocks: List[Dict[str, Any]] = []
        topic_kws = {w.lower() for w in (topic or "").split() if w}

        def _add_md(path: Path, block_type: str, max_chars: int):
            if path.exists():
                text = path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
                blocks.append({
                    "source": path.name,
                    "block_type": block_type,
                    "content": text,
                    "confidence": 1.0,
                    "key_terms": _extract_key_terms_from_text(text),
                })
                logger.debug("加载 %s (%d 字符)", path.name, len(text))

        _add_md(master_dir / "master_ontology_v2_summary.md", "master_summary", 6000)
        _add_md(master_dir / "master_ontology_v2.md", "master_v2", 15000)
        _add_md(master_dir / "master_ontology_l2_summary.md", "l2_summary", 4000)
        _add_md(master_dir / "master_ontology_l2.md", "l2_detail", 12000)

        rag_chunks_path = master_dir / "rag" / "chunks.jsonl"
        if rag_chunks_path.exists():
            try:
                matched: List[tuple] = []
                with open(rag_chunks_path, encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        content = chunk.get("content", "")
                        chunk_id = chunk.get("chunk_id", "")
                        chunk_type = chunk.get("chunk_type", "")
                        content_lower = content.lower()
                        id_lower = chunk_id.lower()
                        if topic_kws:
                            score = sum(
                                2 if kw in id_lower else (1 if kw in content_lower else 0)
                                for kw in topic_kws
                            )
                            if score == 0:
                                continue
                        else:
                            score = 1
                        matched.append((score, chunk_id, chunk_type, content[:500]))

                matched.sort(key=lambda x: -x[0])
                for score, chunk_id, chunk_type, content in matched[:40]:
                    text = f"[chunk_id={chunk_id}] [{chunk_type}]\n{content}"
                    blocks.append({
                        "source": f"rag/chunks.jsonl#{chunk_id}",
                        "block_type": "rag_chunk",
                        "content": text,
                        "confidence": min(1.0, 0.5 + score * 0.1),
                        "key_terms": _extract_key_terms_from_text(content),
                    })
                logger.debug("从 RAG chunks 中匹配到 %d 条（topic=%r）", len(matched[:40]), topic)
            except Exception as e:
                logger.warning("加载 RAG chunks 失败: %s", e)

        logger.info("load_from_master_dir: 共加载 %d 个 ontology 块", len(blocks))
        return blocks

    def load_master_ontology(self, master_path: str | Path) -> List[Dict[str, Any]]:
        """
        加载主本体文件作为对比上下文。

        Args:
            master_path: master_ontology_v2.json 文件路径

        Returns:
            List[Dict]: 主本体节点块列表
        """
        master_file = Path(master_path)
        if not master_file.exists():
            raise FileNotFoundError(f"主本体文件不存在: {master_file}")

        try:
            with open(master_file, encoding="utf-8") as f:
                master_data = json.load(f)
        except Exception as e:
            raise ValueError(f"主本体文件解析失败: {e}")

        blocks: List[Dict[str, Any]] = []

        nodes = master_data.get("nodes", [])
        for node in nodes[:50]:
            node_text = json.dumps(node, ensure_ascii=False)
            key_terms = _extract_key_terms_from_json(node)
            blocks.append({
                "source": str(master_file.name),
                "block_type": "master_node",
                "content": node_text,
                "confidence": 1.0,
                "key_terms": key_terms,
            })

        edges = master_data.get("edges", [])
        for edge in edges[:30]:
            edge_text = json.dumps(edge, ensure_ascii=False)
            key_terms = _extract_key_terms_from_json(edge)
            blocks.append({
                "source": str(master_file.name),
                "block_type": "master_edge",
                "content": edge_text,
                "confidence": 1.0,
                "key_terms": key_terms,
            })

        return blocks


# =============================================================================
# 统一入口工厂
# =============================================================================

class ContextFactory:
    """统一上下文入口工厂。"""

    @staticmethod
    def create_master_ontology_loader(custom_paths: Optional[List[str]] = None) -> "MasterOntologyContext":
        """创建结构化本体上下文加载器（用于编译流程）。"""
        return MasterOntologyContext(custom_paths)

    @staticmethod
    def create_analysis_loader(run_dir: str | Path) -> "AnalysisContextLoader":
        """创建数据分析上下文加载器（用于分析流程）。"""
        return AnalysisContextLoader(run_dir)

    @staticmethod
    def load_master_context(path: Optional[Path] = None) -> LoadedOntologyContext:
        """快速加载结构化本体上下文。"""
        loader = MasterOntologyContext()
        return loader.load_master_ontology(path)

    @staticmethod
    def load_analysis_context(run_dir: str | Path) -> List[Dict[str, Any]]:
        """快速加载数据分析上下文。"""
        loader = AnalysisContextLoader(run_dir)
        return loader.load()


# 向后兼容别名
OntologyContextLoader = MasterOntologyContext
