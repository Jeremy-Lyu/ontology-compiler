"""
Cross File Association Service

Discovers relationships between entities across different files and with master ontology.
Uses similarity algorithms to find same_as, related_to, and parent_child relationships.

[PERF-OPT-1] Added inverted index for token-based candidate matching to reduce
O(N) scans to O(M) where M is the number of matching tokens.

[LLM-INTEGRATION] Added SemanticEquivalenceJudge for LLM-driven entity matching.
"""

import re
import logging
import json
from typing import Dict, List, Optional, Any, Set, Tuple
from difflib import SequenceMatcher
from collections import defaultdict
import hashlib
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class EntityRelationship:
    """Represents a relationship between two entities."""
    source_id: str
    target_id: str
    relationship_type: str  # same_as, related_to, parent_of, child_of, depends_on
    confidence: float
    evidence: List[str]
    similarity_score: float = 0.0
    matched_by: str = "rule"  # "llm" or "rule"


@dataclass
class EquivalenceResult:
    """Result of semantic equivalence judgment."""
    is_same: bool
    confidence: float
    reasoning: str
    matched_by: str  # "llm" or "rule"


class SemanticEquivalenceJudge:
    """
    LLM-driven semantic equivalence judge for entity matching.
    
    Two-phase strategy:
    1. Fast string similarity pre-filtering:
       - Exact match or very high similarity (>0.85): Directly return True
       - Low similarity (<0.3): Directly return False
       - meta_class mismatch: Directly return False
       - layer mismatch: Directly return False
       - Length difference >60%: Directly return False
       - 0.3-0.85 with domain term boost: May return True
       - Ambiguous cases: Call LLM for semantic judgment
    
    2. LLM semantic judgment:
       - Uses UnifiedLLMClient to compare entity descriptions
       - Returns structured result with confidence and reasoning
       - Supports both Chinese and English entity names
    
    Cost control: With enhanced pre-filtering, only ~20-30% of candidate pairs require LLM calls.
    """
    
    HIGH_CONFIDENCE_THRESHOLD = 0.55
    LOW_CONFIDENCE_THRESHOLD = 0.35
    EXACT_MATCH_THRESHOLD = 0.8
    DESCRIPTION_SIMILARITY_THRESHOLD = 0.7
    MAX_CANDIDATES_PER_ENTITY = 10
    FAST_SIMILARITY_THRESHOLD = 0.2
    
    DOMAIN_TERMS = {
        "bom", "ebom", "mbom", "supply", "chain", "manufacturing", "process",
        "system", "capability", "data", "asset", "actor", "inventory", "order",
        "product", "component", "material", "requirement", "specification",
        "configuration", "engineering", "quality", "planning", "management",
        "service", "integration", "interface", "api", "erp", "wms", "scm",
        "kpi", "metric", "report", "analysis", "validation", "verification",
    }
    
    def __init__(self, llm_client=None):
        """Initialize with optional LLM client.
        
        Args:
            llm_client: Pre-configured LLM client (UnifiedLLMClient or BaseLLMClient).
                        If None, will lazily create UnifiedLLMClient on first use.
        """
        self._llm_client = llm_client
        self._llm_calls = 0
        self._rule_based_decisions = 0
        self._rate_limit_exceeded = False
        self._rate_limit_cooldown_until = 0
        self._judgment_cache = {}
        
    @property
    def llm_client(self):
        """Lazy initialize LLM client (D-1: unified factory)."""
        if self._llm_client is None:
            try:
                from ontology_compiler.rigor.llm_backend import get_llm_client
                self._llm_client = get_llm_client()
            except Exception as e:
                logger.warning(f"Failed to initialize LLM client: {e}")
                self._llm_client = None
        return self._llm_client
    
    async def judge(
        self,
        entity_a: Dict[str, Any],
        entity_b: Dict[str, Any],
        force_llm: bool = False
    ) -> EquivalenceResult:
        """
        Judge if two entities are semantically equivalent.
        
        Args:
            entity_a: First entity dict
            entity_b: Second entity dict
            force_llm: If True, skip rule-based pre-filtering and call LLM directly
        
        Returns:
            EquivalenceResult with is_same, confidence, reasoning, and matched_by
        """
        name_a = self._get_entity_name(entity_a)
        name_b = self._get_entity_name(entity_b)
        desc_a = entity_a.get("description", "")
        desc_b = entity_b.get("description", "")
        class_a = entity_a.get("meta_class", "") or entity_a.get("class", "")
        class_b = entity_b.get("meta_class", "") or entity_b.get("class", "")
        layer_a = entity_a.get("layer", "")
        layer_b = entity_b.get("layer", "")
        
        cache_key = tuple(sorted([name_a, name_b]))
        if cache_key in self._judgment_cache:
            cached = self._judgment_cache[cache_key]
            logger.debug(f"Cache hit for entity pair: '{name_a}' vs '{name_b}'")
            return cached
        
        if not force_llm:
            rule_result = self._rule_based_judge(name_a, name_b, desc_a, desc_b, class_a, class_b, layer_a, layer_b)
            if rule_result is not None:
                self._rule_based_decisions += 1
                self._judgment_cache[cache_key] = rule_result
                return rule_result
        
        if not self.llm_client:
            logger.warning("LLM client not available, falling back to rule-based")
            rule_result = self._rule_based_judge(name_a, name_b, desc_a, desc_b)
            if rule_result is not None:
                return rule_result
            return EquivalenceResult(
                is_same=False,
                confidence=0.5,
                reasoning="LLM client not available, cannot determine equivalence",
                matched_by="rule"
            )
        
        import time
        if self._rate_limit_exceeded and time.time() < self._rate_limit_cooldown_until:
            logger.debug("Rate limit exceeded (cooldown active), falling back to rule-based for entity pair")
            rule_result = self._rule_based_judge(name_a, name_b, desc_a, desc_b)
            if rule_result is not None:
                return rule_result
            return EquivalenceResult(
                is_same=False,
                confidence=0.5,
                reasoning="LLM rate limit exceeded, falling back to rule-based",
                matched_by="rule"
            )
        elif self._rate_limit_exceeded and time.time() >= self._rate_limit_cooldown_until:
            self._rate_limit_exceeded = False
            logger.info("Rate limit cooldown expired, resuming LLM-based judgment")
        
        self._llm_calls += 1
        result = await self._llm_based_judge(entity_a, entity_b)
        cache_key = tuple(sorted([self._get_entity_name(entity_a), self._get_entity_name(entity_b)]))
        self._judgment_cache[cache_key] = result
        return result
    
    def _rule_based_judge(
        self,
        name_a: str,
        name_b: str,
        desc_a: str,
        desc_b: str,
        class_a: str = "",
        class_b: str = "",
        layer_a: str = "",
        layer_b: str = ""
    ) -> Optional[EquivalenceResult]:
        """
        Rule-based judgment for fast pre-filtering.
        
        Enhanced with:
        - meta_class consistency check
        - layer consistency check
        - Length difference check
        - Domain term boost for near-matches
        - Exact match shortcut
        
        Returns:
            EquivalenceResult if confident decision can be made, None otherwise.
        """
        if not name_a or not name_b:
            return None
        
        if name_a.lower() == name_b.lower():
            return EquivalenceResult(
                is_same=True,
                confidence=1.0,
                reasoning=f"Exact name match: '{name_a}'",
                matched_by="rule"
            )
        
        if class_a and class_b and class_a.strip().lower() != class_b.strip().lower():
            return EquivalenceResult(
                is_same=False,
                confidence=0.95,
                reasoning=f"meta_class mismatch: '{class_a}' vs '{class_b}'",
                matched_by="rule"
            )
        
        if layer_a and layer_b and layer_a.strip().lower() != layer_b.strip().lower():
            return EquivalenceResult(
                is_same=False,
                confidence=0.90,
                reasoning=f"layer mismatch: '{layer_a}' vs '{layer_b}'",
                matched_by="rule"
            )
        
        len_diff = abs(len(name_a) - len(name_b)) / max(len(name_a), len(name_b))
        if len_diff > 0.6:
            return EquivalenceResult(
                is_same=False,
                confidence=0.85,
                reasoning=f"Significant length difference ({len_diff:.1%})",
                matched_by="rule"
            )
        
        seq_sim = SequenceMatcher(None, name_a.lower(), name_b.lower()).ratio()
        
        words_a = set(self._tokenize(name_a))
        words_b = set(self._tokenize(name_b))
        
        if words_a and words_b:
            jaccard = len(words_a & words_b) / len(words_a | words_b)
        else:
            jaccard = 0.0
        
        combined = (seq_sim + jaccard) / 2
        
        shared_domain_terms = words_a & words_b & self.DOMAIN_TERMS
        domain_bonus = min(len(shared_domain_terms) * 0.05, 0.15)
        
        boosted_score = combined + domain_bonus
        
        if boosted_score >= self.EXACT_MATCH_THRESHOLD:
            return EquivalenceResult(
                is_same=True,
                confidence=min(0.98, boosted_score),
                reasoning=f"High similarity with domain boost (SequenceMatcher={seq_sim:.2f}, Jaccard={jaccard:.2f}, domain_terms={len(shared_domain_terms)})",
                matched_by="rule"
            )
        
        if combined >= self.HIGH_CONFIDENCE_THRESHOLD:
            return EquivalenceResult(
                is_same=True,
                confidence=min(0.95, boosted_score),
                reasoning=f"High string similarity (SequenceMatcher={seq_sim:.2f}, Jaccard={jaccard:.2f})",
                matched_by="rule"
            )
        
        if combined < self.LOW_CONFIDENCE_THRESHOLD:
            return EquivalenceResult(
                is_same=False,
                confidence=1.0 - combined,
                reasoning=f"Low string similarity (SequenceMatcher={seq_sim:.2f}, Jaccard={jaccard:.2f})",
                matched_by="rule"
            )
        
        if desc_a and desc_b:
            desc_words_a = set(self._tokenize(desc_a))
            desc_words_b = set(self._tokenize(desc_b))
            if desc_words_a and desc_words_b:
                desc_jaccard = len(desc_words_a & desc_words_b) / len(desc_words_a | desc_words_b)
                if desc_jaccard >= self.DESCRIPTION_SIMILARITY_THRESHOLD:
                    return EquivalenceResult(
                        is_same=True,
                        confidence=min(0.92, combined + 0.05),
                        reasoning=f"High description similarity (Jaccard={desc_jaccard:.2f})",
                        matched_by="rule"
                    )
                elif desc_jaccard < 0.1:
                    return EquivalenceResult(
                        is_same=False,
                        confidence=0.85,
                        reasoning=f"Very low description similarity (Jaccard={desc_jaccard:.2f})",
                        matched_by="rule"
                    )
        
        name_words_a = set(name_a.lower().split())
        name_words_b = set(name_b.lower().split())
        if not name_words_a & name_words_b:
            return EquivalenceResult(
                is_same=False,
                confidence=0.80,
                reasoning=f"No shared words between names",
                matched_by="rule"
            )
        
        return None
    
    async def _llm_based_judge(
        self,
        entity_a: Dict[str, Any],
        entity_b: Dict[str, Any]
    ) -> EquivalenceResult:
        """
        LLM-based semantic judgment for ambiguous cases.
        
        Uses structured prompt to get deterministic JSON output.
        """
        name_a = self._get_entity_name(entity_a)
        name_b = self._get_entity_name(entity_b)
        desc_a = entity_a.get("description", "")
        desc_b = entity_b.get("description", "")
        class_a = entity_a.get("meta_class", "") or entity_a.get("class", "")
        class_b = entity_b.get("meta_class", "") or entity_b.get("class", "")
        layer_a = entity_a.get("layer", "")
        layer_b = entity_b.get("layer", "")
        
        prompt = f"""
You are an expert in enterprise ontology design. Your task is to determine if two entities are SEMANTICALLY EQUIVALENT - meaning they represent the SAME concept or thing.

Consider these cases as equivalent:
- "物料管理系统" and "BOM Management System"
- "供应链管理" and "Supply Chain Management"
- "需求规格说明书" and "Requirements Specification Document"

Consider these cases as NOT equivalent:
- "物料管理系统" and "采购系统" (different systems)
- "供应链管理" and "物流管理" (related but different)
- "用户管理" and "权限管理" (different capabilities)

Entity A:
- Name: {name_a}
- Class: {class_a}
- Layer: {layer_a}
- Description: {desc_a}

Entity B:
- Name: {name_b}
- Class: {class_b}
- Layer: {layer_b}
- Description: {desc_b}

OUTPUT ONLY VALID JSON. DO NOT include markdown code blocks (```json), explanations, or any other text.

{{
  "is_same": true or false,
  "confidence": number between 0 and 1,
  "reasoning": "Your reasoning in 1-2 sentences"
}}
"""
        
        try:
            response = await self.llm_client.call(prompt, temperature=0.1, max_tokens=500)
            
            try:
                result = json.loads(response.strip())
            except json.JSONDecodeError:
                logger.warning(f"LLM returned invalid JSON: {response[:200]}")
                result = self._parse_llm_response(response)
            
            is_same = bool(result.get("is_same", False))
            confidence = float(result.get("confidence", 0.5))
            reasoning = str(result.get("reasoning", "No reasoning provided"))
            
            return EquivalenceResult(
                is_same=is_same,
                confidence=confidence,
                reasoning=reasoning,
                matched_by="llm"
            )
            
        except Exception as e:
            import time
            error_str = str(e).lower()
            if "rate limit" in error_str or "rate_limit" in error_str or "429" in str(e):
                logger.warning(f"LLM rate limit exceeded, enabling fallback mode for 60 seconds")
                self._rate_limit_exceeded = True
                self._rate_limit_cooldown_until = time.time() + 60
                return EquivalenceResult(
                    is_same=False,
                    confidence=0.5,
                    reasoning="LLM rate limit exceeded, falling back to rule-based",
                    matched_by="rule"
                )
            logger.error(f"LLM judgment failed: {e}")
            return EquivalenceResult(
                is_same=False,
                confidence=0.0,
                reasoning=f"LLM error: {str(e)}",
                matched_by="llm"
            )
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse non-JSON LLM response, handling markdown code blocks."""
        cleaned = response.strip()
        
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        
        cleaned = cleaned.strip()
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                first_brace = cleaned.find("{")
                last_brace = cleaned.rfind("}")
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    json_str = cleaned[first_brace:last_brace + 1]
                    return json.loads(json_str)
            except json.JSONDecodeError:
                pass
            
            response_lower = cleaned.lower()
            is_same = "true" in response_lower or "equivalent" in response_lower or "same" in response_lower
            
            confidence_match = re.search(r'confidence[:\s]*(\d+\.?\d*)', cleaned)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.5
            
            return {
                "is_same": is_same,
                "confidence": confidence,
                "reasoning": cleaned[:200]
            }
    
    def get_stats(self) -> Dict[str, int]:
        """Get usage statistics."""
        return {
            "llm_calls": self._llm_calls,
            "rule_based_decisions": self._rule_based_decisions,
        }
    
    def reset_stats(self) -> None:
        """Reset statistics and clear cache."""
        self._llm_calls = 0
        self._rule_based_decisions = 0
        self._judgment_cache.clear()
    
    def _get_entity_name(self, entity: Dict[str, Any]) -> str:
        """Extract name from entity dict."""
        for key in ["name", "scenario_name", "meta_name", "title", "label"]:
            if key in entity and entity[key]:
                return str(entity[key])
        return ""
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        words = re.findall(r'\b\w+\b', text.lower())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        return [w for w in words if w not in stop_words and len(w) > 2]
    
    async def find_same_as_relationships_llm(
        self,
        new_entities: List[Dict[str, Any]],
        master_entities: List[Dict[str, Any]],
        use_optimized: bool = True
    ) -> List[EntityRelationship]:
        """
        Find same_as relationships using LLM-driven semantic equivalence judgment.
        
        Optimizations:
        1. Fast pre-filtering using string similarity to eliminate obvious mismatches
        2. Candidate pool limitation: only top N most similar candidates per entity
        3. Rule-based decisions for clear cases before LLM call
        
        Args:
            new_entities: Newly extracted entities
            master_entities: Entities from master ontology
            use_optimized: If True, use indexed batch processing for pre-filtering
        
        Returns:
            List of same_as relationships with confidence scores
        """
        relationships = []
        seen_pairs = set()
        skipped_due_to_fast_filter = 0
        skipped_due_to_candidate_limit = 0
        
        for new_ent in new_entities:
            new_id = new_ent.get("id", new_ent.get("meta_id", ""))
            new_name = self._get_entity_name(new_ent)
            if not new_id or not new_name:
                continue
            
            candidates = []
            for master_ent in master_entities:
                master_id = master_ent.get("id", master_ent.get("meta_id", ""))
                master_name = self._get_entity_name(master_ent)
                if not master_id or not master_name:
                    continue
                
                pair_key = tuple(sorted([new_id, master_id]))
                if pair_key in seen_pairs or new_id == master_id:
                    continue
                
                fast_sim = self._calculate_fast_similarity(new_name, master_name)
                if fast_sim < self.FAST_SIMILARITY_THRESHOLD:
                    skipped_due_to_fast_filter += 1
                    continue
                
                candidates.append((fast_sim, master_ent, master_id, master_name))
            
            candidates.sort(key=lambda x: -x[0])
            top_candidates = candidates[:self.MAX_CANDIDATES_PER_ENTITY]
            skipped_due_to_candidate_limit += len(candidates) - len(top_candidates)
            
            for _, master_ent, master_id, _ in top_candidates:
                pair_key = tuple(sorted([new_id, master_id]))
                seen_pairs.add(pair_key)
                
                result = await self.judge(new_ent, master_ent)
                if result.is_same and result.confidence >= 0.7:
                    relationships.append(EntityRelationship(
                        source_id=new_id,
                        target_id=master_id,
                        relationship_type="same_as",
                        evidence=[result.reasoning],
                        confidence=result.confidence,
                        matched_by=result.matched_by
                    ))
        
        logger.info(f"Conflict detection optimization stats: "
                    f"fast_filter_skipped={skipped_due_to_fast_filter}, "
                    f"candidate_limit_skipped={skipped_due_to_candidate_limit}, "
                    f"llm_calls={self._llm_calls}, "
                    f"rule_based={self._rule_based_decisions}")
        
        return relationships
    
    def _calculate_fast_similarity(self, name_a: str, name_b: str) -> float:
        """Fast similarity calculation for pre-filtering."""
        from difflib import SequenceMatcher
        
        name_a_lower = name_a.lower()
        name_b_lower = name_b.lower()
        
        seq_sim = SequenceMatcher(None, name_a_lower, name_b_lower).ratio()
        
        words_a = set(name_a_lower.split())
        words_b = set(name_b_lower.split())
        
        if words_a and words_b:
            jaccard = len(words_a & words_b) / len(words_a | words_b)
        else:
            jaccard = 0.0
        
        return (seq_sim + jaccard) / 2
    
    async def find_same_as_relationships_llm_as_dicts(
        self,
        new_entities: List[Dict[str, Any]],
        master_entities: List[Dict[str, Any]],
        use_optimized: bool = True
    ) -> List[Dict[str, Any]]:
        """Return LLM-based same_as relationships as plain dictionaries."""
        relationships = await self.find_same_as_relationships_llm(new_entities, master_entities, use_optimized=use_optimized)
        return [
            {
                "from": rel.source_id,
                "to": rel.target_id,
                "type": rel.relationship_type,
                "description": "; ".join(rel.evidence),
                "confidence": rel.confidence,
                "matched_by": getattr(rel, "matched_by", "rule"),
            }
            for rel in relationships
        ]


class CrossFileAssociationService:
    """
    Service for discovering cross-file entity relationships.

    Capabilities:
    1. Find same_as relationships using name similarity
    2. Discover related entities using semantic analysis
    3. Build parent-child hierarchies
    4. Detect dependency relationships

    R7.5 Performance Enhancements:
    - Entity name index for fast candidate filtering
    - Batch similarity computation
    - Normalized name caching

    [PERF-OPT-1] Inverted Index Enhancement:
    - Token → Entity IDs inverted index for O(M) token matching
    - Expected benefit: 50-70% reduction in candidate matches
    - 40-60% reduction in similarity computation
    """

    # Similarity thresholds
    SAME_AS_THRESHOLD = 0.85
    RELATED_THRESHOLD = 0.60
    PARENT_CHILD_THRESHOLD = 0.70

    # R7.5: Batch processing config
    BATCH_SIZE = 100  # Process entities in batches for memory efficiency
    INDEX_MIN_CHARS = 3  # Minimum characters for index entries

    def __init__(self):
        self._entity_cache: Dict[str, Dict[str, Any]] = {}
        self._relationships: List[EntityRelationship] = []
        # R7.5: Name index for fast lookup (prefix -> entity_ids)
        self._name_index: Dict[str, Set[str]] = defaultdict(set)
        # R7.5: Normalized name cache (entity_id -> normalized_name)
        self._normalized_name_cache: Dict[str, str] = {}
        # R7.5: Token cache (entity_id -> tokens)
        self._token_cache: Dict[str, Set[str]] = {}
        
        # [PERF-OPT-1] Inverted index for token-based matching
        # Maps token → entity_ids for fast candidate lookup
        self._token_to_entities: Dict[str, Set[str]] = defaultdict(set)
        
        # [PERF-OPT-1] Performance stats
        self._perf_stats: Dict[str, int] = {
            "candidates_scanned": 0,
            "index_hits": 0,
            "token_matches": 0,
        }
        
        # [LLM-INTEGRATION] Semantic equivalence judge
        self._equivalence_judge = SemanticEquivalenceJudge()

    def _build_entity_index(self, entities: List[Dict[str, Any]]) -> None:
        """Build name index for fast candidate lookup (R7.5).

        Creates prefix index from normalized entity names for efficient filtering.

        [PERF-OPT-1] Also builds inverted index for token-based matching.
        """
        # Clear all indexes
        self._name_index.clear()
        self._normalized_name_cache.clear()
        self._token_cache.clear()
        self._token_to_entities.clear()

        for entity in entities:
            entity_id = entity.get("meta_id", "")
            if not entity_id:
                continue

            name = self._get_entity_name(entity)
            if not name:
                continue

            # Cache normalized name
            norm_name = self._normalize_name(name)
            self._normalized_name_cache[entity_id] = norm_name

            # Cache tokens
            tokens = set(self._tokenize(name))
            self._token_cache[entity_id] = tokens

            # Build prefix index (first N characters)
            prefix_len = min(self.INDEX_MIN_CHARS, len(norm_name))
            while prefix_len <= len(norm_name):
                prefix = norm_name[:prefix_len]
                self._name_index[prefix].add(entity_id)
                prefix_len += 1

            # [PERF-OPT-1] Build inverted index for tokens
            for token in tokens:
                self._token_to_entities[token].add(entity_id)

        logger.debug(f"Built index for {len(entities)} entities: "
                    f"{len(self._name_index)} prefixes, {len(self._token_to_entities)} tokens")

    def _get_candidate_matches(
        self,
        entity_name: str,
        threshold: float = 0.5
    ) -> Set[str]:
        """Get candidate entity IDs that might match the given name (R7.5).

        Uses prefix index for fast filtering before expensive similarity computation.

        [PERF-OPT-1] Uses inverted index for token matching instead of O(N) scan.
        """
        norm_name = self._normalize_name(entity_name)
        if len(norm_name) < self.INDEX_MIN_CHARS:
            # Short names: return all entities
            self._perf_stats["candidates_scanned"] += len(self._normalized_name_cache)
            return set(self._normalized_name_cache.keys())

        # Get candidates sharing prefix of minimum length
        prefix = norm_name[:self.INDEX_MIN_CHARS]
        candidates = self._name_index.get(prefix, set()).copy()
        self._perf_stats["index_hits"] += len(candidates)

        # [PERF-OPT-1] Check token overlap using inverted index (O(M) instead of O(N))
        query_tokens = set(self._tokenize(entity_name))
        if query_tokens:
            # Collect entities that share any token
            for token in query_tokens:
                entities_with_token = self._token_to_entities.get(token, set())
                candidates.update(entities_with_token)
                self._perf_stats["token_matches"] += len(entities_with_token)

        self._perf_stats["candidates_scanned"] += len(candidates)
        return candidates

    def _batch_compute_similarity(
        self,
        new_entities: List[Dict[str, Any]],
        candidate_map: Dict[str, Set[str]],
        batch_size: int = None
    ) -> List[Tuple[str, str, float]]:
        """Batch compute similarity scores (R7.5).

        Processes entities in batches for memory efficiency.
        Returns list of (new_id, master_id, score) tuples.
        """
        batch_size = batch_size or self.BATCH_SIZE
        results = []

        for i in range(0, len(new_entities), batch_size):
            batch = new_entities[i:i + batch_size]

            for new_entity in batch:
                new_id = new_entity.get("meta_id", "")
                if not new_id:
                    continue

                candidates = candidate_map.get(new_id, set())
                if not candidates:
                    continue

                new_name = self._get_entity_name(new_entity)
                new_norm = self._normalized_name_cache.get(new_id) or self._normalize_name(new_name)
                new_tokens = self._token_cache.get(new_id) or set(self._tokenize(new_name))

                for master_id in candidates:
                    if master_id == new_id:
                        continue

                    master_norm = self._normalized_name_cache.get(master_id, "")
                    if not master_norm:
                        continue

                    # Fast path: exact match
                    if new_norm == master_norm:
                        results.append((new_id, master_id, 1.0))
                        continue

                    # Compute similarity using cached values
                    name_score = self._fast_name_similarity(new_norm, master_norm)

                    if name_score >= self.SAME_AS_THRESHOLD:
                        # Verify with token overlap
                        master_tokens = self._token_cache.get(master_id, set())
                        if new_tokens and master_tokens:
                            jaccard = len(new_tokens & master_tokens) / len(new_tokens | master_tokens)
                            combined = name_score * 0.7 + jaccard * 0.3
                        else:
                            combined = name_score

                        if combined >= self.SAME_AS_THRESHOLD:
                            results.append((new_id, master_id, combined))

        return results

    def _fast_name_similarity(self, norm_a: str, norm_b: str) -> float:
        """Fast name similarity using pre-normalized names (R7.5).

        Avoids redundant normalization for cached names.
        """
        if not norm_a or not norm_b:
            return 0.0

        if norm_a == norm_b:
            return 1.0

        # Levenshtein-based similarity
        seq_sim = SequenceMatcher(None, norm_a, norm_b).ratio()

        # Quick Jaccard for word overlap
        words_a = set(norm_a.split())
        words_b = set(norm_b.split())
        if words_a and words_b:
            intersection = words_a & words_b
            union = words_a | words_b
            jaccard_sim = len(intersection) / len(union) if union else 0.0
        else:
            jaccard_sim = 0.0

        return max(seq_sim, jaccard_sim * 0.9)

    def find_same_as_relationships(
        self,
        new_entities: List[Dict[str, Any]],
        master_entities: List[Dict[str, Any]],
        use_optimized: bool = True
    ) -> List[EntityRelationship]:
        """
        Find entities that are the same across files using similarity algorithms.

        Uses Jaccard similarity for word overlap and Levenshtein for name similarity.

        R7.5: Added optimized path with indexing for large entity sets.

        Args:
            new_entities: Newly extracted entities
            master_entities: Entities from master ontology
            use_optimized: If True, use indexed batch processing (recommended for >100 entities)

        Returns:
            List of same_as relationships with confidence scores
        """
        if use_optimized and len(new_entities) > 50:
            return self._find_same_as_optimized(new_entities, master_entities)
        return self._find_same_as_legacy(new_entities, master_entities)

    def find_same_as_relationships_as_dicts(
        self,
        new_entities: List[Dict[str, Any]],
        master_entities: List[Dict[str, Any]],
        use_optimized: bool = True
    ) -> List[Dict[str, Any]]:
        """Return same_as relationships as plain dictionaries for pipeline usage."""
        relationships = self.find_same_as_relationships(new_entities, master_entities, use_optimized=use_optimized)
        return [
            {
                "from": rel.source_id,
                "to": rel.target_id,
                "type": rel.relationship_type,
                "description": "; ".join(rel.evidence),
                "confidence": rel.confidence,
            }
            for rel in relationships
        ]

    def find_relationships(
        self,
        entities: List[Dict[str, Any]],
        master_entities: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Find both cross-file and master ontology relationships as dictionaries."""
        master_entities = master_entities or []
        rels = self.find_same_as_relationships_as_dicts(entities, master_entities)
        rels.extend([
            {
                "from": rel.source_id,
                "to": rel.target_id,
                "type": rel.relationship_type,
                "description": "; ".join(rel.evidence),
                "confidence": rel.confidence,
            }
            for rel in self.find_dependency_chains(entities)
        ])
        return rels

    def find_same_as_relationships_with_dicts(
        self,
        new_entities: List[Dict[str, Any]],
        master_entities: List[Dict[str, Any]],
        use_optimized: bool = True
    ) -> List[Dict[str, Any]]:
        """Backward compatible alias returning dict relationships."""
        return self.find_same_as_relationships_as_dicts(new_entities, master_entities, use_optimized=use_optimized)
    
    async def find_same_as_relationships_llm(
        self,
        new_entities: List[Dict[str, Any]],
        master_entities: List[Dict[str, Any]],
        use_optimized: bool = True
    ) -> List[EntityRelationship]:
        """
        Find same_as relationships using LLM-driven semantic equivalence judgment.
        
        Two-phase approach:
        1. Use string similarity to pre-filter candidates
        2. Use LLM to judge semantic equivalence for ambiguous cases
        
        Args:
            new_entities: Newly extracted entities
            master_entities: Entities from master ontology
            use_optimized: If True, use indexed batch processing for pre-filtering
        
        Returns:
            List of same_as relationships with confidence scores
        """
        if use_optimized and len(new_entities) > 50:
            return await self._find_same_as_llm_optimized(new_entities, master_entities)
        return await self._find_same_as_llm_legacy(new_entities, master_entities)
    
    async def find_same_as_relationships_llm_as_dicts(
        self,
        new_entities: List[Dict[str, Any]],
        master_entities: List[Dict[str, Any]],
        use_optimized: bool = True
    ) -> List[Dict[str, Any]]:
        """Return LLM-based same_as relationships as plain dictionaries."""
        relationships = await self.find_same_as_relationships_llm(new_entities, master_entities, use_optimized=use_optimized)
        return [
            {
                "from": rel.source_id,
                "to": rel.target_id,
                "type": rel.relationship_type,
                "description": "; ".join(rel.evidence),
                "confidence": rel.confidence,
                "matched_by": getattr(rel, "matched_by", "rule"),
            }
            for rel in relationships
        ]

    def _find_same_as_optimized(
        self,
        new_entities: List[Dict[str, Any]],
        master_entities: List[Dict[str, Any]]
    ) -> List[EntityRelationship]:
        """Optimized same_as detection using indexing and batch processing (R7.5)."""
        import time
        start_time = time.time()

        relationships = []

        # Build index for master entities
        self._build_entity_index(master_entities)

        # Build candidate map using index
        candidate_map: Dict[str, Set[str]] = {}
        for new_entity in new_entities:
            new_id = new_entity.get("meta_id", "")
            if not new_id:
                continue
            new_name = self._get_entity_name(new_entity)
            candidates = self._get_candidate_matches(new_name)
            candidate_map[new_id] = candidates

        # Batch compute similarities
        matches = self._batch_compute_similarity(new_entities, candidate_map)

        # Create relationships from matches
        master_index = {e.get("meta_id", ""): e for e in master_entities if e.get("meta_id")}

        for new_id, master_id, score in matches:
            new_entity = next((e for e in new_entities if e.get("meta_id") == new_id), None)
            master_entity = master_index.get(master_id)

            if not new_entity or not master_entity:
                continue

            new_name = self._get_entity_name(new_entity)
            new_desc = new_entity.get("description", "")
            master_desc = master_entity.get("description", "")

            relationship = EntityRelationship(
                source_id=new_id,
                target_id=master_id,
                relationship_type="same_as",
                confidence=score,
                similarity_score=score,
                evidence=[
                    f"Name similarity: {score:.2f}",
                    f"Description overlap: {self._jaccard_similarity(new_desc, master_desc):.2f}",
                    "R7.5 optimized matching"
                ]
            )
            relationships.append(relationship)

            logger.debug(
                f"Found same_as (optimized): {new_name} -> {self._get_entity_name(master_entity)} "
                f"(confidence: {score:.2f})"
            )

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"Optimized same_as matching: {len(new_entities)} new vs {len(master_entities)} master "
            f"-> {len(relationships)} matches in {elapsed_ms}ms"
        )

        return relationships

    def _find_same_as_legacy(
        self,
        new_entities: List[Dict[str, Any]],
        master_entities: List[Dict[str, Any]]
    ) -> List[EntityRelationship]:
        """Legacy same_as detection (for small entity sets)."""
        relationships = []

        for new_entity in new_entities:
            new_id = new_entity.get("meta_id", "")
            new_name = self._get_entity_name(new_entity)
            new_desc = new_entity.get("description", "")

            best_match = None
            best_score = 0.0

            for master_entity in master_entities:
                master_id = master_entity.get("meta_id", "")
                if not master_id:
                    continue

                master_name = self._get_entity_name(master_entity)
                master_desc = master_entity.get("description", "")

                # Calculate combined similarity score
                name_score = self._name_similarity(new_name, master_name)
                desc_score = self._description_similarity(new_desc, master_desc) if new_desc and master_desc else 0.0

                # Weighted combination (name is more important)
                combined_score = name_score * 0.7 + desc_score * 0.3

                if combined_score > best_score and combined_score >= self.SAME_AS_THRESHOLD:
                    best_score = combined_score
                    best_match = master_entity

            if best_match:
                relationship = EntityRelationship(
                    source_id=new_id,
                    target_id=best_match.get("meta_id", ""),
                    relationship_type="same_as",
                    confidence=best_score,
                    similarity_score=best_score,
                    evidence=[
                        f"Name similarity: {self._name_similarity(new_name, self._get_entity_name(best_match)):.2f}",
                        f"Description overlap: {self._jaccard_similarity(new_desc, best_match.get('description', '')):.2f}"
                    ]
                )
                relationships.append(relationship)

                logger.debug(
                    f"Found same_as: {new_name} -> {self._get_entity_name(best_match)} "
                    f"(confidence: {best_score:.2f})"
                )

        return relationships
    
    async def _find_same_as_llm_optimized(
        self,
        new_entities: List[Dict[str, Any]],
        master_entities: List[Dict[str, Any]]
    ) -> List[EntityRelationship]:
        """LLM-enhanced same_as detection using optimized indexing."""
        import time
        start_time = time.time()
        
        relationships = []
        
        self._build_entity_index(master_entities)
        
        candidate_map: Dict[str, Set[str]] = {}
        for new_entity in new_entities:
            new_id = new_entity.get("meta_id", "")
            if not new_id:
                continue
            new_name = self._get_entity_name(new_entity)
            candidates = self._get_candidate_matches(new_name)
            candidate_map[new_id] = candidates
        
        master_index = {e.get("meta_id", ""): e for e in master_entities if e.get("meta_id")}
        
        for new_entity in new_entities:
            new_id = new_entity.get("meta_id", "")
            if not new_id:
                continue
            
            candidates = candidate_map.get(new_id, set())
            if not candidates:
                continue
            
            new_name = self._get_entity_name(new_entity)
            
            for master_id in candidates:
                if master_id == new_id:
                    continue
                
                master_entity = master_index.get(master_id)
                if not master_entity:
                    continue
                
                master_name = self._get_entity_name(master_entity)
                
                norm_new = self._normalized_name_cache.get(new_id) or self._normalize_name(new_name)
                norm_master = self._normalized_name_cache.get(master_id, "")
                
                if norm_new == norm_master:
                    relationships.append(EntityRelationship(
                        source_id=new_id,
                        target_id=master_id,
                        relationship_type="same_as",
                        confidence=1.0,
                        similarity_score=1.0,
                        evidence=["Exact name match"]
                    ))
                    continue
                
                seq_sim = self._fast_name_similarity(norm_new, norm_master)
                
                if seq_sim >= self.SAME_AS_THRESHOLD:
                    relationships.append(EntityRelationship(
                        source_id=new_id,
                        target_id=master_id,
                        relationship_type="same_as",
                        confidence=seq_sim,
                        similarity_score=seq_sim,
                        evidence=[f"High name similarity: {seq_sim:.2f}"]
                    ))
                    continue
                
                if seq_sim >= self.RELATED_THRESHOLD:
                    result = await self._equivalence_judge.judge(new_entity, master_entity)
                    
                    if result.is_same:
                        relationships.append(EntityRelationship(
                            source_id=new_id,
                            target_id=master_id,
                            relationship_type="same_as",
                            confidence=result.confidence,
                            similarity_score=seq_sim,
                            evidence=[
                                f"LLM judgment: {result.reasoning}",
                                f"String similarity: {seq_sim:.2f}"
                            ]
                        ))
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        stats = self._equivalence_judge.get_stats()
        
        logger.info(
            f"LLM-enhanced same_as matching: {len(new_entities)} new vs {len(master_entities)} master "
            f"-> {len(relationships)} matches in {elapsed_ms}ms. LLM calls: {stats['llm_calls']}, "
            f"Rule-based decisions: {stats['rule_based_decisions']}"
        )
        
        return relationships
    
    async def _find_same_as_llm_legacy(
        self,
        new_entities: List[Dict[str, Any]],
        master_entities: List[Dict[str, Any]]
    ) -> List[EntityRelationship]:
        """LLM-enhanced same_as detection for small entity sets."""
        relationships = []
        
        for new_entity in new_entities:
            new_id = new_entity.get("meta_id", "")
            new_name = self._get_entity_name(new_entity)
            new_desc = new_entity.get("description", "")
            
            for master_entity in master_entities:
                master_id = master_entity.get("meta_id", "")
                if not master_id:
                    continue
                
                master_name = self._get_entity_name(master_entity)
                master_desc = master_entity.get("description", "")
                
                name_score = self._name_similarity(new_name, master_name)
                
                if name_score >= self.SAME_AS_THRESHOLD:
                    relationships.append(EntityRelationship(
                        source_id=new_id,
                        target_id=master_id,
                        relationship_type="same_as",
                        confidence=name_score,
                        similarity_score=name_score,
                        evidence=[f"High name similarity: {name_score:.2f}"]
                    ))
                    continue
                
                if name_score >= self.RELATED_THRESHOLD:
                    result = await self._equivalence_judge.judge(new_entity, master_entity)
                    
                    if result.is_same:
                        relationships.append(EntityRelationship(
                            source_id=new_id,
                            target_id=master_id,
                            relationship_type="same_as",
                            confidence=result.confidence,
                            similarity_score=name_score,
                            evidence=[
                                f"LLM judgment: {result.reasoning}",
                                f"String similarity: {name_score:.2f}"
                            ]
                        ))
        
        stats = self._equivalence_judge.get_stats()
        logger.info(
            f"LLM legacy same_as matching: {stats['llm_calls']} LLM calls, "
            f"{stats['rule_based_decisions']} rule-based decisions"
        )
        
        return relationships
    
    def get_equivalence_judge_stats(self) -> Dict[str, int]:
        """Get statistics for the semantic equivalence judge."""
        return self._equivalence_judge.get_stats()
    
    def find_cross_file_relationships(
        self,
        per_file_entities: Dict[str, List[Dict[str, Any]]]
    ) -> List[EntityRelationship]:
        """
        Find relationships between entities from different files.
        
        Args:
            per_file_entities: Dict mapping file path to list of entities
            
        Returns:
            List of cross-file relationships
        """
        relationships = []
        all_entities = []
        
        # Flatten all entities with file source
        for file_path, entities in per_file_entities.items():
            for entity in entities:
                entity["_source_file"] = file_path
                all_entities.append(entity)
        
        # Find relationships between entities from different files
        for i, entity_a in enumerate(all_entities):
            for entity_b in all_entities[i+1:]:
                # Skip if from same file
                if entity_a.get("_source_file") == entity_b.get("_source_file"):
                    continue
                
                rel = self._analyze_relationship(entity_a, entity_b)
                if rel and rel.confidence >= self.RELATED_THRESHOLD:
                    relationships.append(rel)
        
        return relationships
    
    def build_entity_hierarchy(
        self,
        entities: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """
        Build parent-child hierarchy from entities.
        
        Args:
            entities: List of entities with possible parent references
            
        Returns:
            Dict mapping parent_id to list of child_ids
        """
        hierarchy: Dict[str, List[str]] = defaultdict(list)
        
        # Index all entities by ID
        entity_index = {e.get("meta_id", ""): e for e in entities if e.get("meta_id")}
        
        for entity in entities:
            entity_id = entity.get("meta_id", "")
            parent_refs = self._extract_parent_references(entity)
            
            for parent_id in parent_refs:
                if parent_id in entity_index:
                    hierarchy[parent_id].append(entity_id)
        
        return dict(hierarchy)
    
    def find_dependency_chains(
        self,
        entities: List[Dict[str, Any]]
    ) -> List[EntityRelationship]:
        """
        Find dependency chains between entities.
        
        Args:
            entities: List of entities
            
        Returns:
            List of dependency relationships
        """
        relationships = []
        
        for entity in entities:
            entity_id = entity.get("meta_id", "")
            dependencies = entity.get("dependencies", [])
            
            for dep in dependencies:
                dep_id = dep.get("meta_id") if isinstance(dep, dict) else dep
                if dep_id:
                    relationships.append(EntityRelationship(
                        source_id=entity_id,
                        target_id=str(dep_id),
                        relationship_type="depends_on",
                        confidence=0.9,
                        evidence=["Explicit dependency declaration"]
                    ))
        
        return relationships
    
    def _analyze_relationship(
        self,
        entity_a: Dict[str, Any],
        entity_b: Dict[str, Any]
    ) -> Optional[EntityRelationship]:
        """Analyze relationship between two entities."""
        name_a = self._get_entity_name(entity_a)
        name_b = self._get_entity_name(entity_b)
        id_a = entity_a.get("meta_id", "")
        id_b = entity_b.get("meta_id", "")
        
        # Check for same_as
        name_sim = self._name_similarity(name_a, name_b)
        if name_sim >= self.SAME_AS_THRESHOLD:
            return EntityRelationship(
                source_id=id_a,
                target_id=id_b,
                relationship_type="same_as",
                confidence=name_sim,
                similarity_score=name_sim,
                evidence=[f"High name similarity: {name_sim:.2f}"]
            )
        
        # Check for parent-child (naming patterns)
        if name_b.startswith(name_a) or name_a.startswith(name_b):
            if name_sim >= self.PARENT_CHILD_THRESHOLD:
                rel_type = "parent_of" if len(name_a) < len(name_b) else "child_of"
                return EntityRelationship(
                    source_id=id_a,
                    target_id=id_b,
                    relationship_type=rel_type,
                    confidence=name_sim,
                    similarity_score=name_sim,
                    evidence=[f"Naming hierarchy pattern: {name_sim:.2f}"]
                )
        
        # Check for related (domain/category similarity)
        domain_a = entity_a.get("domain", entity_a.get("meta_domain", ""))
        domain_b = entity_b.get("domain", entity_b.get("meta_domain", ""))
        category_a = entity_a.get("category", "")
        category_b = entity_b.get("category", "")
        
        if domain_a and domain_b and domain_a == domain_b:
            if category_a == category_b:
                return EntityRelationship(
                    source_id=id_a,
                    target_id=id_b,
                    relationship_type="related_to",
                    confidence=0.75,
                    similarity_score=0.75,
                    evidence=["Same domain and category"]
                )
        
        return None
    
    def _name_similarity(self, name_a: str, name_b: str) -> float:
        """Calculate name similarity using multiple methods."""
        if not name_a or not name_b:
            return 0.0
        
        # Normalize names
        norm_a = self._normalize_name(name_a)
        norm_b = self._normalize_name(name_b)
        
        # Exact match
        if norm_a == norm_b:
            return 1.0
        
        # Levenshtein-based similarity
        seq_sim = SequenceMatcher(None, norm_a, norm_b).ratio()
        
        # Jaccard similarity for word overlap
        jaccard_sim = self._jaccard_similarity(norm_a, norm_b)
        
        # Combine scores
        return max(seq_sim, jaccard_sim * 0.9)
    
    def _description_similarity(self, desc_a: str, desc_b: str) -> float:
        """Calculate description similarity."""
        if not desc_a or not desc_b:
            return 0.0
        
        # Normalize and tokenize
        words_a = set(self._tokenize(desc_a))
        words_b = set(self._tokenize(desc_b))
        
        if not words_a or not words_b:
            return 0.0
        
        # Jaccard similarity
        intersection = words_a & words_b
        union = words_a | words_b
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _jaccard_similarity(self, str_a: str, str_b: str) -> float:
        """Calculate Jaccard similarity between two strings."""
        words_a = set(self._tokenize(str_a))
        words_b = set(self._tokenize(str_b))
        
        if not words_a or not words_b:
            return 0.0
        
        intersection = words_a & words_b
        union = words_a | words_b
        
        return len(intersection) / len(union) if union else 0.0
    
    def _normalize_name(self, name: str) -> str:
        """Normalize entity name for comparison."""
        # Convert to lowercase
        name = name.lower()
        # Remove special characters except alphanumeric and spaces
        name = re.sub(r'[^\w\s]', ' ', name)
        # Normalize whitespace
        name = ' '.join(name.split())
        return name.strip()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple word tokenization
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        return [w for w in words if w not in stop_words and len(w) > 2]
    
    def _get_entity_name(self, entity: Dict[str, Any]) -> str:
        """Extract name from entity dict."""
        for key in ["name", "scenario_name", "meta_name", "title", "label"]:
            if key in entity and entity[key]:
                return str(entity[key])
        return ""
    
    def _extract_parent_references(self, entity: Dict[str, Any]) -> List[str]:
        """Extract parent entity references from entity."""
        parents = []
        
        # Check various possible parent reference fields
        for key in ["parent_id", "parent_meta_id", "belongs_to", "parent"]:
            if key in entity and entity[key]:
                val = entity[key]
                if isinstance(val, list):
                    parents.extend([str(v) for v in val if v])
                else:
                    parents.append(str(val))
        
        # Check for parent reference in relationships
        relationships = entity.get("relationships", [])
        for rel in relationships:
            if isinstance(rel, dict) and rel.get("type") in ["child_of", "belongs_to"]:
                target = rel.get("target_id") or rel.get("target_meta_id")
                if target:
                    parents.append(str(target))
        
        return parents
    
    def get_perf_stats(self) -> Dict[str, int]:
        """Get performance statistics for candidate matching.

        [PERF-OPT-1] Monitor inverted index effectiveness.

        Returns:
            Dict with "candidates_scanned", "index_hits", "token_matches"
        """
        return self._perf_stats.copy()
    
    def reset_perf_stats(self) -> None:
        """Reset performance statistics counters."""
        self._perf_stats = {
            "candidates_scanned": 0,
            "index_hits": 0,
            "token_matches": 0,
        }
    
    def get_index_stats(self) -> Dict[str, int]:
        """Get index statistics.

        [PERF-OPT-1] Monitor index size and coverage.

        Returns:
            Dict with "prefix_index_size", "token_index_size", "entity_count"
        """
        return {
            "prefix_index_size": len(self._name_index),
            "token_index_size": len(self._token_to_entities),
            "entity_count": len(self._normalized_name_cache),
        }
    
    def compute_entity_fingerprint(self, entity: Dict[str, Any]) -> str:
        """
        Compute a fingerprint hash for an entity.
        
        Args:
            entity: Entity dict
            
        Returns:
            MD5 hash string
        """
        # Create fingerprint from key fields
        name = self._get_entity_name(entity)
        domain = entity.get("domain", entity.get("meta_domain", ""))
        category = entity.get("category", "")
        
        fingerprint_str = f"{name}|{domain}|{category}"
        return hashlib.md5(fingerprint_str.encode()).hexdigest()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""
        rel_types = defaultdict(int)
        for rel in self._relationships:
            rel_types[rel.relationship_type] += 1
        
        return {
            "total_relationships": len(self._relationships),
            "by_type": dict(rel_types),
            "high_confidence": len([r for r in self._relationships if r.confidence >= 0.8]),
        }
