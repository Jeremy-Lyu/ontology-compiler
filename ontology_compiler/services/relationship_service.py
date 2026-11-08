"""Relationship extraction services extracted from ``TrueUnifiedExtractor``.

Two-phase relationship extraction plus its entity-centric context helper
(2026-08-11).
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def build_entity_context_text(text: str, entities_list: list) -> str:
    """Build entity-centric context text for relationship extraction of large documents.

    Instead of blind truncation, extracts paragraphs containing entity mentions
    plus surrounding context, ensuring all entities have relevant context.
    """
    max_chars = 12000
    paragraphs = text.split('\n')
    
    # Score each paragraph by entity mention count
    scored = []
    entity_names = [e.get("name", "") for e in entities_list if e.get("name")]
    
    for i, para in enumerate(paragraphs):
        para_lower = para.lower()
        score = sum(1 for name in entity_names if name and name.lower() in para_lower)
        # Always include slide markers and first paragraph
        if para.strip().startswith('--- Slide') or i == 0:
            score += 2
        scored.append((i, para, score))
    
    # Sort by score descending, take top paragraphs
    scored.sort(key=lambda x: x[2], reverse=True)
    
    # Select paragraphs until we reach max_chars
    selected_indices = set()
    selected_text = []
    total_len = 0
    
    for idx, para, score in scored:
        if score == 0 and total_len > max_chars * 0.7:
            break
        # Include paragraph plus adjacent context
        for offset in (-1, 0, 1):
            target = idx + offset
            if 0 <= target < len(paragraphs) and target not in selected_indices:
                p = paragraphs[target]
                if total_len + len(p) + 1 > max_chars:
                    continue
                selected_indices.add(target)
                selected_text.append((target, p))
                total_len += len(p) + 1
    
    # Sort back by original order
    selected_text.sort(key=lambda x: x[0])
    return '\n'.join(p for _, p in selected_text)


async def extract_relationships_separate(
    text: str,
    entities: List[Dict[str, Any]],
    domain: Optional[str] = None,
    *,
    llm,
    max_tokens,
    temperature,
    timeout,
    prompt_loader,
    build_entity_context_text,
) -> List[Dict[str, Any]]:
    """
    Two-phase extraction: Extract relationships separately after entities are extracted.

    Args:
        text: Document text
        entities: List of already extracted entities
        domain: Optional domain hint

    Returns:
        List of relationship dictionaries
    """
    import json
    
    entities_list = []
    for entity in entities:
        if isinstance(entity, dict):
            e = entity
        elif hasattr(entity, "model_dump"):
            e = entity.model_dump()
        else:
            e = dict(entity) if hasattr(entity, "__dict__") else {}
        entity_info = {
            "meta_id": e.get("meta_id", ""),
            "name": e.get("name", ""),
            "meta_class": e.get("meta_class", ""),
            "layer": e.get("meta_id", "")[:3] if e.get("meta_id") else ""
        }
        entities_list.append(entity_info)
    
    entities_json = json.dumps(entities_list, ensure_ascii=False, indent=2)
    
    prompt_template = prompt_loader.get_template("relation_extraction")
    if not prompt_template:
        logger.warning("relation_extraction template not found, using fallback")
        return []
    
    # For large texts, use entity-centric context extraction instead of blind truncation
    if len(text) > 12000:
        relation_text = build_entity_context_text(text, entities_list)
    else:
        relation_text = text
    
    prompt = prompt_template.render(
        text=relation_text,
        entities=entities_json,
        domain=domain or ""
    )
    
    logger.info(f"[RELATION-EXTRACT] Two-phase relation extraction: {len(entities)} entities, prompt={len(prompt)} chars")
    
    try:
        response_text = await asyncio.wait_for(
            llm.call_with_retry(prompt, temperature=temperature, max_tokens=max_tokens, complexity="auto", use_cache=True),
            timeout=timeout
        )
        
        if not response_text or not response_text.strip():
            logger.warning("[RELATION-EXTRACT] Empty response from LLM")
            return []
        
        cleaned = response_text.strip().replace('```json', '').replace('```', '').strip()
        
        try:
            result = json.loads(cleaned)
            if isinstance(result, list):
                relationships = result
            elif isinstance(result, dict):
                relationships = result.get("relationships", result.get("edges", result.get("relations", [])))
            else:
                relationships = []
            
            for rel in relationships:
                if isinstance(rel, dict) and "confidence" not in rel:
                    rel["confidence"] = "0.7"
            
            logger.info(f"[RELATION-EXTRACT] Extracted {len(relationships)} relationships")
            return relationships
        except json.JSONDecodeError as e:
            logger.warning(f"[RELATION-EXTRACT] Failed to parse relation response: {e}")
            return []
            
    except Exception as e:
        logger.error(f"[RELATION-EXTRACT] Two-phase extraction failed: {e}")
        return []
