"""ExtractorPlugin entry-point helpers extracted from ``TrueUnifiedExtractor``.

Shared by the sync/async plugin entry points: source reading, failure-result
construction, and the sync result-to-Fragment conversion (2026-08-11).
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ontology_compiler.agents.extractor_plugin import ExtractorResult

try:
    from core.events import FragmentExtractedEvent, ExtractionFailedEvent
    from core.events.event_consumer_validator import EventConsumerRegistry
    EVENT_ENABLED = True
except ImportError:
    EVENT_ENABLED = False

logger = logging.getLogger(__name__)


def read_source_content(source, start_time, extractor_name):
    """Read file content if source is a path; else return source as-is.

    Returns (content, None) on success or (None, ExtractorResult) on read error.
    """
    if isinstance(source, (str, Path)) and os.path.exists(source):
        try:
            with open(source, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            return None, ExtractorResult(
                success=False,
                error_message=f"Failed to read file: {e}",
                execution_time_ms=(time.time() - start_time) * 1000,
                extractor_name=extractor_name
            )
        return content, None
    return source, None


def build_failure_result(error_msg, execution_time, extractor_name):
    """Construct a failed ExtractorResult for the plugin entry points."""
    return ExtractorResult(
        success=False,
        error_message=error_msg,
        execution_time_ms=execution_time,
        extractor_name=extractor_name
    )


def build_sync_extractor_result(result, domain, file_path, ctx, extractor_name):
    """Convert a UnifiedExtractionResult into the sync ExtractorResult format.

    Flattens entities into dict-based nodes with L0 validation, builds
    Fragment domain models, and publishes the FragmentExtractedEvent.
    """
    # Flatten all entities into nodes
    nodes = []
    source_file_str_sync = str(file_path) if file_path else ""
    try:
        from core.schema_registry import get_schema_registry
        registry = get_schema_registry()
        VALID_L0_CLASSES = set(registry.main_classes)
    except ImportError:
        VALID_L0_CLASSES = {"Capability", "System", "Actor", "Asset", "Data", "Process", "Concept"}
    
    VALID_L0_CLASSES_LOWER = {c.lower() for c in VALID_L0_CLASSES}
    
    for entity in result.l0_entities:
        entity_name = entity.get("name", "").strip()
        entity_meta_class = entity.get("meta_class", "").strip()
        if entity_name.lower() in VALID_L0_CLASSES_LOWER:
            for valid_class in VALID_L0_CLASSES:
                if valid_class.lower() == entity_name.lower():
                    entity_name = valid_class
                    break
            nodes.append({
                "meta_id": entity.get("meta_id"),
                "meta_name": entity_name,
                "meta_class": entity.get("meta_class", "Capability"),
                "layer": "L0",
                "source_file": source_file_str_sync,
                **entity
            })
        elif entity_meta_class.lower() in VALID_L0_CLASSES_LOWER:
            logger.warning(f"[L0 VALIDATION] Entity name '{entity_name}' is a domain-specific concept, not an abstract L0 class. Moving to L1 with parent_class='{entity_meta_class}'.")
            meta_id = entity.get("meta_id", "")
            nodes.append({
                "meta_id": meta_id.replace("L0-", "L1-") if meta_id else "",
                "meta_name": entity_name,
                "meta_class": entity_meta_class,
                "layer": "L1",
                "parent_class": entity_meta_class,
                "source_file": source_file_str_sync,
                **entity
            })
        else:
            logger.warning(f"[L0 VALIDATION] Entity '{entity_name}' is not a valid L0 class, moved to L2")
            meta_id = entity.get("meta_id", "")
            nodes.append({
                "meta_id": meta_id.replace("L0-", "L2-") if meta_id else "",
                "meta_name": entity.get("name"),
                "meta_class": entity.get("meta_class", "System"),
                "layer": "L2",
                "source_file": source_file_str_sync,
                **entity
            })
    for entity in result.l1_entities:
        nodes.append({
            "meta_id": entity.get("meta_id"),
            "meta_name": entity.get("name"),
            "meta_class": entity.get("meta_class", "System"),
            "layer": "L1",
            "source_file": source_file_str_sync,
            **entity
        })

    # L2 as business scenario nodes
    for entity in result.l2_entities:
        nodes.append({
            "meta_id": entity.get("meta_id"),
            "meta_name": entity.get("scenario_name", entity.get("name", "")),
            "meta_class": entity.get("meta_class", "System"),
            "layer": "L2",
            "source_file": source_file_str_sync,
            **entity
        })

    # L2.5 as nodes (rules/mappings)
    for entity in result.l25_entities:
        nodes.append({
            "meta_id": entity.get("meta_id"),
            "meta_name": entity.get("name", ""),
            "meta_class": entity.get("meta_class", "Data"),
            "layer": "L2.5",
            "source_file": source_file_str_sync,
            **entity
        })

    # L3 instances
    instances = []
    for entity in result.l3_entities:
        instances.append({
            "meta_id": entity.get("meta_id"),
            "meta_name": entity.get("name"),
            "meta_class": entity.get("meta_class", "Asset"),
            "layer": "L3",
            "source_file": source_file_str_sync,
            **entity
        })
    
    # L2.5 as edges/rules
    edges = []
    for entity in result.l25_entities:
        edges.append({
            "meta_id": entity.get("meta_id"),
            "meta_name": entity.get("name"),
            "meta_class": entity.get("meta_class", "Data"),
            "layer": "L2.5",
            **entity
        })
    
    # [MONITOR-3] Publish FragmentExtractedEvent via EventConsumerRegistry
    if EVENT_ENABLED:
        try:
            all_entities = result.l1_entities + result.l2_entities + result.l25_entities
            event = FragmentExtractedEvent(
                source_file=str(file_path) if file_path else "memory",
                domain=domain or "SUPPLY_CHAIN",
                entities=all_entities,
                relations=result.relationships,
                run_id=ctx.get("run_id"),
                extraction_strategy=ctx.get("prompt_strategy", "unified"),
                quality_score=result.quality_score,
                extractor_name=extractor_name,
            )
            registry = EventConsumerRegistry()
            publish_result = registry.publish(event)
            logger.info(
                f"Published FragmentExtractedEvent: {event} "
                f"(consumed by {publish_result.successful}/{publish_result.total_consumers} consumers)"
            )
        except Exception as event_err:
            logger.warning(f"Failed to publish FragmentExtractedEvent: {event_err}")
    
    from src.domain.models import Fragment, FragmentNode, FragmentRelationship, FragmentInstance
    
    fragment_nodes = [
        FragmentNode(
            meta_id=n.get("meta_id", ""),
            meta_name=n.get("meta_name", n.get("name", "")),
            meta_class=n.get("meta_class", "System"),
            domain=n.get("domain", "SUPPLY_CHAIN"),
            layer=n.get("layer", ""),
            description=n.get("description", ""),
            properties=n,
        )
        for n in nodes
    ]
    
    fragment_instances = [
        FragmentInstance(
            meta_id=i.get("meta_id", ""),
            meta_name=i.get("meta_name", i.get("name", "")),
            meta_class=i.get("meta_class", "Asset") if i.get("meta_class") not in ("Instance", "") else "Asset",
            domain=i.get("domain", "SUPPLY_CHAIN"),
            layer=i.get("layer", "L3"),
            description=i.get("description", ""),
            properties=i,
        )
        for i in instances
    ]
    
    fragment_relationships = []
    for rel in result.relationships:
        fragment_relationships.append(FragmentRelationship(
            source=rel.get("source", rel.get("from", "")),
            target=rel.get("target", rel.get("to", "")),
            type=rel.get("relation_type", rel.get("type", "associated_with")),
            domain=rel.get("domain", "SUPPLY_CHAIN"),
            properties=rel,
        ))
    
    fragment = Fragment(
        source_file=str(file_path) if file_path else "",
        domain=domain or "SUPPLY_CHAIN",
        nodes=fragment_nodes,
        instances=fragment_instances,
        relationships=fragment_relationships,
        metadata={
            "l1_count": len(result.l1_entities),
            "l2_count": len(result.l2_entities),
            "l25_count": len(result.l25_entities),
            "quality_score": result.quality_score,
            "token_usage": result.token_usage,
            "domain": domain,
            "source_file": str(file_path) if file_path else None,
            "extractor_name": extractor_name,
        },
    )
    
    return ExtractorResult.from_fragment(fragment, success=True)
