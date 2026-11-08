"""Smart-chunking extraction service extracted from ``TrueUnifiedExtractor``.

``extract_with_chunking`` depends only on its collaborators (chunker + four
callbacks) and the result class, so it lives here as a plain module function
(2026-08-11). The extractor keeps a thin delegating wrapper.
"""

import asyncio
import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from core.contracts import CacheVersionBundle
from ontology_compiler.services.validation_service import validate_with_schema_registry
from ontology_compiler.services.quality_assurance.quality_rules import calculate_unified_quality_score

try:
    from core.schema_registry import get_schema_registry
except ImportError:
    get_schema_registry = None

logger = logging.getLogger(__name__)


# Loaded once at import time so the env lookup doesn't run per-call.
_CHUNK_MAX_OUTPUT_TOKENS_ENV = int(os.environ.get("CHUNK_MAX_OUTPUT_TOKENS", "0") or 0)


def _chunk_output_budget(chunk_chars: int) -> int:
    """Map a chunk's input size to a sane LLM output-token budget.

    [OPTIMIZATION 2026-08-15] The default 32K-token output budget let the
    model "fill the space" on any chunk: a 6834-char chunk once produced a
    ~29K-token non-JSON narrative (215s + 333s of wasted retries). Extraction
    of entities/relationships is typically 1.5-2x the input length; budgeting
    output tokens ≈ input chars covers worst case without inviting drift.

    [OPTIMIZATION 2026-08-18] Cap via CHUNK_MAX_OUTPUT_TOKENS env var. DIFY's
    DIFY_MAX_TOKENS=16000 so chunk budgets larger than that get silently
    truncated upstream — set this 1-2K below the provider max to leave room
    for the JSON wrapper / stop tokens.
    """
    if chunk_chars <= 0:
        return 4096
    ceiling = _CHUNK_MAX_OUTPUT_TOKENS_ENV or 32000
    return max(4096, min(ceiling, chunk_chars))


def _merge_entity_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    def _entity_key(item: dict[str, Any]) -> str:
        meta_id = str(item.get("meta_id") or item.get("id") or "").strip()
        if meta_id:
            return f"id:{meta_id}"
        name = str(item.get("name") or item.get("meta_name") or "").strip().lower()
        layer = str(item.get("layer") or "").strip().lower()
        meta_class = str(item.get("meta_class") or item.get("entity_type") or "").strip().lower()
        return f"sig:{layer}:{meta_class}:{name}"

    def _merge_list_field(target: dict[str, Any], source: dict[str, Any], field_name: str) -> None:
        left = target.get(field_name)
        right = source.get(field_name)
        if isinstance(left, list) or isinstance(right, list):
            combined: list[Any] = []
            for value in (left if isinstance(left, list) else []):
                if value not in combined:
                    combined.append(value)
            for value in (right if isinstance(right, list) else []):
                if value not in combined:
                    combined.append(value)
            if combined:
                target[field_name] = combined
            return

    for record in records:
        if not isinstance(record, dict):
            continue
        key = _entity_key(record)
        if key not in merged:
            item = dict(record)
            item.setdefault("source_chunk_ids", [])
            item.setdefault("source_sections", [])
            chunk_index = item.get("chunk_index")
            chunk_section = item.get("chunk_section")
            if chunk_index is not None and chunk_index not in item["source_chunk_ids"]:
                item["source_chunk_ids"].append(chunk_index)
            if chunk_section and chunk_section not in item["source_sections"]:
                item["source_sections"].append(chunk_section)
            merged[key] = item
            order.append(key)
            continue

        target = merged[key]
        for field_name, value in record.items():
            if value in (None, "", [], {}):
                continue
            if field_name in {"source_chunk_ids", "source_sections"}:
                continue
            if field_name == "decision_trace" and isinstance(value, list):
                target.setdefault("decision_trace", [])
                target["decision_trace"].extend([item for item in value if item not in target["decision_trace"]])
                continue
            if field_name == "provenance" and isinstance(value, dict):
                target.setdefault("provenance", {})
                for prov_key, prov_value in value.items():
                    if prov_key not in target["provenance"] and prov_value not in (None, "", [], {}):
                        target["provenance"][prov_key] = prov_value
                continue
            if isinstance(value, list):
                _merge_list_field(target, record, field_name)
                continue
            if field_name not in target or target[field_name] in (None, "", [], {}):
                target[field_name] = value

        target.setdefault("source_chunk_ids", [])
        target.setdefault("source_sections", [])
        chunk_index = record.get("chunk_index")
        chunk_section = record.get("chunk_section")
        if chunk_index is not None and chunk_index not in target["source_chunk_ids"]:
            target["source_chunk_ids"].append(chunk_index)
        if chunk_section and chunk_section not in target["source_sections"]:
            target["source_sections"].append(chunk_section)
        target.setdefault("decision_trace", []).append({
            "action": "merge_chunk_entity",
            "reason": "deduplicated across chunked extraction",
            "source_chunk_index": chunk_index,
            "source_chunk_section": chunk_section,
        })

    return [merged[key] for key in order]


def _merge_relationship_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    def _relation_key(item: dict[str, Any]) -> str:
        source = str(item.get("from") or item.get("source") or item.get("source_id") or "").strip()
        target = str(item.get("to") or item.get("target") or item.get("target_id") or "").strip()
        rel_type = str(item.get("type") or item.get("relation_type") or item.get("rel_type") or "").strip().lower()
        return f"{source}->{target}:{rel_type}"

    for record in records:
        if not isinstance(record, dict):
            continue
        record = dict(record)
        if record.get("source") and not record.get("from"):
            record["from"] = record.get("source")
        if record.get("target") and not record.get("to"):
            record["to"] = record.get("target")
        key = _relation_key(record)
        if key not in merged:
            item = dict(record)
            item.setdefault("source_chunk_ids", [])
            item.setdefault("source_sections", [])
            chunk_index = item.get("chunk_index")
            chunk_section = item.get("chunk_section")
            if chunk_index is not None and chunk_index not in item["source_chunk_ids"]:
                item["source_chunk_ids"].append(chunk_index)
            if chunk_section and chunk_section not in item["source_sections"]:
                item["source_sections"].append(chunk_section)
            merged[key] = item
            order.append(key)
            continue

        target = merged[key]
        for field_name, value in record.items():
            if value in (None, "", [], {}):
                continue
            if field_name in {"source_chunk_ids", "source_sections"}:
                continue
            if field_name == "decision_trace" and isinstance(value, list):
                target.setdefault("decision_trace", [])
                target["decision_trace"].extend([item for item in value if item not in target["decision_trace"]])
                continue
            if field_name == "provenance" and isinstance(value, dict):
                target.setdefault("provenance", {})
                for prov_key, prov_value in value.items():
                    if prov_key not in target["provenance"] and prov_value not in (None, "", [], {}):
                        target["provenance"][prov_key] = prov_value
                continue
            if isinstance(value, list):
                left = target.get(field_name)
                combined: list[Any] = []
                for item in (left if isinstance(left, list) else []):
                    if item not in combined:
                        combined.append(item)
                for item in value:
                    if item not in combined:
                        combined.append(item)
                if combined:
                    target[field_name] = combined
                continue
            if field_name not in target or target[field_name] in (None, "", [], {}):
                target[field_name] = value

        target.setdefault("source_chunk_ids", [])
        target.setdefault("source_sections", [])
        chunk_index = record.get("chunk_index")
        chunk_section = record.get("chunk_section")
        if chunk_index is not None and chunk_index not in target["source_chunk_ids"]:
            target["source_chunk_ids"].append(chunk_index)
        if chunk_section and chunk_section not in target["source_sections"]:
            target["source_sections"].append(chunk_section)
        target.setdefault("decision_trace", []).append({
            "action": "merge_chunk_relationship",
            "reason": "deduplicated across chunked extraction",
            "source_chunk_index": chunk_index,
            "source_chunk_section": chunk_section,
        })

    return [merged[key] for key in order]


async def extract_with_chunking(
    text: str,
    file_path: Optional[Path] = None,
    domain: Optional[str] = None,
    use_context: bool = True,
    enable_validation: bool = True,
    max_retries: int = 1,
    document_metadata: Optional[Dict[str, Any]] = None,
    *,
    chunker,
    build_prompt,
    call_llm,
    parse_response,
    extract_relationships,
    result_cls,
    strategy: str = "",
    cache_enabled: bool = False,
    version_bundle: Optional[CacheVersionBundle] = None,
) -> "UnifiedExtractionResult":

    chunk_start_time = time.time()
    source_file_name = str(file_path) if file_path else ""
    strategy = (strategy or "").lower()
    cache_enabled = cache_enabled is True

    print(f"[EXTRACT] 📄 Large document detected ({len(text)} chars), using smart chunking (strategy={strategy or 'auto'})")
    
    # [OPTIMIZATION]: For xlsx/spreadsheet files, use moderate chunk size to balance chunk count vs prompt length
    # Strategy: Use 3000 chars chunk size (prompt ~5500-6000 chars) to reduce chunk count
    # xlsx files have structured tabular data, larger chunks preserve context better
    original_max_chunk_size = None
    is_spreadsheet = source_file_name and any(source_file_name.lower().endswith(ext) for ext in ['.xlsx', '.xls', '.csv'])
    if is_spreadsheet:
        if hasattr(chunker, 'config') and isinstance(chunker.config, dict):
            original_max_chunk_size = chunker.config.get('max_chunk_size')
            chunker.config['max_chunk_size'] = 3000
            chunker.config['overlap_size'] = 200
            print(f"[EXTRACT] 📊 Spreadsheet detected, using moderate chunk size (3000 chars) for balanced processing")
    
    chunks = chunker.smart_chunk(text, source_file=source_file_name, strategy=strategy) if strategy else chunker.smart_chunk(text, source_file=source_file_name)
    strategy_used = strategy or "auto"
    
    # Restore original config
    if original_max_chunk_size is not None:
        chunker.config['max_chunk_size'] = original_max_chunk_size
    
    print(f"[EXTRACT] Split into {len(chunks)} chunks")
    
    all_l0 = []
    all_l1 = []
    all_l2 = []
    all_l25 = []
    all_l3 = []
    all_relationships = []
    all_responses = []
    chunk_metrics: list[dict[str, Any]] = []
    total_tokens = 0
    rate_limit_delay = 1.0  # 基础延迟1秒，避免限流
    
    # [OPTIMIZATION]: 使用Semaphore控制并发，平衡速度和限流
    # 可通过 CHUNK_EXTRACTION_CONCURRENCY 环境变量覆盖
    # xlsx文件使用更大的并发度，因为chunk较小且处理快
    # [OPTIMIZATION]: 大文件（>5MB）降低并发度到默认值，避免API限流
    file_size = os.path.getsize(file_path) if file_path and file_path.exists() else len(text)
    is_large_file = file_size > 5 * 1024 * 1024  # >5MB
    default_concurrency = int(os.environ.get("CHUNK_EXTRACTION_CONCURRENCY", "4"))
    
    if is_spreadsheet:
        concurrency = max(4, default_concurrency)
    elif is_large_file:
        concurrency = max(2, default_concurrency // 2)
        print(f"[EXTRACT] 📦 Large file detected ({file_size/1024/1024:.1f}MB), using concurrency={concurrency}")
    else:
        concurrency = default_concurrency
    
    semaphore = asyncio.Semaphore(concurrency)
    chunk_cache: dict[str, tuple[list, list, list, list, list, list]] = {}
    cache_hits = 0
    cache_misses = 0

    # P1 / C-4: chunk cache keys include the version bundle fingerprint so
    # that a prompt / schema / policy / model / mode bump invalidates all
    # cached chunk results — never reuse stale extraction output.
    _vb_fingerprint = version_bundle.to_fingerprint() if version_bundle and not version_bundle.is_empty() else ""

    def _chunk_cache_key(content: str) -> str:
        normalized = "\n".join(content.split())
        if _vb_fingerprint:
            payload = f"{_vb_fingerprint}\x1f{normalized}"
        else:
            payload = normalized
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    async def process_chunk(chunk, chunk_index):
        nonlocal total_tokens, rate_limit_delay, cache_hits, cache_misses
        chunk_content = chunk.get("content", "")
        chunk_section = chunk.get("section", "")
        
        print(f"[EXTRACT] Processing chunk {chunk_index+1}/{len(chunks)}: {chunk_section} ({len(chunk_content)} chars)")
        
        if cache_enabled and chunk_content:
            cache_key = _chunk_cache_key(chunk_content)
            if cache_key in chunk_cache:
                cache_hits += 1
                l0, l1, l2, l25, l3, rels = chunk_cache[cache_key]
                originals = (l0, l1, l2, l25, l3, rels)
                l0, l1, l2, l25, l3, rels = ([dict(e) for e in lst] for lst in originals)
                for e in l0:
                    e["chunk_section"] = chunk_section
                    e["chunk_index"] = chunk_index
                for e in l1:
                    e["chunk_section"] = chunk_section
                    e["chunk_index"] = chunk_index
                for e in l2:
                    e["chunk_section"] = chunk_section
                    e["chunk_index"] = chunk_index
                for e in l25:
                    e["chunk_section"] = chunk_section
                    e["chunk_index"] = chunk_index
                for e in l3:
                    e["chunk_section"] = chunk_section
                    e["chunk_index"] = chunk_index
                for rel in rels:
                    if isinstance(rel, dict):
                        rel["chunk_section"] = chunk_section
                        rel["chunk_index"] = chunk_index
                chunk_metrics.append({
                    "chunk_index": chunk_index,
                    "chunk_section": chunk_section,
                    "entities": len(l0) + len(l1) + len(l2) + len(l25) + len(l3),
                    "relationships": len(rels),
                    "tokens": 0,
                    "status": "cache_hit",
                })
                return l0, l1, l2, l25, l3, rels
        
        cache_misses += 1
        
        prompt_metrics = {}
        prompt = build_prompt(chunk_content, domain, use_context, skip_truncate=True, document_metadata=document_metadata, metrics=prompt_metrics)
        
        # [TIMEOUT BUDGET] Retries are tunable via EXTRACT_CHUNK_MAX_RETRIES.
        # The hardcoded ``max_retries = 3`` made every chunk attempt up to 4
        # llm calls, each of which already loops internally (JSON + routing
        # retries). Worst case that was ~12 API round-trips per chunk — enough
        # to blow past the whole-file timeout on any slow response. Default 1
        # (2 attempts per chunk) keeps the budget bounded.
        max_retries = int(os.environ.get("EXTRACT_CHUNK_MAX_RETRIES", "1"))
        retry_delay = rate_limit_delay
        
        for attempt in range(max_retries + 1):
            try:
                # [OUTPUT BUDGET] 按 chunk 内容缩放 max_tokens，防止模型放飞。
                # 32K 默认输出预算会让模型把任何 chunk 都"写满"（实测 6834-char
                # chunk 诱发 29K-token 废话响应，215s+333s 两轮重试全报废）。
                # 抽 JSON 实体/关系的输出约为输入 1.5-2x，给出 4x 足够且可控。
                chunk_max_tokens = _chunk_output_budget(len(chunk_content))
                async with semaphore:
                    llm_result = await call_llm(prompt, max_tokens=chunk_max_tokens)
                
                response_text = llm_result["response"]
                all_responses.append(response_text)
                total_tokens += len(response_text) // 4
                
                l0, l1, l2, l25, l3, rels = parse_response(response_text)
                
                if cache_enabled and chunk_content:
                    chunk_cache[_chunk_cache_key(chunk_content)] = (
                        [dict(e) for e in l0],
                        [dict(e) for e in l1],
                        [dict(e) for e in l2],
                        [dict(e) for e in l25],
                        [dict(e) for e in l3],
                        [dict(e) for e in rels] if all(isinstance(r, dict) for r in rels) else list(rels),
                    )
                
                for e in l0:
                    e["chunk_section"] = chunk_section
                    e["chunk_index"] = chunk_index
                for e in l1:
                    e["chunk_section"] = chunk_section
                    e["chunk_index"] = chunk_index
                for e in l2:
                    e["chunk_section"] = chunk_section
                    e["chunk_index"] = chunk_index
                for e in l25:
                    e["chunk_section"] = chunk_section
                    e["chunk_index"] = chunk_index
                for e in l3:
                    e["chunk_section"] = chunk_section
                    e["chunk_index"] = chunk_index
                for rel in rels:
                    if isinstance(rel, dict):
                        rel["chunk_section"] = chunk_section
                        rel["chunk_index"] = chunk_index
                
                rate_limit_delay = max(0.5, rate_limit_delay * 0.9)
                
                chunk_metrics.append({
                    "chunk_index": chunk_index,
                    "chunk_section": chunk_section,
                    "entities": len(l0) + len(l1) + len(l2) + len(l25) + len(l3),
                    "relationships": len(rels),
                    "tokens": len(response_text) // 4,
                    "status": "ok",
                    "prompt_metrics": prompt_metrics,
                })
                return l0, l1, l2, l25, l3, rels
            except Exception as e:
                if "Rate limit exceeded" in str(e) and attempt < max_retries:
                    rate_limit_delay = min(30.0, rate_limit_delay * 2)
                    print(f"[EXTRACT] ⚠️ Rate limit exceeded for chunk {chunk_index}, retrying in {rate_limit_delay:.1f}s (attempt {attempt+1}/{max_retries+1})")
                    await asyncio.sleep(rate_limit_delay)
                    continue
                elif attempt < max_retries:
                    print(f"[EXTRACT] ⚠️ Error processing chunk {chunk_index}, retrying (attempt {attempt+1}/{max_retries+1}): {e}")
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    logger.error(f"Error processing chunk {chunk_index} after {max_retries+1} attempts: {e}")
                    chunk_metrics.append({
                        "chunk_index": chunk_index,
                        "chunk_section": chunk_section,
                        "entities": 0,
                        "relationships": 0,
                        "tokens": 0,
                        "status": "failed",
                        "error": str(e),
                    })
                    return [], [], [], [], [], []
    
    chunk_tasks = []
    for i, chunk in enumerate(chunks):
        chunk_tasks.append(process_chunk(chunk, i))
    
    results = await asyncio.gather(*chunk_tasks)
    
    for l0, l1, l2, l25, l3, rels in results:
        all_l0.extend(l0)
        all_l1.extend(l1)
        all_l2.extend(l2)
        all_l25.extend(l25)
        all_l3.extend(l3)
        all_relationships.extend(rels)

    all_l0 = _merge_entity_records(all_l0)
    all_l1 = _merge_entity_records(all_l1)
    all_l2 = _merge_entity_records(all_l2)
    all_l25 = _merge_entity_records(all_l25)
    all_l3 = _merge_entity_records(all_l3)
    
    processing_time = (time.time() - chunk_start_time) * 1000
    
    quality_score = await calculate_unified_quality_score(
        l1_entities=all_l1,
        l2_entities=all_l2,
        l25_entities=all_l25,
        relationships=all_relationships,
        domain=domain
    )
    chunk_quality = "success"
    if chunk_metrics and any(item.get("status") == "failed" for item in chunk_metrics):
        chunk_quality = "partial_success"
    if quality_score < 0.6:
        chunk_quality = "degraded_success" if all_l1 or all_l2 or all_l25 or all_relationships else "failure"
    
    print(f"[EXTRACT] Chunking complete: {len(all_l1)+len(all_l2)+len(all_l25)} entities from {len(chunks)} chunks in {processing_time:.1f}ms")
    
    # [TWO-PHASE EXTRACTION]: Always perform separate relation extraction to ensure relationships are captured
    all_entities = all_l0 + all_l1 + all_l2 + all_l25 + all_l3
    if len(all_entities) > 0:
        logger.info(f"[EXTRACT] 📊 Two-phase extraction (chunking): {len(all_entities)} entities, triggering separate relation extraction")
        print(f"[EXTRACT] 📊 Two-phase (chunking): {len(all_entities)} entities, triggering relation extraction...")
        
        try:
            relation_result = await extract_relationships(text, all_entities, domain)
            if relation_result and len(relation_result) > 0:
                all_relationships.extend(relation_result)
                all_relationships = _merge_relationship_records(all_relationships)
                logger.info(f"[EXTRACT] 📊 Two-phase extraction (chunking) completed: added {len(relation_result)} relationships")
                print(f"[EXTRACT] ✅ Two-phase (chunking): added {len(relation_result)} relationships")
        except Exception as e:
            logger.warning(f"[EXTRACT] Two-phase relation extraction (chunking) failed: {e}")
            print(f"[EXTRACT] ⚠️ Two-phase relation extraction (chunking) failed: {e}")

    all_relationships = _merge_relationship_records(all_relationships)
    
    # [SCHEMA_VALIDATION] Validate meta_class for chunking path (fixes BusinessRule/ValidationRule/MappingRule not being auto-mapped)
    if enable_validation and get_schema_registry:
        print(f"[DEBUG] Before schema validation (chunking): L1={len(all_l1)}, L2={len(all_l2)}, L2.5={len(all_l25)}")
        all_l1, all_l2, all_l25, all_relationships = validate_with_schema_registry(
            all_l1, all_l2, all_l25, all_relationships, domain
        )
        print(f"[DEBUG] After schema validation (chunking): L1={len(all_l1)}, L2={len(all_l2)}, L2.5={len(all_l25)}")
    
    # P1-01: L0 auto-supplement moved to extract.py to avoid duplication across files
    # Each chunk should NOT auto-supplement L0 base classes individually
    # L0 base classes will be added once at the extract stage after all files are processed
    
    result_metadata = {
        "document_type": (document_metadata or {}).get("document_type") if isinstance(document_metadata, dict) else None,
        "document_type_confidence": (document_metadata or {}).get("document_type_confidence") if isinstance(document_metadata, dict) else None,
        "secondary_types": (document_metadata or {}).get("secondary_types", []) if isinstance(document_metadata, dict) else [],
        "source_path": str(file_path) if file_path else (document_metadata or {}).get("source_path", ""),
        "matched_skills": (document_metadata or {}).get("matched_skills", []) if isinstance(document_metadata, dict) else [],
        "skill_stats": (document_metadata or {}).get("skill_stats", {}) if isinstance(document_metadata, dict) else {},
    }
    if not all_relationships and (all_l1 or all_l2 or all_l25 or all_l3):
        chunk_quality = "partial_success"

    result = result_cls(
        l0_entities=all_l0,
        l1_entities=all_l1,
        l2_entities=all_l2,
        l25_entities=all_l25,
        l3_entities=all_l3,
        relationships=all_relationships,
        raw_response="\n---CHUNK_SEPARATOR---\n".join(all_responses),
        processing_time_ms=processing_time,
        token_usage={"estimated": total_tokens},
        quality_score=quality_score,
        degraded=chunk_quality != "success"
    )
    setattr(result, "metadata", {
        **result_metadata,
        "chunk_metrics": chunk_metrics,
        "chunk_quality": chunk_quality,
        "chunk_strategy": strategy_used,
        "cache_stats": {"hits": cache_hits, "misses": cache_misses, "enabled": cache_enabled},
    })
    return result
