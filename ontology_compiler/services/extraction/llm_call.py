"""LLM call machinery extracted from ``TrueUnifiedExtractor``.

The cluster {``_call_llm``, ``_validate_json_response``, ``_build_retry_prompt``}
depends on extractor state only through the explicitly passed collaborators
(LLM client, token/temperature/timeout config, tracer, metadata hook), so it
lives here as plain module functions (2026-08-11).

The extractor keeps a thin ``_call_llm`` delegating wrapper so the original
call sites and trace step names stay unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, Union

try:
    from ontology_compiler.services.failure_memory import (
        record_extraction_failure,
        record_llm_timeout_failure,
    )
except ImportError:
    def record_extraction_failure(*args, **kwargs):
        pass

    def record_llm_timeout_failure(*args, **kwargs):
        pass

# C-3 / C-1治理层：接入 LLMMonitor，让预算闸门和成本汇总能看见真实调用量
try:
    from src.application.services.llm_monitor import record_llm_call as _monitor_record_llm
    _MONITOR_AVAILABLE = True
except Exception:
    _MONITOR_AVAILABLE = False

    def _monitor_record_llm(*_args, **_kwargs):
        pass

logger = logging.getLogger(__name__)

_TRANSIENT_MARKERS = (
    "rate limit",
    "timeout",
    "timed out",
    "temporarily unavailable",
    "connection reset",
    "connection refused",
    "connection aborted",
    "remote disconnected",
    "nodename nor servname",
    "name or service not known",
    "temporary failure in name resolution",
    "network is unreachable",
    "server disconnected",
    "502",
    "503",
    "504",
    "429",
)


def is_transient_provider_error(exc: Union[BaseException, str]) -> bool:
    """True for timeouts, DNS, connection drops, and rate limits."""
    text = str(exc).lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def _empty_ontology_json() -> str:
    return json.dumps(
        {
            "L0_entities": [],
            "L1_entities": [],
            "L2_entities": [],
            "L2_5_entities": [],
            "L3_entities": [],
            "relationships": [],
        },
        ensure_ascii=False,
    )


def _degraded_payload(
    *,
    input_chars: int,
    duration_ms: float,
    retry_count: int,
    error: str,
    offline_fallback: bool = False,
) -> Dict[str, Any]:
    return {
        "response": _empty_ontology_json(),
        "truncated": False,
        "input_chars": input_chars,
        "output_chars": 0,
        "duration_ms": duration_ms,
        "retry_count": retry_count,
        "parse_success": True,
        "degraded": True,
        "provider_error": str(error)[:300],
        "offline_fallback": offline_fallback,
    }


def validate_json_response(text: str) -> Dict[str, Any]:
    """Validate JSON response format and structure.

    [ENHANCEMENT] Supports multiple field naming conventions:
    - L0_entities, L1_entities, L2_entities, L2_5_entities, L3_entities, relationships
    - l0_entities, l1_entities, l2_entities, l25_entities, l3_entities
    - L0, L1, L2, L2.5, L3 (simplified format)
    - business_scenarios, mapping_rules, instances (alternative naming)
    """
    if not text or not text.strip():
        return {"valid": False, "error": "Empty response"}

    try:
        cleaned = text.strip().replace('```json', '').replace('```', '').strip()
        data = json.loads(cleaned)

        if not isinstance(data, dict):
            return {"valid": False, "error": "Response is not a JSON object"}

        entity_key_groups = [
            ["L0_entities", "l0_entities", "L0"],
            ["L1_entities", "l1_entities", "L1"],
            ["L2_entities", "l2_entities", "L2", "business_scenarios"],
            ["L2_5_entities", "L25_entities", "l25_entities", "L2.5", "mapping_rules"],
            ["L3_entities", "l3_entities", "L3", "instances"],
        ]

        relationship_keys = ["relationships", "RELATIONSHIPS"]

        for key_group in entity_key_groups:
            found_key = None
            for key in key_group:
                if key in data:
                    found_key = key
                    break

            if found_key is None:
                return {"valid": False, "error": f"Missing required key (one of): {', '.join(key_group)}"}

            if not isinstance(data[found_key], list):
                return {"valid": False, "error": f"{found_key} is not an array"}

            for idx, item in enumerate(data[found_key]):
                if not isinstance(item, dict):
                    return {"valid": False, "error": f"{found_key}[{idx}] is not an object"}

        has_relationships = False
        for key in relationship_keys:
            if key in data:
                has_relationships = True
                if isinstance(data[key], list):
                    for idx, item in enumerate(data[key]):
                        if not isinstance(item, dict):
                            return {"valid": False, "error": f"{key}[{idx}] is not an object"}
                break

        return {"valid": True, "error": None}

    except json.JSONDecodeError as e:
        return {"valid": False, "error": f"JSON parse error: {str(e)}"}
    except Exception as e:
        return {"valid": False, "error": f"Validation error: {str(e)}"}


_TRUNCATION_MARKERS = (
    "Unterminated",
    "Expecting value",
    "Expecting ','",
    "Expecting ':'",
    "Expecting property name",
)


def _json_body(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    return cleaned


def json_response_was_truncated(text: str, error: str) -> bool:
    """True when the model stopped inside a JSON value, not when it wrote prose.

    The char offset has to sit at the end of the body. A syntax error in the
    middle of an otherwise finished object is not truncation.
    """
    body = _json_body(text)
    if not body.startswith("{") and not body.startswith("["):
        return False
    err = error or ""
    if not any(marker in err for marker in _TRUNCATION_MARKERS):
        return False
    match = re.search(r"char (\d+)", err)
    if match is None:
        return True
    return int(match.group(1)) >= max(0, len(body) - 80)


def build_retry_prompt(original_prompt: str, failed_response: str, error: str) -> str:
    """Build enhanced retry prompt with error feedback."""
    retry_instructions = f"""

========================================
IMPORTANT CORRECTION INSTRUCTIONS:
Your previous response failed JSON validation with error: {error}
Please fix the following issues:
1. Ensure ALL commas are present between array/object elements
2. Close ALL brackets and braces properly
3. Use ONLY double quotes (no single quotes)
4. Escape internal quotes within strings using backslash (\\)
5. NO trailing commas before closing brackets/braces
6. Include ALL required fields: L0_entities, L1_entities, L2_entities, L2_5_entities, L3_entities, relationships
7. Output ONLY pure JSON, no markdown code blocks, no explanations

Previous invalid response (for reference):
{failed_response[:500]}...
========================================

"""
    return original_prompt + retry_instructions


async def _heartbeat_ticker(
    done_event: asyncio.Event,
    started_at: float,
    interval: float = 20.0,
    label: str = "Waiting for API response",
) -> None:
    """Emit periodic liveness ticks while an async LLM call is in flight.

    DeepSeek calls routinely take 1-6 minutes; without a heartbeat the terminal
    shows a single ``→ Waiting...`` line and looks frozen. This prints elapsed
    seconds every ``interval`` so the user can confirm the call is still connected.
    """
    last_tick = started_at
    try:
        while not done_event.is_set():
            await asyncio.sleep(1.0)
            if done_event.is_set():
                break
            now = time.time()
            if now - last_tick >= interval:
                elapsed = now - started_at
                print(f"[LLM CALL] ⏳ {label}... {elapsed:.0f}s elapsed", flush=True)
                last_tick = now
    except asyncio.CancelledError:
        pass


async def call_llm(
    llm, max_tokens, temperature, timeout,
    current_run_id, tracer, get_metadata,
    prompt: str, **kwargs,
) -> Dict[str, Any]:
    """Make async LLM call with tracing, smart routing, and JSON validation with retries.

    ``get_metadata`` is a zero-arg callable returning the active skill metadata
    (used for failure memory attribution); ``tracer`` / ``current_run_id`` are
    used for optional execution tracing.

    Returns:
        Dict containing:
            - response: The LLM response text
            - truncated: Boolean indicating if response likely truncated
            - input_chars: Input prompt character count
            - output_chars: Output response character count
            - duration_ms: Call duration in milliseconds
            - retry_count: Number of retries used
            - parse_success: Whether JSON parse succeeded
    """
    call_start = time.time()

    llm_provider = os.environ.get("LLM_PROVIDER", "unknown").upper()
    llm_class_name = llm.__class__.__name__
    llm_model = getattr(llm, 'model', 'unknown') if hasattr(llm, 'model') else 'unknown'

    MAX_INPUT_CHARS = int(os.environ.get("LLM_MAX_INPUT_CHARS", "80000"))
    MAX_JSON_RETRIES = int(os.environ.get("LLM_MAX_JSON_RETRIES", "2"))
    RETRY_DELAY_SECONDS = int(os.environ.get("LLM_RETRY_DELAY_SECONDS", "3"))
    offline_mode = os.environ.get("LLM_OFFLINE_MODE", "0").lower() in {"1", "true", "yes"}
    # Default ON: transient provider failures degrade instead of aborting the pipeline.
    degraded_on_failure = os.environ.get("LLM_DEGRADED_ON_FAILURE", "1").lower() in {
        "1", "true", "yes", "on",
    }

    input_chars = len(prompt)

    if input_chars > MAX_INPUT_CHARS:
        warning_msg = f"[LLM CALL] ⚠️ Input budget exceeded: {input_chars} > {MAX_INPUT_CHARS}. Truncating prompt."
        logger.warning(warning_msg)
        print(warning_msg)
        prompt = prompt[:MAX_INPUT_CHARS]
        input_chars = MAX_INPUT_CHARS

    complexity = kwargs.get("complexity", "auto")
    if complexity == "auto":
        if len(prompt) > 15000:
            complexity = "complex"
            print(f"[LLM CALL] ⚠️ Long prompt detected, routing to complex model")

    print(f"[LLM CALL] Provider: {llm_provider} | Client: {llm_class_name} | Model: {llm_model}")
    print(f"[LLM CALL] Prompt: {input_chars} chars | Max Tokens: {max_tokens} | Temperature: {temperature} | Complexity: {complexity}")
    print(f"[LLM CALL] Timeout: {timeout} seconds | Max JSON Retries: {MAX_JSON_RETRIES}")
    print(f"[LLM CALL] Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}")

    response_text = ""
    truncated = False
    retry_count = 0
    parse_success = False
    total_duration_ms = 0
    # C-3：累计最后一次 attempt 的 token 估算用于治理层监控
    last_prompt_tokens = 0
    last_response_tokens = 0

    if offline_mode:
        return _degraded_payload(
            input_chars=input_chars,
            duration_ms=0,
            retry_count=0,
            error="LLM_OFFLINE_MODE",
            offline_fallback=True,
        )

    while retry_count <= MAX_JSON_RETRIES:
        try:
            call_start_inner = time.time()
            logger.info(f"Calling LLM (attempt {retry_count + 1}/{MAX_JSON_RETRIES + 1}, provider={llm_provider}, model={llm_model})")
            print(f"[LLM CALL] → Waiting for API response (attempt {retry_count + 1})...", flush=True)

            done_event = asyncio.Event()
            heartbeat = asyncio.ensure_future(
                _heartbeat_ticker(done_event, call_start_inner)
            )
            try:
                response_text = await asyncio.wait_for(
                    llm.call_with_retry(prompt, temperature=temperature, max_tokens=max_tokens, complexity=complexity, use_cache=True),
                    timeout=timeout
                )
            finally:
                done_event.set()
                heartbeat.cancel()

            inner_duration = (time.time() - call_start_inner) * 1000
            total_duration_ms += inner_duration

            prompt_tokens = len(prompt) // 4
            response_tokens = len(response_text) // 4 if response_text else 0
            # C-3：记录最后一次 attempt 的估算用于治理层 monitor 汇总
            last_prompt_tokens = max(last_prompt_tokens, prompt_tokens)
            last_response_tokens = max(last_response_tokens, response_tokens)

            max_response_chars = max_tokens * 4
            if response_text and len(response_text) >= max_response_chars * 0.9:
                truncated = True
                logger.warning(f"LLM response likely truncated: {len(response_text)} chars near max {max_response_chars}")
                print(f"[LLM CALL] ⚠️ Response likely truncated (near max_tokens)")

            if not response_text or not response_text.strip():
                logger.error(f"LLM returned empty response")
                print(f"[LLM CALL] ❌ Empty response | Duration: {inner_duration:.2f}ms")
                if retry_count < MAX_JSON_RETRIES:
                    retry_count += 1
                    print(f"[LLM CALL] 🔄 Retrying (attempt {retry_count + 1}/{MAX_JSON_RETRIES + 1}) after {RETRY_DELAY_SECONDS}s delay...")
                    await asyncio.sleep(RETRY_DELAY_SECONDS)
                    continue
                else:
                    print(f"[LLM CALL] ⚠️ Max retries reached for empty response")
                    break
            else:
                logger.info(f"LLM response received: {len(response_text)} chars, preview: {response_text[:200]}")
                print(f"[LLM CALL] ✅ Success | Duration: {inner_duration:.2f}ms | Prompt tokens: ~{prompt_tokens} | Response tokens: ~{response_tokens}")
                print(f"[LLM CALL] Response preview: {response_text[:150]}...")

            parse_result = validate_json_response(response_text)
            if parse_result["valid"]:
                parse_success = True
                print(f"[LLM CALL] ✅ JSON validation passed")
                break
            else:
                print(f"[LLM CALL] ❌ JSON validation failed: {parse_result['error']}")
                logger.warning(f"JSON validation failed: {parse_result['error']}, attempt {retry_count + 1}")

                # Skip a same-prompt retry when it cannot change the outcome:
                #  - Truncated JSON: the body starts with { or [ and the parse
                #    error sits at the end. The extractor shortens the document.
                #  - Short (< 100 chars): a fragment, not a schema payload.
                #  - Long prose: HTML, a refusal, or an explanation. Repair
                #    cannot turn that into the schema, and a second call repeats it.
                _resp_stripped_len = len(response_text.strip())
                # Truncated JSON still starts with '{' or '['. Retrying the same
                # prompt reproduces the cutoff. Mark it truncated and return so
                # the extractor shortens the document and calls once more.
                if truncated or json_response_was_truncated(response_text, parse_result.get("error") or ""):
                    truncated = True
                    print(
                        f"[LLM CALL] ⏭️ Truncated JSON ({_resp_stripped_len} chars); "
                        "skipping same-prompt retry"
                    )
                    logger.warning(
                        "Truncated JSON (%d chars); deferring to shorter-input retry",
                        _resp_stripped_len,
                    )
                    break
                if _resp_stripped_len < 100:
                    print(f"[LLM CALL] ⏭️ Skipping retry for short response ({_resp_stripped_len} chars) - likely LLM error")
                    logger.warning(f"Skipping retry: response too short ({_resp_stripped_len} chars), likely LLM error")
                    break
                if _resp_stripped_len > 500:
                    print(f"[LLM CALL] ⏭️ Skipping retry for long non-JSON response ({_resp_stripped_len} chars) - prose/refusal won't fix by retry")
                    logger.warning(
                        "Skipping retry: response long (%d chars) but non-JSON — "
                        "likely prose/reasoning/refusal, prompt schema mismatch; "
                        "preview: %s",
                        _resp_stripped_len,
                        response_text[:120].replace("\n", " "),
                    )
                    break

                if retry_count < MAX_JSON_RETRIES:
                    retry_count += 1
                    print(f"[LLM CALL] 🔄 Retrying (attempt {retry_count + 1}/{MAX_JSON_RETRIES + 1}) after {RETRY_DELAY_SECONDS}s delay...")
                    await asyncio.sleep(RETRY_DELAY_SECONDS)

                    prompt = build_retry_prompt(prompt, response_text, parse_result["error"])
                    input_chars = len(prompt)
                else:
                    print(f"[LLM CALL] ⚠️ Max retries ({MAX_JSON_RETRIES}) reached, using repair mechanism")
                    break

        except Exception as e:
            inner_duration = (time.time() - call_start_inner) * 1000 if 'call_start_inner' in locals() else 0
            total_duration_ms += inner_duration
            transient = is_transient_provider_error(e)

            if transient and retry_count < MAX_JSON_RETRIES:
                retry_count += 1
                delay = RETRY_DELAY_SECONDS * (2 ** (retry_count - 1))
                print(
                    f"[LLM CALL] ⚠️ Transient provider error ({e.__class__.__name__}): "
                    f"waiting {delay}s before retry {retry_count + 1}/{MAX_JSON_RETRIES + 1}"
                )
                await asyncio.sleep(delay)
                continue

            if current_run_id:
                tracer.record_llm_call(
                    run_id=current_run_id,
                    step_name="TrueUnifiedExtractor._call_llm",
                    prompt=prompt[:500] + "..." if len(prompt) > 500 else prompt,
                    response=f"ERROR: {str(e)}",
                    model=getattr(llm, '__class__.__name__', 'unknown'),
                    duration_ms=total_duration_ms,
                    token_usage={},
                    entity_ids=[],
                    metadata={
                        "input_chars": input_chars,
                        "output_chars": 0,
                        "truncated": False,
                        "parse_success": False,
                        "retry_count": retry_count,
                        "error": str(e),
                        "transient": transient,
                        "degraded": degraded_on_failure,
                    }
                )

            # C-3 / C-1治理层：记录异常失败调用，让成本/预算统计完整
            try:
                _monitor_record_llm(
                    provider=llm_provider,
                    model=str(llm_model),
                    prompt_tokens=int(last_prompt_tokens),
                    completion_tokens=int(last_response_tokens),
                    total_tokens=int(last_prompt_tokens + last_response_tokens),
                    duration_ms=int(total_duration_ms),
                    success=False,
                    error_message=str(e)[:200] if e else "",
                )
            except Exception:
                pass

            if offline_mode or (degraded_on_failure and transient):
                logger.warning(
                    "LLM call failed (%s); returning degraded empty ontology: %s",
                    "offline" if offline_mode else "transient",
                    e,
                )
                if "timeout" in str(e).lower() or "504" in str(e):
                    record_llm_timeout_failure(
                        error_message=f"LLM timeout: {e}",
                        skill_used=get_metadata().name,
                    )
                else:
                    record_extraction_failure(
                        error_message=f"LLM call degraded: {e}",
                        failure_type="llm_call_degraded",
                        severity="medium",
                    )
                return _degraded_payload(
                    input_chars=input_chars,
                    duration_ms=total_duration_ms,
                    retry_count=retry_count,
                    error=str(e),
                    offline_fallback=offline_mode,
                )

            if "timeout" in str(e).lower() or "504" in str(e):
                record_llm_timeout_failure(
                    error_message=f"LLM timeout: {e}",
                    skill_used=get_metadata().name
                )
            else:
                record_extraction_failure(
                    error_message=f"LLM call failed: {e}",
                    failure_type="llm_call_failed",
                    severity="high"
                )

            logger.error(f"LLM call failed: {e}")
            raise

    if current_run_id:
        entity_ids = []
        try:
            cleaned = response_text.strip().replace('```json', '').replace('```', '').strip()
            data = json.loads(cleaned)
            if isinstance(data, dict):
                for key in ["L1_entities", "L2_entities", "L2_5_entities", "l1_entities", "l2_entities", "l25_entities"]:
                    entities = data.get(key, [])
                    if isinstance(entities, list):
                        for e in entities:
                            if isinstance(e, dict) and e.get("meta_id"):
                                entity_ids.append(e["meta_id"])
        except Exception:
            pass

        tracer.record_llm_call(
            run_id=current_run_id,
            step_name="TrueUnifiedExtractor._call_llm",
            prompt=prompt[:500] + "..." if len(prompt) > 500 else prompt,
            response=response_text[:500] + "..." if len(response_text) > 500 else response_text,
            model=getattr(llm, '__class__.__name__', 'unknown'),
            duration_ms=total_duration_ms,
            token_usage={"estimated": len(response_text) // 4},
            entity_ids=entity_ids[:20],
            metadata={
                "input_chars": input_chars,
                "output_chars": len(response_text),
                "truncated": truncated,
                "parse_success": parse_success,
                "retry_count": retry_count
            }
        )

    # C-3 / C-1 治理层：记录成功/失败调用（无论 current_run_id 是否有值都上报）
    try:
        _monitor_record_llm(
            provider=llm_provider,
            model=str(llm_model),
            prompt_tokens=int(last_prompt_tokens),
            completion_tokens=int(last_response_tokens),
            total_tokens=int(last_prompt_tokens + last_response_tokens),
            duration_ms=int(total_duration_ms),
            success=bool(parse_success),
        )
    except Exception:
        pass

    return {
        "response": response_text,
        "truncated": truncated,
        "input_chars": input_chars,
        "output_chars": len(response_text),
        "duration_ms": total_duration_ms,
        "retry_count": retry_count,
        "parse_success": parse_success,
        "degraded": False,
    }
