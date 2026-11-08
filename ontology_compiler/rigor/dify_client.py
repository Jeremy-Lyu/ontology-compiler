"""Dify Knowledge Base Q&A client for Ontology-RAG.

This module provides a thin HTTP client for Dify "chat" apps (chat-messages).
It is intentionally dependency-light and uses `httpx` (already in requirements).

Sensitive information (API keys) must NOT be hardcoded. All configuration is read
from environment variables.

No exceptions are swallowed.
"""

from __future__ import annotations

import logging
import os
import re
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from ontology_compiler.config import load_runtime_env
from src.domain.errors import DataContractError, TransientInfraError

from .models import RESPONSE_MODE_BLOCKING, RESPONSE_MODE_STREAMING


_CAP_ID_RE = re.compile(r"\bCAP_[A-Z0-9_]+\b")

logger = logging.getLogger(__name__)


def _extract_text_from_outputs(outputs: Any) -> Optional[str]:
    if isinstance(outputs, str) and outputs.strip():
        return outputs.strip()
    if isinstance(outputs, dict):
        for k in ("answer", "text", "result", "output"):
            v = outputs.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        for v in outputs.values():
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def _parse_dify_error_message(payload: Any) -> str:
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    return ""


def _is_transient_dify_error(status_code: int, payload: Any, text: str) -> bool:
    if status_code in {429, 502, 503, 504}:
        return True

    lowered = " ".join(
        part.lower()
        for part in [
            text or "",
            _parse_dify_error_message(payload),
            json.dumps(payload, ensure_ascii=False) if isinstance(payload, dict) else "",
        ]
        if part
    )
    return any(token in lowered for token in ("no_tokens", "no available tokens", "server_error"))


def _raise_for_dify_http_error(resp: httpx.Response, *, cfg: DifyConfig, url: str, input_keys: List[str]) -> None:
    endpoint = url.split(cfg.base_url)[-1] if cfg.base_url in url else url
    raw_text = (resp.text or "")[:500]

    payload: Any = None
    try:
        payload = resp.json()
    except Exception:
        payload = None

    message = _parse_dify_error_message(payload) or raw_text or f"HTTP {resp.status_code}"
    detail = (
        f"dify http {resp.status_code}: {message}; "
        f"app_type={cfg.app_type}; endpoint={endpoint}; inputs_keys={input_keys}"
    )

    if _is_transient_dify_error(resp.status_code, payload, raw_text):
        raise TransientInfraError(detail)
    raise DataContractError(detail)


@dataclass(frozen=True)
class DifyConfig:
    """Dify config loaded from environment variables."""

    base_url: str
    api_key: str
    username: str
    timeout_s: float
    app_type: str
    response_mode: str = RESPONSE_MODE_BLOCKING
    input_fields: tuple[str, ...] = ()
    inputs_json: Dict[str, Any] = field(default_factory=dict)
    input_values: Dict[str, str] = field(default_factory=dict)


def _maybe_load_env_from_files() -> None:
    """Load env vars from standard project env files.

    Delegates to ontology_compiler.config.load_env_file to ensure compiler/rag
    share exactly the same loading behavior and precedence.
    """

    # Priority: config/env.env (base) -> env.env -> env.local.env (override)
    load_runtime_env()


def load_dify_config_from_env() -> DifyConfig:
    """Load Dify config from environment.

    Required:
    - DIFY_BASE_URL
    - DIFY_API_KEY

    Optional:
    - DIFY_USERNAME (default: "ontology_rag")
    - LLM_TIMEOUT (default: 300)

    Raises:
        DataContractError: if required variables are missing.
    """

    _maybe_load_env_from_files()

    base_url = (os.getenv("DIFY_BASE_URL") or "").strip().rstrip("/")
    api_key = (os.getenv("DIFY_API_KEY") or "").strip()
    username = (os.getenv("DIFY_USERNAME") or "ontology_rag").strip()
    timeout_raw = (os.getenv("LLM_TIMEOUT") or "300").strip()
    app_type = (os.getenv("DIFY_APP_TYPE") or "chat").strip().lower()
    response_mode = (os.getenv("DIFY_RESPONSE_MODE") or RESPONSE_MODE_BLOCKING).strip().lower()
    input_fields_raw = (os.getenv("DIFY_INPUT_FIELDS") or "").strip()
    inputs_json_raw = (os.getenv("DIFY_INPUTS_JSON") or "").strip()

    if not base_url:
        raise DataContractError("missing env var: DIFY_BASE_URL")
    if not api_key:
        raise DataContractError("missing env var: DIFY_API_KEY")

    if api_key.strip().upper() in {"<REDACTED>", "REDACTED"}:
        raise DataContractError(
            "DIFY_API_KEY is a placeholder ('<REDACTED>'). Set a real app token in env.local.env or process environment."
        )
    if len(api_key.strip()) < 20:
        raise DataContractError(
            f"DIFY_API_KEY looks too short (len={len(api_key.strip())}). "
            "Check for truncation, extra spaces after '=', or wrong variable name."
        )

    if app_type not in {"chat", "workflow"}:
        raise DataContractError(f"unsupported DIFY_APP_TYPE={app_type!r}; expected 'chat'|'workflow'")

    if response_mode not in {RESPONSE_MODE_BLOCKING, RESPONSE_MODE_STREAMING}:
        raise DataContractError(
            f"invalid env var DIFY_RESPONSE_MODE={response_mode!r}; expected blocking|streaming"
        )

    try:
        timeout_s = float(timeout_raw)
    except Exception as exc:
        raise DataContractError(f"invalid env var LLM_TIMEOUT={timeout_raw!r}: {exc}") from exc

    # Parse optional inputs_json
    inputs_json: Dict[str, Any] = {}
    if inputs_json_raw:
        try:
            obj = json.loads(inputs_json_raw)
        except Exception as exc:
            raise DataContractError(f"invalid env var DIFY_INPUTS_JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise DataContractError("invalid env var DIFY_INPUTS_JSON: must be a JSON object")
        inputs_json = obj

    # Collect per-field overrides: DIFY_INPUT_VALUE_<field>
    input_values: Dict[str, str] = {}
    prefix = "DIFY_INPUT_VALUE_"
    for k, v in os.environ.items():
        if not isinstance(k, str) or not k.startswith(prefix):
            continue
        field = k[len(prefix) :]
        if not field:
            continue
        if field not in input_values and isinstance(v, str) and v.strip():
            input_values[field] = v.strip()

    input_fields = tuple(x.strip() for x in input_fields_raw.split(",") if x.strip())

    return DifyConfig(
        base_url=base_url,
        api_key=api_key,
        username=username,
        timeout_s=timeout_s,
        app_type=app_type,
        response_mode=response_mode,
        input_fields=input_fields,
        inputs_json=inputs_json,
        input_values=input_values,
    )


def extract_cap_node_ids(text: str) -> List[str]:
    """Extract CAP_* node ids from a text response."""

    if not isinstance(text, str) or not text:
        return []
    return sorted(set(_CAP_ID_RE.findall(text)))


def _parse_sse_answer_lines(lines: List[str]) -> Dict[str, Any]:
    """Parse Dify SSE lines and extract an answer.

    We only extract minimal information and never return secrets.
    """

    answer: Optional[str] = None
    last_event: Dict[str, Any] = {}

    for raw in lines:
        line = (raw or "").strip()
        if not line:
            continue
        if not line.startswith("data:"):
            continue
        data_part = line[len("data:") :].strip()
        if not data_part or data_part == "[DONE]":
            continue
        try:
            evt = json.loads(data_part)
        except Exception:
            continue
        if isinstance(evt, dict):
            last_event = evt
            a = evt.get("answer")
            if isinstance(a, str) and a.strip():
                answer = a.strip()
            if answer is None:
                t = evt.get("text")
                if isinstance(t, str) and t.strip():
                    answer = t.strip()
            if answer is None:
                out = evt.get("outputs")
                answer = _extract_text_from_outputs(out) or answer
            if answer is None:
                data_obj = evt.get("data")
                if isinstance(data_obj, dict):
                    answer = _extract_text_from_outputs(data_obj.get("outputs")) or answer

    if answer:
        out = dict(last_event)
        out["answer"] = answer
        return out

    keys = sorted([k for k in last_event.keys() if isinstance(k, str)]) if isinstance(last_event, dict) else []
    raise DataContractError(f"dify sse response missing answer; last_event_keys={keys}")


def _stream_and_parse(
    client: httpx.Client,
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    cfg: DifyConfig,
    input_keys: List[str],
) -> Dict[str, Any]:
    """Make a streaming Dify request and parse the SSE response."""

    collected: List[str] = []
    try:
        with client.stream("POST", url, json=payload, headers=headers) as sresp:
            if sresp.status_code >= 400:
                _raise_for_dify_http_error(sresp, cfg=cfg, url=url, input_keys=input_keys)
            for line in sresp.iter_lines():
                if line is None:
                    continue
                collected.append(line)
    except (DataContractError, TransientInfraError):
        raise
    except Exception as exc:
        raise DataContractError(f"dify streaming request failed: {exc}") from exc

    return _parse_sse_answer_lines(collected)


def ask_dify_kb_qa(
    *,
    query: str,
    inputs: Optional[Dict[str, Any]] = None,
    config: Optional[DifyConfig] = None,
) -> Dict[str, Any]:
    """Ask Dify Knowledge Base Q&A using chat-messages.

    This function uses the Dify app configured by the API key.

    Returns:
        Raw JSON response object from Dify.

    Raises:
        DataContractError: for HTTP / response contract errors.
    """

    if not isinstance(query, str) or not query.strip():
        raise DataContractError("dify query must be non-empty string")

    cfg = config or load_dify_config_from_env()
    if cfg.app_type == "workflow":
        url = f"{cfg.base_url}/workflows/run"
    else:
        url = f"{cfg.base_url}/chat-messages"

    default_inputs: Dict[str, Any] = dict(cfg.inputs_json)
    default_inputs.update(cfg.input_values)

    # Convenience: populate specified input fields with query.
    if not inputs and cfg.input_fields:
        for name in cfg.input_fields:
            default_inputs.setdefault(name, query)

    final_inputs = inputs or default_inputs
    if cfg.app_type == "workflow":
        # For workflows, the prompt is usually passed through an input variable.
        # If no input is specified, default to a generic 'query' field.
        if not final_inputs:
            final_inputs = {"query": query}
        payload: Dict[str, Any] = {
            "inputs": final_inputs,
            "response_mode": cfg.response_mode,
            "user": cfg.username,
        }
    else:
        payload = {
            "inputs": final_inputs,
            "query": query,
            "response_mode": cfg.response_mode,
            "user": cfg.username,
        }

    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }

    input_keys = sorted([k for k in (final_inputs or {}).keys() if isinstance(k, str)])

    logger.info(
        "dify request start: app_type=%s response_mode=%s url=%s input_keys=%s",
        cfg.app_type,
        cfg.response_mode,
        url,
        input_keys,
    )

    try:
        # Dify calls must not use environment proxies by default.
        # This aligns with config/env.env guidance and avoids accidental corporate proxy routing.
        with httpx.Client(timeout=cfg.timeout_s, trust_env=False) as client:
            if cfg.response_mode == RESPONSE_MODE_STREAMING:
                return _stream_and_parse(client, url, payload, headers, cfg, input_keys)
            resp = client.post(url, json=payload, headers=headers)
    except TransientInfraError:
        raise
    except Exception as exc:
        raise DataContractError(f"dify request failed: {exc}") from exc

    if resp.status_code >= 400:
        _raise_for_dify_http_error(resp, cfg=cfg, url=url, input_keys=input_keys)

    ctype = (resp.headers.get("content-type") or "").lower()
    if "text/event-stream" in ctype:
        return _parse_sse_answer_lines(resp.text.splitlines())

    try:
        data = resp.json()
    except Exception as exc:
        raise DataContractError(f"dify invalid json response: {exc}") from exc

    if not isinstance(data, dict):
        raise DataContractError("dify response is not an object")

    # Workflow blocking response typically returns outputs under data.outputs.
    if cfg.app_type == "workflow":
        d = data.get("data")
        if isinstance(d, dict):
            out = d.get("outputs")
            extracted = _extract_text_from_outputs(out)
            if extracted:
                norm = dict(data)
                norm["answer"] = extracted
                return norm

    # Some Dify deployments return an event-like JSON (SSE shape) even in blocking mode.
    # If answer is empty but event is present, retry with streaming mode once.
    if cfg.response_mode == RESPONSE_MODE_BLOCKING:
        a = data.get("answer")
        ev = data.get("event")
        if isinstance(ev, str) and (not isinstance(a, str) or not a.strip()):
            logger.info("dify retry with streaming mode: url=%s", url)
            with httpx.Client(timeout=cfg.timeout_s, trust_env=False) as client:
                payload2 = dict(payload)
                payload2["response_mode"] = RESPONSE_MODE_STREAMING
                return _stream_and_parse(client, url, payload2, headers, cfg, input_keys)

    logger.info("dify request complete: app_type=%s response_mode=%s", cfg.app_type, cfg.response_mode)
    return data


def get_dify_answer_text(resp: Dict[str, Any]) -> str:
    """Extract answer text from Dify response.

    Dify chat-messages typically returns an object including an 'answer' field.

    Raises:
        DataContractError: if answer cannot be found.
    """

    if not isinstance(resp, dict):
        raise DataContractError("dify response is not a dict")

    def _first_nonempty_str(*vals: Any) -> Optional[str]:
        for v in vals:
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    def _coerce_text(v: Any) -> Optional[str]:
        if isinstance(v, str):
            return v.strip() if v.strip() else None
        # Some deployments may return a list of message parts.
        if isinstance(v, list):
            parts: List[str] = []
            for item in v:
                if isinstance(item, str) and item.strip():
                    parts.append(item.strip())
                elif isinstance(item, dict):
                    t = _first_nonempty_str(item.get("text"), item.get("content"), item.get("answer"))
                    if t:
                        parts.append(t)
            joined = "\n".join(parts).strip()
            return joined if joined else None
        if isinstance(v, dict):
            t = _first_nonempty_str(v.get("text"), v.get("content"), v.get("answer"))
            return t
        return None

    # Common shapes for Dify chat-messages.
    ans = _coerce_text(resp.get("answer")) or _first_nonempty_str(resp.get("output"), resp.get("text"))
    if ans:
        return ans

    # Some deployments wrap message content.
    msg = resp.get("message")
    if isinstance(msg, dict):
        ans = _coerce_text(msg.get("content")) or _coerce_text(msg.get("answer")) or _first_nonempty_str(msg.get("text"))
        if ans:
            return ans

    # Some setups wrap everything under `data`.
    data = resp.get("data")
    if isinstance(data, dict):
        ans = _coerce_text(data.get("answer")) or _first_nonempty_str(data.get("output"), data.get("text"))
        if ans:
            return ans
        outputs = data.get("outputs")
        if isinstance(outputs, dict):
            # Heuristic: try common output keys.
            ans = _coerce_text(outputs.get("answer")) or _first_nonempty_str(outputs.get("text"), outputs.get("result"))
            if ans:
                return ans

    meta = resp.get("metadata")
    if isinstance(meta, dict):
        # Some workflow/chatflow apps may stash output in metadata.
        ans = _first_nonempty_str(meta.get("answer"), meta.get("output"), meta.get("text"), meta.get("result"))
        if ans:
            return ans

        # Dify may return an empty `answer` but include an annotation reply.
        ann = meta.get("annotation_reply")
        if isinstance(ann, dict):
            ans = _coerce_text(ann.get("content")) or _coerce_text(ann.get("answer")) or _coerce_text(ann.get("text"))
            if ans:
                return ans

    # Safe diagnostics: show keys/types/lengths only, never dump full response.
    top_keys = sorted([k for k in resp.keys() if isinstance(k, str)])
    ans_val = resp.get("answer") if isinstance(resp, dict) else None
    ans_type = type(ans_val).__name__
    ans_len = len(ans_val) if isinstance(ans_val, str) else None
    data_keys: List[str] = []
    if isinstance(data, dict):
        data_keys = sorted([k for k in data.keys() if isinstance(k, str)])
    meta_keys: List[str] = []
    if isinstance(meta, dict):
        meta_keys = sorted([k for k in meta.keys() if isinstance(k, str)])
    raise DataContractError(
        "dify response missing answer; "
        f"top_keys={top_keys}; "
        f"data_keys={data_keys}; "
        f"meta_keys={meta_keys}; "
        f"answer_type={ans_type}; answer_len={ans_len}"
    )
