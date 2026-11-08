"""LLM Provider - Unified interface for LLM backends.

[RESTORED-2026-06-10] Recreated from .pyc backup to fix RAG CLI imports.
[FIXED-2026-07-21] DIFY backend now uses DIFY native API (/chat-messages) instead of OpenAI-compatible endpoint.
"""
from __future__ import annotations

import os
import json
import requests
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List


class LLMBackend(Enum):
    """Supported LLM backends."""
    DEEPSEEK = "deepseek"
    OPENAI = "openai"
    AZURE = "azure"
    DIFY = "dify"


@dataclass
class LLMRequest:
    """LLM request parameters."""
    prompt: str
    max_tokens: int = 2000
    temperature: float = 0.3
    system_prompt: Optional[str] = None
    

@dataclass
class LLMResponse:
    """LLM response wrapper."""
    content: str
    model: str = "unknown"
    tokens_used: int = 0
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class UnifiedLLMProvider:
    """Unified provider for LLM operations.
    
    Supports multiple backends with automatic failover.
    Reads configuration from environment variables:
    - LLM_PROVIDER: primary provider (dify/deepseek)
    - LLM_FALLBACK_PROVIDERS: comma-separated fallback providers
    - DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_BASE_URL
    - DIFY_API_KEY, DIFY_BASE_URL, DIFY_MODEL, DIFY_APP_TYPE, DIFY_USERNAME
    """
    
    def __init__(self, backend: LLMBackend = None, api_key: str = None):
        provider_env = os.getenv("LLM_PROVIDER", "deepseek").lower().strip()
        if backend:
            self.backend = backend
        elif provider_env == "dify":
            self.backend = LLMBackend.DIFY
        else:
            self.backend = LLMBackend.DEEPSEEK
        self.api_key = api_key
        self._client = None
        
    def _get_client(self):
        """Lazy initialization of LLM client."""
        if self._client is None:
            if self.backend == LLMBackend.DEEPSEEK:
                try:
                    import openai
                    key = self.api_key or os.getenv("DEEPSEEK_API_KEY", "")
                    self._client = openai.OpenAI(
                        api_key=key,
                        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
                    )
                except ImportError:
                    raise RuntimeError("openai package required for DeepSeek backend")
            elif self.backend == LLMBackend.DIFY:
                # DIFY使用原生API，不需要openai客户端
                # 通过complete_dify方法直接调用DIFY原生端点
                self._client = "dify_native"
            else:
                raise NotImplementedError(f"Backend {self.backend} not yet implemented")
        return self._client
    
    def _get_model_name(self) -> str:
        """Get model name from environment based on backend."""
        if self.backend == LLMBackend.DEEPSEEK:
            return os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        elif self.backend == LLMBackend.DIFY:
            return os.getenv("DIFY_MODEL", "dify-default")
        return "unknown"
    
    def _complete_dify(self, request: LLMRequest) -> LLMResponse:
        """Call DIFY native API (/chat-messages or /completion-messages).

        DIFY API format:
        - Endpoint: {base_url}/chat-messages (chat) or {base_url}/completion-messages (completion)
        - Headers: Authorization: Bearer {api_key}
        - Payload: {inputs: {}, query: prompt, response_mode: "streaming", user: username}
        - Response: SSE stream, parse "answer" field from data lines

        [FIX-2026-08-17] honor project_memory hard constraint:
          - internal domains (*.internal.example.com, *.corp.example.com, localhost)
            → trust_env=False (bypass enterprise proxy that pollutes SSE)
          - external domains → trust_env=True
          - global kill-switch: LLM_DISABLE_PROXY=true forces trust_env=False
            for ALL domains.
        """
        base_url = os.getenv("DIFY_BASE_URL", "https://api.dify.ai/v1")
        api_key = self.api_key or os.getenv("DIFY_API_KEY", "")
        app_type = os.getenv("DIFY_APP_TYPE", "chat").lower()
        username = os.getenv("DIFY_USERNAME", "rag-user")
        timeout = int(os.getenv("DIFY_TIMEOUT_SECONDS", "300"))

        # Determine trust_env based on target domain.
        # Internal enterprise domains must bypass the corporate proxy; the proxy
        # occasionally returns empty SSE streams for internal endpoints.
        disable_proxy_kill_switch = os.getenv("LLM_DISABLE_PROXY", "").lower() == "true"
        internal_domain_markers = (
            "internal.example.com", "corp.example.com",
            "localhost", "127.0.0.1",
        )
        is_internal = any(marker in base_url for marker in internal_domain_markers)
        trust_env = (not is_internal) and (not disable_proxy_kill_switch)

        # Build prompt with system prompt if provided
        prompt = request.prompt
        if request.system_prompt:
            prompt = f"{request.system_prompt}\n\n{request.prompt}"

        # Build URL and payload based on app_type
        if app_type == "chat":
            url = f"{base_url}/chat-messages"
            payload = {
                "inputs": {},
                "query": prompt,
                "response_mode": "streaming",
                "user": username,
                "auto_generate_name": False,
            }
        else:
            url = f"{base_url}/completion-messages"
            payload = {
                "inputs": {},
                "query": prompt,
                "response_mode": "streaming",
                "user": username,
            }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Force a fresh TCP connection per request. The corporate Dify
            # reverse proxy sometimes drops HTTP/1.1 keep-alive sockets that
            # were already closed server-side (mirrors the infrastructure
            # dify_client.py hard constraint).
            "Connection": "close",
        }

        # Use a Session so we can pin trust_env (requests.post has no such kwarg).
        session = requests.Session()
        session.trust_env = trust_env
        # Explicit no-proxy for internal endpoints (belt-and-suspenders).
        if not trust_env:
            session.proxies.update({"http": "", "https": ""})

        try:
            response = session.post(
                url, json=payload, headers=headers, timeout=timeout, stream=True
            )
            response.raise_for_status()

            # Parse SSE streaming response
            answer_parts = []
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
                        if data.get("event") == "message" or data.get("answer"):
                            chunk = data.get("answer", "")
                            if chunk:
                                answer_parts.append(chunk)
                    except json.JSONDecodeError:
                        continue

            answer = "".join(answer_parts).strip()
            model_name = self._get_model_name()

            if not answer:
                return LLMResponse(
                    content="[LLM Error: Empty response from DIFY]",
                    model=model_name,
                    confidence=0.0,
                )

            return LLMResponse(
                content=answer,
                model=model_name,
                tokens_used=0,
            )
        except Exception as e:
            return LLMResponse(
                content=f"[LLM Error: {e}]",
                model="error",
                confidence=0.0,
            )
        finally:
            session.close()
    
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute LLM completion request."""
        # DIFY backend uses native API
        if self.backend == LLMBackend.DIFY:
            return self._complete_dify(request)
        
        # Other backends use OpenAI-compatible client
        client = self._get_client()
        model_name = self._get_model_name()
        
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                tokens_used=response.usage.total_tokens if response.usage else 0,
            )
        except Exception as e:
            return LLMResponse(
                content=f"[LLM Error: {e}]",
                model="error",
                confidence=0.0,
            )
    
    async def complete_async(self, request: LLMRequest) -> LLMResponse:
        """Async version of complete."""
        # For simplicity, delegate to sync version
        # In production, this should use async client
        return self.complete(request)

    def call_llm(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.3,
    ) -> str:
        """Public LLM call interface — simple string-in / string-out.

        [H7 FIX] Exposes a stable public method instead of letting callers poke
        inside ``self._client`` (private attribute).  Callers (e.g. QueryRewriter)
        MUST use this method and never rely on ``_client`` directly.

        Args:
            prompt: User-facing query / instruction text.
            system_prompt: Optional system instruction prepended to the prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Raw LLM answer text (stripped).
        """
        request = LLMRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        resp = self.complete(request)
        return (resp.content or "").strip()

    async def call_llm_async(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.3,
    ) -> str:
        """Async variant of :meth:`call_llm`."""
        return self.call_llm(
            prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
