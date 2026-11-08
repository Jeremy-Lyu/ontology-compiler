"""LLM helper functions for the RAG validator.

[P2 RAG 收口 2026-09-03]: extracted from validator.py so the LLM provider
abstraction, B-mode caching, and response parsing own their own module.
validator.py re-imports these symbols for backward compatibility.
"""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from src.domain.errors import DataContractError
from ontology_compiler.config import get_config
from src.application.ports.llm_ports import LLMPort

from .infrastructure.dify_client import get_dify_answer_text

# G26-OPT-01: 统一 LLM Provider 抽象层
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from ontology_compiler.rigor.llm_provider import UnifiedLLMProvider, LLMRequest, LLMBackend  # noqa: E402

logger = logging.getLogger(__name__)

# B-mode 缓存：用于 strict_b=True 时的降级策略
# 缓存结构: {query_hash: {"answer": str, "timestamp": float, "confidence": float}}
_b_mode_cache: Dict[str, Dict[str, Any]] = {}


def _get_cached_answer(query: str) -> Optional[Dict[str, Any]]:
    """获取缓存的 B-mode 答案（如果存在且未过期）"""
    import time
    query_hash = hashlib.md5(query.encode()).hexdigest()[:16]
    cached = _b_mode_cache.get(query_hash)
    if cached:
        # 缓存有效期：24小时
        if time.time() - cached.get("timestamp", 0) < 86400:
            return cached
    return None


def _cache_answer(query: str, answer: str, confidence: float) -> None:
    """缓存 B-mode 成功的答案"""
    import time
    query_hash = hashlib.md5(query.encode()).hexdigest()[:16]
    _b_mode_cache[query_hash] = {
        "answer": answer,
        "timestamp": time.time(),
        "confidence": confidence,
    }


def _rag_llm_provider() -> str:
    cfg = get_config()
    return (cfg.ontology_rag_llm_provider or cfg.llm_provider or "dify").strip().lower()


# G26-OPT-01: 统一 LLM Provider 实例（延迟初始化）
_llm_provider: Optional[UnifiedLLMProvider] = None

def _get_llm_provider() -> UnifiedLLMProvider:
    """获取或创建统一 LLM Provider 实例"""
    global _llm_provider
    if _llm_provider is None:
        _llm_provider = UnifiedLLMProvider()
    return _llm_provider


def _ask_llm_unified(query: str, backend: Optional[LLMBackend] = None) -> Dict[str, Any]:
    """
    G26-OPT-01: 使用 UnifiedLLMProvider 的统一 LLM 调用

    替代原有的 _ask_deepseek_kb_qa + ask_dify_kb_qa 分散调用
    支持自动故障转移和多后端切换
    """
    if not isinstance(query, str) or not query.strip():
        raise DataContractError("query must be non-empty string")

    # 创建指定后端的 provider
    provider = UnifiedLLMProvider(backend=backend) if backend else _get_llm_provider()

    # 构建标准化请求
    request = LLMRequest(
        prompt=query.strip(),
        temperature=0.1,
        max_tokens=1200
    )

    try:
        # 执行调用
        response = provider.complete(request)

        return {
            "answer": response.content,
            "model": response.model,
            "usage": response.tokens_used,
            "latency_ms": getattr(response, "latency_ms", 0)
        }
    except Exception as exc:
        raise DataContractError(f"llm invocation failed: {exc}") from exc


# 保留旧函数名以保持兼容性，但内部使用新的统一实现
def _ask_deepseek_kb_qa(*, query: str, llm_port: Optional[LLMPort] = None) -> Dict[str, Any]:
    """【已迁移】使用 UnifiedLLMProvider 调用 DeepSeek"""
    return _ask_llm_unified(query, backend=LLMBackend.DEEPSEEK)


def _ask_dify_kb_qa(*, query: str) -> Dict[str, Any]:
    """【已迁移】使用 UnifiedLLMProvider 调用 Dify"""
    return _ask_llm_unified(query, backend=LLMBackend.DIFY)


ask_dify_kb_qa = _ask_dify_kb_qa


def _get_llm_answer_text(response: Dict[str, Any]) -> str:
    """
    G26-OPT-01: 统一的响应解析函数

    兼容旧的 get_dify_answer_text 和新的 UnifiedLLMProvider 响应格式
    """
    # 新的 UnifiedLLMProvider 格式
    if "answer" in response:
        return response.get("answer", "")

    # 旧的 Dify 格式（通过原始 import）
    return get_dify_answer_text(response)
