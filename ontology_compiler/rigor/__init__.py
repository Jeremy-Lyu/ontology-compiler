"""
RIGOR Module - Iterative Pattern Refinement

Provides RIGOR (Recursive Iterative Generation Of Refined Ontology)
extraction for improving ontology fragment quality.

Note: RIGOR engine (RIGORExtractor, RIGORState, etc.) has been archived
and is no longer available. Use UnifiedLLMClient for LLM operations.
"""

try:
    from .engine import RIGORExtractor, RIGORState, PatternRefiner, RefineFailedError, SampleValidator
except ImportError:
    class RIGORExtractor: pass
    class RIGORState: pass
    class PatternRefiner: pass
    class RefineFailedError(Exception): pass
    class SampleValidator: pass

from .llm_backend import (
    BaseLLMClient,
    DifyAPIError,
    DifyClient,
    DeepSeekClient,
    UnifiedLLMClient,
)

__all__ = [
    'RIGORExtractor',
    'RIGORState',
    'PatternRefiner',
    'RefineFailedError',
    'SampleValidator',
    'BaseLLMClient',
    'DifyAPIError',
    'DeepSeekClient',
    'DifyClient',
    'UnifiedLLMClient',
]
