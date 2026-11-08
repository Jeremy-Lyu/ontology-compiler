"""Domain-layer RAG models (B-5, 2026-08-17).

Canonical home of RAG data contracts (retrieval/answer/query DTOs). The
top-level ``src.application.rag.models`` / ``src.application.rag.unified_query``
now re-export from here for backward compatibility; new code should import from
``src.application.rag.domain``.
"""