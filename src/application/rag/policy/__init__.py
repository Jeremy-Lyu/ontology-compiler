"""RAG policy layer (B-6, 2026-08-17).

Single home for cross-cutting retrieval/answer policies: the fallback cascade
order (RAG-06) and citation shape. Application use-cases read these policies
instead of scattering fallback ordering across call sites.
"""