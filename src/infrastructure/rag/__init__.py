"""RAG infrastructure providers (B-4, 2026-08-17).

Owns the third-party SDK imports (neo4j / openai / cohere / sentence_transformers)
so ``src.application.rag`` stays SDK-free (RAG-05). Application code accesses
infrastructure through these modules or through the RAG search ports.
"""