"""RAG infrastructure layer (B-5, 2026-08-17).

Owns the SDK-touching adapters: dify_client, embedding, vector_index_manager,
neo4j_retriever, neo4j_sync_service. Application use-cases import these only
through the top-level compat shims or the RAG search ports; the bounded-context
rule is ``application -> domain / ports`` and ``infrastructure referenced only
by application``.
"""