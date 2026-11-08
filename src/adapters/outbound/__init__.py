"""Outbound adapters for external system integration.

BATCH 0 Consolidation (2026-06-29):
- Unified exports for canonical adapters
- Marked deprecated modules
- Adapters only handle protocol conversion, no business logic
"""

# Core adapters
from .fs_outputs_artifact_store import OutputsArtifactStore
from .deepseek_llm_adapter import DeepSeekLLMAdapter
from .dify_llm_adapter import DifyLLMAdapter
from .local_rag_adapter import LocalRAGAdapter
from .neo4j_rag_adapter import Neo4jRAGAdapter
from .session_replay_store_adapter import SessionReplayStoreAdapter

# Skill registry adapter
from .skills.skill_registry_adapter import SkillRegistryAdapter

# Document reader adapter
from .parsers.file_parser_document_reader import FileParserDocumentReader

# DEPRECATED - kept for backward compatibility
import warnings

__all__ = [
    # Core adapters
    "OutputsArtifactStore",
    "DeepSeekLLMAdapter",
    "DifyLLMAdapter",
    "LocalRAGAdapter",
    "Neo4jRAGAdapter",
    "SessionReplayStoreAdapter",
    # Skill registry
    "SkillRegistryAdapter",
    # Document reader
    "FileParserDocumentReader",
]
