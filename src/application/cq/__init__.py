"""CQ (Competency Question) domain package — CQ_PIPELINE_IMPLEMENTATION_PLAN M0.

Pure data models + registry + artifact writers for the CQ-driven pipeline.
No side effects at import time; the pipeline stays byte-identical when no CQ
catalog exists (M2 验收约束).

Layering (check_import_boundary safe): ``src.application.cq`` never imports
``ontology_compiler`` implementation modules.
"""

from src.application.cq.cq_models import (
    CQ,
    CQCatalog,
    CQMapping,
    CQValidationResult,
    CQStatus,
    allowed_transitions,
)
from src.application.cq.cq_registry import CQRegistry, default_catalog_path

__all__ = [
    "CQ",
    "CQCatalog",
    "CQMapping",
    "CQValidationResult",
    "CQStatus",
    "allowed_transitions",
    "CQRegistry",
    "default_catalog_path",
]
