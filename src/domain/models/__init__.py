"""
Domain Models - 领域层数据模型

Stable entry point:
- import from here if you need backward-compatible flat exports.

Migration entry points:
- `src.domain.models.core`
- `src.domain.models.pipeline`
- `src.domain.models.rag`
- `src.domain.models.agent`
- `src.domain.models.quality`

Recommended import order for new code:
1. Prefer the specific subpackage that owns the model.
2. Use `src.domain.models` only for backward-compatible flat imports.
3. Keep flat imports temporary during migration.
4. Move imports to the subpackage as soon as the call site is updated.
"""

from .core import *
from .pipeline import *
from .rag import *
from .agent import *
from .quality import *
from .enums import *

import sys as _sys

from .core import __all__ as _core_all
from .pipeline import __all__ as _pipeline_all
from .rag import __all__ as _rag_all
from .agent import __all__ as _agent_all
from .quality import __all__ as _quality_all
from .enums import (
    LayerType,
    OntologyRelationType,
    OntologyEntityType,
    CanonicalStatus,
    EntityType,
    RelationType,
    AttrValueType,
    PipelinePhase,
)

__all__ = [
    *_core_all,
    *_pipeline_all,
    *_rag_all,
    *_agent_all,
    *_quality_all,
    "LayerType",
    "OntologyRelationType",
    "OntologyEntityType",
    "CanonicalStatus",
    "EntityType",
    "RelationType",
    "AttrValueType",
    "PipelinePhase",
]

# Backward-compat alias for legacy imports that still referenced the removed
# flat module path during migration.
_sys.modules.setdefault(f"{__name__}.core_contracts", _sys.modules.get(f"{__name__}.core"))
