"""Compat shim → canonical ``ontology_compiler.services.extraction.extraction_agents``.

TODO(remove 2026-11-30): dated shim deletion (PROJECT_CURRENT_STATUS §4-4).
"""
import importlib as _il
import sys as _sys

_c = _il.import_module("ontology_compiler.services.extraction.extraction_agents")
_sys.modules[__name__] = _c
