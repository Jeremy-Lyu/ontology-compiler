"""Compat shim (2026-10-16). TODO(remove 2026-11-30): 带日期删除，见 STATUS_MAP §4-4.

Alias 到 canonical 模块对象（非转发），使 mock.patch("...services.extraction_packet_adapter.X")
与 canonical 命名空间共享同一 dict —— 迁移前测试无需改动即可生效。
"""
import importlib as _il
import sys as _sys

_c = _il.import_module("ontology_compiler.services.extraction.extraction_packet_adapter")
_sys.modules[__name__] = _c
