"""CQ data models (M0 建契约).

CQ_PIPELINE_IMPLEMENTATION_PLAN.md §6 / §9-M0:

- 最小必填字段：``cq_id / cq_text / entities / relations / query_path / status``
- 状态机：``draft -> reviewed -> frozen -> deprecated``（仅 ``frozen`` 进入
  M2/M3 的验证与回归）
- 纯数据模型，无 IO、无副作用；schema 的机器可读版本在
  ``config/cq/cq_schema.json``，人类可读规范在 ``docs/CQ_SCHEMA_SPEC.md``。
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class CQStatus(str, Enum):
    """CQ 生命周期状态（线性流转，deprecated 为终态）。"""

    DRAFT = "draft"
    REVIEWED = "reviewed"
    FROZEN = "frozen"
    DEPRECATED = "deprecated"


# 状态机：仅允许线性前移或进入 deprecated；deprecated 不允许再流转。
_ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    CQStatus.DRAFT.value: frozenset({CQStatus.REVIEWED.value, CQStatus.DEPRECATED.value}),
    CQStatus.REVIEWED.value: frozenset({CQStatus.FROZEN.value, CQStatus.DEPRECATED.value}),
    CQStatus.FROZEN.value: frozenset({CQStatus.DEPRECATED.value}),
    CQStatus.DEPRECATED.value: frozenset(),
}


def allowed_transitions(status: str | CQStatus) -> frozenset[str]:
    """Return the statuses ``status`` may transition to (empty for deprecated)."""
    key = status.value if isinstance(status, CQStatus) else str(status)
    return _ALLOWED_TRANSITIONS.get(key, frozenset())


class CQ(BaseModel):
    """一条 Competency Question（业务语言，实体化在 IT 侧）。

    必填：``cq_id / cq_text / entities / relations / query_path / status``；
    其余字段可选（见 CQ_PIPELINE_IMPLEMENTATION_PLAN.md §6.1）。
    """

    cq_id: str = Field(..., min_length=1)
    cq_text: str = Field(..., min_length=1)
    entities: List[str] = Field(..., min_length=1)
    relations: List[str] = Field(..., min_length=1)
    query_path: str = Field(..., min_length=1)
    status: str = CQStatus.DRAFT.value

    # 可选字段
    original_text: Optional[str] = None
    pattern_type: Optional[str] = None
    attributes: List[str] = Field(default_factory=list)
    owner: Optional[str] = None
    version: str = "v1"
    evidence: List[str] = Field(default_factory=list)
    sample_binding: Optional[str] = None
    open_questions: List[str] = Field(default_factory=list)

    @field_validator("status")
    @classmethod
    def _status_must_be_known(cls, value: str) -> str:
        if value not in _ALLOWED_TRANSITIONS:
            raise ValueError(
                f"unknown CQ status {value!r}; expected one of {sorted(_ALLOWED_TRANSITIONS)}"
            )
        return value

    @field_validator("entities", "relations", "attributes", "evidence", "open_questions")
    @classmethod
    def _strip_string_lists(cls, value: List[str]) -> List[str]:
        return [str(item).strip() for item in value if str(item).strip()]

    def can_transition_to(self, target: str | CQStatus) -> bool:
        target_key = target.value if isinstance(target, CQStatus) else str(target)
        return target_key in allowed_transitions(self.status)


class CQCatalog(BaseModel):
    """CQ 目录（可从 JSONL 加载；输入文件为 ``config/cq/seed_cq.jsonl``）。"""

    items: List[CQ] = Field(default_factory=list)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "CQCatalog":
        """Strict loader — raises on unreadable file or malformed JSON lines.

        Tolerant loading (skip + warn) lives in
        :class:`src.application.cq.cq_registry.CQRegistry`.
        """
        items: List[CQ] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            items.append(CQ.model_validate(json.loads(line)))
        return cls(items=items)

    def to_jsonl(self, path: str | Path) -> None:
        payload = "\n".join(item.model_dump_json() for item in self.items)
        Path(path).write_text(payload + ("\n" if payload else ""), encoding="utf-8")

    def get(self, cq_id: str) -> Optional[CQ]:
        for item in self.items:
            if item.cq_id == cq_id:
                return item
        return None

    def by_status(self, status: str | CQStatus) -> List[CQ]:
        key = status.value if isinstance(status, CQStatus) else str(status)
        return [item for item in self.items if item.status == key]

    def frozen(self) -> List[CQ]:
        return self.by_status(CQStatus.FROZEN)

    def __len__(self) -> int:
        return len(self.items)


class CQMapping(BaseModel):
    """CQ ↔ 实体/关系/属性/样本 的映射记录（M3 落地使用，M0 先立契约）。"""

    cq_id: str = Field(..., min_length=1)
    entities: List[str] = Field(default_factory=list)
    relations: List[str] = Field(default_factory=list)
    attributes: List[str] = Field(default_factory=list)
    sample_binding: Optional[str] = None


class CQValidationResult(BaseModel):
    """单条 CQ 的只读可执行性检查结果（M2 产物字段契约，M3 扩展缺口明细）。

    M2：``missing_entities / missing_relations``；
    M3：新增 ``missing_attributes``（CQ 关心但 master 未承载的属性）与
    ``chain_breaks``（query_path 链式断点定位，"position N: ..." 格式）。
    两者均为可选字段，M2 产物读取方不受影响。
    """

    cq_id: str = Field(..., min_length=1)
    executable: bool
    missing_entities: List[str] = Field(default_factory=list)
    missing_relations: List[str] = Field(default_factory=list)
    missing_attributes: List[str] = Field(default_factory=list)
    chain_breaks: List[str] = Field(default_factory=list)
    notes: str = ""

    @property
    def has_gap(self) -> bool:
        return bool(
            self.missing_entities
            or self.missing_relations
            or self.missing_attributes
            or self.chain_breaks
        )
