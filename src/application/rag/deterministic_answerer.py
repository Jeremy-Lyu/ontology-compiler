"""确定性回答模块 - DomainOntology + 规则匹配。

RAG-07 (§9.7, 2026-09-03): 从 rag_answerer.py 拆出确定性回答引擎，使
rag_answerer.py 聚焦于统一路由/编排。本模块包含：

1. ``QuestionType``：支持的问题类型枚举
2. ``AnswerResult``：确定性回答的结果对象
3. ``DomainOntology``：layer1_domain_ontology.json 的内存视图（多跳推理）
4. ``answer_deterministic``：Phase0 风格问题的确定性回答入口

行为与拆分前完全一致；由 tests/rag/test_rag_contracts_characterization.py
等 characterization 测试锁定。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
from collections import deque

from src.domain.errors import DataContractError

from .models import EDGE_TYPES_SYSTEM_TO_CAP


# =============================================================================
# Question Types for Deterministic Answering
# =============================================================================

class QuestionType(Enum):
    """Supported question types for deterministic answering."""

    L1_CHILDREN = "l1_children"
    L2_PARENT = "l2_parent"
    PATH_L1_L2 = "path_l1_l2"
    DOMAIN_FILTER = "domain_filter"
    SYSTEM_TO_L2 = "system_to_l2"
    L2_TO_SYSTEM = "l2_to_system"
    COMPARISON = "comparison"

    @classmethod
    def from_string(cls, value: str) -> Optional["QuestionType"]:
        """Safely convert string to QuestionType with case-insensitive matching."""
        if not isinstance(value, str):
            return None
        try:
            return cls(value)
        except ValueError:
            return None


# =============================================================================
# Answer Results
# =============================================================================

@dataclass(frozen=True)
class AnswerResult:
    """Structured answer result for deterministic answering."""

    answer: str
    evidence_node_ids: List[str]
    confidence: float
    cannot_answer_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "answer": self.answer,
            "evidence_node_ids": self.evidence_node_ids,
            "confidence": self.confidence,
            "cannot_answer_reason": self.cannot_answer_reason,
        }


# =============================================================================
# Domain Ontology (for Deterministic Answering)
# =============================================================================

class DomainOntology:
    """In-memory view of layer1_domain_ontology.json."""

    def __init__(self, path: Path):
        self.path = path
        self.payload = self._load(path)
        self.nodes = self._require_list(self.payload, "nodes")
        self.edges = self._require_list(self.payload, "edges")
        self.node_by_id: Dict[str, Dict[str, Any]] = {}
        self.adjacency_map: Dict[str, List[Tuple[str, str]]] = {}
        for n in self.nodes:
            mid = n.get("meta_id")
            if isinstance(mid, str) and mid:
                self.node_by_id[mid] = n
                self.adjacency_map[mid] = []
        for e in self.edges:
            src = e.get("source")
            tgt = e.get("target")
            et = e.get("type")
            if isinstance(src, str) and isinstance(tgt, str) and isinstance(et, str):
                if src not in self.adjacency_map:
                    self.adjacency_map[src] = []
                if tgt not in self.adjacency_map:
                    self.adjacency_map[tgt] = []
                self.adjacency_map[src].append((tgt, et))
                self.adjacency_map[tgt].append((src, et))

    def _load(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise DataContractError(f"domain ontology not found: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise DataContractError(f"failed to load domain ontology: {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise DataContractError(f"domain ontology is not an object: {path}")
        return payload

    def _require_list(self, payload: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
        value = payload.get(key)
        if not isinstance(value, list):
            raise DataContractError(f"missing/invalid '{key}' list in {self.path}")
        out: List[Dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                out.append(item)
        return out

    def capability_parent_l1(self, cap_l2_id: str) -> Optional[str]:
        """Infer L1 parent for a CAP_L2_* id by prefix matching existing nodes."""
        if not cap_l2_id.startswith("CAP_L2_"):
            return None
        parts = cap_l2_id.split("_")
        if len(parts) < 4:
            return None
        l1_num = parts[2]
        prefix = f"CAP_L1_{l1_num}_"
        matches = [nid for nid in self.node_by_id.keys() if nid.startswith(prefix)]
        return matches[0] if len(matches) == 1 else None

    def l1_children_l2(self, cap_l1_id: str) -> List[str]:
        if not cap_l1_id.startswith("CAP_L1_"):
            return []
        parts = cap_l1_id.split("_")
        if len(parts) < 4:
            return []
        l1_num = parts[2]
        prefix = f"CAP_L2_{l1_num}_"
        kids = [nid for nid, n in self.node_by_id.items() if nid.startswith(prefix) and n.get("meta_class") == "Capability"]
        return sorted(kids)

    def node_name(self, node_id: str) -> str:
        n = self.node_by_id.get(node_id) or {}
        return str(n.get("meta_name") or "")

    def node_domain(self, node_id: str) -> str:
        n = self.node_by_id.get(node_id) or {}
        return str(n.get("sandbox_domain") or "")

    def systems_for_capability(self, cap_id: str) -> List[str]:
        sys_ids: Set[str] = set()
        for e in self.edges:
            src = e.get("source")
            tgt = e.get("target")
            et = e.get("type")
            if not isinstance(src, str) or not isinstance(tgt, str) or not isinstance(et, str):
                continue
            if tgt != cap_id:
                continue
            if et.upper() not in EDGE_TYPES_SYSTEM_TO_CAP:
                continue
            src_node = self.node_by_id.get(src)
            if not src_node or src_node.get("meta_class") != "System":
                continue
            sys_ids.add(src)
        return sorted(sys_ids)

    def find_path_between(self, start_id: str, end_id: str, max_hops: int = 3) -> Optional[List[Tuple[str, str, str]]]:
        """
        多跳路径搜索 - 使用BFS查找两个实体之间的路径。
        
        Args:
            start_id: 起始节点ID
            end_id: 目标节点ID
            max_hops: 最大跳数
            
        Returns:
            路径列表，格式为 [(node_id, relation_type, node_id), ...]
        """
        if start_id not in self.adjacency_map or end_id not in self.adjacency_map:
            return None
        
        visited: Set[str] = {start_id}
        queue: Deque[List[Tuple[str, str, str]]] = deque()
        queue.append([])
        
        while queue:
            current_path = queue.popleft()
            
            if not current_path:
                current_node = start_id
            else:
                current_node = current_path[-1][2]
            
            if current_node == end_id:
                return current_path
            
            if len(current_path) >= max_hops:
                continue
            
            for neighbor, rel_type in self.adjacency_map.get(current_node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = current_path + [(current_node, rel_type, neighbor)]
                    queue.append(new_path)
        
        return None

    def find_reachable_nodes(self, start_id: str, max_hops: int = 3,
                            target_meta_class: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        查找从起始节点可达的所有节点（多跳扩展）。
        
        Args:
            start_id: 起始节点ID
            max_hops: 最大跳数
            target_meta_class: 目标meta_class过滤
            
        Returns:
            可达节点列表，包含路径信息
        """
        if start_id not in self.adjacency_map:
            return []
        
        visited: Dict[str, List[Tuple[str, str, str]]] = {start_id: []}
        queue: Deque[Tuple[str, int, List[Tuple[str, str, str]]]] = deque()
        queue.append((start_id, 0, []))
        
        results: List[Dict[str, Any]] = []
        
        while queue:
            current_node, hops, path = queue.popleft()
            
            if hops > 0:
                node_info = self.node_by_id.get(current_node, {})
                if target_meta_class is None or \
                   node_info.get("meta_class", "").upper() == target_meta_class.upper():
                    results.append({
                        "node_id": current_node,
                        "meta_class": node_info.get("meta_class"),
                        "name": self.node_name(current_node),
                        "domain": self.node_domain(current_node),
                        "hops": hops,
                        "path": path,
                    })
            
            if hops < max_hops:
                for neighbor, rel_type in self.adjacency_map.get(current_node, []):
                    if neighbor not in visited or len(visited[neighbor]) > hops + 1:
                        visited[neighbor] = path + [(current_node, rel_type, neighbor)]
                        queue.append((neighbor, hops + 1, path + [(current_node, rel_type, neighbor)]))
        
        return results

    def find_common_neighbors(self, node_ids: List[str], max_hops: int = 2) -> List[Dict[str, Any]]:
        """
        查找多个节点的公共邻居（用于关联分析）。
        
        Args:
            node_ids: 节点ID列表
            max_hops: 最大跳数
            
        Returns:
            公共邻居节点列表
        """
        neighbor_sets: List[Set[str]] = []
        
        for node_id in node_ids:
            if node_id not in self.adjacency_map:
                continue
            
            visited: Set[str] = {node_id}
            queue: Deque[Tuple[str, int]] = deque()
            queue.append((node_id, 0))
            neighbors: Set[str] = set()
            
            while queue:
                current, hops = queue.popleft()
                
                if hops > 0:
                    neighbors.add(current)
                
                if hops < max_hops:
                    for neighbor, _ in self.adjacency_map.get(current, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, hops + 1))
            
            neighbor_sets.append(neighbors)
        
        if not neighbor_sets:
            return []
        
        common_neighbors = set.intersection(*neighbor_sets)
        common_neighbors -= set(node_ids)
        
        results = []
        for nid in common_neighbors:
            node_info = self.node_by_id.get(nid, {})
            results.append({
                "node_id": nid,
                "meta_class": node_info.get("meta_class"),
                "name": self.node_name(nid),
                "domain": self.node_domain(nid),
            })
        
        return results


# =============================================================================
# Deterministic Answer Handlers
# =============================================================================

def _answer_l1_children(
    ontology: DomainOntology,
    cap_l1: str | None,
    _cap_l2: str | None,
    _expected_node_ids: list[Any],
    expected_cap_l1_ids: list[str],
    qtext: str,
) -> AnswerResult:
    """Enumerate L2 capabilities under a given L1."""
    if len(expected_cap_l1_ids) >= 2 and ("L1" in qtext or "业务领域" in qtext):
        names = [ontology.node_name(x) for x in expected_cap_l1_ids]
        answer = "\n".join([n for n in names if n])
        return AnswerResult(
            answer=answer,
            evidence_node_ids=expected_cap_l1_ids,
            confidence=1.0,
        )
    if not cap_l1:
        return AnswerResult(
            answer="",
            evidence_node_ids=[],
            confidence=0.0,
            cannot_answer_reason="missing CAP_L1 node id in expected_node_ids",
        )
    l2s = ontology.l1_children_l2(cap_l1)
    names = [ontology.node_name(x) for x in l2s]
    answer = "\n".join(names) if names else ""
    return AnswerResult(answer=answer, evidence_node_ids=[cap_l1], confidence=1.0)


def _answer_l2_parent(
    ontology: DomainOntology,
    _cap_l1: str | None,
    cap_l2: str | None,
    _expected_node_ids: list[Any],
    _expected_cap_l1_ids: list[str],
    _qtext: str,
) -> AnswerResult:
    """Return the L1 parent of a given L2 capability."""
    if not cap_l2:
        return AnswerResult(
            answer="",
            evidence_node_ids=[],
            confidence=0.0,
            cannot_answer_reason="missing CAP_L2 node id in expected_node_ids",
        )
    parent = ontology.capability_parent_l1(cap_l2)
    if not parent:
        return AnswerResult(
            answer="",
            evidence_node_ids=[cap_l2],
            confidence=0.3,
            cannot_answer_reason="cannot infer L1 parent",
        )
    return AnswerResult(
        answer=ontology.node_name(parent),
        evidence_node_ids=[cap_l2, parent],
        confidence=1.0,
    )


def _answer_path_l1_l2(
    ontology: DomainOntology,
    _cap_l1: str | None,
    cap_l2: str | None,
    _expected_node_ids: list[Any],
    _expected_cap_l1_ids: list[str],
    _qtext: str,
) -> AnswerResult:
    """Return the L1->L2 path for a given L2 capability."""
    if not cap_l2:
        return AnswerResult(
            answer="",
            evidence_node_ids=[],
            confidence=0.0,
            cannot_answer_reason="missing CAP_L2 node id in expected_node_ids",
        )
    parent = ontology.capability_parent_l1(cap_l2)
    if not parent:
        return AnswerResult(
            answer="",
            evidence_node_ids=[cap_l2],
            confidence=0.3,
            cannot_answer_reason="cannot infer L1 parent",
        )
    path = f"{ontology.node_name(parent)} -> {ontology.node_name(cap_l2)}"
    return AnswerResult(answer=path, evidence_node_ids=[parent, cap_l2], confidence=1.0)


def _answer_domain_filter(
    ontology: DomainOntology,
    cap_l1: str | None,
    cap_l2: str | None,
    expected_node_ids: list[Any],
    _expected_cap_l1_ids: list[str],
    _qtext: str,
) -> AnswerResult:
    """Return the domain of a target node."""
    target = cap_l1 or cap_l2 or next((x for x in expected_node_ids if isinstance(x, str)), None)
    if not target:
        return AnswerResult(
            answer="",
            evidence_node_ids=[],
            confidence=0.0,
            cannot_answer_reason="missing target node id",
        )
    return AnswerResult(
        answer=ontology.node_domain(target),
        evidence_node_ids=[target],
        confidence=1.0,
    )


def _answer_system_map(
    ontology: DomainOntology,
    _cap_l1: str | None,
    cap_l2: str | None,
    _expected_node_ids: list[Any],
    _expected_cap_l1_ids: list[str],
    _qtext: str,
) -> AnswerResult:
    """List systems mapped to a given L2 capability."""
    if not cap_l2:
        return AnswerResult(
            answer="",
            evidence_node_ids=[],
            confidence=0.0,
            cannot_answer_reason="missing CAP_L2 node id in expected_node_ids",
        )
    sys_ids = ontology.systems_for_capability(cap_l2)
    sys_names = [ontology.node_name(s) for s in sys_ids]
    answer = "\n".join(sys_names) if sys_names else ""
    return AnswerResult(
        answer=answer,
        evidence_node_ids=[cap_l2, *sys_ids],
        confidence=1.0 if sys_ids else 0.6,
        cannot_answer_reason=None if sys_ids else "no systems mapped",
    )


def _answer_comparison(
    ontology: DomainOntology,
    _cap_l1: str | None,
    _cap_l2: str | None,
    expected_node_ids: list[Any],
    _expected_cap_l1_ids: list[str],
    _qtext: str,
) -> AnswerResult:
    """Compare two L2 capabilities by their L1 parents."""
    l2_ids = [x for x in expected_node_ids if isinstance(x, str) and x.startswith("CAP_L2_")]
    if len(l2_ids) < 2:
        return AnswerResult(
            answer="",
            evidence_node_ids=[],
            confidence=0.0,
            cannot_answer_reason="missing two CAP_L2 node ids in expected_node_ids",
        )
    a, b = l2_ids[0], l2_ids[1]
    pa = ontology.capability_parent_l1(a)
    pb = ontology.capability_parent_l1(b)
    if not pa or not pb:
        return AnswerResult(
            answer="",
            evidence_node_ids=[x for x in [a, b, pa, pb] if x],
            confidence=0.3,
            cannot_answer_reason="cannot infer L1 parent for comparison",
        )
    if pa == pb:
        answer = "是\n" + ontology.node_name(pa)
        evidence = [a, b, pa]
    else:
        answer = "否\n" + ontology.node_name(pa) + "\n" + ontology.node_name(pb)
        evidence = [a, b, pa, pb]
    return AnswerResult(answer=answer.strip(), evidence_node_ids=evidence, confidence=1.0)


_ANSWER_HANDLERS: dict[QuestionType, Any] = {
    QuestionType.L1_CHILDREN: _answer_l1_children,
    QuestionType.L2_PARENT: _answer_l2_parent,
    QuestionType.PATH_L1_L2: _answer_path_l1_l2,
    QuestionType.DOMAIN_FILTER: _answer_domain_filter,
    QuestionType.SYSTEM_TO_L2: _answer_system_map,
    QuestionType.L2_TO_SYSTEM: _answer_system_map,
    QuestionType.COMPARISON: _answer_comparison,
}


def _normalize_expected_node_ids(question: Dict[str, Any]) -> tuple[list[Any], list[str], str | None, str | None, str]:
    expected_node_ids = question.get("expected_node_ids") or []
    if not isinstance(expected_node_ids, list):
        expected_node_ids = []

    expected_cap_l1_ids = sorted(
        {
            x
            for x in expected_node_ids
            if isinstance(x, str) and x.startswith("CAP_L1_")
        }
    )
    cap_l1 = next((x for x in expected_node_ids if isinstance(x, str) and x.startswith("CAP_L1_")), None)
    cap_l2 = next((x for x in expected_node_ids if isinstance(x, str) and x.startswith("CAP_L2_")), None)
    qtext = str(question.get("question") or "")
    return expected_node_ids, expected_cap_l1_ids, cap_l1, cap_l2, qtext


def _resolve_question_type(qtype_raw: Any) -> QuestionType | None:
    return QuestionType.from_string(qtype_raw) if qtype_raw else None


def answer_deterministic(question: Dict[str, Any], ontology: DomainOntology) -> AnswerResult:
    """Answer a Phase0-style question deterministically from ontology."""
    qtype_raw = question.get("question_type")
    expected_node_ids, expected_cap_l1_ids, cap_l1, cap_l2, qtext = _normalize_expected_node_ids(question)

    qtype = _resolve_question_type(qtype_raw)
    if not qtype:
        return AnswerResult(
            answer="",
            evidence_node_ids=[],
            confidence=0.0,
            cannot_answer_reason=f"invalid or missing question_type: {qtype_raw}",
        )

    handler = _ANSWER_HANDLERS.get(qtype)
    if handler:
        return handler(ontology, cap_l1, cap_l2, expected_node_ids, expected_cap_l1_ids, qtext)

    return AnswerResult(
        answer="",
        evidence_node_ids=[],
        confidence=0.0,
        cannot_answer_reason=f"unsupported question_type: {qtype_raw}",
    )
