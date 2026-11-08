"""证据绑定器 - 为检索结果绑定证据

本模块提供证据绑定能力，支持：
1. 证据溯源（来源追踪）
2. 证据聚合（多源证据合并）
3. 证据置信度评估
4. 证据冲突检测
5. 证据可视化

架构目标：
- G4: 统一知识消费层
- G5: 统一质量闭环
"""
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass

from pydantic import BaseModel, Field

from src.domain.models.evidence import Evidence
from src.domain.models.canonical import CanonicalEvidence


@dataclass
class EvidenceSource:
    """证据来源"""
    source_id: str
    source_type: str
    source_name: str
    confidence: float
    created_at: str


@dataclass
class EvidenceBinding:
    """证据绑定"""
    binding_id: str
    target_id: str
    target_type: str
    evidences: List[Evidence]
    aggregated_confidence: float
    sources: List[EvidenceSource]
    conflicts: List[str]
    created_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "binding_id": self.binding_id,
            "target_id": self.target_id,
            "target_type": self.target_type,
            "evidence_count": len(self.evidences),
            "aggregated_confidence": self.aggregated_confidence,
            "evidences": [e.model_dump() for e in self.evidences],
            "sources": [
                {
                    "source_id": s.source_id,
                    "source_type": s.source_type,
                    "source_name": s.source_name,
                    "confidence": s.confidence,
                }
                for s in self.sources
            ],
            "conflicts": self.conflicts,
            "created_at": self.created_at,
        }


class EvidenceBinder:
    """证据绑定器
    
    为检索结果绑定证据，提供溯源和置信度评估。
    """
    
    def __init__(self) -> None:
        self._bindings: Dict[str, EvidenceBinding] = {}
        self._evidence_index: Dict[str, List[str]] = defaultdict(list)  # target_id -> evidence_ids
    
    def bind(
        self,
        target_id: str,
        target_type: str,
        evidences: List[Evidence],
    ) -> EvidenceBinding:
        """绑定证据
        
        Args:
            target_id: 目标 ID（实体/关系 ID）
            target_type: 目标类型（entity/relation）
            evidences: 证据列表
        
        Returns:
            EvidenceBinding: 证据绑定结果
        """
        # 聚合置信度
        aggregated_confidence = self._aggregate_confidence(evidences)
        
        # 提取来源
        sources = self._extract_sources(evidences)
        
        # 检测冲突
        conflicts = self._detect_conflicts(evidences)
        
        binding = EvidenceBinding(
            binding_id=f"binding-{target_id}",
            target_id=target_id,
            target_type=target_type,
            evidences=evidences,
            aggregated_confidence=aggregated_confidence,
            sources=sources,
            conflicts=conflicts,
            created_at=datetime.now().isoformat(),
        )
        
        self._bindings[binding.binding_id] = binding
        self._evidence_index[target_id].extend([e.evidence_id for e in evidences])
        
        return binding
    
    def get_binding(self, target_id: str) -> Optional[EvidenceBinding]:
        """获取证据绑定"""
        for binding in self._bindings.values():
            if binding.target_id == target_id:
                return binding
        return None
    
    def get_evidences(self, target_id: str) -> List[CanonicalEvidence]:
        """获取证据列表"""
        binding = self.get_binding(target_id)
        if binding:
            return binding.evidences
        return []
    
    def get_sources(self, target_id: str) -> List[EvidenceSource]:
        """获取来源列表"""
        binding = self.get_binding(target_id)
        if binding:
            return binding.sources
        return []
    
    def get_conflicts(self, target_id: str) -> List[str]:
        """获取冲突列表"""
        binding = self.get_binding(target_id)
        if binding:
            return binding.conflicts
        return []
    
    def _aggregate_confidence(self, evidences: List[Evidence]) -> float:
        """聚合置信度"""
        if not evidences:
            return 0.0
        
        # 使用加权平均
        total_confidence = sum(e.confidence for e in evidences)
        return total_confidence / len(evidences)
    
    def _extract_sources(self, evidences: List[Evidence]) -> List[EvidenceSource]:
        """提取来源"""
        source_map: Dict[str, EvidenceSource] = {}
        
        for evidence in evidences:
            # 兼容 CanonicalEvidence 的字段名
            source_id = evidence.source_file or evidence.source_url or evidence.metadata.get("source", "")
            source_type = evidence.metadata.get("source_type", "")
            source_name = source_id
            
            if source_id not in source_map:
                source_map[source_id] = EvidenceSource(
                    source_id=source_id,
                    source_type=source_type,
                    source_name=source_name,
                    confidence=evidence.confidence,
                    created_at=evidence.created_at,
                )
            else:
                # 更新置信度（取最大值）
                source_map[source_id].confidence = max(
                    source_map[source_id].confidence,
                    evidence.confidence
                )
        
        return list(source_map.values())
    
    def _detect_conflicts(self, evidences: List[Evidence]) -> List[str]:
        """检测冲突"""
        conflicts = []
        
        # 检查内容冲突
        content_map: Dict[str, List[str]] = defaultdict(list)
        for evidence in evidences:
            source_id = evidence.source_file or evidence.source_url or ""
            content_map[evidence.content].append(source_id)
        
        for content, sources in content_map.items():
            if len(sources) > 1:
                # 多个来源有相同内容，可能是冲突
                conflicts.append(f"多源相同内容: {content[:50]}...")
        
        # 检查置信度冲突
        confidences = [e.confidence for e in evidences]
        if confidences:
            max_conf = max(confidences)
            min_conf = min(confidences)
            if max_conf - min_conf > 0.5:
                conflicts.append(f"置信度差异较大: {min_conf:.2f} - {max_conf:.2f}")
        
        return conflicts
    
    def visualize_evidence(self, target_id: str) -> str:
        """可视化证据"""
        binding = self.get_binding(target_id)
        if not binding:
            return f"No evidence found for {target_id}"
        
        lines = [
            f"## Evidence Binding for {target_id}",
            f"",
            f"**Type**: {binding.target_type}",
            f"**Evidence Count**: {len(binding.evidences)}",
            f"**Aggregated Confidence**: {binding.aggregated_confidence:.2f}",
            f"",
            f"### Sources",
            f"",
        ]
        
        for source in binding.sources:
            lines.append(f"- **{source.source_name}** ({source.source_type}): {source.confidence:.2f}")
        
        lines.append("")
        lines.append("### Evidences")
        lines.append("")
        
        for i, evidence in enumerate(binding.evidences, 1):
            source_id = evidence.source_file or evidence.source_url or "unknown"
            lines.append(f"{i}. **{source_id}**")
            lines.append(f"   {evidence.content[:100]}...")
            lines.append(f"   Confidence: {evidence.confidence:.2f}")
            lines.append("")
        
        if binding.conflicts:
            lines.append("### Conflicts")
            lines.append("")
            for conflict in binding.conflicts:
                lines.append(f"- {conflict}")
            lines.append("")
        
        return "\n".join(lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        total_bindings = len(self._bindings)
        total_evidences = sum(len(b.evidences) for b in self._bindings.values())
        total_sources = sum(len(b.sources) for b in self._bindings.values())
        total_conflicts = sum(len(b.conflicts) for b in self._bindings.values())
        
        return {
            "total_bindings": total_bindings,
            "total_evidences": total_evidences,
            "total_sources": total_sources,
            "total_conflicts": total_conflicts,
            "avg_evidences_per_binding": total_evidences / total_bindings if total_bindings > 0 else 0,
            "avg_sources_per_binding": total_sources / total_bindings if total_bindings > 0 else 0,
        }
    
    def clear(self) -> None:
        """清空绑定"""
        self._bindings.clear()
        self._evidence_index.clear()