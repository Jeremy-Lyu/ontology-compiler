"""Conflict Deduplication and Counting Service - 统一冲突去重和统计服务

目标：解决冲突数量不稳定、重复统计和阶段间口径不一致的问题。

提供的能力：
1. ConflictDedupService - 冲突去重服务
2. ConflictStats - 冲突统计结果
3. dedup_conflicts - 阶段内去重
4. merge_conflicts - 跨阶段归并
5. calculate_stats - 统一统计口径
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from .base import Conflict, with_conflict_stage


@dataclass
class ConflictStats:
    """统一的冲突统计结果"""
    
    unique_conflict_count: int = 0
    duplicate_discovery_count: int = 0
    total_detected_count: int = 0
    
    critical_conflict_count: int = 0
    warning_conflict_count: int = 0
    info_conflict_count: int = 0
    
    entity_conflict_count: int = 0
    relation_conflict_count: int = 0
    schema_conflict_count: int = 0
    
    by_type: Dict[str, int] = field(default_factory=dict)
    by_severity: Dict[str, int] = field(default_factory=dict)
    by_stage: Dict[str, int] = field(default_factory=dict)
    by_source_file: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "unique_conflict_count": self.unique_conflict_count,
            "duplicate_discovery_count": self.duplicate_discovery_count,
            "total_detected_count": self.total_detected_count,
            "critical_conflict_count": self.critical_conflict_count,
            "warning_conflict_count": self.warning_conflict_count,
            "info_conflict_count": self.info_conflict_count,
            "entity_conflict_count": self.entity_conflict_count,
            "relation_conflict_count": self.relation_conflict_count,
            "schema_conflict_count": self.schema_conflict_count,
            "by_type": self.by_type,
            "by_severity": self.by_severity,
            "by_stage": self.by_stage,
            "by_source_file": self.by_source_file,
        }


class ConflictDedupService:
    """冲突去重服务"""
    
    def __init__(self):
        self._seen_dedup_keys: Set[str] = set()
        self._conflict_store: Dict[str, Conflict] = {}
        self._discovery_count: Dict[str, int] = {}
    
    def reset(self):
        """重置状态"""
        self._seen_dedup_keys.clear()
        self._conflict_store.clear()
        self._discovery_count.clear()
    
    def dedup_conflicts(self, conflicts: List[Conflict], stage: str = "") -> List[Conflict]:
        """阶段内去重
        
        Args:
            conflicts: 待去重的冲突列表
            stage: 当前阶段名称
            
        Returns:
            去重后的冲突列表
        """
        unique_conflicts = []
        
        for conflict in conflicts:
            dedup_key = conflict.dedup_key
            
            if dedup_key not in self._seen_dedup_keys:
                self._seen_dedup_keys.add(dedup_key)
                self._conflict_store[dedup_key] = conflict
                self._discovery_count[dedup_key] = 1
                
                unique_conflicts.append(with_conflict_stage(conflict, stage))
            else:
                self._discovery_count[dedup_key] += 1
        
        return unique_conflicts
    
    def merge_conflicts(self, conflicts: List[Conflict]) -> List[Conflict]:
        """跨阶段冲突归并
        
        Args:
            conflicts: 待归并的冲突列表
            
        Returns:
            归并后的唯一冲突列表
        """
        for conflict in conflicts:
            dedup_key = conflict.dedup_key
            
            if dedup_key not in self._conflict_store:
                self._conflict_store[dedup_key] = conflict
                self._discovery_count[dedup_key] = 1
            else:
                existing = self._conflict_store[dedup_key]
                
                if conflict.severity == "blocking" and existing.severity != "blocking":
                    self._conflict_store[dedup_key] = conflict
                
                self._discovery_count[dedup_key] += 1
        
        return list(self._conflict_store.values())
    
    def get_unique_conflicts(self) -> List[Conflict]:
        """获取所有唯一冲突"""
        return list(self._conflict_store.values())
    
    def get_stats(self) -> ConflictStats:
        """计算统一统计口径"""
        stats = ConflictStats()
        
        stats.total_detected_count = sum(self._discovery_count.values())
        stats.unique_conflict_count = len(self._conflict_store)
        stats.duplicate_discovery_count = stats.total_detected_count - stats.unique_conflict_count
        
        for dedup_key, conflict in self._conflict_store.items():
            stats.by_type[conflict.conflict_type] = stats.by_type.get(conflict.conflict_type, 0) + 1
            stats.by_severity[conflict.severity] = stats.by_severity.get(conflict.severity, 0) + 1
            
            if conflict.stage:
                stats.by_stage[conflict.stage] = stats.by_stage.get(conflict.stage, 0) + 1
            
            if conflict.source_file:
                stats.by_source_file[conflict.source_file] = stats.by_source_file.get(conflict.source_file, 0) + 1
            
            if conflict.severity == "blocking":
                stats.critical_conflict_count += 1
            elif conflict.severity == "warning":
                stats.warning_conflict_count += 1
            elif conflict.severity == "info":
                stats.info_conflict_count += 1
            
            if conflict.conflict_type.startswith("entity_") or conflict.conflict_type in [
                "duplicate_nodes", "semantic_conflict", "meta_name_drift", 
                "meta_class_drift", "instance_binding_conflict"
            ]:
                stats.entity_conflict_count += 1
            elif conflict.conflict_type.startswith("relation_") or conflict.conflict_type in [
                "circular_relationships", "relationship_key_conflict", 
                "relation_whitelist_violation", "same_as_conflict"
            ]:
                stats.relation_conflict_count += 1
            elif conflict.conflict_type in ["schema_violation", "attribute_type_conflict"]:
                stats.schema_conflict_count += 1
        
        return stats


def dedup_conflicts(conflicts: List[Conflict], stage: str = "") -> List[Conflict]:
    """阶段内冲突去重（便捷函数）"""
    service = ConflictDedupService()
    return service.dedup_conflicts(conflicts, stage)


def merge_conflicts(all_conflicts: List[List[Conflict]]) -> List[Conflict]:
    """跨阶段冲突归并（便捷函数）"""
    service = ConflictDedupService()
    for conflicts in all_conflicts:
        service.merge_conflicts(conflicts)
    return service.get_unique_conflicts()


def calculate_stats(conflicts: List[Conflict]) -> ConflictStats:
    """计算冲突统计（便捷函数）"""
    service = ConflictDedupService()
    service.merge_conflicts(conflicts)
    return service.get_stats()