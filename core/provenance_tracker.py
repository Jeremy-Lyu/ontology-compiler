"""Provenance Tracker - 溯源追踪系统

本模块实现：
1. ProvenanceRecord - 单个溯源记录
2. ExtractionProvenance - 抽取过程完整溯源
3. ProvenanceTracker - 溯源追踪器

溯源信息包含：
- 哪个 skill 导致的
- 哪个 prompt 片段影响的
- 哪个 agent 做出的
- 哪条 evidence 支撑的
- 是什么置信度
- 是直接抽取还是推断

这对后续修正非常重要。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum


class ProvenanceSource(Enum):
    """溯源来源类型"""
    LLM = "llm"
    SKILL = "skill"
    RULE = "rule"
    INFERRED = "inferred"
    MANUAL = "manual"
    MERGED = "merged"


class ActionType(Enum):
    """操作类型"""
    EXTRACT = "extract"
    INFER = "infer"
    MERGE = "merge"
    DEDUPLICATE = "deduplicate"
    ENRICH = "enrich"
    VALIDATE = "validate"
    TRANSFORM = "transform"


@dataclass
class ProvenanceRecord:
    """单个溯源记录"""
    
    record_id: str
    entity_id: str
    entity_type: str
    action_type: ActionType
    source: ProvenanceSource
    
    timestamp: datetime = field(default_factory=datetime.now)
    agent_name: Optional[str] = None
    skill_id: Optional[str] = None
    prompt_version: Optional[str] = None
    llm_model: Optional[str] = None
    evidence_ref: Optional[str] = None
    
    confidence: float = 0.0
    input_text: Optional[str] = None
    output_text: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "action_type": self.action_type.value,
            "source": self.source.value,
            "timestamp": self.timestamp.isoformat(),
            "agent_name": self.agent_name,
            "skill_id": self.skill_id,
            "prompt_version": self.prompt_version,
            "llm_model": self.llm_model,
            "evidence_ref": self.evidence_ref,
            "confidence": self.confidence,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceRecord":
        return cls(
            record_id=data["record_id"],
            entity_id=data["entity_id"],
            entity_type=data["entity_type"],
            action_type=ActionType(data["action_type"]),
            source=ProvenanceSource(data["source"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            agent_name=data.get("agent_name"),
            skill_id=data.get("skill_id"),
            prompt_version=data.get("prompt_version"),
            llm_model=data.get("llm_model"),
            evidence_ref=data.get("evidence_ref"),
            confidence=data.get("confidence", 0.0),
            input_text=data.get("input_text"),
            output_text=data.get("output_text"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ExtractionProvenance:
    """抽取过程完整溯源"""
    
    extraction_id: str
    document_id: str
    file_name: str
    
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    records: List[ProvenanceRecord] = field(default_factory=list)
    agents_used: List[str] = field(default_factory=list)
    skills_used: List[str] = field(default_factory=list)
    llm_calls: int = 0
    retries: int = 0
    
    def add_record(self, record: ProvenanceRecord) -> None:
        """添加溯源记录"""
        self.records.append(record)
        
        if record.agent_name and record.agent_name not in self.agents_used:
            self.agents_used.append(record.agent_name)
        
        if record.skill_id and record.skill_id not in self.skills_used:
            self.skills_used.append(record.skill_id)
    
    def mark_completed(self) -> None:
        """标记抽取完成"""
        self.end_time = datetime.now()
    
    def get_duration(self) -> float:
        """获取抽取耗时（秒）"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def get_records_by_entity(self, entity_id: str) -> List[ProvenanceRecord]:
        """按实体ID获取溯源记录"""
        return [r for r in self.records if r.entity_id == entity_id]
    
    def get_records_by_source(self, source: ProvenanceSource) -> List[ProvenanceRecord]:
        """按来源获取溯源记录"""
        return [r for r in self.records if r.source == source]
    
    def get_records_by_agent(self, agent_name: str) -> List[ProvenanceRecord]:
        """按Agent获取溯源记录"""
        return [r for r in self.records if r.agent_name == agent_name]
    
    def get_summary(self) -> Dict[str, Any]:
        """获取溯源摘要"""
        source_counts = {}
        action_counts = {}
        
        for record in self.records:
            source_counts[record.source.value] = source_counts.get(record.source.value, 0) + 1
            action_counts[record.action_type.value] = action_counts.get(record.action_type.value, 0) + 1
        
        return {
            "extraction_id": self.extraction_id,
            "document_id": self.document_id,
            "file_name": self.file_name,
            "duration": self.get_duration(),
            "total_records": len(self.records),
            "agents_used": self.agents_used,
            "skills_used": self.skills_used,
            "llm_calls": self.llm_calls,
            "retries": self.retries,
            "by_source": source_counts,
            "by_action": action_counts,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "extraction_id": self.extraction_id,
            "document_id": self.document_id,
            "file_name": self.file_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "records": [r.to_dict() for r in self.records],
            "agents_used": self.agents_used,
            "skills_used": self.skills_used,
            "llm_calls": self.llm_calls,
            "retries": self.retries,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractionProvenance":
        records = [ProvenanceRecord.from_dict(r) for r in data.get("records", [])]
        
        return cls(
            extraction_id=data["extraction_id"],
            document_id=data["document_id"],
            file_name=data["file_name"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            records=records,
            agents_used=data.get("agents_used", []),
            skills_used=data.get("skills_used", []),
            llm_calls=data.get("llm_calls", 0),
            retries=data.get("retries", 0),
        )


class ProvenanceTracker:
    """溯源追踪器"""
    
    def __init__(self) -> None:
        self._provenance_by_extraction: Dict[str, ExtractionProvenance] = {}
        self._provenance_by_document: Dict[str, List[str]] = {}
        self._next_record_id = 1
    
    def create_extraction_provenance(
        self, extraction_id: str, document_id: str, file_name: str
    ) -> ExtractionProvenance:
        """创建抽取溯源"""
        provenance = ExtractionProvenance(
            extraction_id=extraction_id,
            document_id=document_id,
            file_name=file_name,
        )
        
        self._provenance_by_extraction[extraction_id] = provenance
        
        if document_id not in self._provenance_by_document:
            self._provenance_by_document[document_id] = []
        self._provenance_by_document[document_id].append(extraction_id)
        
        return provenance
    
    def get_provenance(self, extraction_id: str) -> Optional[ExtractionProvenance]:
        """获取抽取溯源"""
        return self._provenance_by_extraction.get(extraction_id)
    
    def get_provenance_by_document(self, document_id: str) -> List[ExtractionProvenance]:
        """按文档获取所有抽取溯源"""
        extraction_ids = self._provenance_by_document.get(document_id, [])
        return [self._provenance_by_extraction[eid] for eid in extraction_ids if eid in self._provenance_by_extraction]
    
    def track_extraction(
        self,
        extraction_id: str,
        entity_id: str,
        entity_type: str,
        agent_name: str,
        skill_id: Optional[str] = None,
        prompt_version: Optional[str] = None,
        llm_model: Optional[str] = None,
        evidence_ref: Optional[str] = None,
        confidence: float = 0.0,
        input_text: Optional[str] = None,
        output_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceRecord:
        """追踪实体抽取"""
        record = ProvenanceRecord(
            record_id=f"REC-{str(self._next_record_id).zfill(6)}",
            entity_id=entity_id,
            entity_type=entity_type,
            action_type=ActionType.EXTRACT,
            source=ProvenanceSource.LLM if llm_model else ProvenanceSource.SKILL if skill_id else ProvenanceSource.INFERRED,
            agent_name=agent_name,
            skill_id=skill_id,
            prompt_version=prompt_version,
            llm_model=llm_model,
            evidence_ref=evidence_ref,
            confidence=confidence,
            input_text=input_text,
            output_text=output_text,
            metadata=metadata or {},
        )
        
        self._next_record_id += 1
        
        provenance = self._provenance_by_extraction.get(extraction_id)
        if provenance:
            provenance.add_record(record)
            provenance.llm_calls += 1 if llm_model else 0
        
        return record
    
    def track_inference(
        self,
        extraction_id: str,
        entity_id: str,
        entity_type: str,
        agent_name: str,
        evidence_ref: Optional[str] = None,
        confidence: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceRecord:
        """追踪实体推断"""
        record = ProvenanceRecord(
            record_id=f"REC-{str(self._next_record_id).zfill(6)}",
            entity_id=entity_id,
            entity_type=entity_type,
            action_type=ActionType.INFER,
            source=ProvenanceSource.INFERRED,
            agent_name=agent_name,
            evidence_ref=evidence_ref,
            confidence=confidence,
            metadata=metadata or {},
        )
        
        self._next_record_id += 1
        
        provenance = self._provenance_by_extraction.get(extraction_id)
        if provenance:
            provenance.add_record(record)
        
        return record
    
    def track_merge(
        self,
        extraction_id: str,
        entity_id: str,
        entity_type: str,
        merged_from: List[str],
        agent_name: str,
        confidence: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceRecord:
        """追踪实体合并"""
        record = ProvenanceRecord(
            record_id=f"REC-{str(self._next_record_id).zfill(6)}",
            entity_id=entity_id,
            entity_type=entity_type,
            action_type=ActionType.MERGE,
            source=ProvenanceSource.MERGED,
            agent_name=agent_name,
            confidence=confidence,
            metadata=metadata or {"merged_from": merged_from},
        )
        
        self._next_record_id += 1
        
        provenance = self._provenance_by_extraction.get(extraction_id)
        if provenance:
            provenance.add_record(record)
        
        return record
    
    def get_entity_history(self, entity_id: str) -> List[ProvenanceRecord]:
        """获取实体完整历史"""
        history = []
        for provenance in self._provenance_by_extraction.values():
            history.extend(provenance.get_records_by_entity(entity_id))
        history.sort(key=lambda r: r.timestamp)
        return history
    
    def get_all_summaries(self) -> List[Dict[str, Any]]:
        """获取所有溯源摘要"""
        summaries = []
        for provenance in self._provenance_by_extraction.values():
            summaries.append(provenance.get_summary())
        return summaries
    
    def save(self, output_path: str) -> None:
        """保存所有溯源信息"""
        import json
        data = {
            "provenance": {
                eid: prov.to_dict() for eid, prov in self._provenance_by_extraction.items()
            },
            "document_index": self._provenance_by_document,
            "_metadata": {
                "saved_at": datetime.now().isoformat(),
                "total_extractions": len(self._provenance_by_extraction),
                "total_records": sum(len(p.records) for p in self._provenance_by_extraction.values()),
            },
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, input_path: str) -> None:
        """加载溯源信息"""
        import json
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for eid, prov_data in data.get("provenance", {}).items():
            self._provenance_by_extraction[eid] = ExtractionProvenance.from_dict(prov_data)
        
        self._provenance_by_document = data.get("document_index", {})


tracker = ProvenanceTracker()