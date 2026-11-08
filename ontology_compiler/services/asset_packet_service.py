"""Asset Packet Service - Generate standardized asset packages from pipeline outputs.

[DATA CONTINUITY PLAN 2026-07-09]: Implements standard asset package generation:
- entities.json: Entity data with provenance
- relations.json: Relation data with provenance
- evidence.json: Evidence chunks
- manifest.json: Asset manifest and metadata
- quality_report.json: Quality assessment report
- provenance.json: Complete provenance trace

This module enables:
- Standardized pipeline output format
- Direct RAG consumption of governed assets
- Complete traceability of all outputs
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class EntityAsset:
    """Entity asset with provenance."""
    id: str
    code: str = ""
    name: str = ""
    type: str = ""
    layer: str = ""
    status: str = "active"
    description: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    evidence_ref: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RelationAsset:
    """Relation asset with provenance."""
    id: str
    source_id: str = ""
    target_id: str = ""
    relation_type: str = ""
    confidence: float = 0.0
    evidence_ref: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceAsset:
    """Evidence asset."""
    id: str
    content: str = ""
    source_file: str = ""
    source_line: int = 0
    source_page: int = 0
    entity_refs: List[str] = field(default_factory=list)
    relation_refs: List[str] = field(default_factory=list)
    confidence: float = 0.0
    chunk_id: str = ""
    embedding: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AssetPacketQualityReport:
    """Quality assessment report for an asset packet.

    Distinct context from the canonical domain `QualityReport`
    (`src.domain.models.quality_report`, contract `quality_report` v1.0.0):
    this DTO summarizes packet-level quality for the governed/raw asset-packet
    artifacts and carries its own payload version "2.0.0". Kept per-context per
    Phase 4 (统一事实来源) naming rule to avoid same-name cross-context reuse.
    """
    run_id: str
    overall_score: float = 0.0
    review_required: bool = False
    entity_quality: Dict[str, Any] = field(default_factory=dict)
    relation_quality: Dict[str, Any] = field(default_factory=dict)
    governance_checks: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": "2.0.0",
            "run_id": self.run_id,
            "generated_at": datetime.now().isoformat(),
            "overall_score": self.overall_score,
            "review_required": self.review_required,
            "entity_quality": self.entity_quality,
            "relation_quality": self.relation_quality,
            "governance_checks": self.governance_checks,
            "recommendations": self.recommendations,
        }


@dataclass
class ProvenanceRecord:
    """Provenance record for the asset package."""
    run_id: str
    workflow: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    rules: Dict[str, Any] = field(default_factory=dict)
    models: List[Dict[str, Any]] = field(default_factory=list)
    skills: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": "2.0.0",
            "run_id": self.run_id,
            "generated_at": datetime.now().isoformat(),
            "workflow": self.workflow,
            "sources": self.sources,
            "rules": self.rules,
            "models": self.models,
            "skills": self.skills,
        }


class AssetPacketService:
    """Service for generating standardized asset packages."""

    def __init__(self, output_base_path: Optional[Path] = None):
        self.output_base_path = output_base_path or _PROJECT_ROOT / "outputs" / "asset_packets"
        self.output_base_path.mkdir(parents=True, exist_ok=True)
        self._raw_path = self.output_base_path / "raw"
        self._governed_path = self.output_base_path / "governed"
        self._raw_path.mkdir(parents=True, exist_ok=True)
        self._governed_path.mkdir(parents=True, exist_ok=True)

    def generate_asset_packet(
        self,
        run_id: str,
        session_id: str,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]] = None,
        evidence: List[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        governed: bool = False,
    ) -> str:
        """Generate a complete asset package.

        Args:
            run_id: Run ID
            session_id: Session ID
            entities: List of entities
            relations: List of relations
            evidence: List of evidence chunks
            metadata: Additional metadata
            governed: Whether this is a governed package

        Returns:
            Path to the asset package directory
        """
        relations = relations or []
        evidence = evidence or []
        metadata = metadata or {}

        target_path = self._governed_path if governed else self._raw_path
        package_dir = target_path / run_id
        package_dir.mkdir(parents=True, exist_ok=True)

        metadata = self._normalize_package_metadata(metadata)
        quality_report = self._generate_quality_report(run_id, entities, relations, metadata)
        provenance_record = self._generate_provenance(run_id, session_id, entities, relations, evidence, metadata)

        self._write_entities(package_dir, run_id, entities)
        self._write_relations(package_dir, run_id, relations)
        self._write_evidence(package_dir, run_id, evidence)
        self._write_quality_report(package_dir, quality_report)
        self._write_provenance(package_dir, provenance_record)
        self._write_manifest(package_dir, run_id, session_id, entities, relations, evidence, metadata, governed)
        self._write_feedback(package_dir, metadata, quality_report)

        if governed:
            self._copy_to_governed(run_id, package_dir)

        logger.info(f"Generated asset package at {package_dir}")
        return str(package_dir)

    def add_provenance_to_entities(
        self,
        entities: List[Dict[str, Any]],
        source_file: str,
        run_id: str,
        rule_version: str,
        model_version: str,
        skill_version: str = "2.0.0",
    ) -> List[Dict[str, Any]]:
        """Add provenance information to entities."""
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            if "provenance" not in entity:
                entity["provenance"] = {}

            entity["provenance"].update({
                "source_file": source_file,
                "source_run_id": run_id,
                "rule_version": rule_version,
                "model_version": model_version,
                "skill_version": skill_version,
                "extracted_at": datetime.now().isoformat(),
                "confidence": entity.get("confidence", 0.0),
            })

            if "id" not in entity:
                entity["id"] = entity.get("meta_id", "") or f"ent_{hash(entity.get('meta_id', '') + entity.get('meta_name', '')) % 1000000}"
            if "code" not in entity:
                entity["code"] = entity.get("id", "") or entity.get("meta_id", "")
            if "layer" not in entity:
                entity["layer"] = "L2"
            if "status" not in entity:
                entity["status"] = "active"

        return entities

    def add_provenance_to_relations(
        self,
        relations: List[Dict[str, Any]],
        source_file: str,
        run_id: str,
        rule_version: str,
        model_version: str,
        skill_version: str = "2.0.0",
    ) -> List[Dict[str, Any]]:
        """Add provenance information to relations.
        
        [ENHANCEMENT] Normalize relation fields to match quality gate expectations:
        - source -> source_id
        - target -> target_id
        - type -> relation_type
        - Convert confidence string to float (HIGH/MEDIUM/LOW -> 0.95/0.70/0.50)
        """
        confidence_map = {
            "HIGH": 0.95,
            "MEDIUM": 0.70,
            "LOW": 0.50,
            "high": 0.95,
            "medium": 0.70,
            "low": 0.50,
        }
        
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            if "provenance" not in relation:
                relation["provenance"] = {}

            source = relation.get("source", relation.get("source_id", ""))
            target = relation.get("target", relation.get("target_id", ""))
            rtype = relation.get("type", relation.get("relation_type", ""))
            
            if source and "source_id" not in relation:
                relation["source_id"] = source
            if target and "target_id" not in relation:
                relation["target_id"] = target
            if rtype and "relation_type" not in relation:
                relation["relation_type"] = rtype
            
            confidence_val = relation.get("confidence", 0.0)
            if isinstance(confidence_val, str):
                confidence_val = confidence_map.get(confidence_val.upper(), 0.70)
            elif isinstance(confidence_val, float) or isinstance(confidence_val, int):
                confidence_val = float(confidence_val)
            else:
                confidence_val = 0.70
            
            relation["confidence"] = confidence_val

            relation["provenance"].update({
                "source_file": source_file,
                "source_run_id": run_id,
                "rule_version": rule_version,
                "model_version": model_version,
                "skill_version": skill_version,
                "extracted_at": datetime.now().isoformat(),
                "confidence": confidence_val,
            })

            if "id" not in relation:
                relation["id"] = f"rel_{hash(f'{source}_{target}_{rtype}') % 1000000}"

        return relations

    def _write_entities(self, package_dir: Path, run_id: str, entities: List[Dict[str, Any]]) -> None:
        """Write entities to entities.json."""
        data = {
            "version": "2.0.0",
            "generated_at": datetime.now().isoformat(),
            "run_id": run_id,
            "total_count": len(entities),
            "entities": entities,
        }
        with open(package_dir / "entities.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _write_relations(self, package_dir: Path, run_id: str, relations: List[Dict[str, Any]]) -> None:
        """Write relations to relations.json."""
        data = {
            "version": "2.0.0",
            "generated_at": datetime.now().isoformat(),
            "run_id": run_id,
            "total_count": len(relations),
            "relations": relations,
        }
        with open(package_dir / "relations.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _write_evidence(self, package_dir: Path, run_id: str, evidence: List[Dict[str, Any]]) -> None:
        """Write evidence to evidence.json."""
        data = {
            "version": "2.0.0",
            "generated_at": datetime.now().isoformat(),
            "run_id": run_id,
            "total_count": len(evidence),
            "evidence": evidence,
        }
        with open(package_dir / "evidence.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _normalize_package_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        metadata = dict(metadata or {})
        metadata.setdefault("document_type", metadata.get("document_type", ""))
        metadata.setdefault("document_type_confidence", metadata.get("document_type_confidence", 0.0))
        metadata.setdefault("matched_skills", metadata.get("matched_skills", []))
        metadata.setdefault("skill_stats", metadata.get("skill_stats", {}))
        metadata.setdefault("classification_hints", metadata.get("classification_hints", {}))
        return metadata

    def _write_manifest(self, package_dir: Path, run_id: str, session_id: str,
                        entities: List[Dict[str, Any]], relations: List[Dict[str, Any]],
                        evidence: List[Dict[str, Any]], metadata: Dict[str, Any], governed: bool) -> None:
        """Write manifest to manifest.json."""
        mbom_stats = self._calculate_mbom_signals(entities, relations)
        data = {
            "version": "2.0.0",
            "run_id": run_id,
            "session_id": session_id,
            "generated_at": datetime.now().isoformat(),
            "status": "completed",
            "input_files": metadata.get("input_files", []),
            "pipeline_version": metadata.get("pipeline_version", ""),
            "rule_version": metadata.get("rule_version", ""),
            "model_version": metadata.get("model_version", ""),
            "skill_versions": metadata.get("skill_versions", {}),
            "matched_skills": metadata.get("matched_skills", []),
            "document_type": metadata.get("document_type", ""),
            "document_type_confidence": metadata.get("document_type_confidence", 0.0),
            "skill_stats": metadata.get("skill_stats", {}),
            "prompt_version": metadata.get("prompt_version", ""),
            "output_path": str(package_dir),
            "execution_duration_ms": metadata.get("execution_duration_ms", 0),
            "llm_call_count": metadata.get("llm_call_count", 0),
            "governed": governed,
            "output_tier": "governed" if governed else "raw",
            "assets": {
                "entities": "entities.json",
                "relations": "relations.json",
                "evidence": "evidence.json",
                "quality_report": "quality_report.json",
                "provenance": "provenance.json",
            },
            "statistics": {
                "total_files": len(metadata.get("input_files", [])),
                "entity_count": len(entities),
                "relation_count": len(relations),
                "evidence_count": len(evidence),
                "quality_score": metadata.get("quality_score", 0.0),
                "review_required": metadata.get("review_required", False),
                "mbom_signals": mbom_stats,
                "critical_signals": metadata.get("critical_signals", {}),
                "document_type": metadata.get("document_type", ""),
                "document_type_confidence": metadata.get("document_type_confidence", 0.0),
                "matched_skills": metadata.get("matched_skills", []),
            },
            "errors": metadata.get("errors", []),
            "warnings": metadata.get("warnings", []),
        }
        with open(package_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _write_quality_report(self, package_dir: Path, report: AssetPacketQualityReport) -> None:
        """Write quality report to quality_report.json."""
        with open(package_dir / "quality_report.json", "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

    def _calculate_mbom_signals(self, entities: List[Dict[str, Any]], relations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate MBOM-oriented signal coverage for asset packaging."""
        signal_counts = {
            "occurrence": 0,
            "position": 0,
            "variant": 0,
            "provenance": 0,
            "source_structure": 0,
            "alignment": 0,
            "dedup": 0,
            "change": 0,
        }

        def _mark(name: str):
            lname = name.lower()
            for token, key in (
                ("occurrence", "occurrence"),
                ("position", "position"),
                ("variant", "variant"),
                ("provenance", "provenance"),
                ("source_structure", "source_structure"),
                ("alignment", "alignment"),
                ("dedup", "dedup"),
                ("merge", "dedup"),
                ("change", "change"),
                ("revision", "change"),
                ("version", "change"),
            ):
                if token in lname:
                    signal_counts[key] += 1

        for ent in entities:
            if not isinstance(ent, dict):
                continue
            text = " ".join([
                str(ent.get("name", "")),
                str(ent.get("meta_name", "")),
                str(ent.get("description", "")),
                json.dumps(ent.get("provenance", {}), ensure_ascii=False),
            ])
            _mark(text)
            if ent.get("provenance"):
                signal_counts["provenance"] += 1
            if ent.get("source_structure") or ent.get("source_document") or ent.get("source_file"):
                signal_counts["source_structure"] += 1
            if any(ent.get(k) for k in ("variant", "variant_code", "variant_designation", "position", "position_code", "position_label", "occurrence", "occurrence_id")):
                if any(ent.get(k) for k in ("variant", "variant_code", "variant_designation")):
                    signal_counts["variant"] += 1
                if any(ent.get(k) for k in ("position", "position_code", "position_label")):
                    signal_counts["position"] += 1
                if any(ent.get(k) for k in ("occurrence", "occurrence_id")):
                    signal_counts["occurrence"] += 1

        for rel in relations:
            if not isinstance(rel, dict):
                continue
            text = " ".join([
                str(rel.get("relation_type", "")),
                str(rel.get("type", "")),
                json.dumps(rel.get("provenance", {}), ensure_ascii=False),
            ])
            _mark(text)
            if rel.get("provenance"):
                signal_counts["provenance"] += 1

        total_entities = len([e for e in entities if isinstance(e, dict)]) or 1
        total_relations = len([r for r in relations if isinstance(r, dict)]) or 1
        return {
            "counts": signal_counts,
            "entity_coverage": {
                "occurrence": round(signal_counts["occurrence"] / total_entities, 3),
                "position": round(signal_counts["position"] / total_entities, 3),
                "variant": round(signal_counts["variant"] / total_entities, 3),
                "provenance": round(signal_counts["provenance"] / total_entities, 3),
                "source_structure": round(signal_counts["source_structure"] / total_entities, 3),
            },
            "relation_coverage": {
                "alignment": round(signal_counts["alignment"] / total_relations, 3),
                "dedup": round(signal_counts["dedup"] / total_relations, 3),
                "change": round(signal_counts["change"] / total_relations, 3),
            },
        }

    def _write_provenance(self, package_dir: Path, provenance: ProvenanceRecord) -> None:
        """Write provenance to provenance.json."""
        with open(package_dir / "provenance.json", "w", encoding="utf-8") as f:
            json.dump(provenance.to_dict(), f, indent=2, ensure_ascii=False)

    def _get_entity_confidence(self, entity: Dict[str, Any]) -> float:
        """Get entity confidence from multiple possible locations."""
        if not isinstance(entity, dict):
            return 0.7
        confidence = entity.get("confidence")
        if confidence is not None:
            return float(confidence)
        
        evidence = entity.get("evidence", {})
        if isinstance(evidence, dict):
            evidence_conf = evidence.get("confidence")
            if evidence_conf is not None:
                return float(evidence_conf)
        
        provenance = entity.get("provenance", {})
        if isinstance(provenance, dict):
            provenance_conf = provenance.get("confidence")
            if provenance_conf is not None:
                return float(provenance_conf)
        
        return 0.7

    def _generate_quality_report(self, run_id: str, entities: List[Dict[str, Any]],
                                 relations: List[Dict[str, Any]],
                                 metadata: Optional[Dict[str, Any]] = None) -> AssetPacketQualityReport:
        """Generate quality report from entities and relations."""
        entity_confidences = [self._get_entity_confidence(e) for e in entities]
        relation_confidences = [r.get("confidence", 0.0) for r in relations if r.get("confidence")]

        avg_entity_conf = sum(entity_confidences) / len(entity_confidences) if entity_confidences else 0.0
        avg_relation_conf = sum(relation_confidences) / len(relation_confidences) if relation_confidences else 0.0

        mbom_signals = self._calculate_mbom_signals(entities, relations)

        entity_quality = {
            "total_count": len(entities),
            "valid_count": len([e for e in entities if self._get_entity_confidence(e) >= 0.5]),
            "invalid_count": len([e for e in entities if self._get_entity_confidence(e) < 0.5]),
            "confidence_mean": avg_entity_conf,
            "completeness_score": self._calculate_completeness(entities),
            "duplicate_rate": 0.0,
            "violations": [],
            "mbom_signals": mbom_signals.get("entity_coverage", {}),
        }

        relation_quality = {
            "total_count": len(relations),
            "valid_count": len([r for r in relations if r.get("confidence", 0) >= 0.5]),
            "invalid_count": len([r for r in relations if r.get("confidence", 0) < 0.5]),
            "confidence_mean": avg_relation_conf,
            "conflict_count": 0,
            "coverage_score": self._calculate_coverage(entities, relations),
            "violations": [],
            "mbom_signals": mbom_signals.get("relation_coverage", {}),
        }

        overall_score = (entity_quality["confidence_mean"] + relation_quality["confidence_mean"]) / 2

        matched_skills = metadata.get("matched_skills", []) if isinstance(metadata, dict) else []
        document_type = metadata.get("document_type", "") if isinstance(metadata, dict) else ""
        doc_gate = self._build_document_type_gate(document_type, entities, relations, metadata)
        recommendations = []
        if matched_skills:
            recommendations.append(f"Matched {len(matched_skills)} skill(s): {', '.join(str(s.get('skill_name', s.get('skill_id', 'unknown'))) for s in matched_skills[:5] if isinstance(s, dict))}")
        recommendations.extend(doc_gate.get("recommendations", []))
        quality_report = AssetPacketQualityReport(
            run_id=run_id,
            overall_score=overall_score,
            review_required=(overall_score < 0.7) or bool(doc_gate.get("review_required")),
            entity_quality={**entity_quality, "matched_skills": matched_skills, "document_type": document_type},
            relation_quality={**relation_quality, "matched_skills": matched_skills, "document_type": document_type},
            governance_checks=doc_gate.get("governance_checks", []),
            recommendations=recommendations,
        )
        return quality_report

    def _calculate_completeness(self, entities: List[Dict[str, Any]]) -> float:
        """Calculate entity completeness score."""
        if not entities:
            return 0.0

        required_fields = ["id", "name", "type", "layer"]
        valid_entities = [e for e in entities if isinstance(e, dict)]
        total_checks = len(valid_entities) * len(required_fields)
        passed_checks = sum(
            1 for e in valid_entities for field in required_fields
            if e.get(field) and str(e[field]).strip()
        )
        return passed_checks / total_checks if total_checks > 0 else 0.0

    def _calculate_coverage(self, entities: List[Dict[str, Any]], relations: List[Dict[str, Any]]) -> float:
        """Calculate relation coverage score."""
        if not entities or not relations:
            return 0.0

        valid_entities = [e for e in entities if isinstance(e, dict)]
        valid_relations = [r for r in relations if isinstance(r, dict)]
        
        if not valid_entities or not valid_relations:
            return 0.0

        entity_ids = set(e.get("id", "") for e in valid_entities)
        connected_entities = set()
        for r in valid_relations:
            if r.get("source_id") in entity_ids:
                connected_entities.add(r["source_id"])
            if r.get("target_id") in entity_ids:
                connected_entities.add(r["target_id"])

        return len(connected_entities) / len(entity_ids)

    def _build_document_type_gate(self, document_type: str, entities: List[Dict[str, Any]], relations: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        document_type = (document_type or "").lower()
        recommendations = []
        governance_checks = []
        review_required = False

        def has_term(items: List[Dict[str, Any]], terms: List[str]) -> bool:
            for item in items:
                if not isinstance(item, dict):
                    continue
                text = " ".join(str(item.get(k, "")) for k in ("name", "meta_name", "description", "meta_class", "type", "relation_type"))
                text_l = text.lower()
                if any(term in text_l for term in terms):
                    return True
            return False

        if document_type == "mbom":
            required = {
                "position": has_term(entities, ["position"]),
                "variant": has_term(entities, ["variant"]),
                "occurrence": has_term(entities, ["occurrence"]),
                "provenance": has_term(entities, ["provenance"]),
                "source_structure": has_term(entities, ["source structure", "source_structure", "structure_level"]),
                "alignment": has_term(relations, ["align", "alignment"]),
            }
            for key, ok in required.items():
                governance_checks.append({"check": key, "passed": ok})
                if not ok:
                    review_required = True
                    recommendations.append(f"MBOM gate: missing {key}")
        elif document_type == "benchmark":
            required = {
                "scope": has_term(entities, ["scope"]),
                "kpi": has_term(entities, ["kpi"]),
                "question": has_term(entities, ["question"]),
            }
            for key, ok in required.items():
                governance_checks.append({"check": key, "passed": ok})
                if not ok:
                    review_required = True
                    recommendations.append(f"Benchmark gate: missing {key}")
        elif document_type == "process_governance":
            required = {
                "rule": has_term(entities, ["rule"]),
                "workflow": has_term(entities, ["workflow", "process"]),
                "owner": has_term(entities, ["owner", "manager", "approver"]),
            }
            for key, ok in required.items():
                governance_checks.append({"check": key, "passed": ok})
                if not ok:
                    review_required = True
                    recommendations.append(f"Governance gate: missing {key}")
        else:
            governance_checks.append({"check": "document_type", "passed": False, "note": "unknown or mixed"})

        if isinstance(metadata, dict) and metadata.get("matched_skills"):
            governance_checks.append({"check": "matched_skills", "passed": True, "count": len(metadata.get("matched_skills", []))})

        return {"review_required": review_required, "governance_checks": governance_checks, "recommendations": recommendations}

    def _generate_provenance(self, run_id: str, session_id: str, entities: List[Dict[str, Any]],
                            relations: List[Dict[str, Any]], evidence: List[Dict[str, Any]],
                            metadata: Dict[str, Any]) -> ProvenanceRecord:
        """Generate provenance record."""
        workflow = [
            {
                "step_id": "extract",
                "step_name": "Entity/Relation Extraction",
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "input_refs": metadata.get("input_files", []),
                "output_refs": ["entities.json", "relations.json"],
                "skill_name": "entity_recognition",
                "skill_version": metadata.get("skill_versions", {}).get("entity_recognition", "2.0.0"),
                "model_name": metadata.get("model_version", ""),
                "model_version": metadata.get("model_version", ""),
            },
            {
                "step_id": "govern",
                "step_name": "Governance & Validation",
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "input_refs": ["entities.json", "relations.json"],
                "output_refs": ["quality_report.json"],
                "skill_name": "quality_gate",
                "skill_version": "2.0.0",
            },
            {
                "step_id": "package",
                "step_name": "Asset Packaging",
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "input_refs": ["entities.json", "relations.json", "evidence.json", "quality_report.json"],
                "output_refs": ["manifest.json", "provenance.json"],
                "skill_name": "asset_packet",
                "skill_version": "2.0.0",
            },
        ]

        sources = []
        for file_path in metadata.get("input_files", []):
            sources.append({
                "file_path": file_path,
                "file_hash": "",
                "processed_at": datetime.now().isoformat(),
                "entities_extracted": 0,
                "relations_extracted": 0,
            })

        models = []
        if metadata.get("model_version"):
            models.append({
                "name": metadata.get("model_version", ""),
                "version": "",
                "call_count": metadata.get("llm_call_count", 0),
                "total_tokens": 0,
                "total_cost": 0,
            })

        skills = []
        for skill_name, skill_version in metadata.get("skill_versions", {}).items():
            skills.append({
                "name": skill_name,
                "version": skill_version,
                "executed_at": datetime.now().isoformat(),
                "duration_ms": 0,
                "input_count": 0,
                "output_count": 0,
            })
        for hit in metadata.get("matched_skills", []) if isinstance(metadata, dict) else []:
            if isinstance(hit, dict):
                skills.append({
                    "name": hit.get("skill_name", hit.get("skill_id", "unknown")),
                    "version": hit.get("version", ""),
                    "executed_at": datetime.now().isoformat(),
                    "duration_ms": 0,
                    "input_count": 0,
                    "output_count": 0,
                })

        return ProvenanceRecord(
            run_id=run_id,
            workflow=workflow,
            sources=sources,
            rules={
                "version": metadata.get("rule_version", ""),
                "classification_rules": [],
                "relation_rules": [],
                "validation_rules": [],
            },
            models=models,
            skills=skills,
        )

    def _copy_to_governed(self, run_id: str, source_dir: Path):
        """Copy raw package to governed location (legacy support)."""
        pass

    def _write_feedback(self, package_dir: Path, metadata: Dict[str, Any], quality_report: AssetPacketQualityReport) -> None:
        """Write lightweight feedback loop artifact for downstream analysis."""
        feedback = {
            "version": "1.0.0",
            "run_id": metadata.get("run_id", ""),
            "session_id": metadata.get("session_id", ""),
            "generated_at": datetime.now().isoformat(),
            "document_type": metadata.get("document_type", ""),
            "document_type_confidence": metadata.get("document_type_confidence", 0.0),
            "matched_skills": metadata.get("matched_skills", []),
            "schema_report": metadata.get("schema_report", {}),
            "quality_score": quality_report.overall_score,
            "review_required": quality_report.review_required,
            "governance_checks": quality_report.governance_checks,
            "recommendations": quality_report.recommendations,
            "critical_signals": metadata.get("critical_signals", {}),
        }
        with open(package_dir / "feedback.json", "w", encoding="utf-8") as f:
            json.dump(feedback, f, indent=2, ensure_ascii=False)

    def load_asset_packet(self, run_id: str, governed: Optional[bool] = None) -> Optional[Dict[str, Any]]:
        """Load an asset package by run ID.

        Args:
            run_id: Run ID
            governed: If True, load from governed; If False, load from raw;
                      If None, search both locations

        Returns:
            Asset package data or None if not found
        """
        if governed is True:
            package_dir = self._governed_path / run_id
            if package_dir.exists():
                return self._load_package_from_dir(package_dir)
            return None
        elif governed is False:
            package_dir = self._raw_path / run_id
            if package_dir.exists():
                return self._load_package_from_dir(package_dir)
            return None
        else:
            for path in [self._governed_path, self._raw_path]:
                package_dir = path / run_id
                if package_dir.exists():
                    return self._load_package_from_dir(package_dir)
            return None

    def _load_package_from_dir(self, package_dir: Path) -> Dict[str, Any]:
        """Load asset package from directory."""
        result = {}
        files_to_load = ["entities.json", "relations.json", "evidence.json",
                         "manifest.json", "quality_report.json", "provenance.json"]

        for file_name in files_to_load:
            file_path = package_dir / file_name
            if file_path.exists():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        result[file_name.replace(".json", "")] = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load {file_name}: {e}")

        result["package_path"] = str(package_dir)
        return result

    def get_governed_packages(self) -> List[str]:
        """Get all governed asset package IDs."""
        return [d.name for d in self._governed_path.iterdir() if d.is_dir()]

    def get_raw_packages(self) -> List[str]:
        """Get all raw asset package IDs."""
        return [d.name for d in self._raw_path.iterdir() if d.is_dir()]

    def promote_to_governed(self, run_id: str):
        """Promote a raw package to governed."""
        raw_dir = self._raw_path / run_id
        if not raw_dir.exists():
            logger.warning(f"Raw package not found: {run_id}")
            return

        governed_dir = self._governed_path / run_id
        if governed_dir.exists():
            import shutil
            shutil.rmtree(governed_dir)

        import shutil
        shutil.copytree(raw_dir, governed_dir)
        logger.info(f"Promoted package {run_id} to governed")

    def list_packages(self, governed_only: bool = False, raw_only: bool = False) -> List[Dict[str, Any]]:
        """List all asset packages with metadata."""
        packages = []
        
        if not raw_only:
            for pkg_dir in self._governed_path.iterdir():
                if pkg_dir.is_dir():
                    manifest_path = pkg_dir / "manifest.json"
                    if manifest_path.exists():
                        with open(manifest_path, "r", encoding="utf-8") as f:
                            manifest = json.load(f)
                            packages.append({
                                "run_id": pkg_dir.name,
                                "governed": True,
                                "generated_at": manifest.get("generated_at", ""),
                                "status": manifest.get("status", ""),
                                "entity_count": manifest.get("statistics", {}).get("entity_count", 0),
                                "quality_score": manifest.get("statistics", {}).get("quality_score", 0),
                            })

        if not governed_only:
            for pkg_dir in self._raw_path.iterdir():
                if pkg_dir.is_dir():
                    manifest_path = pkg_dir / "manifest.json"
                    if manifest_path.exists():
                        with open(manifest_path, "r", encoding="utf-8") as f:
                            manifest = json.load(f)
                            packages.append({
                                "run_id": pkg_dir.name,
                                "governed": False,
                                "generated_at": manifest.get("generated_at", ""),
                                "status": manifest.get("status", ""),
                                "entity_count": manifest.get("statistics", {}).get("entity_count", 0),
                                "quality_score": manifest.get("statistics", {}).get("quality_score", 0),
                            })

        return packages


_global_asset_packet_service = None


def get_asset_packet_service() -> AssetPacketService:
    """Get or create global asset packet service instance."""
    global _global_asset_packet_service
    if _global_asset_packet_service is None:
        _global_asset_packet_service = AssetPacketService()
    return _global_asset_packet_service


__all__ = [
    "EntityAsset",
    "RelationAsset",
    "EvidenceAsset",
    "AssetPacketQualityReport",
    "ProvenanceRecord",
    "AssetPacketService",
    "get_asset_packet_service",
]