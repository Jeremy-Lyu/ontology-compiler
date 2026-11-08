from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RETRIEVAL_PACKET_ROOT = PROJECT_ROOT / "outputs" / "runs" / "ontology" / "retrieval_packets"


@dataclass
class RetrievalAssetPacketService:
    """Build and persist retrieval asset packets.

    This service centralizes the packet contract so MainPipeline can focus on
    orchestration and stage coordination.
    """

    storage_root: Path = DEFAULT_RETRIEVAL_PACKET_ROOT
    _last_packet_paths: dict[str, Path] = field(default_factory=dict)

    def _packet_path(self, run_id: str) -> Path:
        return Path(self.storage_root) / run_id / "retrieval_asset_packet.contract.json"

    @staticmethod
    def _serialize_trace_context(trace_context: Any) -> dict[str, Any]:
        if trace_context is None:
            return {}
        if isinstance(trace_context, dict):
            return dict(trace_context)
        if hasattr(trace_context, "model_dump"):
            try:
                return trace_context.model_dump(exclude_none=True)
            except Exception:
                pass
        if hasattr(trace_context, "to_dict"):
            try:
                return trace_context.to_dict()
            except Exception:
                pass
        if hasattr(trace_context, "__dict__"):
            return {k: v for k, v in vars(trace_context).items() if not k.startswith("_")}
        return {"value": str(trace_context)}

    def build_packet(
        self,
        *,
        run_id: str,
        ontology_version: str,
        trace_context: Any,
        document_ref: dict[str, Any],
        chunk_ref: dict[str, Any],
        entity_refs: list[dict[str, Any]],
        relation_refs: list[dict[str, Any]],
        evidence_refs: list[dict[str, Any]],
        structured_filters: dict[str, Any],
        retrieval_text: list[str],
        quality: dict[str, Any],
        provenance: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "packet_id": f"retrieval-{run_id}",
            "ontology_version": ontology_version,
            "run_id": run_id,
            "trace_context": self._serialize_trace_context(trace_context),
            "document_refs": [document_ref],
            "chunk_refs": [chunk_ref],
            "entity_refs": entity_refs,
            "relation_refs": relation_refs,
            "evidence_refs": evidence_refs,
            "structured_filters": structured_filters,
            "retrieval_text": retrieval_text,
            "quality": quality,
            "provenance": provenance,
        }

    def save_packet(self, *, run_id: str, packet: dict[str, Any]) -> str:
        packet_path = self._packet_path(run_id)
        packet_path.parent.mkdir(parents=True, exist_ok=True)
        packet_path.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
        self._last_packet_paths[run_id] = packet_path
        return str(packet_path)

    def build_and_save_packet(
        self,
        *,
        run_id: str,
        ontology_version: str,
        trace_context: Any,
        document_ref: dict[str, Any],
        chunk_ref: dict[str, Any],
        entity_refs: list[dict[str, Any]],
        relation_refs: list[dict[str, Any]],
        evidence_refs: list[dict[str, Any]],
        structured_filters: dict[str, Any],
        retrieval_text: list[str],
        quality: dict[str, Any],
        provenance: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        packet = self.build_packet(
            run_id=run_id,
            ontology_version=ontology_version,
            trace_context=trace_context,
            document_ref=document_ref,
            chunk_ref=chunk_ref,
            entity_refs=entity_refs,
            relation_refs=relation_refs,
            evidence_refs=evidence_refs,
            structured_filters=structured_filters,
            retrieval_text=retrieval_text,
            quality=quality,
            provenance=provenance,
        )
        return self.save_packet(run_id=run_id, packet=packet), packet

    def load_packet(self, run_id: str) -> Optional[dict[str, Any]]:
        packet_path = self._packet_path(run_id)
        if not packet_path.exists():
            return None
        try:
            return json.loads(packet_path.read_text(encoding="utf-8"))
        except Exception:
            return None


_SERVICE: Optional[RetrievalAssetPacketService] = None


def get_retrieval_asset_packet_service() -> RetrievalAssetPacketService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = RetrievalAssetPacketService()
    return _SERVICE
