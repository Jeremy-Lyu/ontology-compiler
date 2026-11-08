"""MetadataEnrichmentSkill — deterministic structured metadata extraction (P2-2 F7).

Wraps :mod:`src.pipeline.metadata_enrichment` (already pure, zero LLM cost).
The service extracts dates / security classes / frequencies / conditions /
formulas / source hints / time windows / percentages / ranges from free text.
This skill lifts it into the skill contract so it can be discovered, versioned,
and regression-tested like every other skill.

Boundaries (§6.2):
- Input:  entities + document_content (or entity descriptions when content absent)
- Output: enriched_entities + enrichment_count + context_types_triggered
- Trigger: extract stage, after entity_classification, when document_type ∈ {pptx,docx,md,pdf}
- No LLM, no side effects — pure in-memory transform.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..base import Skill, SkillContext, SkillPriority, SkillResult
from src.pipeline.metadata_enrichment import extract_metadata


class MetadataEnrichmentSkill(Skill):
    """Attach structured metadata to entities by scanning document text."""

    @property
    def priority(self) -> SkillPriority:
        return SkillPriority.NORMAL

    @property
    def description(self) -> str:
        return "为实体补充结构化 metadata（日期/密级/频率/公式/来源提示等，确定性零成本）"

    async def can_apply(self, context: SkillContext) -> bool:
        return bool(self._entities_from(context))

    async def execute(self, context: SkillContext) -> SkillResult:
        entities = self._entities_from(context)
        if not entities:
            return SkillResult.skipped_result(self.name, "no entities to enrich")

        doc_text = self._document_text_from(context, entities)
        meta = extract_metadata(doc_text)

        enriched: List[Dict[str, Any]] = []
        for ent in entities:
            c = dict(ent)
            c["structured_metadata"] = meta.to_dict()
            c.setdefault("enrichment_source", "metadata_enrichment_skill")
            # context type derived from pipeline's SOURCE_HINTS mapping
            c.setdefault("metadata_context_type", self._context_type(doc_text))
            enriched.append(c)

        # context_types_triggered mirrors pipeline's context_mapping contract
        triggered = [self._context_type(doc_text)] if doc_text.strip() else []

        return SkillResult(
            success=True,
            skill_name=self.name,
            applied=True,
            output={
                "enriched_entities": enriched,
                "enrichment_count": len(enriched),
                "context_types_triggered": triggered,
                "extracted_metadata": meta.to_dict(),
            },
        )

    # ---------- extraction helpers ----------

    def _entities_from(self, context: SkillContext) -> List[Dict[str, Any]]:
        if context.entities:
            return list(context.entities)
        cfg = context.config or {}
        for key in ("entities", "enriched_entities"):
            v = cfg.get(key)
            if isinstance(v, list) and v:
                return list(v)
        ext = context.extraction_result or {}
        for key in ("entities", "nodes"):
            v = ext.get(key)
            if isinstance(v, list) and v:
                return list(v)
        return []

    def _document_text_from(self, context: SkillContext, entities: List[Dict[str, Any]]) -> str:
        cfg = context.config or {}
        for key in ("document_content", "document_text", "text"):
            v = cfg.get(key)
            if isinstance(v, str) and v.strip():
                return v
        if context.document_content and context.document_content.strip():
            return context.document_content
        # fallback: concatenate entity descriptions / names
        parts = []
        for e in entities:
            for k in ("description", "meta_description", "name", "meta_name"):
                v = e.get(k)
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip())
                    break
        return "\n".join(parts)

    @staticmethod
    def _context_type(text: str) -> str:
        low = (text or "").lower()
        if any(t in low for t in ("wip", "kpi", "report", "benchmark")):
            return "reporting_context"
        if any(t in low for t in ("mbom", "bom", "bop", "plant", "factory")):
            return "manufacturing_context"
        if any(t in low for t in ("inventory", "supply", "stock")):
            return "operational_context"
        if any(t in low for t in ("compliance", "safety", "proprietary", "confidential")):
            return "policy_context"
        return "simple"
