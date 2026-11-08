"""Built-in Skills — QA 报告识别的六大技能实现 + 治理技能

R38-QA: 六大隐式技能显式实现:
1. CostAwareSkipSkill - 成本意识跳过 ✅
2. HierarchicalExtractionSkill - 一键多层抽取 ✅
3. ContextBudgetingSkill - 上下文浓缩 ✅ [2026-06-18]
4. IndexedBatchAssociationSkill - 批量关联 ✅ [2026-06-18]
5. TransactionalPatchSkill - 并发安全 ✅ [2026-06-18]
6. AgentOrchestrationSkill - 分析策略编排 ✅ [2026-06-18]

治理技能:
7. CheckConfigSkill - 配置检查 ✅
8. ValidateOutputContractsSkill - 输出契约验证 ✅
9. OntologyValidatorSkill - 本体验证 ✅
10. OntologyMergerSkill - 本体合并 ✅
11. ConsensusResolverSkill - 冲突解决 ✅
12. DriftDetectorSkill - 漂移检测 ✅
13. RunOntologyRagValidateSkill - RAG验证 ✅
14. CapsulePublisherSkill - 胶囊发布 ✅
"""

from .cost_aware_skip import CostAwareSkipSkill
from .hierarchical_extraction import HierarchicalExtractionSkill
from .context_budgeting import ContextBudgetingSkill
from .indexed_batch_association import IndexedBatchAssociationSkill
from .transactional_patch import TransactionalPatchSkill
from .agent_orchestration import AgentOrchestrationSkill
from .governance_skills import CheckConfigSkill, ValidateOutputContractsSkill, OntologyValidatorSkill
from .ontology_skills import OntologyMergerSkill, ConsensusResolverSkill, DriftDetectorSkill, RunOntologyRagValidateSkill
from .publisher_skills import CapsulePublisherSkill
from .relation_rewrite_skill import RelationRewriteSkill
from .metadata_enrichment_skill import MetadataEnrichmentSkill
from .cross_file_association_skill import CrossFileAssociationSkill
from .quality_gate_skill import QualityGateSkill

__all__ = [
    "CostAwareSkipSkill",
    "HierarchicalExtractionSkill",
    "ContextBudgetingSkill",
    "IndexedBatchAssociationSkill",
    "TransactionalPatchSkill",
    "AgentOrchestrationSkill",
    "CheckConfigSkill",
    "ValidateOutputContractsSkill",
    "OntologyValidatorSkill",
    "OntologyMergerSkill",
    "ConsensusResolverSkill",
    "DriftDetectorSkill",
    "RunOntologyRagValidateSkill",
    "CapsulePublisherSkill",
    "RelationRewriteSkill",
    "MetadataEnrichmentSkill",
    "CrossFileAssociationSkill",
    "QualityGateSkill",
]
