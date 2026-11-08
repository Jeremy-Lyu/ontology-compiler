"""SKILLS Framework — 显式技能抽象与编排

R38-QA: 将隐式技能封装为显式 Skill 抽象类，支持配置动态组合技能链。

Usage:
    from ontology_compiler.skills import SkillRegistry, SkillChain
    from ontology_compiler.skills.builtin import CostAwareSkipSkill
    
    # Register skills
    registry = SkillRegistry()
    registry.register(CostAwareSkipSkill())
    
    # Execute skill chain
    chain = SkillChain(["cost_aware_skip", "hierarchical_extraction"])
    result = await chain.execute(context)
"""

from .base import Skill, SkillContext, SkillResult, SkillPriority
from .registry import SkillRegistry, get_skill_registry, reset_skill_registry
from .chain import SkillChain

from .builtin.governance_skills import (
    CheckConfigSkill,
    ValidateOutputContractsSkill,
    OntologyValidatorSkill,
)
from .builtin.ontology_skills import (
    OntologyMergerSkill,
    ConsensusResolverSkill,
    DriftDetectorSkill,
    RunOntologyRagValidateSkill,
)
from .builtin.publisher_skills import (
    CapsulePublisherSkill,
)

registry = get_skill_registry()
registry.register(CheckConfigSkill(), "validation")
registry.register(ValidateOutputContractsSkill(), "validation")
registry.register(OntologyValidatorSkill(), "validation")
registry.register(OntologyMergerSkill(), "post_processing")
registry.register(ConsensusResolverSkill(), "post_processing")
registry.register(DriftDetectorSkill(), "optimization")
registry.register(RunOntologyRagValidateSkill(), "validation")
registry.register(CapsulePublisherSkill(), "post_processing")

__all__ = [
    "Skill",
    "SkillContext",
    "SkillResult",
    "SkillPriority",
    "SkillRegistry",
    "get_skill_registry",
    "reset_skill_registry",
    "SkillChain",
    "CheckConfigSkill",
    "ValidateOutputContractsSkill",
    "OntologyValidatorSkill",
    "OntologyMergerSkill",
    "ConsensusResolverSkill",
    "DriftDetectorSkill",
    "RunOntologyRagValidateSkill",
    "CapsulePublisherSkill",
]
