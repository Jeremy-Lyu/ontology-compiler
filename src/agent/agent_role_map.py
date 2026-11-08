"""Agent Role Mapping - v2 统一 Agent 角色矩阵

定义每个 Agent 的职责边界、可调用技能、输出 schema 和兜底策略。

核心原则：
1. Agent 负责调度（理解、路由、兜底、协调）
2. Skill 负责能力（具体功能实现）
3. Schema 负责结构（输入输出约束）
4. Config 负责策略（规则、阈值）
5. Canonical contract 优先于别名字段
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Set


class AgentCapability(str, Enum):
    """Agent 能力分类"""
    PLANNING = "planning"           # 任务规划
    EXECUTION = "execution"         # 执行操作
    VALIDATION = "validation"       # 结果校验
    SUMMARY = "summary"             # 结果总结
    FALLBACK = "fallback"           # 失败兜底
    REVIEW = "review"               # 人工审核
    AUDIT = "audit"                 # 审计检查
    REPORTING = "reporting"         # 报告生成
    ROUTING = "routing"             # 任务路由
    ANALYSIS = "analysis"           # 数据分析
    GOVERNANCE = "governance"       # 治理检查


@dataclass
class AgentSkillMapping:
    """Agent 可调用技能映射"""
    skill_id: str
    description: str
    input_schema: str = ""
    output_schema: str = ""


@dataclass
class AgentRoleConfig:
    """单个 Agent 角色配置"""
    agent_id: str
    name: str
    role: str
    description: str
    
    capabilities: List[AgentCapability] = field(default_factory=list)
    forbidden_capabilities: List[AgentCapability] = field(default_factory=list)
    
    allowed_skills: List[AgentSkillMapping] = field(default_factory=list)
    forbidden_skills: List[str] = field(default_factory=list)
    
    output_schema: str = ""
    input_schema: str = ""
    
    fallback_policy: str = "retry"
    max_retries: int = 3
    timeout_seconds: int = 300
    
    depends_on: List[str] = field(default_factory=list)
    priority: int = 10


class AgentRoleMap:
    """Agent 角色矩阵 - 统一管理所有 Agent 的职责边界"""
    
    def __init__(self):
        self._roles: Dict[str, AgentRoleConfig] = {}
        self._load_default_roles()
    
    def _load_default_roles(self):
        """加载默认 Agent 角色配置"""
        self._roles = {
            "document_router": AgentRoleConfig(
                agent_id="document_router",
                name="Document Router",
                role="PLANNER",
                description="文档类型识别与路由选择",
                capabilities=[AgentCapability.ROUTING, AgentCapability.PLANNING],
                forbidden_capabilities=[AgentCapability.EXECUTION, AgentCapability.VALIDATION],
                allowed_skills=[
                    AgentSkillMapping(skill_id="document_router", description="文档分类与路由"),
                ],
                output_schema="document_type, business_domain, route, routing_reason",
                fallback_policy="default_route",
                priority=1,
            ),
            "entity_extractor": AgentRoleConfig(
                agent_id="entity_extractor",
                name="Entity Extractor",
                role="EXTRACTOR",
                description="实体识别与抽取",
                capabilities=[AgentCapability.EXECUTION],
                forbidden_capabilities=[AgentCapability.PLANNING, AgentCapability.VALIDATION],
                allowed_skills=[
                    AgentSkillMapping(skill_id="true_unified_extractor", description="统一实体抽取"),
                    AgentSkillMapping(skill_id="relation_extraction", description="关系抽取"),
                    AgentSkillMapping(skill_id="entity_linking", description="实体链接"),
                ],
                output_schema="nodes[], instances[], relationships[], evidence[], llm_usage{}",
                input_schema="source_path, domain, run_id, use_context, fast_mode",
                fallback_policy="retry",
                max_retries=3,
                depends_on=["document_router"],
                priority=2,
            ),
            "semantic_organizer": AgentRoleConfig(
                agent_id="semantic_organizer",
                name="Semantic Organizer",
                role="ANALYST",
                description="语义组织与业务结构化",
                capabilities=[AgentCapability.EXECUTION, AgentCapability.ANALYSIS],
                forbidden_capabilities=[AgentCapability.VALIDATION, AgentCapability.REVIEW],
                allowed_skills=[
                    AgentSkillMapping(skill_id="process_skeleton_extraction", description="流程骨架提取"),
                    AgentSkillMapping(skill_id="business_rule_extraction", description="业务规则提取"),
                    AgentSkillMapping(skill_id="scenario_case_extraction", description="场景案例提取"),
                    AgentSkillMapping(skill_id="concept_role_mapping", description="概念角色映射"),
                ],
                output_schema="processes[], rules[], scenarios[], mappings[], normalized_entities[]",
                input_schema="nodes[], instances[], relationships[]",
                fallback_policy="skip",
                depends_on=["entity_extractor"],
                priority=3,
            ),
            "quality_guard": AgentRoleConfig(
                agent_id="quality_guard",
                name="Quality Guard",
                role="VERIFIER",
                description="质量门禁与结果校验",
                capabilities=[AgentCapability.VALIDATION, AgentCapability.AUDIT],
                forbidden_capabilities=[AgentCapability.EXECUTION, AgentCapability.PLANNING],
                allowed_skills=[
                    AgentSkillMapping(skill_id="quality_gate", description="质量门禁"),
                    AgentSkillMapping(skill_id="consensus_resolution", description="一致性解决"),
                    AgentSkillMapping(skill_id="schema_conformance", description="Schema 一致性验证"),
                    AgentSkillMapping(skill_id="relation_validator", description="关系规则验证"),
                ],
                output_schema="valid, status, violations[], semantic_validation, review_required",
                input_schema="output_data, contract_schema, validation_options",
                fallback_policy="fail",
                priority=4,
            ),
            "graph_merger": AgentRoleConfig(
                agent_id="graph_merger",
                name="Graph Merger",
                role="MERGER",
                description="图合并与去重",
                capabilities=[AgentCapability.EXECUTION],
                forbidden_capabilities=[AgentCapability.PLANNING, AgentCapability.REVIEW],
                allowed_skills=[
                    AgentSkillMapping(skill_id="graph_merge_and_dedup", description="图合并去重"),
                    AgentSkillMapping(skill_id="ontology_projection_pipeline", description="本体投影"),
                ],
                output_schema="merged_nodes[], merged_edges[], deduplicated_count",
                fallback_policy="skip",
                depends_on=["quality_guard"],
                priority=5,
            ),
            "human_reviewer": AgentRoleConfig(
                agent_id="human_reviewer",
                name="Human Reviewer",
                role="REVIEWER",
                description="人工审核队列管理",
                capabilities=[AgentCapability.REVIEW, AgentCapability.VALIDATION],
                forbidden_capabilities=[AgentCapability.EXECUTION, AgentCapability.PLANNING],
                allowed_skills=[
                    AgentSkillMapping(skill_id="human_review_queue", description="人工审核队列"),
                ],
                output_schema="approved, rejected, comments[], review_timestamp",
                fallback_policy="escalate",
                priority=6,
            ),
            "publisher": AgentRoleConfig(
                agent_id="publisher",
                name="Publisher",
                role="EXPORTER",
                description="资产发布与回滚",
                capabilities=[AgentCapability.EXECUTION],
                forbidden_capabilities=[AgentCapability.PLANNING, AgentCapability.VALIDATION],
                allowed_skills=[
                    AgentSkillMapping(skill_id="publish_or_rollback", description="发布或回滚"),
                ],
                output_schema="published_artifacts[], rollback_package, publish_timestamp",
                fallback_policy="rollback",
                depends_on=["human_reviewer", "graph_merger"],
                priority=7,
            ),
            "rag_answerer": AgentRoleConfig(
                agent_id="rag_answerer",
                name="RAG Answerer",
                role="RAG_ANSWERER",
                description="RAG 问答服务",
                capabilities=[AgentCapability.EXECUTION, AgentCapability.SUMMARY],
                forbidden_capabilities=[AgentCapability.PLANNING, AgentCapability.VALIDATION],
                allowed_skills=[],
                output_schema="answer, confidence, evidence_chunks[], sources[]",
                fallback_policy="no_answer",
                priority=10,
            ),
            "orchestrator": AgentRoleConfig(
                agent_id="orchestrator",
                name="Orchestrator",
                role="PLANNER",
                description="全局任务编排与协调",
                capabilities=[AgentCapability.PLANNING, AgentCapability.ROUTING, AgentCapability.FALLBACK, AgentCapability.SUMMARY],
                forbidden_capabilities=[AgentCapability.EXECUTION, AgentCapability.VALIDATION],
                allowed_skills=[],
                output_schema="task_chain[], execution_plan[], status_summary",
                fallback_policy="manual",
                priority=0,
            ),
            "entity_classifier": AgentRoleConfig(
                agent_id="entity_classifier",
                name="Entity Classifier",
                role="EXTRACTOR",
                description="基于规则的实体分类",
                capabilities=[AgentCapability.EXECUTION, AgentCapability.VALIDATION],
                forbidden_capabilities=[AgentCapability.PLANNING],
                allowed_skills=[
                    AgentSkillMapping(skill_id="entity_recognition", description="实体识别"),
                ],
                output_schema="classified_entities[], confidence_scores[], layer_assignments[]",
                fallback_policy="llm_fallback",
                depends_on=["document_router"],
                priority=2,
            ),
            "schema_conformance": AgentRoleConfig(
                agent_id="schema_conformance",
                name="Schema Conformance Agent",
                role="VERIFIER",
                description="Schema 一致性验证",
                capabilities=[AgentCapability.VALIDATION],
                forbidden_capabilities=[AgentCapability.EXECUTION, AgentCapability.PLANNING],
                allowed_skills=[],
                output_schema="valid, status, schema_validation, semantic_validation, violations[], summary",
                input_schema="output_data, contract_schema, validation_options",
                fallback_policy="fail",
                priority=4,
            ),
            "domain_detector": AgentRoleConfig(
                agent_id="domain_detector",
                name="Domain Detection Agent",
                role="PLANNER",
                description="领域自动检测",
                capabilities=[AgentCapability.ROUTING, AgentCapability.PLANNING],
                forbidden_capabilities=[AgentCapability.EXECUTION],
                allowed_skills=[
                    AgentSkillMapping(skill_id="document_router", description="文档路由"),
                ],
                output_schema="domain, confidence, suggested_route",
                fallback_policy="default_domain",
                priority=1,
            ),
            "relation_validator": AgentRoleConfig(
                agent_id="relation_validator",
                name="Relation Validator",
                role="VERIFIER",
                description="关系规则验证",
                capabilities=[AgentCapability.VALIDATION],
                forbidden_capabilities=[AgentCapability.EXECUTION],
                allowed_skills=[
                    AgentSkillMapping(skill_id="relation_extraction", description="关系抽取"),
                    AgentSkillMapping(skill_id="validate-output-contracts", description="输出契约验证"),
                ],
                output_schema="valid_relations[], invalid_relations[], violations[], semantic_validation",
                input_schema="relationships[], entity_ids[], allowed_relation_types",
                fallback_policy="filter",
                priority=4,
            ),
            "conflict_governor": AgentRoleConfig(
                agent_id="conflict_governor",
                name="Conflict Governor",
                role="AUDITOR",
                description="冲突治理扫描",
                capabilities=[AgentCapability.GOVERNANCE, AgentCapability.ANALYSIS],
                forbidden_capabilities=[AgentCapability.EXECUTION],
                allowed_skills=[],
                output_schema="conflicts[], risk_level, recommendations[]",
                fallback_policy="report",
                priority=8,
            ),
            "data_continuity": AgentRoleConfig(
                agent_id="data_continuity",
                name="Data Continuity Agent",
                role="AUDITOR",
                description="数据连续性扫描",
                capabilities=[AgentCapability.GOVERNANCE, AgentCapability.AUDIT],
                forbidden_capabilities=[AgentCapability.EXECUTION],
                allowed_skills=[],
                output_schema="fingerprint_status, manifest_status, processed_index_status",
                fallback_policy="report",
                priority=8,
            ),
            "contract_aligner": AgentRoleConfig(
                agent_id="contract_aligner",
                name="Contract Aligner",
                role="VERIFIER",
                description="契约对齐扫描",
                capabilities=[AgentCapability.VALIDATION, AgentCapability.GOVERNANCE],
                forbidden_capabilities=[AgentCapability.EXECUTION],
                allowed_skills=[],
                output_schema="contract_status, violations[], alignment_score, semantic_validation",
                input_schema="output_data, contract_schema, validation_options",
                fallback_policy="report",
                priority=8,
            ),
        }
    
    def get_role(self, agent_id: str) -> Optional[AgentRoleConfig]:
        """获取 Agent 角色配置"""
        return self._roles.get(agent_id)
    
    def list_agents(self) -> List[str]:
        """列出所有 Agent ID"""
        return list(self._roles.keys())
    
    def get_agents_by_role(self, role: str) -> List[AgentRoleConfig]:
        """按角色类型获取 Agent 列表"""
        return [config for config in self._roles.values() if config.role == role]
    
    def can_perform(self, agent_id: str, capability: AgentCapability) -> bool:
        """检查 Agent 是否具有指定能力"""
        role = self._roles.get(agent_id)
        if not role:
            return False
        
        if capability in role.forbidden_capabilities:
            return False
        
        if role.capabilities and capability not in role.capabilities:
            return False
        
        return True
    
    def get_allowed_skills(self, agent_id: str) -> List[str]:
        """获取 Agent 可调用的技能列表"""
        role = self._roles.get(agent_id)
        if not role:
            return []
        return [s.skill_id for s in role.allowed_skills]
    
    def is_skill_allowed(self, agent_id: str, skill_id: str) -> bool:
        """检查 Agent 是否允许调用指定技能"""
        role = self._roles.get(agent_id)
        if not role:
            return False
        
        if skill_id in role.forbidden_skills:
            return False
        
        if not role.allowed_skills:
            return True
        
        return any(s.skill_id == skill_id for s in role.allowed_skills)
    
    def get_dependencies(self, agent_id: str) -> List[str]:
        """获取 Agent 的前置依赖"""
        role = self._roles.get(agent_id)
        if not role:
            return []
        return role.depends_on
    
    def to_dict(self) -> Dict[str, Dict]:
        """转换为字典格式，便于序列化"""
        result = {}
        for agent_id, config in self._roles.items():
            result[agent_id] = {
                "agent_id": config.agent_id,
                "name": config.name,
                "role": config.role,
                "description": config.description,
                "capabilities": [c.value for c in config.capabilities],
                "forbidden_capabilities": [c.value for c in config.forbidden_capabilities],
                "allowed_skills": [s.skill_id for s in config.allowed_skills],
                "forbidden_skills": config.forbidden_skills,
                "output_schema": config.output_schema,
                "input_schema": config.input_schema,
                "fallback_policy": config.fallback_policy,
                "max_retries": config.max_retries,
                "timeout_seconds": config.timeout_seconds,
                "depends_on": config.depends_on,
                "priority": config.priority,
            }
        return result


_agent_role_map = None


def get_agent_role_map() -> AgentRoleMap:
    """获取全局 Agent 角色矩阵实例"""
    global _agent_role_map
    if _agent_role_map is None:
        _agent_role_map = AgentRoleMap()
    return _agent_role_map