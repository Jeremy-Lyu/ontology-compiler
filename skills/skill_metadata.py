"""Skill Metadata Models - 技能元数据定义

本模块定义技能的元数据结构，支持：
1. 技能注册与版本管理
2. 技能触发条件定义
3. 输入输出 schema 描述
4. 技能评分与统计
5. 技能血缘追踪
6. 业务域支持 (2026-06-27 新增)

架构目标：
- G3: Agent/Skill 体系
- G4: 统一知识消费层
"""
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class TriggerCondition(BaseModel):
    """技能触发条件"""
    keywords: List[str] = Field(default_factory=list, description="触发关键词")
    file_patterns: List[str] = Field(default_factory=list, description="匹配的文件模式")
    min_confidence: float = Field(0.7, description="最小置信度阈值")
    supported_layers: List[str] = Field(default_factory=list, description="支持的层级")
    domain_tags: List[str] = Field(default_factory=list, description="适用领域标签")


class InputSchema(BaseModel):
    """技能输入 schema"""
    type: str = Field("object", description="输入类型")
    required: List[str] = Field(default_factory=list, description="必需字段")
    properties: Dict[str, Any] = Field(default_factory=dict, description="字段定义")


class OutputSchema(BaseModel):
    """技能输出 schema"""
    type: str = Field("object", description="输出类型")
    properties: Dict[str, Any] = Field(default_factory=dict, description="字段定义")


class SkillScript(BaseModel):
    """技能脚本配置"""
    analysis: Optional[str] = Field(None, description="分析脚本路径")
    validation: Optional[str] = Field(None, description="验证脚本路径")
    main: Optional[str] = Field(None, description="主执行脚本路径")


class SkillDocumentation(BaseModel):
    """技能文档配置"""
    methodology: Optional[str] = Field(None, description="方法论文档路径")
    implementation: Optional[str] = Field(None, description="实现指南路径")
    validation: Optional[str] = Field(None, description="验证清单路径")


class SkillExample(BaseModel):
    """技能使用示例"""
    name: str = Field(..., description="示例名称")
    description: str = Field("", description="示例描述")
    input: Dict[str, Any] = Field(default_factory=dict, description="示例输入")
    expected_output: Optional[Dict[str, Any]] = Field(None, description="期望输出")


class SkillStats(BaseModel):
    """技能统计信息"""
    usage_count: int = Field(0, description="使用次数")
    success_count: int = Field(0, description="成功次数")
    failure_count: int = Field(0, description="失败次数")
    avg_execution_time: float = Field(0.0, description="平均执行时间(秒)")
    avg_confidence: float = Field(0.0, description="平均置信度")
    last_used_at: Optional[str] = Field(None, description="最后使用时间")
    last_success_at: Optional[str] = Field(None, description="最后成功时间")


class SkillRating(BaseModel):
    """技能评分"""
    overall_score: float = Field(0.0, description="综合评分(0-1)")
    accuracy_score: float = Field(0.0, description="准确率评分")
    efficiency_score: float = Field(0.0, description="效率评分")
    reliability_score: float = Field(0.0, description="可靠性评分")
    version: str = Field("1.0", description="评分版本")


class SkillLineage(BaseModel):
    """技能血缘信息"""
    parent_skill_ids: List[str] = Field(default_factory=list, description="父技能ID")
    derived_from: List[str] = Field(default_factory=list, description="派生来源")
    created_by: Optional[str] = Field(None, description="创建者")
    created_at: str = Field("", description="创建时间")
    updated_at: str = Field("", description="更新时间")


class SkillMetadata(BaseModel):
    """技能元数据 - 技能的完整描述信息"""
    skill_id: str = Field(..., description="技能唯一标识")
    name: str = Field(..., description="技能名称")
    description: str = Field("", description="技能描述")
    version: str = Field("1.0.0", description="技能版本")
    category: str = Field("general", description="技能类别")
    tags: List[str] = Field(default_factory=list, description="技能标签")
    
    trigger: TriggerCondition = Field(default_factory=TriggerCondition, description="触发条件")
    input_schema: InputSchema = Field(default_factory=InputSchema, description="输入schema")
    output_schema: OutputSchema = Field(default_factory=OutputSchema, description="输出schema")
    scripts: SkillScript = Field(default_factory=SkillScript, description="脚本配置")
    documentation: SkillDocumentation = Field(default_factory=SkillDocumentation, description="文档配置")
    examples: List[SkillExample] = Field(default_factory=list, description="使用示例")
    
    stats: SkillStats = Field(default_factory=SkillStats, description="统计信息")
    rating: SkillRating = Field(default_factory=SkillRating, description="评分信息")
    lineage: SkillLineage = Field(default_factory=SkillLineage, description="血缘信息")
    
    skill_path: Optional[str] = Field(None, description="技能目录路径")
    business_domain: Optional[Union[str, List[str]]] = Field(None, description="业务域 (字符串或列表)")
    active: bool = Field(True, description="是否激活")
    deprecated: bool = Field(False, description="是否已废弃")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")

    @model_validator(mode='after')
    def set_timestamps(self) -> 'SkillMetadata':
        if not self.lineage.created_at:
            self.lineage.created_at = datetime.now().isoformat()
        self.lineage.updated_at = datetime.now().isoformat()
        return self

    @field_validator('category')
    def validate_category(cls, v):
        valid_categories = [
            'general', 'workflow', 'analysis', 'validation', 'pipeline', 
            'rag', 'transformation', 'reporting', 'ontology', 'ops', 
            'maintenance', 'pipeline_stage', 'stage'
        ]
        if v not in valid_categories:
            return 'general'
        return v

    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.stats.usage_count == 0:
            return 0.0
        return self.stats.success_count / self.stats.usage_count

    @property
    def match_score(self) -> float:
        """匹配评分（综合考虑置信度、成功率、评分）"""
        confidence_weight = 0.3
        success_weight = 0.4
        rating_weight = 0.3
        return (
            self.stats.avg_confidence * confidence_weight +
            self.success_rate * success_weight +
            self.rating.overall_score * rating_weight
        )

    def update_usage(self, success: bool, execution_time: float, confidence: float = 0.0) -> None:
        """更新使用统计"""
        self.stats.usage_count += 1
        self.stats.last_used_at = datetime.now().isoformat()
        
        if success:
            self.stats.success_count += 1
            self.stats.last_success_at = datetime.now().isoformat()
        else:
            self.stats.failure_count += 1
        
        if self.stats.avg_execution_time == 0:
            self.stats.avg_execution_time = execution_time
        else:
            self.stats.avg_execution_time = (
                self.stats.avg_execution_time * (self.stats.usage_count - 1) + execution_time
            ) / self.stats.usage_count
        
        if confidence > 0:
            if self.stats.avg_confidence == 0:
                self.stats.avg_confidence = confidence
            else:
                self.stats.avg_confidence = (
                    self.stats.avg_confidence * (self.stats.usage_count - 1) + confidence
                ) / self.stats.usage_count
        
        self.lineage.updated_at = datetime.now().isoformat()

    def matches_keyword(self, keyword: str) -> bool:
        """检查是否匹配关键词"""
        keyword_lower = keyword.lower()
        return any(
            keyword_lower in kw.lower() 
            for kw in self.trigger.keywords + self.tags + [self.name, self.description]
        )

    def matches_file_pattern(self, file_name: str) -> bool:
        """检查是否匹配文件模式"""
        for pattern in self.trigger.file_patterns:
            import fnmatch
            if fnmatch.fnmatch(file_name, pattern):
                return True
        return False

    def is_compatible_with_layer(self, layer: str) -> bool:
        """检查是否兼容指定层级"""
        if not self.trigger.supported_layers:
            return True
        return layer in self.trigger.supported_layers

    def is_compatible_with_business_domain(self, business_domain: str) -> bool:
        """检查是否属于指定业务域 (2026-06-27 新增)"""
        if not self.business_domain:
            return True
        if isinstance(self.business_domain, list):
            return any(bd.lower() == business_domain.lower() for bd in self.business_domain)
        return self.business_domain.lower() == business_domain.lower()

    def is_compatible_with_domain(self, domain: str) -> bool:
        """检查是否兼容指定领域"""
        if not self.trigger.domain_tags:
            return True
        return domain.lower() in [dt.lower() for dt in self.trigger.domain_tags]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillMetadata":
        """从字典创建"""
        return cls(**data)

    @classmethod
    def from_skill_json(cls, skill_json_path: str) -> "SkillMetadata":
        """从 skill.json 文件创建
        
        自动从目录结构推断 business_domain：
        - skills/mfg/* → mfg
        - skills/ontology/* → ontology
        - skills/pipeline/* → pipeline
        - skills/rag/* → rag
        - skills/system/* → system
        
        兼容 skill.json 的不同字段格式：
        - script_path → scripts.main
        - command → metadata.command
        """
        import json
        import os
        
        with open(skill_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        skill_path = os.path.dirname(os.path.dirname(skill_json_path))
        
        if 'skill_id' not in data:
            data['skill_id'] = data.get('name', os.path.basename(skill_path))
        
        # 自动推断 business_domain 从路径
        if 'business_domain' not in data:
            parts = skill_path.split(os.sep)
            if 'skills' in parts:
                idx = parts.index('skills')
                if len(parts) > idx + 1:
                    domain = parts[idx + 1]
                    valid_domains = ['mfg', 'ontology', 'pipeline', 'rag', 'system']
                    if domain in valid_domains:
                        data['business_domain'] = domain
        
        # 兼容 script_path 字段 → scripts.main
        if 'script_path' in data and 'scripts' not in data:
            data['scripts'] = {'main': data['script_path']}
            data.pop('script_path')
        elif 'script_path' in data and isinstance(data.get('scripts'), dict):
            data['scripts']['main'] = data['script_path']
            data.pop('script_path')
        
        # 将 command 字段存入 metadata
        if 'command' in data:
            if 'metadata' not in data:
                data['metadata'] = {}
            data['metadata']['command'] = data['command']
            data.pop('command')
        
        # 将 args_schema 转换为 input_schema
        if 'args_schema' in data and 'input_schema' not in data:
            args_schema = data.pop('args_schema')
            required_fields = [k for k, v in args_schema.items() if v.get('required')]
            data['input_schema'] = {
                'type': 'object',
                'required': required_fields,
                'properties': args_schema
            }
        
        return cls(
            skill_path=skill_path,
            **data
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.model_dump(exclude={'skill_path'}, by_alias=True)

    def save_to_json(self, output_path: str) -> None:
        """保存为 JSON 文件"""
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
