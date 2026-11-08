"""Skill Memory - 技能记忆与失败样本记录

本模块提供技能执行过程中的记忆管理功能，支持：
1. 成功执行经验记录
2. 失败样本记录与分析
3. Prompt 记忆优化
4. 技能执行上下文追溯
5. 自动反馈闭环

架构目标：
- G3: Agent/Skill 体系
- G5: 增量与版本管理
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class ExecutionTrace(BaseModel):
    """技能执行轨迹"""
    trace_id: str = Field(..., description="轨迹唯一标识")
    skill_id: str = Field(..., description="技能ID")
    run_id: str = Field("", description="运行ID")
    session_id: str = Field("", description="会话ID")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="输入数据")
    output_data: Dict[str, Any] = Field(default_factory=dict, description="输出数据")
    execution_time: float = Field(0.0, description="执行时间(秒)")
    confidence: float = Field(0.0, description="置信度")
    success: bool = Field(True, description="是否成功")
    error_message: str = Field("", description="错误信息")
    error_type: str = Field("", description="错误类型")
    created_at: str = Field("", description="创建时间")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")

    @model_validator(mode='after')
    def set_created_at(self) -> 'ExecutionTrace':
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        return self


class FailureSample(BaseModel):
    """失败样本"""
    sample_id: str = Field(..., description="样本唯一标识")
    skill_id: str = Field(..., description="技能ID")
    trace_id: str = Field("", description="关联轨迹ID")
    run_id: str = Field("", description="运行ID")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="导致失败的输入")
    expected_output: Optional[Dict[str, Any]] = Field(None, description="期望输出")
    actual_output: Dict[str, Any] = Field(default_factory=dict, description="实际输出")
    error_message: str = Field("", description="错误信息")
    error_type: str = Field("", description="错误类型")
    failure_category: str = Field("unknown", description="失败类别")
    root_cause: str = Field("", description="根本原因分析")
    suggested_fix: str = Field("", description="建议修复方案")
    retry_count: int = Field(0, description="重试次数")
    resolved: bool = Field(False, description="是否已解决")
    created_at: str = Field("", description="创建时间")
    resolved_at: Optional[str] = Field(None, description="解决时间")

    @model_validator(mode='after')
    def set_created_at(self) -> 'FailureSample':
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        return self


class PromptMemory(BaseModel):
    """Prompt 记忆"""
    memory_id: str = Field(..., description="记忆唯一标识")
    skill_id: str = Field(..., description="技能ID")
    prompt_template: str = Field("", description="Prompt 模板")
    prompt_variables: Dict[str, Any] = Field(default_factory=dict, description="Prompt 变量")
    successful_count: int = Field(0, description="成功使用次数")
    failed_count: int = Field(0, description="失败使用次数")
    avg_confidence: float = Field(0.0, description="平均置信度")
    last_used_at: Optional[str] = Field(None, description="最后使用时间")
    created_at: str = Field("", description="创建时间")
    updated_at: str = Field("", description="更新时间")

    @model_validator(mode='after')
    def set_timestamps(self) -> 'PromptMemory':
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        return self

    @property
    def success_rate(self) -> float:
        """成功率"""
        total = self.successful_count + self.failed_count
        if total == 0:
            return 0.0
        return self.successful_count / total


class DomainMemory(BaseModel):
    """领域记忆"""
    memory_id: str = Field(..., description="记忆唯一标识")
    domain: str = Field(..., description="领域名称")
    key_concepts: List[str] = Field(default_factory=list, description="关键概念")
    patterns: List[str] = Field(default_factory=list, description="常见模式")
    constraints: List[str] = Field(default_factory=list, description="约束条件")
    best_practices: List[str] = Field(default_factory=list, description="最佳实践")
    pitfalls: List[str] = Field(default_factory=list, description="常见陷阱")
    created_at: str = Field("", description="创建时间")
    updated_at: str = Field("", description="更新时间")

    @model_validator(mode='after')
    def set_timestamps(self) -> 'DomainMemory':
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        return self


class SkillMemory:
    """技能记忆管理器"""

    def __init__(self) -> None:
        self._traces: Dict[str, ExecutionTrace] = {}
        self._failures: Dict[str, FailureSample] = {}
        self._prompt_memories: Dict[str, PromptMemory] = {}
        self._domain_memories: Dict[str, DomainMemory] = {}
        self._skill_traces: Dict[str, List[str]] = {}
        self._skill_failures: Dict[str, List[str]] = {}

    def record_trace(self, trace: ExecutionTrace) -> None:
        """记录执行轨迹"""
        self._traces[trace.trace_id] = trace
        
        if trace.skill_id not in self._skill_traces:
            self._skill_traces[trace.skill_id] = []
        self._skill_traces[trace.skill_id].append(trace.trace_id)

    def record_failure(self, failure: FailureSample) -> None:
        """记录失败样本"""
        self._failures[failure.sample_id] = failure
        
        if failure.skill_id not in self._skill_failures:
            self._skill_failures[failure.skill_id] = []
        self._skill_failures[failure.skill_id].append(failure.sample_id)

    def record_prompt_usage(self, memory_id: str, success: bool, confidence: float = 0.0) -> None:
        """记录 Prompt 使用情况"""
        memory = self._prompt_memories.get(memory_id)
        if not memory:
            return

        if success:
            memory.successful_count += 1
        else:
            memory.failed_count += 1
        
        if confidence > 0:
            if memory.avg_confidence == 0:
                memory.avg_confidence = confidence
            else:
                total = memory.successful_count + memory.failed_count
                memory.avg_confidence = (
                    memory.avg_confidence * (total - 1) + confidence
                ) / total
        
        memory.last_used_at = datetime.now().isoformat()
        memory.updated_at = datetime.now().isoformat()

    def add_prompt_memory(self, memory: PromptMemory) -> None:
        """添加 Prompt 记忆"""
        self._prompt_memories[memory.memory_id] = memory

    def add_domain_memory(self, memory: DomainMemory) -> None:
        """添加领域记忆"""
        self._domain_memories[memory.domain] = memory

    def get_trace(self, trace_id: str) -> Optional[ExecutionTrace]:
        """获取执行轨迹"""
        return self._traces.get(trace_id)

    def get_traces_by_skill(self, skill_id: str, limit: int = 20) -> List[ExecutionTrace]:
        """获取技能的执行轨迹"""
        trace_ids = self._skill_traces.get(skill_id, [])[-limit:]
        return [self._traces[t_id] for t_id in trace_ids if t_id in self._traces]

    def get_failure(self, sample_id: str) -> Optional[FailureSample]:
        """获取失败样本"""
        return self._failures.get(sample_id)

    def get_failures_by_skill(self, skill_id: str, limit: int = 20) -> List[FailureSample]:
        """获取技能的失败样本"""
        failure_ids = self._skill_failures.get(skill_id, [])[-limit:]
        return [self._failures[f_id] for f_id in failure_ids if f_id in self._failures]

    def get_prompt_memory(self, memory_id: str) -> Optional[PromptMemory]:
        """获取 Prompt 记忆"""
        return self._prompt_memories.get(memory_id)

    def get_domain_memory(self, domain: str) -> Optional[DomainMemory]:
        """获取领域记忆"""
        return self._domain_memories.get(domain)

    def get_skill_failure_stats(self, skill_id: str) -> Dict[str, Any]:
        """获取技能失败统计"""
        failures = self.get_failures_by_skill(skill_id)
        if not failures:
            return {
                "skill_id": skill_id,
                "total_failures": 0,
                "failure_categories": {},
                "error_types": {},
                "resolved_count": 0
            }

        categories: Dict[str, int] = {}
        error_types: Dict[str, int] = {}
        resolved_count = 0

        for failure in failures:
            categories[failure.failure_category] = categories.get(failure.failure_category, 0) + 1
            error_types[failure.error_type] = error_types.get(failure.error_type, 0) + 1
            if failure.resolved:
                resolved_count += 1

        return {
            "skill_id": skill_id,
            "total_failures": len(failures),
            "failure_categories": categories,
            "error_types": error_types,
            "resolved_count": resolved_count,
            "resolution_rate": resolved_count / len(failures)
        }

    def resolve_failure(self, sample_id: str, root_cause: str = "", suggested_fix: str = "") -> None:
        """标记失败样本为已解决"""
        failure = self._failures.get(sample_id)
        if failure:
            failure.resolved = True
            failure.resolved_at = datetime.now().isoformat()
            if root_cause:
                failure.root_cause = root_cause
            if suggested_fix:
                failure.suggested_fix = suggested_fix

    def generate_failure_summary(self, skill_id: str) -> str:
        """生成失败摘要"""
        stats = self.get_skill_failure_stats(skill_id)
        if stats["total_failures"] == 0:
            return f"技能 {skill_id} 暂无失败记录"

        summary = f"技能 {skill_id} 失败分析:\n"
        summary += f"- 总失败次数: {stats['total_failures']}\n"
        summary += f"- 已解决: {stats['resolved_count']} ({stats['resolution_rate']:.1%})\n"
        
        if stats["failure_categories"]:
            summary += "- 失败类别分布:\n"
            for cat, count in sorted(stats["failure_categories"].items(), key=lambda x: x[1], reverse=True):
                summary += f"  * {cat}: {count}次\n"
        
        if stats["error_types"]:
            summary += "- 错误类型分布:\n"
            for err_type, count in sorted(stats["error_types"].items(), key=lambda x: x[1], reverse=True):
                summary += f"  * {err_type}: {count}次\n"

        return summary

    def suggest_improvements(self, skill_id: str) -> List[str]:
        """基于失败样本提供改进建议"""
        failures = self.get_failures_by_skill(skill_id)
        if not failures:
            return []

        suggestions = []
        seen_suggestions = set()

        for failure in failures:
            if failure.suggested_fix and failure.suggested_fix not in seen_suggestions:
                suggestions.append(failure.suggested_fix)
                seen_suggestions.add(failure.suggested_fix)
            
            if failure.root_cause:
                if "输入格式错误" in failure.root_cause and "优化输入验证" not in seen_suggestions:
                    suggestions.append("优化输入验证逻辑")
                    seen_suggestions.add("优化输入验证逻辑")
                if "模型输出解析失败" in failure.root_cause and "增强输出解析" not in seen_suggestions:
                    suggestions.append("增强输出解析和错误恢复")
                    seen_suggestions.add("增强输出解析和错误恢复")
                if "超时" in failure.root_cause and "优化执行效率" not in seen_suggestions:
                    suggestions.append("优化执行效率或增加超时重试")
                    seen_suggestions.add("优化执行效率或增加超时重试")

        return suggestions

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "traces": {k: v.model_dump() for k, v in self._traces.items()},
            "failures": {k: v.model_dump() for k, v in self._failures.items()},
            "prompt_memories": {k: v.model_dump() for k, v in self._prompt_memories.items()},
            "domain_memories": {k: v.model_dump() for k, v in self._domain_memories.items()},
            "_stats": {
                "total_traces": len(self._traces),
                "total_failures": len(self._failures),
                "total_prompt_memories": len(self._prompt_memories),
                "total_domain_memories": len(self._domain_memories),
                "skills_with_traces": len(self._skill_traces),
                "skills_with_failures": len(self._skill_failures)
            }
        }

    def save_memory(self, output_path: str) -> None:
        """保存记忆到文件"""
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


skill_memory = SkillMemory()
