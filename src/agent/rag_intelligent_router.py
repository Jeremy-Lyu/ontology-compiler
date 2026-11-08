#!/usr/bin/env python3
"""Intelligent RAG Router - 自动判断查询复杂度并路由到适当分析路径"""

import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from src.agent.multi_agent.orchestrator import Orchestrator
from src.agent.multi_agent.llm_scenario_classifier import classify_scenario


class QueryComplexity(Enum):
    """查询复杂度等级"""
    SIMPLE = "simple"           # 单实体定义查询
    MODERATE = "moderate"       # 双实体关系查询
    COMPLEX = "complex"         # 多实体+流程+问题诊断


@dataclass
class ComplexityAssessment:
    """复杂度评估结果"""
    level: QueryComplexity
    score: float  # 0-1
    reasoning: str
    suggested_agents: list[str]
    required_skills: list[str]


class RAGIntelligentRouter:
    """智能 RAG 路由器"""
    
    # 复杂查询关键词模式（基于 process.md 分析经验）
    COMPLEX_PATTERNS = [
        # 系统协作类
        r"如何.*联动|如何.*协作|系统.*分工|数据.*流转|集成.*接口|如何使用.*应对",
        # 问题诊断类
        r"问题|痛点|难点|瓶颈|延迟|错误|偏差|缺口|断层|真空|影响|冻结",
        # 流程优化类
        r"优化|改进|提效|闭环|建议|方案|模型|重构|分析.*流程|数据模型",
        # 多场景类
        r"多场景|场景.*支持|场景.*切换|灵活.*适配|支持.*场景",
        # 数据关联类
        r"JOIN|关联|匹配|对应|映射|对齐|一致|结合|关联.*分析",
        # 时间序列类
        r"时序|先后|前置期|lead.*time|周期|窗口|生效|变更",
        # 业务动作类
        r"排产变化|物料预测|物料管理|calloff|Call-off|交货|供应商",
    ]

    # 系统/实体名称（用于计数）
    SYSTEM_ENTITIES = [
        "OHF", "ARO", "TOPAS", "PLUS", "MPP", "CMP", "Partcache", "TBS130",
        "HERCULES", "BIS", "CIP", "C-Note", "CNote", "MES", "ERP", "VVS", "SI Plus",
        "TCP700", "TCP712", "TBS330", "TBS310", "MBOM", "EBOM", "PML", "BOM",
        "Call-off", "Calloff", "MRP", "排产", "物料", "供应商", "装配线",
        "排产计划", "物料需求", "预测", "Forecast", "Call-off"
    ]
    
    def __init__(self, orchestrator: Optional[Orchestrator] = None):
        self.orchestrator = orchestrator or Orchestrator(use_dag=True)
    
    def assess_complexity(self, query: str) -> ComplexityAssessment:
        """评估查询复杂度"""
        score = 0.0
        reasons = []
        suggested_agents = []
        required_skills = []
        
        # 1. 检测复杂模式
        complex_matches = sum(1 for p in self.COMPLEX_PATTERNS if re.search(p, query, re.I))
        if complex_matches >= 2:
            score += 0.4
            reasons.append(f"匹配 {complex_matches} 个复杂查询模式")
            suggested_agents.extend(["system_analyst", "process_optimizer"])
            required_skills.extend(["system-researcher", "consensus-resolver"])
        elif complex_matches == 1:
            score += 0.2
            reasons.append("匹配 1 个复杂查询模式")
        
        # 2. 实体计数
        entities_found = [e for e in self.SYSTEM_ENTITIES if e.lower() in query.lower()]
        entity_count = len(set(entities_found))
        if entity_count >= 4:
            score += 0.3
            reasons.append(f"涉及 {entity_count} 个系统/实体")
            suggested_agents.append("data_integrator")
            required_skills.append("dag-builder")
        elif entity_count >= 2:
            score += 0.15
            reasons.append(f"涉及 {entity_count} 个系统/实体")
        
        # 3. 查询长度（间接反映复杂度）
        char_count = len(query)
        if char_count > 100:
            score += 0.15
            reasons.append("长查询，可能包含多维度需求")
        
        # 4. 问题数量
        question_marks = query.count("?") + query.count("？")
        if question_marks >= 2:
            score += 0.15
            reasons.append(f"包含 {question_marks} 个问题点")
        
        # 检查是否为 MBOM 相关查询（涉及数据分析的必须走复杂流程）
        is_mbom = self._is_mbom_related(query)
        
        # 确定复杂度等级
        if score >= 0.6 or is_mbom:
            # MBOM 查询强制提升到 COMPLEX，确保使用 Multi-Agent + bom_analyzer
            level = QueryComplexity.COMPLEX
            if is_mbom and score < 0.6:
                reasons.append("MBOM/BOM 数据驱动查询，自动启用复杂分析流程")
                suggested_agents.extend(["data_integrator", "bom_analyzer"])
                required_skills.extend(["bom-analyzer", "data-analysis"])
            suggested_agents = list(set(suggested_agents + ["scenario_classifier", "multi_round_planner", "data_validator", "synthesizer"]))
            required_skills = list(set(required_skills + ["dag-builder", "drift-detector", "consensus-resolver", "scene-analysis"]))
        elif score >= 0.3:
            level = QueryComplexity.MODERATE
            suggested_agents = list(set(suggested_agents + ["retriever", "validator"]))
            required_skills = list(set(required_skills + ["dag-builder"]))
        else:
            level = QueryComplexity.SIMPLE
            suggested_agents = ["retriever", "answerer"]
            required_skills = []
        
        return ComplexityAssessment(
            level=level,
            score=min(score, 1.0),
            reasoning="; ".join(reasons) if reasons else "标准定义查询",
            suggested_agents=suggested_agents,
            required_skills=required_skills
        )
    
    def route(self, query: str, use_skills: bool = True) -> dict:
        """路由查询并执行分析"""
        assessment = self.assess_complexity(query)
        
        result = {
            "query": query,
            "complexity": assessment.level.value,
            "score": assessment.score,
            "reasoning": assessment.reasoning,
            "path": None,
            "result": None
        }
        
        if assessment.level == QueryComplexity.SIMPLE:
            # 简单查询：直接 RAG
            result["path"] = "simple_rag"
            result["result"] = self._simple_rag(query)
            
        elif assessment.level == QueryComplexity.MODERATE:
            # 中等复杂度：RAG + 单轮验证
            result["path"] = "moderate_rag_with_validation"
            result["result"] = self._moderate_analysis(query, assessment)
            
        else:
            # 复杂查询：Multi-Agent + Skills 全流程
            result["path"] = "complex_multi_agent"
            result["agents"] = assessment.suggested_agents
            result["skills"] = assessment.required_skills if use_skills else []
            result["result"] = self._complex_analysis(query, assessment)
        
        return result
    
    def _simple_rag(self, query: str) -> dict:
        """简单查询：标准 RAG"""
        # 调用现有 EnhancedRAG
        from ontology_rag.cli.enhanced_rag_cli import EnhancedRAG
        from ontology_rag.config import RAGConfig
        
        rag = EnhancedRAG(RAGConfig())
        results = rag.retrieve(query, mode="hybrid", topk=5)
        answer = rag.answer(query, results)
        
        return {
            "answer": answer,
            "sources": len(results),
            "method": "direct_rag"
        }
    
    def _moderate_analysis(self, query: str, assessment: ComplexityAssessment) -> dict:
        """中等复杂度：RAG + 场景分类 + 单轮验证"""
        # 场景分类
        scenario = classify_scenario(query)
        
        # 使用 Orchestrator 单轮分析 (max_rounds通过构造函数设置)
        report = self.orchestrator.run(query, scenario=scenario.scenario)
        
        return {
            "answer": report.narrative,
            "scenario": report.scenario,
            "confidence": report.final_confidence,
            "rounds": len(report.rounds),
            "method": "orchestrator_2rounds"
        }
    
    def _complex_analysis(self, query: str, assessment: ComplexityAssessment) -> dict:
        """复杂查询：Multi-Agent + Skills 全流程"""
        # 拆解子任务（模拟 process.md 的 3 层分析框架）
        sub_tasks = self._decompose_query(query)
        
        # 使用 Orchestrator 多轮分析（DAG 并行，max_rounds通过构造函数设置）
        report = self.orchestrator.run(
            query,
            scenario="complex_bom_analysis"
        )
        
        return {
            "answer": report.narrative,
            "scenario": report.scenario,
            "confidence": report.final_confidence,
            "rounds": len(report.rounds),
            "sub_tasks": sub_tasks,
            "method": "multi_agent_full_pipeline"
        }
    
    def _is_mbom_related(self, query: str) -> bool:
        """检测查询是否与 MBOM 相关"""
        mbom_keywords = [
            "mbom", "ebom", "bom", "tbs130", "partcache", "物料清单",
            "工程变更", "c-note", "cnote", "排产", "call-off", "calloff",
            "物料预测", "物料管理", "mpp", "plus", "ohf", "hercules",
            "装配", "零件", "物料需求", "mrp"
        ]
        query_lower = query.lower()
        return any(kw in query_lower for kw in mbom_keywords)
    
    def _call_bom_analyzer(self, query: str) -> dict:
        """调用 bom_analyzer 进行 MBOM 数据分析"""
        try:
            # 导入 bom_analyzer 相关组件
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from ontology_compiler.services.data_analysis import (
                DataAnalysisOrchestrator,
                OntologyContextLoader,
            )
            
            # 初始化 orchestrator
            orchestrator = DataAnalysisOrchestrator()
            
            # 加载 ontology 上下文
            context_loader = OntologyContextLoader()
            context = context_loader.load_context()
            
            # 运行分析
            analysis_result = orchestrator.analyze(
                query=query,
                context=context,
                analysis_type="mbom_deep_dive"
            )
            
            return {
                "bom_analysis": analysis_result,
                "data_driven": True,
                "sources": ["bom_analyzer", "ontology_context"]
            }
            
        except Exception as e:
            # 如果 bom_analyzer 调用失败，返回降级信息
            return {
                "bom_analysis": None,
                "data_driven": False,
                "error": str(e),
                "fallback": "使用标准 RAG 回答"
            }
    
    def _decompose_query(self, query: str) -> list[dict]:
        """查询拆解为子任务（基于 process.md 的 3 层框架）"""
        # 模拟 process.md 中 MBOM 分析的子任务结构
        sub_tasks = [
            {
                "layer": 1,
                "name": "field_exploration",
                "description": "字段探索：识别关键表和字段（TBS130/Partcache/TCP700）",
                "skills": ["system-researcher"]
            },
            {
                "layer": 2,
                "name": "precise_join_analysis",
                "description": "精确关联：验证系统间数据 JOIN 关系",
                "skills": ["dag-builder", "drift-detector"]
            },
            {
                "layer": 3,
                "name": "scenario_synthesis",
                "description": "场景综合：RAG + 业务洞察生成运营建议",
                "skills": ["consensus-resolver", "scene-analysis"]
            }
        ]
        
        # 如果检测到 MBOM 相关查询，添加 bom_analyzer 子任务
        if self._is_mbom_related(query):
            sub_tasks.insert(1, {
                "layer": 1.5,
                "name": "bom_data_analysis",
                "description": "MBOM 数据驱动分析：调用 bom_analyzer 进行精确数据分析",
                "skills": ["bom-analyzer", "data-analysis"],
                "data_source": "bom_analyzer"
            })
        
        return sub_tasks


def analyze_query(query: str) -> dict:
    """便捷函数：分析单个查询"""
    router = RAGIntelligentRouter()
    return router.route(query)


if __name__ == "__main__":
    # 测试用例
    test_queries = [
        # 简单
        "什么是 MBOM？",
        # 中等
        "MBOM 和 EBOM 有什么区别？",
        # 复杂（来自 process.md）
        "如果考虑到排产和物料 calloff 管理，如何对目前的流程、系统、数据进行分析后做一个数据模型来支持日常的排产和物料管理？",
        "分析工厂如何使用 MBOM 来应对排产变化及物料预测。PARTCACHE 不是用于跑物料预测的。",
    ]
    
    for q in test_queries:
        print("="*80)
        print(f"查询: {q}")
        result = analyze_query(q)
        print(f"复杂度: {result['complexity']} (得分: {result['score']:.2f})")
        print(f"路由: {result['path']}")
        print(f"原因: {result['reasoning']}")
        if 'agents' in result:
            print(f"建议 Agents: {result['agents']}")
        if 'skills' in result:
            print(f"所需 Skills: {result['skills']}")
        print()
