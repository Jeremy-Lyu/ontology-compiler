"""Pipeline Skill Router — 配置驱动的技能路由

根据 PIPELINE_REFACTOR_EXECUTION_CHECKLIST.md 任务要求：
- 将 stage→skill 的硬编码映射迁出 main_pipeline.py
- 使用 config/pipeline_skill_routing.json 进行配置驱动

设计原则：
1. Pipeline 不定义能力，只负责编排
2. 路由规则外置到 config 文件
3. 支持域级路由和通用路由
4. 支持 fallback 机制
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PipelineSkillRouter:
    """配置驱动的 Pipeline 技能路由器"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self._config_path = config_path or Path(__file__).resolve().parents[2] / "config" / "pipeline_skill_routing.json"
        self._rules: List[Dict[str, Any]] = []
        self._fallback_rules: List[Dict[str, Any]] = []
        self._loaded = False
    
    def load_config(self) -> bool:
        """加载路由配置"""
        if self._loaded:
            return True
        
        if not self._config_path.exists():
            logger.warning(f"Pipeline skill routing config not found: {self._config_path}")
            return False
        
        try:
            with open(self._config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self._rules = config.get("routing_rules", [])
            self._fallback_rules = config.get("fallback_rules", [])
            self._loaded = True
            
            logger.info(f"Loaded {len(self._rules)} routing rules and {len(self._fallback_rules)} fallback rules")
            return True
        except Exception as e:
            logger.error(f"Failed to load pipeline skill routing config: {e}")
            return False
    
    def get_skills_for_stage(self, stage: str, domain: str = "*") -> List[str]:
        """根据阶段和域获取适用技能
        
        匹配优先级：
        1. 精确匹配：stage + domain
        2. 通配匹配：stage + "*"
        3. Fallback：stage

        C-1 (2026-08-15): relationship-stage relation_rewrite rule is only
        active when ``PIPELINE_SKILL_RELATION_REWRITE=1`` so production
        behavior is unchanged by default; the rule stays documented in JSON.
        """
        import os
        if not self._loaded:
            self.load_config()

        # C-1 gate: the only non-empty rule is relationship -> relation_rewrite.
        # Without the env flag it is suppressed (keeps production identical).
        if stage == "relationship":
            enabled = os.environ.get("PIPELINE_SKILL_RELATION_REWRITE", "0").lower() in {"1", "true", "yes"}
            if not enabled:
                return []
        
        for rule in self._rules:
            if rule.get("stage") == stage:
                rule_domain = rule.get("domain", "*")
                if rule_domain == domain or rule_domain == "*":
                    return rule.get("skills", [])
        
        for fallback in self._fallback_rules:
            if fallback.get("stage") == stage:
                return fallback.get("skills", [])
        
        return []
    
    def get_all_routing_info(self) -> Dict[str, Any]:
        """获取所有路由规则信息"""
        if not self._loaded:
            self.load_config()
        
        return {
            "rules": self._rules,
            "fallback_rules": self._fallback_rules,
        }


_pipeline_skill_router: Optional[PipelineSkillRouter] = None


def get_pipeline_skill_router() -> PipelineSkillRouter:
    """获取全局 Pipeline 技能路由器实例"""
    global _pipeline_skill_router
    if _pipeline_skill_router is None:
        _pipeline_skill_router = PipelineSkillRouter()
    return _pipeline_skill_router


def reset_pipeline_skill_router():
    """重置 Pipeline 技能路由器"""
    global _pipeline_skill_router
    _pipeline_skill_router = None
