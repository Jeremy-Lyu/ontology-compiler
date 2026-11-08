"""L2.5 Skill Loader — 加载 YAML 技能配置并集成到流水线

[BATCH 4 CONSOLIDATION 2026-06-29]:
- This module provides **L2.5 skill configuration loading** from YAML files
- Integrates with skill_registry_adapter.py for broader skill management
- Used by TrueUnifiedExtractor to activate domain-specific extraction patterns

R41: 激活 config/l25_skills/ 配置，集成到 TrueUnifiedExtractor

Features:
- Load skill configurations from YAML files
- Match skills to text content via keywords/regex
- Build LLM prompts from skill templates
- Evaluate skill coverage across sample texts
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent.parent
L25_SKILLS_DIR = _PROJECT_ROOT / "skills" / "l25"
STANDARD_SKILLS_DIR = _PROJECT_ROOT / "skills"


class L25SkillConfig:
    """L2.5 技能配置数据类"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
        self.skill_id = config_dict.get("skill_id", "")
        self.skill_name = config_dict.get("skill_name", "")
        self.description = config_dict.get("description", "")
        self.version = config_dict.get("version", "1.0.0")
        self.domain = config_dict.get("domain", [])
        self.enabled = config_dict.get("enabled", True)
        
        # 触发条件
        self.triggers = config_dict.get("triggers", {})
        self.keywords = self.triggers.get("keywords", [])
        self.regex_patterns = self.triggers.get("regex_patterns", [])
        
        # [Optimization] 优先级和互斥组
        self.priority = self.triggers.get("priority", 0)
        self.exclusive_group = self.triggers.get("exclusive_group", "")
        self.cost_hint = self.triggers.get("cost_hint", "low")
        self.requires_llm = self.triggers.get("requires_llm", False)
        
        # [Optimization] 候选筛选关键词（快速筛选用）
        self.candidate_keywords = self.triggers.get("candidate_keywords", [])
        
        # 实体类型定义
        self.entity_types = config_dict.get("entity_types", {})
        
        # LLM 提示词模板
        self.llm_prompt_template = config_dict.get("llm_prompt_template", "")
        
        # 治理规则
        self.governance = config_dict.get("governance", {})

        # T3.2 契约化三层（classification/entity/relation）
        self.contract = config_dict.get("contract", {})
        
        # 后置处理
        self.post_processing = config_dict.get("post_processing", {})
        
        # 跨层映射
        self.cross_layer_mapping = config_dict.get("cross_layer_mapping", {})
    
    def is_candidate(self, text: str) -> bool:
        """[Optimization] 快速候选筛选 - 使用少量核心关键词做初步过滤
        
        这是两段式触发的第一步：先快速筛选出可能匹配的技能，
        再做精确匹配，减少不必要的全量扫描。
        """
        if not self.candidate_keywords:
            return True
        
        text_lower = text.lower()
        for keyword in self.candidate_keywords:
            if keyword.lower() in text_lower:
                return True
        
        return False
    
    def should_activate(self, text: str) -> bool:
        """检查文本是否匹配技能触发条件（精确匹配）
        
        [Optimization] 这是两段式触发的第二步：在候选筛选通过后，
        再做完整的关键词和正则匹配。
        """
        text_lower = text.lower()
        
        # 检查关键词
        for keyword in self.keywords:
            if keyword.lower() in text_lower:
                return True
        
        # 检查正则模式
        for pattern in self.regex_patterns:
            try:
                if re.search(pattern, text):
                    return True
            except re.error:
                logger.warning(f"Invalid regex pattern in skill {self.skill_id}: {pattern}")
        
        return False
    
    def build_prompt(self, content: str) -> str:
        """构建 LLM 提示词"""
        if not self.llm_prompt_template:
            return f"""从以下文档中提取 {self.skill_name} 相关实体。

文档内容：
{content[:8000]}

请识别实体并返回 JSON 格式。"""
        
        return self.llm_prompt_template.format(
            content=content[:8000],
            entity_types=json.dumps(self.entity_types, ensure_ascii=False, indent=2)
        )


class L25SkillLoader:
    """L2.5 技能加载器"""
    
    def __init__(self, skills_dir: Path = L25_SKILLS_DIR):
        self.skills_dir = skills_dir
        self._skills: Dict[str, L25SkillConfig] = {}
        self._loaded = False
        self._trigger_cache: Dict[str, List[str]] = {}
    
    def load_all(self) -> Dict[str, L25SkillConfig]:
        """加载所有 L2.5 技能配置
        
        支持两种格式：
        1. 标准格式：skills/cross_layer/*/.ontology-skill/skill.json
        2. YAML 格式：skills/l25/*.yaml
        """
        if self._loaded:
            return self._skills
        
        cross_layer_dir = STANDARD_SKILLS_DIR / "cross_layer"
        if cross_layer_dir.exists():
            for skill_dir in cross_layer_dir.iterdir():
                if not skill_dir.is_dir():
                    continue
                
                skill_json_path = skill_dir / ".ontology-skill" / "skill.json"
                if skill_json_path.exists():
                    try:
                        with open(skill_json_path, 'r', encoding='utf-8') as f:
                            config_dict = json.load(f)
                        
                        config_dict = self._convert_standard_to_l25(config_dict)
                        skill = L25SkillConfig(config_dict)
                        if skill.enabled:
                            self._skills[skill.skill_id] = skill
                            logger.info(f"Loaded L2.5 skill (JSON): {skill.skill_id} ({skill.skill_name})")
                    except Exception as e:
                        logger.error(f"Failed to load skill from {skill_json_path}: {e}")
        
        l25_skills_dir = L25_SKILLS_DIR
        if l25_skills_dir.exists():
            for yaml_file in l25_skills_dir.glob("*.yaml"):
                try:
                    with open(yaml_file, 'r', encoding='utf-8') as f:
                        config_dict = yaml.safe_load(f)
                    
                    skill = L25SkillConfig(config_dict)
                    if skill.enabled:
                        self._skills[skill.skill_id] = skill
                        logger.info(f"Loaded L2.5 skill (YAML): {skill.skill_id} ({skill.skill_name})")
                except Exception as e:
                    logger.error(f"Failed to load skill from {yaml_file}: {e}")
        
        self._loaded = True
        logger.info(f"Total L2.5 skills loaded: {len(self._skills)}")
        return self._skills
    
    def _convert_standard_to_l25(self, standard_dict: Dict[str, Any]) -> Dict[str, Any]:
        """将标准 skill.json 格式转换为 L25SkillConfig 兼容格式"""
        return {
            "skill_id": standard_dict.get("skill_id", ""),
            "skill_name": standard_dict.get("name", ""),
            "description": standard_dict.get("description", ""),
            "version": standard_dict.get("version", "1.0.0"),
            "domain": standard_dict.get("business_domain", []),
            "enabled": standard_dict.get("status", "active") == "active",
            "layer_target": "L2.5",
            "primary_layer": "L2.5",
            "secondary_layers": standard_dict.get("supported_layers", []),
            "triggers": {
                "keywords": standard_dict.get("trigger", {}).get("keywords", []),
                "regex_patterns": standard_dict.get("trigger", {}).get("regex_patterns", []),
                "priority": 0,
                "exclusive_group": "",
                "cost_hint": "low",
                "requires_llm": True,
            },
            "entity_types": self._convert_entity_contracts(standard_dict.get("entity_contracts", [])),
            "llm_prompt_template": standard_dict.get("llm_prompt_template", ""),
            "governance": {
                "allowed_layers": standard_dict.get("supported_layers", []),
                "allowed_relation_types": [
                    rc.get("relation_type") for rc in standard_dict.get("relation_contracts", [])
                    if rc.get("allowed", True)
                ],
                "blocked_relation_types": [
                    rc.get("relation_type") for rc in standard_dict.get("relation_contracts", [])
                    if not rc.get("allowed", True)
                ],
                "preferred_entity_types": standard_dict.get("classification_contract", {}).get("priority_main_classes", []),
                "validation_mode": "strict",
                "gating_level": "hard",
            },
            "post_processing": {
                "merge_similar_entities": True,
                "build_hierarchy": True,
            },
            "cross_layer_mapping": {},
        }
    
    def _convert_entity_contracts(self, entity_contracts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """将标准实体契约转换为 L25 实体类型定义"""
        result = {}
        for ec in entity_contracts:
            entity_type = ec.get("entity_type", "")
            if not entity_type:
                continue
            
            result[entity_type] = {
                "name_pattern": "|".join(ec.get("examples", [])),
                "required_fields": [
                    {rf.get("field_name"): rf.get("description", "")}
                    for rf in ec.get("required_fields", [])
                ],
                "optional_fields": ec.get("optional_fields", []),
                "indicators": [],
                "examples": ec.get("examples", []),
            }
        return result
    
    def get_skill(self, skill_id: str) -> Optional[L25SkillConfig]:
        """获取指定技能配置"""
        if not self._loaded:
            self.load_all()
        return self._skills.get(skill_id)
    
    def find_matching_skills(self, text: str) -> List[L25SkillConfig]:
        """[Optimization] 根据文本内容找到匹配的技能
        
        使用两段式触发策略：
        1. 快速候选筛选：用少量核心关键词过滤大部分不匹配的技能
        2. 精确匹配：对候选技能做完整的关键词和正则匹配
        3. 优先级排序：按优先级排序结果
        4. 互斥组处理：同一互斥组中只保留优先级最高的技能
        """
        if not self._loaded:
            self.load_all()
        
        text_hash = hashlib.md5(text[:1000].encode()).hexdigest() if text else ""
        if text_hash in self._trigger_cache:
            result_ids = self._trigger_cache[text_hash]
            logger.debug(f"Skill trigger cache hit for text (hash={text_hash[:8]}), skills={result_ids}")
            return [self._skills[sid] for sid in result_ids if sid in self._skills]
        
        logger.debug(f"Starting two-stage skill matching for text (len={len(text)} chars)")
        
        candidates = []
        for skill in self._skills.values():
            if skill.is_candidate(text):
                candidates.append(skill)
        
        logger.debug(f"Stage 1 (candidate filter): {len(candidates)}/{len(self._skills)} skills passed")
        
        matched = []
        for skill in candidates:
            if skill.should_activate(text):
                matched.append(skill)
        
        logger.debug(f"Stage 2 (exact match): {len(matched)}/{len(candidates)} skills matched")
        
        matched = self._apply_priority_and_exclusivity(matched)
        
        logger.debug(f"After priority/exclusivity: {len(matched)} skills selected")
        
        if text_hash:
            self._trigger_cache[text_hash] = [s.skill_id for s in matched]
            if len(self._trigger_cache) > 1000:
                oldest_key = min(self._trigger_cache.keys())
                del self._trigger_cache[oldest_key]
        
        return matched
    
    def _apply_priority_and_exclusivity(self, skills: List[L25SkillConfig]) -> List[L25SkillConfig]:
        """[Optimization] 应用优先级排序和互斥组规则
        
        同一互斥组中只保留优先级最高的技能，避免多个技能做重复工作。
        """
        if len(skills) <= 1:
            return skills
        
        sorted_skills = sorted(skills, key=lambda s: s.priority, reverse=True)
        
        exclusive_groups: Dict[str, L25SkillConfig] = {}
        final_skills: List[L25SkillConfig] = []
        
        for skill in sorted_skills:
            if skill.exclusive_group:
                if skill.exclusive_group not in exclusive_groups:
                    exclusive_groups[skill.exclusive_group] = skill
                    final_skills.append(skill)
                else:
                    existing = exclusive_groups[skill.exclusive_group]
                    if skill.priority > existing.priority:
                        final_skills.remove(existing)
                        exclusive_groups[skill.exclusive_group] = skill
                        final_skills.append(skill)
                    logger.debug(f"Skill {skill.skill_id} excluded (same group as {existing.skill_id})")
            else:
                final_skills.append(skill)
        
        return final_skills
    
    def find_matching_skills_by_cost(self, text: str, max_cost: str = "high") -> List[L25SkillConfig]:
        """[Optimization] 根据成本限制查找匹配技能
        
        Args:
            text: 输入文本
            max_cost: 最大允许成本 ("low", "medium", "high")
        
        Returns:
            匹配的技能列表，按成本升序排列
        """
        matched = self.find_matching_skills(text)
        
        cost_order = {"low": 0, "medium": 1, "high": 2}
        max_cost_level = cost_order.get(max_cost, 2)
        
        filtered = [s for s in matched if cost_order.get(s.cost_hint, 0) <= max_cost_level]
        filtered.sort(key=lambda s: cost_order.get(s.cost_hint, 0))
        
        return filtered
    
    def get_stats(self) -> Dict[str, Any]:
        """获取技能统计信息"""
        if not self._loaded:
            self.load_all()
        
        return {
            "total_skills": len(self._skills),
            "skills_by_domain": self._group_by_domain(),
            "skill_list": [
                {
                    "id": s.skill_id,
                    "name": s.skill_name,
                    "domain": s.domain,
                    "keywords_count": len(s.keywords),
                    "entity_types_count": len(s.entity_types)
                }
                for s in self._skills.values()
            ]
        }
    
    def evaluate_coverage(self, sample_texts: List[str]) -> Dict[str, Any]:
        """Evaluate skill coverage against sample texts.
        
        [Phase 6-L25-1]: Assess skill trigger rates and provide
        recommendations for improving coverage.
        
        Args:
            sample_texts: List of sample texts to test skill matching
            
        Returns:
            Coverage report with trigger statistics and recommendations
        """
        if not self._loaded:
            self.load_all()
        
        if not self._skills:
            return {"error": "No skills loaded"}
        
        if not sample_texts:
            return {"error": "No sample texts provided"}
        
        # Track triggers per skill
        skill_triggers: Dict[str, int] = {skill_id: 0 for skill_id in self._skills}
        skill_trigger_texts: Dict[str, List[str]] = {skill_id: [] for skill_id in self._skills}
        
        # Track texts with no matching skills
        uncovered_texts: List[str] = []
        
        for text in sample_texts:
            matched = False
            for skill_id, skill in self._skills.items():
                if skill.should_activate(text):
                    skill_triggers[skill_id] += 1
                    skill_trigger_texts[skill_id].append(text[:200])  # Store preview
                    matched = True
            
            if not matched:
                uncovered_texts.append(text[:200])
        
        # Calculate metrics
        total_texts = len(sample_texts)
        coverage_rate = (total_texts - len(uncovered_texts)) / total_texts if total_texts > 0 else 0
        
        # Per-skill statistics
        skill_stats = []
        for skill_id, skill in self._skills.items():
            trigger_count = skill_triggers[skill_id]
            trigger_rate = trigger_count / total_texts if total_texts > 0 else 0
            
            skill_stats.append({
                "skill_id": skill_id,
                "skill_name": skill.skill_name,
                "trigger_count": trigger_count,
                "trigger_rate": trigger_rate,
                "keywords": skill.keywords,
                "regex_patterns": skill.regex_patterns,
                "sample_triggers": skill_trigger_texts[skill_id][:3],  # First 3 examples
            })
        
        # Sort by trigger rate
        skill_stats.sort(key=lambda x: x["trigger_rate"], reverse=True)
        
        # Identify underutilized skills
        underutilized = [s for s in skill_stats if s["trigger_rate"] < 0.05]
        
        # Generate recommendations
        recommendations = []
        if coverage_rate < 0.8:
            recommendations.append(
                f"Low coverage rate ({coverage_rate:.1%}). "
                f"{len(uncovered_texts)} texts had no matching skills. "
                f"Consider adding more general-purpose skills or broadening keywords."
            )
        
        if underutilized:
            recommendations.append(
                f"{len(underutilized)} skills have <5% trigger rate: "
                f"{', '.join(s['skill_id'] for s in underutilized[:3])}. "
                f"Consider reviewing their trigger keywords/patterns."
            )
        
        # Check for overly broad skills (trigger on >50% of texts)
        overly_broad = [s for s in skill_stats if s["trigger_rate"] > 0.5]
        if overly_broad:
            recommendations.append(
                f"{len(overly_broad)} skills trigger on >50% of texts: "
                f"{', '.join(s['skill_id'] for s in overly_broad[:3])}. "
                f"Consider making triggers more specific."
            )
        
        return {
            "total_texts": total_texts,
            "total_skills": len(self._skills),
            "coverage_rate": coverage_rate,
            "uncovered_texts_count": len(uncovered_texts),
            "uncovered_texts_sample": uncovered_texts[:5],
            "skill_stats": skill_stats,
            "underutilized_skills": underutilized,
            "overly_broad_skills": overly_broad,
            "recommendations": recommendations,
        }
    
    def _group_by_domain(self) -> Dict[str, int]:
        """按领域分组统计"""
        domain_count = {}
        for skill in self._skills.values():
            for domain in skill.domain:
                domain_count[domain] = domain_count.get(domain, 0) + 1
        return domain_count


# 全局加载器实例
_global_loader: Optional[L25SkillLoader] = None


def get_l25_skill_loader() -> L25SkillLoader:
    """获取全局 L2.5 技能加载器"""
    global _global_loader
    if _global_loader is None:
        _global_loader = L25SkillLoader()
    return _global_loader


def reset_l25_skill_loader():
    """重置全局加载器（用于测试）"""
    global _global_loader
    _global_loader = None


# =============================================================================
# [BATCH 4] Skill Integration
# =============================================================================

__all__ = [
    "L25SkillConfig",
    "L25SkillLoader",
    "get_l25_skill_loader",
    "reset_l25_skill_loader",
]
