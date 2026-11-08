"""L2 refinement service extracted from ``TrueUnifiedExtractor`` (2026-08-11).

``refine_l2`` makes an extra LLM call to enrich L2 scenario coverage.
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


async def refine_l2(
    content: str,
    initial_l2: List[Dict[str, Any]],
    domain: str,
    *,
    llm,
    refinement_threshold,
    enable_refinement,
) -> List[Dict[str, Any]]:
    """
    P2-4: Refine L2 entities by making additional LLM call for enhanced extraction.

    This method is integrated from SmartRefiner.refine_l2() to eliminate code duplication.

    Args:
        content: Document content to refine from
        initial_l2: Initial L2 entities from unified extraction
        domain: Domain hint for refinement

    Returns:
        Extended list of L2 entities with refined/added scenarios
    """
    try:
        if len(initial_l2) >= refinement_threshold and not enable_refinement:
            return initial_l2
        
        # Build refinement prompt
        prompt = f"""
从以下文档中提取业务场景（L2层）：

领域: {domain}

文档内容:
{content[:8000]}

请返回JSON格式，仅包含业务场景列表：
{{"business_scenarios": [{{"name": "场景名称", "description": "描述", "participants": ["角色1", "角色2"], "triggers": ["触发条件"], "outcomes": ["结果"]}}]}}
"""
        resp = await llm.call(prompt, max_tokens=4000, temperature=0.3)
        cleaned = resp.strip().replace('```json','').replace('```','').strip()
        
        import json
        result = json.loads(cleaned)
        scenarios = result.get('business_scenarios', [])
        
        if scenarios:
            # Deduplicate by name
            existing_names = {
                e.get('scenario_name', '').lower() for e in initial_l2 if isinstance(e, dict)
            }
            existing_names.update({
                e.get('name', '').lower() for e in initial_l2 if isinstance(e, dict)
            })
            
            new_scenarios = [
                s for s in scenarios if s.get('name', '').lower() not in existing_names
            ]
            
            # Convert to L2 entity format
            for idx, scenario in enumerate(new_scenarios):
                scenario['meta_id'] = f"L2-REF-{idx}"
                scenario['meta_class'] = "System"
                scenario['scenario_name'] = scenario.get('name', '')
                scenario['layer'] = "L2"
            
            logger.info(f"Refinement added {len(new_scenarios)} new L2 scenarios")
            return initial_l2 + new_scenarios
            
    except Exception as e:
        logger.error(f"L2 refinement failed: {e}")
    
    return initial_l2
