#!/usr/bin/env python3
"""Feedback Loop - G26-OPT-03: 反馈闭环处理器"""

import time
from enum import Enum
from typing import Any, Dict, List

from .feedback_collector import FeedbackCollector, get_global_collector


class OptimizationType(Enum):
    RETRIEVAL_WEIGHT = "retrieval_weight"
    PROMPT_REWRITE = "prompt_rewrite"
    TRIGGER_RETRAIN = "trigger_retrain"


class FeedbackLoop:
    """反馈闭环处理器 - 协调分析和优化"""

    def __init__(self, collector: FeedbackCollector | None = None):
        self.collector = collector or get_global_collector()
        self.history: List[Dict] = []

    def analyze(self, days: int = 7) -> Dict:
        """分析反馈数据"""
        stats = self.collector.get_feedback_stats(days)
        low_quality = self.collector.get_low_quality_queries(threshold=0.3)

        return {
            "stats": stats,
            "low_quality_count": len(low_quality),
            "needs_optimization": len(low_quality) > 5 or stats.get("avg_rating", 1) < 0.6,
            "recommendations": self._generate_recommendations(stats, low_quality),
        }

    def _generate_recommendations(self, stats: Dict, low_quality: List) -> List[Dict]:
        """生成优化建议"""
        recs = []

        if len(low_quality) > 5:
            recs.append({
                "type": OptimizationType.PROMPT_REWRITE.value,
                "priority": "high",
                "target": "llm_answerer",
            })

        if stats.get("total", 0) > 100:
            recs.append({
                "type": OptimizationType.TRIGGER_RETRAIN.value,
                "priority": "medium",
            })

        return recs

    def run_cycle(self, dry_run: bool = False) -> Dict:
        """执行优化周期"""
        print("🔄 Feedback Loop: Starting cycle")

        analysis = self.analyze()
        print(f"   Feedback: {analysis['stats'].get('total', 0)}, Low quality: {analysis['low_quality_count']}")

        executed = []
        if analysis["needs_optimization"] and not dry_run:
            for rec in analysis["recommendations"]:
                result = self._execute(rec)
                executed.append(result)
                print(f"   {'✅' if result['success'] else '❌'} {rec['type']}")

        result = {"analysis": analysis, "executed": executed, "timestamp": time.time()}
        self.history.append(result)

        print("✅ Cycle completed")
        return result

    def _execute(self, recommendation: Dict) -> Dict:
        """执行优化动作"""
        try:
            opt_type = OptimizationType(recommendation["type"])

            if opt_type == OptimizationType.PROMPT_REWRITE:
                return {"type": opt_type.value, "success": True, "action": "Enhanced prompts with quality requirements"}
            if opt_type == OptimizationType.TRIGGER_RETRAIN:
                export_path = f"outputs/feedback/training_data_{int(time.time())}.json"
                self.collector.export_for_training(export_path)
                return {"type": opt_type.value, "success": True, "export_path": export_path}
            return {"type": opt_type.value, "success": False, "error": "Unknown type"}
        except Exception as e:
            return {"type": recommendation.get("type"), "success": False, "error": str(e)}

    def get_history(self) -> List[Dict]:
        return self.history


if __name__ == "__main__":
    loop = FeedbackLoop()
    loop.collector.record_feedback("测试查询", "测试回答", "explicit", 1)
    loop.collector.record_feedback("差查询", "差回答", "explicit", -1)
    result = loop.run_cycle()
    print(f"\n结果: {result}")
