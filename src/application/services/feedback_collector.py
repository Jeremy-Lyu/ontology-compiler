#!/usr/bin/env python3
"""
Feedback Collector - G26-OPT-03

收集和管理用户反馈，支持：
- 显式反馈（点赞/点踩）
- 隐式反馈（停留时间、点击行为）
- 检索质量标记
- 答案质量评分

Usage:
    from feedback.feedback_collector import FeedbackCollector
    
    collector = FeedbackCollector()
    collector.record_feedback(
        query="MBOM是什么",
        answer="MBOM是制造物料清单...",
        feedback_type="explicit",
        rating=1,  # 1=好, -1=差
        user_id="user123"
    )
"""

import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime


@dataclass
class FeedbackRecord:
    """反馈记录"""
    id: str
    timestamp: float
    query: str
    answer: str
    feedback_type: str  # explicit, implicit
    rating: int  # 1=好, 0=中, -1=差
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "query": self.query,
            "answer": self.answer[:200],  # 截断存储
            "feedback_type": self.feedback_type,
            "rating": self.rating,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "metadata": self.metadata or {}
        }


class FeedbackCollector:
    """
    反馈收集器
    
    单例模式，支持内存缓存 + 持久化存储
    """
    
    _instance: Optional["FeedbackCollector"] = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, storage_path: str = "outputs/feedback/feedback_db.jsonl"):
        if self._initialized:
            return
        
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._records: List[FeedbackRecord] = []
        self._counter = 0
        
        # 加载已有记录
        self._load_from_file()
        
        self._initialized = True
        print(f"✅ FeedbackCollector initialized: {len(self._records)} records loaded")
    
    def record_feedback(
        self,
        query: str,
        answer: str,
        feedback_type: str = "explicit",
        rating: int = 0,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        记录反馈
        
        Args:
            query: 用户查询
            answer: 系统回答
            feedback_type: explicit(显式) 或 implicit(隐式)
            rating: 1=好, 0=中, -1=差
            user_id: 用户ID
            session_id: 会话ID
            metadata: 附加元数据
            
        Returns:
            反馈记录ID
        """
        self._counter += 1
        record_id = f"fb_{int(time.time())}_{self._counter}"
        
        record = FeedbackRecord(
            id=record_id,
            timestamp=time.time(),
            query=query,
            answer=answer,
            feedback_type=feedback_type,
            rating=rating,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata
        )
        
        self._records.append(record)
        
        # 立即持久化
        self._append_to_file(record)
        
        print(f"✅ Feedback recorded: {record_id} (rating={rating})")
        return record_id
    
    def record_retrieval_feedback(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        relevance_ratings: List[int],
        user_id: Optional[str] = None
    ) -> str:
        """
        记录检索质量反馈
        
        Args:
            query: 查询
            retrieved_chunks: 检索到的 chunks
            relevance_ratings: 每个 chunk 的相关性评分 (1=相关, 0=不相关)
            user_id: 用户ID
        """
        return self.record_feedback(
            query=query,
            answer=f"Retrieval feedback: {len(retrieved_chunks)} chunks",
            feedback_type="retrieval",
            rating=sum(relevance_ratings) / max(len(relevance_ratings), 1),
            user_id=user_id,
            metadata={
                "chunk_count": len(retrieved_chunks),
                "relevance_ratings": relevance_ratings,
                "avg_relevance": sum(relevance_ratings) / max(len(relevance_ratings), 1)
            }
        )
    
    def record_answer_feedback(
        self,
        query: str,
        answer: str,
        helpful: bool,
        user_comment: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> str:
        """
        记录答案质量反馈
        
        Args:
            query: 查询
            answer: 回答
            helpful: 是否有帮助
            user_comment: 用户评论
            user_id: 用户ID
        """
        rating = 1 if helpful else -1
        
        return self.record_feedback(
            query=query,
            answer=answer,
            feedback_type="answer_quality",
            rating=rating,
            user_id=user_id,
            metadata={
                "helpful": helpful,
                "user_comment": user_comment
            }
        )
    
    def get_feedback_stats(self, days: int = 7) -> Dict[str, Any]:
        """获取反馈统计"""
        cutoff = time.time() - days * 86400
        recent = [r for r in self._records if r.timestamp > cutoff]
        
        if not recent:
            return {"total": 0, "message": "No feedback in specified period"}
        
        explicit = [r for r in recent if r.feedback_type == "explicit"]
        implicit = [r for r in recent if r.feedback_type == "implicit"]
        
        ratings = [r.rating for r in recent]
        
        return {
            "period_days": days,
            "total": len(recent),
            "explicit": len(explicit),
            "implicit": len(implicit),
            "positive": sum(1 for r in ratings if r > 0),
            "neutral": sum(1 for r in ratings if r == 0),
            "negative": sum(1 for r in ratings if r < 0),
            "avg_rating": sum(ratings) / len(ratings) if ratings else 0,
            "by_type": self._group_by_type(recent)
        }
    
    def _group_by_type(self, records: List[FeedbackRecord]) -> Dict:
        """按类型分组统计"""
        result = {}
        for r in records:
            t = r.feedback_type
            if t not in result:
                result[t] = {"count": 0, "avg_rating": 0, "ratings": []}
            result[t]["count"] += 1
            result[t]["ratings"].append(r.rating)
        
        for t in result:
            ratings = result[t]["ratings"]
            result[t]["avg_rating"] = sum(ratings) / len(ratings)
            del result[t]["ratings"]
        
        return result
    
    def get_low_quality_queries(self, threshold: float = 0.3) -> List[Dict]:
        """获取低质量查询（用于优化）"""
        # 按查询分组，计算平均评分
        query_ratings: Dict[str, List[int]] = {}
        for r in self._records:
            q = r.query
            if q not in query_ratings:
                query_ratings[q] = []
            query_ratings[q].append(r.rating)
        
        low_quality = []
        for query, ratings in query_ratings.items():
            avg = sum(ratings) / len(ratings)
            if avg < threshold:
                low_quality.append({
                    "query": query,
                    "avg_rating": avg,
                    "feedback_count": len(ratings)
                })
        
        # 按评分升序排列
        low_quality.sort(key=lambda x: x["avg_rating"])
        return low_quality[:20]  # 返回最差的前20个
    
    def list_all_feedback(self, limit: int = 100) -> List[Dict]:
        """列出所有反馈"""
        return [r.to_dict() for r in self._records[-limit:]]
    
    def export_for_training(self, filepath: str):
        """导出用于模型训练的数据"""
        # 只导出高质量标注数据
        good_examples = [r for r in self._records if r.rating > 0]
        bad_examples = [r for r in self._records if r.rating < 0]
        
        training_data = {
            "good": [r.to_dict() for r in good_examples],
            "bad": [r.to_dict() for r in bad_examples],
            "metadata": {
                "total_good": len(good_examples),
                "total_bad": len(bad_examples),
                "exported_at": datetime.now().isoformat()
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Training data exported: {filepath}")
        return training_data
    
    def _append_to_file(self, record: FeedbackRecord):
        """追加单条记录到文件"""
        with open(self.storage_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + '\n')
    
    def _load_from_file(self):
        """从文件加载记录"""
        if not self.storage_path.exists():
            return
        
        with open(self.storage_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    record = FeedbackRecord(
                        id=data["id"],
                        timestamp=data["timestamp"],
                        query=data["query"],
                        answer=data["answer"],
                        feedback_type=data["feedback_type"],
                        rating=data["rating"],
                        user_id=data.get("user_id"),
                        session_id=data.get("session_id"),
                        metadata=data.get("metadata")
                    )
                    self._records.append(record)
                    # 更新计数器
                    counter = int(data["id"].split("_")[-1])
                    self._counter = max(self._counter, counter)
                except Exception as e:
                    print(f"⚠️ Failed to load feedback record: {e}")
    
    def clear_all(self):
        """清空所有反馈（谨慎使用）"""
        self._records = []
        self._counter = 0
        if self.storage_path.exists():
            self.storage_path.unlink()
        print("✅ All feedback cleared")


# 便捷函数
def get_global_collector() -> FeedbackCollector:
    """获取全局 FeedbackCollector 实例"""
    return FeedbackCollector()


if __name__ == "__main__":
    # 测试
    print("🧪 Testing FeedbackCollector")
    
    collector = FeedbackCollector()
    
    # 记录反馈
    fb1 = collector.record_feedback(
        query="什么是MBOM",
        answer="MBOM是制造物料清单",
        feedback_type="explicit",
        rating=1,
        user_id="test_user"
    )
    
    fb2 = collector.record_feedback(
        query="ECO流程",
        answer="ECO是工程变更流程",
        feedback_type="explicit",
        rating=-1,
        user_comment="回答太简单"
    )
    
    # 检索反馈
    fb3 = collector.record_retrieval_feedback(
        query="BOM管理",
        retrieved_chunks=[{"id": "c1"}, {"id": "c2"}],
        relevance_ratings=[1, 0],
        user_id="test_user"
    )
    
    # 统计
    stats = collector.get_feedback_stats(days=7)
    print(f"\n📊 Stats: {stats}")
    
    # 低质量查询
    low_quality = collector.get_low_quality_queries()
    print(f"\n🔍 Low quality queries: {len(low_quality)}")
    
    print("\n✅ FeedbackCollector test passed")
