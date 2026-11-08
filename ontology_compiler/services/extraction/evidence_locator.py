"""
EvidenceLocator — 证据精确度提升 (R7)

将 LLM 提取的实体与原文进行位置映射，提升证据可追溯性。

**架构定位**: 这是 **intra-document**（文档内）级别的证据定位器，负责在单个源文本中精确定位实体的位置。

**互补模块**: 与 `evidence_plane_service.py` 中的 `EvidencePlaneService` 互补：
- EvidenceLocator: 文档内定位（行号、字符偏移、段落）
- EvidencePlaneService: 跨运行追踪（源文件、运行ID、层级）
- 两者通过 `EvidencePlaneService.trace_to_source_with_locator()` 桥接使用

功能：
- 在原文中搜索实体名称/描述，记录行号范围
- 模糊匹配 + 上下文校验，处理 LLM 输出与原文的细微差异
- 支持段落级别和字符级别的位置映射
- 计算匹配置信度，过滤低质量匹配

Usage:
    locator = EvidenceLocator(source_text, file_path="doc.md")
    evidence = locator.locate_entity(entity_name="Order Processing", entity_desc="...")
    # evidence.line_range = (45, 52)
    # evidence.char_offset = (1200, 1580)
    # evidence.confidence = 0.92

    # 与 EvidencePlaneService 配合使用（推荐）:
    # evidence_plane.trace_to_source_with_locator(entity_uri, source_text=text, file_path=path)
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.domain.models import Evidence

logger = logging.getLogger(__name__)


@dataclass
class LocationMatch:
    """单次匹配结果。"""
    line_start: int
    line_end: int
    char_start: int
    char_end: int
    matched_text: str
    similarity: float  # 0.0 - 1.0
    context: str  # 匹配点周围的文本


@dataclass
class LocatorConfig:
    """EvidenceLocator 配置。"""
    # 模糊匹配阈值
    min_similarity: float = 0.50
    # 关键词重叠度阈值（用于 L2/L2.5 模糊匹配）
    min_keyword_overlap: float = 0.40
    # 上下文窗口大小（行数）
    context_window_lines: int = 3
    # 最大匹配数（去重时保留前 N 个）
    max_matches: int = 3
    # 是否启用段落分割
    enable_paragraph_split: bool = True
    # 段落分隔符正则
    paragraph_pattern: str = r"\n\s*\n"
    # 是否优先匹配完整词
    word_boundary_match: bool = True
    # 是否启用关键词重叠匹配
    enable_keyword_match: bool = True
    # 实体类型特定阈值（L1/L2/L2.5）
    entity_type_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "L1": 0.50,
        "L2": 0.40,
        "L2.5": 0.35,
    })


class EvidenceLocator:
    """证据定位器：将实体映射回原文位置。"""

    def __init__(
        self,
        source_text: str,
        file_path: Optional[Path] = None,
        config: Optional[LocatorConfig] = None,
    ):
        self.source_text = source_text
        self.file_path = file_path
        self.config = config or LocatorConfig()

        # 预处理：分割为行和段落
        self.lines = source_text.splitlines()
        self.line_offsets = self._compute_line_offsets()
        self.paragraphs = self._split_paragraphs() if self.config.enable_paragraph_split else []

    def _compute_line_offsets(self) -> List[int]:
        """计算每行在原文中的字符偏移量。"""
        offsets = [0]
        for line in self.lines:
            offsets.append(offsets[-1] + len(line) + 1)  # +1 for newline
        return offsets

    def _split_paragraphs(self) -> List[Tuple[int, int, str]]:
        """将文本分割为段落，返回 (start_line, end_line, text) 列表。"""
        paragraphs = []
        last_end = 0
        for match in re.finditer(self.config.paragraph_pattern, self.source_text):
            para_text = self.source_text[last_end : match.start()]
            start_line = self.source_text[:last_end].count("\n")
            end_line = self.source_text[: match.start()].count("\n")
            if para_text.strip():
                paragraphs.append((start_line, end_line, para_text))
            last_end = match.end()
        # 最后一段
        if last_end < len(self.source_text):
            para_text = self.source_text[last_end:]
            start_line = self.source_text[:last_end].count("\n")
            end_line = self.source_text.count("\n")
            if para_text.strip():
                paragraphs.append((start_line, end_line, para_text))
        if not paragraphs:
            paragraphs.append((0, len(self.lines) - 1, self.source_text))
        return paragraphs

    def locate_entity(
        self,
        entity_name: str,
        entity_desc: str = "",
        entity_type: str = "",
    ) -> Optional[Evidence]:
        """定位单个实体在原文中的位置。

        Args:
            entity_name: 实体名称
            entity_desc: 实体描述（用于辅助匹配）
            entity_type: 实体类型（L1/L2/L2.5）

        Returns:
            Evidence 对象，包含位置信息；如果未找到则返回 None
        """
        if not entity_name:
            return None

        threshold = self.config.entity_type_thresholds.get(entity_type, self.config.min_similarity)

        # 1. 尝试精确匹配名称
        matches = self._find_matches(entity_name, threshold)

        # 2. 如果名称匹配不足，尝试描述匹配
        if not matches and entity_desc:
            desc_snippet = entity_desc[:50].strip()
            if len(desc_snippet) >= 10:
                matches = self._find_matches(desc_snippet, threshold)

        # 3. 如果仍无匹配，尝试模糊匹配
        if not matches:
            matches = self._fuzzy_find(entity_name, threshold)

        # 4. 如果仍无匹配，尝试关键词重叠匹配（主要用于 L2/L2.5）
        if not matches and self.config.enable_keyword_match:
            combined_text = entity_name + " " + entity_desc
            matches = self._keyword_overlap_match(combined_text, threshold)

        if not matches:
            logger.debug(f"No location match for entity: {entity_name}")
            return None

        # 选择最佳匹配
        best = max(matches, key=lambda m: m.similarity)

        # 构建 Evidence
        evidence = Evidence(
            source_file=str(self.file_path) if self.file_path else "",
            line_range=(best.line_start + 1, best.line_end + 1),  # 1-based
            char_offset=(best.char_start, best.char_end),
            paragraph_id=self._get_paragraph_id(best.char_start),
            content=best.matched_text[:500],
            raw_text=best.matched_text[:200],
            context=best.context[:300],
            confidence=best.similarity,
            metadata={
                "match_type": "name" if best.similarity > 0.9 else "fuzzy" if best.similarity > 0.7 else "keyword_overlap",
                "entity_type": entity_type,
            },
        )
        return evidence

    def locate_entities(
        self,
        entities: List[Dict[str, Any]],
        entity_type: str = "",
    ) -> Dict[str, Optional[Evidence]]:
        """批量定位实体位置。

        Args:
            entities: 实体字典列表或Pydantic模型列表，每个需包含 name 或 meta_name
            entity_type: 实体类型标识

        Returns:
            {entity_id: Evidence} 映射
        """
        results = {}
        for entity in entities:
            if isinstance(entity, dict):
                entity_id = entity.get("meta_id") or entity.get("id") or entity.get("name", "")
                name = entity.get("meta_name") or entity.get("name") or entity.get("scenario_name", "")
                desc = entity.get("description") or entity.get("meta_description", "")
            else:
                entity_id = getattr(entity, "meta_id", "") or getattr(entity, "id", "") or getattr(entity, "name", "")
                name = getattr(entity, "meta_name", "") or getattr(entity, "name", "") or getattr(entity, "scenario_name", "")
                desc = getattr(entity, "description", "") or getattr(entity, "meta_description", "")

            evidence = self.locate_entity(name, desc, entity_type)
            if evidence:
                results[entity_id] = evidence
                if isinstance(entity, dict):
                    entity["evidence"] = evidence.model_dump()
                elif hasattr(entity, "evidence"):
                    setattr(entity, "evidence", evidence)
                elif hasattr(entity, "__dict__"):
                    entity.__dict__["evidence"] = evidence

        logger.info(
            f"Located {len(results)}/{len(entities)} {entity_type} entities in source"
        )
        return results

    def _find_matches(self, query: str, threshold: float = None) -> List[LocationMatch]:
        """在原文中搜索查询字符串，返回所有匹配位置。"""
        matches = []
        if threshold is None:
            threshold = self.config.min_similarity
        query_lower = query.lower()
        text_lower = self.source_text.lower()

        # 精确搜索（不区分大小写）
        start = 0
        while True:
            idx = text_lower.find(query_lower, start)
            if idx == -1:
                break
            match = self._create_match(idx, idx + len(query), query)
            if match.similarity >= threshold:
                matches.append(match)
            start = idx + 1

        # 如果精确匹配失败，尝试词边界匹配
        if not matches and self.config.word_boundary_match:
            escaped = re.escape(query)
            pattern = r"\b" + escaped + r"\b"
            for m in re.finditer(pattern, self.source_text, re.IGNORECASE):
                match = self._create_match(m.start(), m.end(), m.group())
                if match.similarity >= threshold:
                    matches.append(match)

        return matches[: self.config.max_matches]

    def _fuzzy_find(self, query: str, threshold: float = None) -> List[LocationMatch]:
        """使用模糊匹配在原文中查找相似文本。"""
        matches = []
        if threshold is None:
            threshold = self.config.min_similarity
        query_lower = query.lower()

        # 逐行模糊匹配
        for i, line in enumerate(self.lines):
            line_lower = line.lower()
            if not set(query_lower) & set(line_lower):
                continue

            similarity = difflib.SequenceMatcher(None, query_lower, line_lower).ratio()
            if similarity >= threshold:
                char_start = self.line_offsets[i]
                char_end = char_start + len(line)
                context = self._get_context_lines(i)
                matches.append(
                    LocationMatch(
                        line_start=i,
                        line_end=i,
                        char_start=char_start,
                        char_end=char_end,
                        matched_text=line[:100],
                        similarity=similarity,
                        context=context,
                    )
                )

        # 如果单行匹配不足，尝试段落级别
        if not matches and self.paragraphs:
            for start_line, end_line, para_text in self.paragraphs:
                para_lower = para_text.lower()
                if not set(query_lower) & set(para_lower):
                    continue
                similarity = difflib.SequenceMatcher(None, query_lower, para_lower).ratio()
                if similarity >= threshold:
                    char_start = self.line_offsets[start_line]
                    char_end = char_start + len(para_text)
                    context = para_text[:200]
                    matches.append(
                        LocationMatch(
                            line_start=start_line,
                            line_end=end_line,
                            char_start=char_start,
                            char_end=char_end,
                            matched_text=para_text[:100],
                            similarity=similarity,
                            context=context,
                        )
                    )

        return sorted(matches, key=lambda m: m.similarity, reverse=True)[: self.config.max_matches]

    def _keyword_overlap_match(self, text: str, threshold: float = None) -> List[LocationMatch]:
        """使用关键词重叠度匹配在原文中查找相关段落。

        对于 L2/L2.5 实体，名称和描述可能与原文差异较大，
        使用关键词重叠度可以更好地识别相关内容。
        """
        matches = []
        if threshold is None:
            threshold = self.config.min_keyword_overlap

        # 提取关键词（移除停用词）
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                      "have", "has", "had", "do", "does", "did", "will", "would", "could",
                      "should", "may", "might", "must", "shall", "can", "need", "dare",
                      "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
                      "from", "as", "into", "through", "during", "before", "after", "above",
                      "below", "between", "under", "again", "further", "then", "once", "here",
                      "there", "when", "where", "why", "how", "all", "each", "few", "more",
                      "most", "other", "some", "such", "no", "nor", "not", "only", "own",
                      "same", "so", "than", "too", "very", "just", "but", "if", "or", "and",
                      "because", "until", "while", "this", "that", "these", "those"}

        text_lower = text.lower()
        text_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', text_lower)) - stop_words
        if not text_words:
            return matches

        # 在段落级别进行关键词匹配
        if self.paragraphs:
            for start_line, end_line, para_text in self.paragraphs:
                para_lower = para_text.lower()
                para_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', para_lower)) - stop_words

                if not para_words:
                    continue

                # 计算关键词重叠度
                overlap = text_words & para_words
                if overlap:
                    overlap_ratio = len(overlap) / min(len(text_words), len(para_words))
                    if overlap_ratio >= threshold:
                        char_start = self.line_offsets[start_line]
                        char_end = char_start + len(para_text)
                        context = para_text[:200]
                        matches.append(
                            LocationMatch(
                                line_start=start_line,
                                line_end=end_line,
                                char_start=char_start,
                                char_end=char_end,
                                matched_text=para_text[:100],
                                similarity=overlap_ratio,
                                context=context,
                            )
                        )

        # 如果段落匹配不足，尝试行级别
        if not matches:
            for i, line in enumerate(self.lines):
                line_lower = line.lower()
                line_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', line_lower)) - stop_words

                if not line_words:
                    continue

                overlap = text_words & line_words
                if overlap:
                    overlap_ratio = len(overlap) / min(len(text_words), len(line_words))
                    if overlap_ratio >= threshold:
                        char_start = self.line_offsets[i]
                        char_end = char_start + len(line)
                        context = self._get_context_lines(i)
                        matches.append(
                            LocationMatch(
                                line_start=i,
                                line_end=i,
                                char_start=char_start,
                                char_end=char_end,
                                matched_text=line[:100],
                                similarity=overlap_ratio,
                                context=context,
                            )
                        )

        return sorted(matches, key=lambda m: m.similarity, reverse=True)[: self.config.max_matches]

    def _create_match(self, char_start: int, char_end: int, matched_text: str) -> LocationMatch:
        """根据字符位置创建 LocationMatch。"""
        line_start = self.source_text[:char_start].count("\n")
        line_end = self.source_text[:char_end].count("\n")
        context = self._get_context_lines(line_start)

        # 计算相似度（精确匹配为 1.0）
        similarity = 1.0

        return LocationMatch(
            line_start=line_start,
            line_end=line_end,
            char_start=char_start,
            char_end=char_end,
            matched_text=matched_text[:100],
            similarity=similarity,
            context=context,
        )

    def _get_context_lines(self, line_idx: int) -> str:
        """获取指定行周围的上下文。"""
        window = self.config.context_window_lines
        start = max(0, line_idx - window)
        end = min(len(self.lines), line_idx + window + 1)
        return "\n".join(self.lines[start:end])

    def _get_paragraph_id(self, char_offset: int) -> Optional[int]:
        """根据字符偏移量确定段落 ID。"""
        if not self.paragraphs:
            return None
        for para_id, (start_line, end_line, _) in enumerate(self.paragraphs):
            para_start = self.line_offsets[start_line]
            para_end = self.line_offsets[min(end_line + 1, len(self.lines) - 1)]
            if para_start <= char_offset < para_end:
                return para_id
        return None


def enrich_entities_with_evidence(
    entities: List[Dict[str, Any]],
    source_text: str,
    file_path: Optional[Path] = None,
    entity_type: str = "",
    config: Optional[LocatorConfig] = None,
) -> Dict[str, Any]:
    """便捷函数：为实体列表批量添加证据位置信息。

    Args:
        entities: 实体字典列表
        source_text: 原始文本
        file_path: 源文件路径
        entity_type: 实体类型标识（L1/L2/L2.5）
        config: 定位器配置

    Returns:
        {entity_id: Evidence} 映射
    """
    if config is None:
        config = LocatorConfig()
    
    locator = EvidenceLocator(source_text, file_path, config)
    return locator.locate_entities(entities, entity_type)
