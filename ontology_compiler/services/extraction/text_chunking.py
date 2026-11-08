"""Intelligent text chunking service for semantic extraction.

[OPTIMIZATION] Enhanced with:
- Dynamic section-aware merging with adaptive thresholds
- Paragraph-level fallback merging for small chunks
- Multi-language topic similarity calculation (Chinese + English)
- Chunk count control mechanism to prevent over-chunking
- Semantic boundary preservation during merging
- Configuration loading from extraction_policies.yaml (single source of truth)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 2026-10-16: 原为 `from src.pipeline.stages.common import PROJECT_ROOT`
# （import-boundary 违例：ontology_compiler → src.pipeline 禁止）。改为本包
# 惯例的本地根推导（同 asset_index_service / data_continuity）。
from pathlib import Path as _Path

PROJECT_ROOT = _Path(__file__).resolve().parents[3]

_EXTRACTION_POLICIES_PATH = PROJECT_ROOT / "config" / "extraction" / "extraction_policies.yaml"
_PROMPT_TOKEN_BUDGET_PATH = PROJECT_ROOT / "config" / "prompts" / "prompt_token_budget.yaml"


def _resolve_chunk_size_from_budget(policy) -> Optional[int]:
    """C-2: derive max_chunk_size (chars) from a PromptBudgetPolicy.

    Resolution order:
    1. ``chunking.max_chunk_size`` from the policy's loaded YAML (authoritative)
    2. ``layers.document`` token budget from ``extraction_technical`` ×
       ``chunking.chunk_text_token_estimate`` (chars per token, CJK-aware)

    Returns None when the policy is None or the values cannot be resolved.
    """
    if policy is None:
        return None
    # 1. Prefer the explicit chunking.max_chunk_size from the policy config.
    chunking = getattr(policy, "chunking", None) or {}
    if isinstance(chunking, dict) and chunking.get("max_chunk_size"):
        try:
            return int(chunking["max_chunk_size"])
        except (TypeError, ValueError):
            pass
    # 2. Derive from layers.document token budget × chars-per-token ratio.
    try:
        budget = policy.resolve(task="extraction_technical")
    except Exception:
        return None
    tasks_cfg = getattr(policy, "tasks", {}) or {}
    profile_cfg = tasks_cfg.get(budget.profile) or {}
    layers = profile_cfg.get("layers") or {}
    if isinstance(layers, dict) and layers.get("document"):
        doc_tokens = int(layers["document"])
        ratio = 1.5
        if isinstance(chunking, dict) and chunking.get("chunk_text_token_estimate"):
            try:
                ratio = float(chunking["chunk_text_token_estimate"])
            except (TypeError, ValueError):
                pass
        return int(doc_tokens / ratio)
    return None


def _load_chunking_config() -> Dict[str, Any]:
    """Load chunking configuration from extraction_policies.yaml.
    
    Returns default config if file not found or YAML parsing fails.
    
    [OPTIMIZATION] 针对大文件优化：
    - max_chunk_size=6000（prompt 模板 ~2000 chars + 6000 chunk = ~8000 < 12000，
      避免触发 compress 丢信息，同时防止模型按大输入放飞输出）
    - 增加max_total_chunks到80（处理更大文档）
    - 降低topic_similarity_threshold到0.35（更容易合并相关chunk）
    """
    default = {
        # [OPTIMIZATION] chunk 大小需以「叠加 prompt 后的总预算」为约束推导：
        # prompt 模板固定开销 ~2000 chars（schema hints + instructions），
        # MAX_PROMPT_LENGTH=12000，超过即触发 compress（丢信息）。
        # 有效内容上限 = prompt_预算(11000) - 模板开销(~2000) ≈ 9000。
        # 但取 6000 而非 9000：模型会按剩余输出预算放飞写满 max_tokens，
        # 输入越大输出越失控（实测 6834-char chunk 诱发 29K-token 废话响应）。
        # 6000 = 均衡点：单 chunk 信息量足够、叠加模板后 ~8000 不触发压缩、
        # 模型输出保持可控。
        "max_chunk_size": 6000,
        "min_chunk_size": 1500,
        "overlap_size": 500,
        "prefer_headings": True,
        "preserve_code_blocks": True,
        "preserve_tables": True,
        "section_merge_threshold": 0.75,
        "topic_similarity_threshold": 0.35,  # 降低阈值，更容易合并
        "min_tokens_for_independent_chunk": 250,
        "max_consecutive_small_chunks": 4,
        "max_total_chunks": 80,  # 增加限制，处理更大文档
        "small_chunk_merge_limit": 4,
        "semantic_boundary_preserve": True,
        "chinese_support": True,
    }
    
    if not _EXTRACTION_POLICIES_PATH.exists():
        logger.warning(f"Extraction policies file not found: {_EXTRACTION_POLICIES_PATH}")
        return default
    
    try:
        import yaml
        with open(_EXTRACTION_POLICIES_PATH, "r", encoding="utf-8") as f:
            policies = yaml.safe_load(f)
        
        if isinstance(policies, dict) and "chunking" in policies:
            chunking = policies["chunking"]
            if isinstance(chunking, dict) and "default" in chunking:
                config = chunking["default"]
                if isinstance(config, dict):
                    merged = {**default, **config}
                    logger.info(f"Loaded chunking config from {_EXTRACTION_POLICIES_PATH}")
                    return merged
    except Exception as e:
        logger.warning(f"Failed to load chunking config from {_EXTRACTION_POLICIES_PATH}: {e}")
    
    return default


class TextChunker:
    """
    Intelligent text chunking service.
    
    [Optimization]: Implements "少切块，但切得更合理" principle from
    PIPELINE_OPTIMIZATION_PLAN_PROPOSAL.md.
    
    Supports multiple chunking strategies:
    - By paragraph (Markdown-aware)
    - By semantic section (heading-based)
    - Fixed-size with overlap
    - Context-aware (preserves related content)
    - Smart chunking with multi-pass optimization
    
    Configuration is loaded from extraction_policies.yaml as the single source of truth.
    """

    DEFAULT_CONFIG = _load_chunking_config()

    def __init__(self, config: Optional[Dict[str, Any]] = None, budget_policy=None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        # Ensure auto_split_threshold is available
        self.config.setdefault("auto_split_threshold", 50000)
        # C-2: when a PromptBudgetPolicy is attached, derive max_chunk_size
        # from the YAML chunking.max_chunk_size (preferred) or the
        # extraction_technical profile's layers.document token budget.
        if budget_policy is not None:
            _chunk_size = _resolve_chunk_size_from_budget(budget_policy)
            if _chunk_size is not None and _chunk_size > 0:
                self.config["max_chunk_size"] = int(_chunk_size)

    def chunk_by_semantic_section(self, text: str, source_file: str = "") -> List[Dict[str, Any]]:
        """
        Chunk text by semantic sections (headings, paragraphs, code blocks).
        
        [Optimization]: Preserves semantic boundaries and handles mixed content.
        """
        chunks: List[Dict[str, Any]] = []
        current_section = "Introduction"
        current_content = ""
        section_stack: List[str] = []

        lines = text.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]

            heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if heading_match:
                if current_content.strip():
                    chunks.append(self._create_chunk(current_content, current_section, source_file))
                
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                
                while section_stack and len(section_stack) >= level:
                    section_stack.pop()
                section_stack.append(title)
                current_section = " > ".join(section_stack)
                current_content = line + "\n"
                i += 1
                continue

            if self.config["preserve_code_blocks"] and line.startswith("```"):
                if current_content.strip():
                    chunks.append(self._create_chunk(current_content, current_section, source_file))
                
                code_block = line + "\n"
                i += 1
                while i < len(lines) and not lines[i].startswith("```"):
                    code_block += lines[i] + "\n"
                    i += 1
                if i < len(lines):
                    code_block += lines[i] + "\n"
                    i += 1
                
                chunks.append(self._create_chunk(code_block, current_section, source_file, chunk_type="code"))
                current_content = ""
                continue

            if self.config["preserve_tables"] and line.startswith("|"):
                if current_content.strip():
                    chunks.append(self._create_chunk(current_content, current_section, source_file))
                
                table = line + "\n"
                i += 1
                while i < len(lines) and lines[i].startswith("|"):
                    table += lines[i] + "\n"
                    i += 1
                
                chunks.append(self._create_chunk(table, current_section, source_file, chunk_type="table"))
                current_content = ""
                continue

            if line.strip() == "" and current_content.strip():
                current_content += line + "\n"
                chunks.append(self._create_chunk(current_content, current_section, source_file))
                current_content = ""
                i += 1
                continue

            current_content += line + "\n"
            i += 1

        if current_content.strip():
            chunks.append(self._create_chunk(current_content, current_section, source_file))

        return chunks

    def chunk_with_stats(self, text: str, source_file: str = "", doc_type: str = "") -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Chunk text and return both chunks and statistics."""
        chunks = self.smart_chunk(text, source_file)
        stats = self.analyze_chunking(chunks, doc_type=doc_type)
        stats["pre_split_triggered"] = len(text) > self.config.get("auto_split_threshold", 50000)
        return chunks, stats

    def chunk_by_fixed_size(self, text: str, source_file: str = "") -> List[Dict[str, Any]]:
        """Chunk text into fixed-size chunks with overlap."""
        chunks: List[Dict[str, Any]] = []
        max_size = self.config["max_chunk_size"]
        overlap = self.config["overlap_size"]
        
        start = 0
        chunk_id = 1

        while start < len(text):
            end = start + max_size
            
            if end < len(text):
                last_period = text.rfind(".", start, end)
                last_newline = text.rfind("\n", start, end)
                
                if last_period > start + max_size // 2:
                    end = last_period + 1
                elif last_newline > start + max_size // 2:
                    end = last_newline
                else:
                    end = start + max_size
            
            chunk_text = text[start:end]
            
            if chunk_text.strip():
                chunks.append({
                    "content": chunk_text,
                    "chunk_id": f"chunk_{chunk_id}",
                    "source_file": source_file,
                    "start_pos": start,
                    "end_pos": end,
                    "chunk_type": "fixed",
                })
            
            chunk_id += 1
            start = end - overlap
            
            if start >= len(text):
                break

        return chunks

    def chunk_by_context(self, text: str, source_file: str = "") -> List[Dict[str, Any]]:
        """Context-aware chunking that preserves related content."""
        chunks: List[Dict[str, Any]] = []
        max_size = self.config["max_chunk_size"]
        
        paragraphs = re.split(r"\n\s*\n", text)
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)
            
            if current_size + para_size > max_size and current_chunk:
                chunks.append(self._create_chunk(
                    "\n\n".join(current_chunk),
                    "Context Group",
                    source_file,
                    chunk_type="context"
                ))
                current_chunk = []
                current_size = 0

            list_items = re.split(r"^\s*[-*•]\s+", para, flags=re.MULTILINE)
            if len(list_items) > 1:
                list_chunk = []
                list_size = 0

                for item in list_items:
                    item_size = len(item)
                    if list_size + item_size > max_size / 2 and list_chunk:
                        chunks.append(self._create_chunk(
                            "\n".join(list_chunk),
                            "List Group",
                            source_file,
                            chunk_type="list"
                        ))
                        list_chunk = []
                        list_size = 0
                    list_chunk.append(item)
                    list_size += item_size

                if list_chunk:
                    chunks.append(self._create_chunk(
                        "\n".join(list_chunk),
                        "List Group",
                        source_file,
                        chunk_type="list"
                    ))
            else:
                current_chunk.append(para)
                current_size += para_size

        if current_chunk:
            chunks.append(self._create_chunk(
                "\n\n".join(current_chunk),
                "Context Group",
                source_file,
                chunk_type="context"
            ))

        return chunks

    _SLIDE_MARKER_RE = re.compile(
        r"^\s*(?:#{1,6}\s+)?(?:Slide|slide|幻灯片|幻灯片第)\s*(\d+)[:：.\-]?\s*$"
    )

    def chunk_by_slide(self, text: str, source_file: str = "") -> List[Dict[str, Any]]:
        """Chunk text by slide boundaries (PPTX-oriented).

        Splits on explicit slide markers ("Slide N", "第N页") and bare "---"
        separators. Slides without a marker keep the previous marker index.
        A slide that alone exceeds ``max_chunk_size`` is sub-split at
        sentence boundaries so no chunk is oversized.
        """
        chunks: List[Dict[str, Any]] = []
        max_size = int(self.config.get("max_chunk_size", 7000))
        lines = text.split("\n")
        current_lines: List[str] = []
        current_slide: Optional[int] = None
        slide_hits = 0

        def flush() -> None:
            nonlocal current_lines, current_slide
            content = "\n".join(current_lines).strip()
            if not content:
                current_lines = []
                return
            section = f"Slide {current_slide}" if current_slide is not None else "Slide ?"
            if len(content) <= max_size:
                chunks.append(self._create_chunk(content, section, source_file, chunk_type="slide"))
            else:
                pieces = re.split(r"(?<=[.!?。！？])\s+", content)
                buffer: List[str] = []
                buffer_size = 0
                for piece in pieces:
                    if buffer_size + len(piece) > max_size and buffer:
                        chunks.append(self._create_chunk("\n".join(buffer), section, source_file, chunk_type="slide"))
                        buffer = []
                        buffer_size = 0
                    buffer.append(piece)
                    buffer_size += len(piece)
                if buffer:
                    chunks.append(self._create_chunk("\n".join(buffer), section, source_file, chunk_type="slide"))
            current_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                current_lines.append("")
                continue
            marker = self._SLIDE_MARKER_RE.match(stripped)
            if marker:
                if current_lines:
                    flush()
                current_slide = int(marker.group(1))
                slide_hits += 1
                current_lines.append(stripped)
                continue
            if stripped == "---":
                if current_lines:
                    flush()
                continue
            current_lines.append(line)

        if current_lines:
            flush()

        if not chunks and text.strip():
            return self.chunk_by_fixed_size(text, source_file)
        if slide_hits == 0 and len(chunks) <= 1:
            return self.chunk_by_fixed_size(text, source_file)
        return chunks

    def chunk_by_strategy(self, strategy: str, text: str, source_file: str = "") -> List[Dict[str, Any]]:
        """Explicitly chunk using the requested strategy profile.

        Strategies: ``slide_based``, ``section_based``, ``hybrid``, or
        ``auto`` (existing smart auto-detection). Unknown strategies fall back
        to ``auto``.
        """
        strategy = (strategy or "auto").lower()
        if strategy == "slide_based":
            return self.chunk_by_slide(text, source_file)
        if strategy == "section_based":
            return self.chunk_by_semantic_section(text, source_file)
        if strategy == "hybrid":
            if re.search(r"^\s*(?:Slide|slide|幻灯片)\s*\d+", text, re.MULTILINE):
                return self.chunk_by_slide(text, source_file)
            if re.search(r"^#{1,6}\s+", text, re.MULTILINE):
                return self.chunk_by_semantic_section(text, source_file)
            return self.chunk_by_fixed_size(text, source_file)
        return self.smart_chunk(text, source_file, strategy="auto")

    def summarize_chunking(self, text: str, source_file: str = "", doc_type: str = "") -> Dict[str, Any]:
        """Create a chunking summary for regression reports."""
        chunks = self.smart_chunk(text, source_file)
        stats = self.analyze_chunking(chunks, doc_type=doc_type)
        quality_score = self._score_chunking_quality(chunks, text)
        stats["chunk_quality_score"] = quality_score
        stats["source_file"] = source_file
        stats["doc_type"] = doc_type
        stats["chunk_details"] = [
            {
                "chunk_id": c.get("chunk_id"),
                "section": c.get("section"),
                "page": c.get("page"),
                "char_count": c.get("char_count"),
                "token_count": c.get("token_count"),
                "chunk_type": c.get("chunk_type"),
            }
            for c in chunks
        ]
        return stats

    def _score_chunking_quality(self, chunks: List[Dict[str, Any]], text: str) -> float:
        """Score chunking quality on a 0-1 scale."""
        if not chunks:
            return 0.0
        lengths = [c.get("char_count", 0) for c in chunks]
        max_size = self.config["max_chunk_size"]
        min_size = self.config["min_chunk_size"]
        oversized_penalty = sum(1 for l in lengths if l > max_size) / len(lengths)
        small_penalty = sum(1 for l in lengths if l < min_size) / len(lengths)
        balance = 1.0 - min(abs(sum(lengths) / len(lengths) - (max_size + min_size) / 2) / max_size, 1.0)
        score = max(0.0, min(1.0, 0.5 * balance + 0.25 * (1 - oversized_penalty) + 0.25 * (1 - small_penalty)))
        return round(score, 4)

    def auto_chunk(self, text: str, source_file: str = "") -> List[Dict[str, Any]]:
        """Automatically select the best chunking strategy based on content type."""
        if re.search(r"^#{1,6}\s+", text, re.MULTILINE):
            logger.debug(f"Using semantic section chunking for {source_file}")
            return self.chunk_by_semantic_section(text, source_file)
        
        if len(text) > self.config["max_chunk_size"] * 3:
            logger.debug(f"Using context-aware chunking for {source_file}")
            return self.chunk_by_context(text, source_file)
        
        logger.debug(f"Using fixed-size chunking for {source_file}")
        return self.chunk_by_fixed_size(text, source_file)

    def get_chunking_strategy_profile(self, doc_type: str, text: str = "") -> str:
        """Return the preferred chunking strategy profile for a document type.

        Profiles:
        - slide_based
        - semantic_section / section_based
        - context
        - fixed
        - smart
        """
        doc_type = (doc_type or "").lower()
        if doc_type in {"pptx", "ppt"}:
            return "slide_based"
        if doc_type in {"md", "docx"}:
            return "section_based"
        if doc_type in {"pdf", "txt"}:
            return "context"
        if doc_type in {"csv", "xlsx"}:
            return "fixed"
        if text and len(text) > self.config["auto_split_threshold"]:
            return "smart"
        if re.search(r"(?i)(?:slide|幻灯片)\s*\d+", text):
            return "slide_based"
        return "section_based" if re.search(r"^#{1,6}\s+", text, re.MULTILINE) else "fixed"

    def analyze_chunking(self, chunks: List[Dict[str, Any]], doc_type: str = "") -> Dict[str, Any]:
        """Return chunking statistics for reporting and regression."""
        lengths = [len(c.get("content", "")) for c in chunks if isinstance(c, dict)]
        oversized = sum(1 for l in lengths if l > self.config["max_chunk_size"])
        small = sum(1 for l in lengths if l < self.config["min_chunk_size"])
        merged_small = max(0, len(lengths) - small) if lengths else 0
        return {
            "doc_type": doc_type,
            "strategy_used": self.get_chunking_strategy_profile(doc_type),
            "total_chunks": len(chunks),
            "input_chars": sum(lengths),
            "output_chars": sum(lengths),
            "avg_chunk_chars": (sum(lengths) / len(lengths)) if lengths else 0,
            "max_chunk_chars": max(lengths) if lengths else 0,
            "min_chunk_chars": min(lengths) if lengths else 0,
            "oversized_chunks": oversized,
            "small_chunks": small,
            "merged_small_chunks": merged_small,
            "semantic_merged_pairs": 0,
            "count_limited_merges": 0,
            "pre_split_triggered": False,
        }

    def _create_chunk(
        self,
        content: str,
        section: str,
        source_file: str,
        chunk_type: str = "text",
    ) -> Dict[str, Any]:
        """Create a chunk dictionary with metadata.

        [E6] Records source_file, section, page (if available), and position.
        """
        return {
            "content": content.strip(),
            "section": section,
            "source_file": source_file,
            "chunk_type": chunk_type,
            "char_count": len(content),
            "token_count": self.estimate_tokens(content),
            "page": self._estimate_page_number(content, section),
            "section_path": section,
        }

    @staticmethod
    def _estimate_page_number(content: str, section: str = "") -> Optional[int]:
        """Best-effort page estimation from content markers.

        Looks for patterns like "Page 3", "第3页", "Slide 5".
        Returns None if no page marker is found.
        """
        import re
        patterns = [
            r'(?:Page|page|pg\.?|p\.)\s*(\d+)',
            r'第\s*(\d+)\s*页',
            r'(?:Slide|slide)\s*(\d+)',
            r'(?:幻灯片|幻灯片第)\s*(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    pass
        return None

    def _merge_undersized_with_neighbors(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fold chunks below min_chunk_size into an adjacent neighbor.

        [OPTIMIZATION 2026-08-15] A 27-char chunk still costs one full LLM
        call whose prompt-template overhead (~2000 chars) dwarfs the content
        — pure waste. After all other merge passes, fold any leftover
        undersized chunk into the previous chunk when it still fits under
        max_chunk_size; if the first chunk is undersized, fold it forward.
        """
        if len(chunks) <= 1:
            return chunks
        min_size = self.config["min_chunk_size"]
        max_size = self.config["max_chunk_size"]

        result: List[Dict[str, Any]] = []
        for chunk in chunks:
            content = chunk.get("content", "")
            if len(content) >= min_size or not result:
                result.append(dict(chunk))
                continue
            prev = result[-1]
            if (prev.get("char_count", 0) + len(content) + 2 <= max_size
                    and prev.get("chunk_type") == chunk.get("chunk_type")):
                prev["content"] += "\n\n" + content
                prev["char_count"] += len(content) + 2
                prev["token_count"] += chunk.get("token_count", 0)
                if prev.get("section") != chunk.get("section"):
                    prev["section"] = f"{prev.get('section')} + {chunk.get('section')}"
            else:
                result.append(dict(chunk))

        # Leading undersized chunk: fold forward into the second chunk if it fits.
        if len(result) >= 2 and result[0].get("char_count", 0) < min_size:
            first, second = result[0], result[1]
            if (first.get("char_count", 0) + second.get("char_count", 0) + 2 <= max_size
                    and first.get("chunk_type") == second.get("chunk_type")):
                second["content"] = first["content"] + "\n\n" + second["content"]
                second["char_count"] = first.get("char_count", 0) + second.get("char_count", 0) + 2
                second["token_count"] = first.get("token_count", 0) + second.get("token_count", 0)
                if first.get("section") != second.get("section"):
                    second["section"] = f"{first.get('section')} + {second.get('section')}"
                result = result[1:]
        return result

    def _merge_small_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge small chunks to meet minimum size requirement.
        
        [Optimization]: Aggressive merging of any small chunks, regardless of position.
        Only start a new chunk when merging would exceed max_chunk_size.
        """
        if len(chunks) <= 1:
            return chunks
        
        merged: List[Dict[str, Any]] = []
        current_chunk: Optional[Dict[str, Any]] = None
        min_size = self.config["min_chunk_size"]
        max_size = self.config["max_chunk_size"]
        
        for chunk in chunks:
            if chunk["char_count"] < min_size:
                if current_chunk is None:
                    current_chunk = chunk.copy()
                else:
                    # Check if merging would exceed max_size
                    new_size = current_chunk["char_count"] + chunk["char_count"] + 2
                    if new_size <= max_size:
                        # Still fits, merge
                        current_chunk["content"] += "\n\n" + chunk["content"]
                        current_chunk["char_count"] += len(chunk["content"]) + 2
                        current_chunk["token_count"] += chunk["token_count"]
                    else:
                        # Would exceed max_size, push current and start new
                        merged.append(current_chunk)
                        current_chunk = chunk.copy()
            else:
                # This chunk is already large enough
                if current_chunk is not None:
                    # If we have accumulated small chunks, push them
                    merged.append(current_chunk)
                # Push the large chunk
                merged.append(chunk)
                current_chunk = None
        
        if current_chunk is not None:
            merged.append(current_chunk)
        
        return merged

    def _merge_by_section(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge chunks within the same section if they're small.
        
        [Optimization]: Enhanced section-aware merging with semantic boundary preservation.
        """
        if not chunks:
            return chunks
        
        merged: List[Dict[str, Any]] = []
        current_section = chunks[0].get("section", "")
        current_section_chunks: List[Dict[str, Any]] = []
        
        for chunk in chunks:
            chunk_section = chunk.get("section", "")
            
            if chunk_section == current_section:
                current_section_chunks.append(chunk)
            else:
                merged.extend(self._merge_section_chunks(current_section_chunks))
                current_section = chunk_section
                current_section_chunks = [chunk]
        
        merged.extend(self._merge_section_chunks(current_section_chunks))
        
        return merged

    def _merge_section_chunks(self, section_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge small chunks within a section with size constraint."""
        if len(section_chunks) <= 1:
            return section_chunks
        
        merged: List[Dict[str, Any]] = []
        current_chunk = section_chunks[0].copy()
        max_size = self.config["max_chunk_size"]
        
        for chunk in section_chunks[1:]:
            if current_chunk["char_count"] + chunk["char_count"] <= max_size:
                current_chunk["content"] += "\n\n" + chunk["content"]
                current_chunk["char_count"] += len(chunk["content"]) + 2
                current_chunk["token_count"] += chunk["token_count"]
            else:
                merged.append(current_chunk)
                current_chunk = chunk.copy()
        
        merged.append(current_chunk)
        
        return merged

    def _calculate_topic_similarity(self, chunk1: Dict[str, Any], chunk2: Dict[str, Any]) -> float:
        """Calculate similarity between two chunks based on shared terms.
        
        [Optimization]: Supports both English and Chinese text analysis.
        """
        text1 = chunk1.get("content", "")
        text2 = chunk2.get("content", "")
        
        words1 = self._extract_key_terms(text1)
        words2 = self._extract_key_terms(text2)
        
        if not words1 and not words2:
            return 0.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0

    def _extract_key_terms(self, text: str) -> set:
        """Extract key terms from text, supporting English and Chinese."""
        terms = set()
        
        terms.update(re.findall(r'\b[a-zA-Z]{3,}\b', text.lower()))
        
        if self.config["chinese_support"]:
            terms.update(re.findall(r'[\u4e00-\u9fff]{2,}', text))
        
        return terms

    def _merge_by_topic(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge chunks that share similar topics even across sections."""
        if len(chunks) <= 1:
            return chunks
        
        merged: List[Dict[str, Any]] = []
        current_chunk = chunks[0].copy()
        max_size = self.config["max_chunk_size"]
        similarity_threshold = self.config["topic_similarity_threshold"]
        
        for chunk in chunks[1:]:
            similarity = self._calculate_topic_similarity(current_chunk, chunk)
            
            if similarity >= similarity_threshold and current_chunk["char_count"] + chunk["char_count"] <= max_size:
                current_chunk["content"] += "\n\n" + chunk["content"]
                current_chunk["char_count"] += len(chunk["content"]) + 2
                current_chunk["token_count"] += chunk["token_count"]
                if "section" in current_chunk and "section" in chunk:
                    if current_chunk["section"] != chunk["section"]:
                        current_chunk["section"] = f"{current_chunk['section']} + {chunk['section']}"
            else:
                merged.append(current_chunk)
                current_chunk = chunk.copy()
        
        merged.append(current_chunk)
        
        return merged

    def _enforce_chunk_count_limit(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """[Optimization] Enforce maximum chunk count by merging adjacent chunks.
        
        When chunk count exceeds the limit, merge adjacent chunks to reduce count.
        This prevents over-chunking which leads to excessive LLM calls.
        """
        max_chunks = self.config["max_total_chunks"]
        if len(chunks) <= max_chunks:
            return chunks
        
        logger.warning(f"Chunk count ({len(chunks)}) exceeds limit ({max_chunks}), merging...")
        
        result = chunks.copy()
        while len(result) > max_chunks:
            best_pair = None
            best_similarity = -1
            
            for i in range(len(result) - 1):
                similarity = self._calculate_topic_similarity(result[i], result[i + 1])
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_pair = (i, i + 1)
            
            if best_pair:
                i, j = best_pair
                merged = result[i].copy()
                merged["content"] += "\n\n" + result[j]["content"]
                merged["char_count"] += len(result[j]["content"]) + 2
                merged["token_count"] += result[j]["token_count"]
                
                result = result[:i] + [merged] + result[j + 1:]
            
            else:
                break
        
        logger.info(f"Reduced chunk count from {len(chunks)} to {len(result)}")
        return result

    def _apply_semantic_boundary_preservation(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """[Optimization] Preserve semantic boundaries during merging.
        
        Ensures that key semantic elements (definitions, lists, tables) are
        kept together and not split across chunks.
        """
        if not self.config["semantic_boundary_preserve"]:
            return chunks
        
        if len(chunks) <= 1:
            return chunks
        
        result = []
        i = 0
        
        while i < len(chunks):
            current = chunks[i]
            
            if i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                
                if self._should_merge_for_semantics(current, next_chunk):
                    merged = current.copy()
                    merged["content"] += "\n\n" + next_chunk["content"]
                    merged["char_count"] += len(next_chunk["content"]) + 2
                    merged["token_count"] += next_chunk["token_count"]
                    result.append(merged)
                    i += 2
                    continue
            
            result.append(current)
            i += 1
        
        return result

    def _should_merge_for_semantics(self, chunk1: Dict[str, Any], chunk2: Dict[str, Any]) -> bool:
        """Determine if two chunks should be merged to preserve semantics."""
        content1 = chunk1.get("content", "")
        content2 = chunk2.get("content", "")
        
        chunk1_type = chunk1.get("chunk_type", "text")
        chunk2_type = chunk2.get("chunk_type", "text")
        
        if chunk1_type != chunk2_type:
            if chunk1_type in ("code", "table") or chunk2_type in ("code", "table"):
                return False
        
        if re.search(r'(定义|定义为|是指|即)\s*$', content1):
            return True
        
        if re.search(r'^\s*[-*•]\s+', content1) and re.search(r'^\s*[-*•]\s+', content2):
            return True
        
        if re.search(r'^\s*\d+\.\s+', content1) and re.search(r'^\s*\d+\.\s+', content2):
            return True
        
        return False

    def smart_chunk(self, text: str, source_file: str = "", strategy: str = "auto") -> List[Dict[str, Any]]:
        """
        Smart chunking with multi-pass optimization:
        
        0. Pre-processing: Analyze document structure and identify oversized content
        1. First pass: semantic section chunking
        2. Second pass: merge by semantic boundary preservation
        3. Third pass: merge by section
        4. Fourth pass: merge by topic similarity
        5. Fifth pass: merge small chunks
        6. Sixth pass: enforce chunk count limit
        7. Seventh pass: split oversized chunks with sentence-level precision
        
        ``strategy``: explicit strategy override ("slide_based", "section_based",
        "hybrid", "fixed", "context" or "auto"); "auto" keeps the historical
        multi-pass smart pipeline (with slide-oriented dispatch for slide
        markers).
        
        [Optimization]: Implements the "少切块，但切得更合理" principle from
        PIPELINE_OPTIMIZATION_PLAN_PROPOSAL.md, ensuring semantic units are kept intact.
        
        [ENHANCEMENT]: Pre-processing stage for very large documents (>50000 chars)
        with fine-grained sentence-level splitting for oversized paragraphs.

        [STATS]: Tracks per-run statistics for reporting (see docs/CHUNKING_STRATEGY_PROFILES.md)
        """
        strategy = (strategy or "auto").lower()
        pre_split_triggered = False
        logger.info(f"Starting smart chunking for {source_file} (text length: {len(text)} chars, strategy={strategy})")
        
        # Explicit strategy dispatch (slide/section/hybrid/fixed/context).
        if strategy in {"slide_based", "hybrid", "fixed", "context"}:
            if strategy == "slide_based":
                return self.chunk_by_slide(text, source_file)
            if strategy == "hybrid":
                return self.chunk_by_strategy("hybrid", text, source_file)
            if strategy == "fixed":
                return self.chunk_by_fixed_size(text, source_file)
            return self.chunk_by_context(text, source_file)
        if strategy == "section_based":
            chunks = self.chunk_by_semantic_section(text, source_file)
            # [FIX 2026-08-15] section_based previously bypassed every size
            # control pass, so a semantic section just under max_chunk_size
            # (e.g. VSM "Original Content (Raw)" = 6834/7000 chars) shipped as
            # one oversized chunk while tiny 3-char chunks stayed separate.
            # Run the same merge/split chain auto mode uses so every chunk
            # lands in the [min, max] window (balanced after prompt overhead).
            chunks = self._merge_by_section(chunks)
            chunks = self._merge_by_topic(chunks)
            chunks = self._merge_small_chunks(chunks)
            chunks = self._split_oversized_chunks(chunks)
            chunks = self._final_size_validation(chunks)
            return self._merge_undersized_with_neighbors(chunks)
        
        # [ENHANCEMENT] Pre-processing for large documents:
        # Analyze and split oversized paragraphs before semantic chunking
        if len(text) > self.config.get("auto_split_threshold", 50000):
            pre_split_triggered = True
            logger.info(f"[PRE-PROCESSING] Large document ({len(text)} chars), applying fine-grained pre-splitting")
            text = self._preprocess_large_document(text)
        
        # Auto mode: for slide-oriented content prefer slide boundaries.
        if re.search(r"^\s*(?:Slide|slide|幻灯片)\s*\d+", text, re.MULTILINE):
            chunks = self.chunk_by_slide(text, source_file)
            logger.debug(f"After slide detection: {len(chunks)} chunks")
            return chunks
        
        chunks = self.chunk_by_semantic_section(text, source_file)
        logger.debug(f"After semantic section chunking: {len(chunks)} chunks")
        
        chunks = self._apply_semantic_boundary_preservation(chunks)
        logger.debug(f"After semantic boundary preservation: {len(chunks)} chunks")
        
        chunks = self._merge_by_section(chunks)
        logger.debug(f"After section merge: {len(chunks)} chunks")
        
        chunks = self._merge_by_topic(chunks)
        logger.debug(f"After topic merge: {len(chunks)} chunks")
        
        chunks = self._merge_small_chunks(chunks)
        logger.debug(f"After small chunk merge: {len(chunks)} chunks")
        
        chunks = self._enforce_chunk_count_limit(chunks)
        logger.debug(f"After chunk count limit enforcement: {len(chunks)} chunks")
        
        chunks = self._split_oversized_chunks(chunks)
        logger.debug(f"After oversized chunk splitting: {len(chunks)} chunks")
        
        # [ENHANCEMENT] Final pass: ensure no chunks exceed max_size with sentence-level precision
        chunks = self._final_size_validation(chunks)
        logger.debug(f"After final size validation: {len(chunks)} chunks")

        chunks = self._merge_undersized_with_neighbors(chunks)
        logger.debug(f"After neighbor merge: {len(chunks)} chunks")

        logger.info(f"Smart chunking completed: {len(chunks)} final chunks")
        
        return chunks
    
    def _preprocess_large_document(self, text: str) -> str:
        """
        [ENHANCEMENT] Pre-process large documents to split oversized paragraphs.
        
        For documents >50000 chars, identify and split paragraphs that exceed
        max_chunk_size before semantic chunking. This prevents single massive
        sections from becoming unmanageable chunks.
        
        Strategy:
        1. Split text by paragraphs
        2. For paragraphs exceeding max_chunk_size, split by sentences
        3. Reconstruct text with proper separators
        """
        max_size = self.config["max_chunk_size"]
        paragraphs = re.split(r'\n\s*\n', text)
        processed_paragraphs = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                processed_paragraphs.append("")
                continue
            
            # If paragraph is already manageable, keep it as-is
            if len(para) <= max_size:
                processed_paragraphs.append(para)
                continue
            
            # [ENHANCEMENT] Oversized paragraph - split by sentence boundaries
            logger.warning(f"[PRE-PROCESSING] Oversized paragraph ({len(para)} chars), splitting by sentences")
            
            # Try sentence-level splitting
            sentences = self._split_by_sentences(para)
            
            # Group sentences into sub-paragraphs under max_size
            current_group = []
            current_size = 0
            
            for sentence in sentences:
                sentence_size = len(sentence)
                
                # If single sentence exceeds max_size, split it further
                if sentence_size > max_size:
                    if current_group:
                        processed_paragraphs.append("\n".join(current_group))
                        current_group = []
                        current_size = 0
                    
                    # Split oversized sentence into chunks
                    sub_chunks = self._split_large_sentence(sentence, max_size)
                    processed_paragraphs.extend(sub_chunks)
                    continue
                
                if current_size + sentence_size + 2 > max_size and current_group:
                    processed_paragraphs.append("\n".join(current_group))
                    current_group = [sentence]
                    current_size = sentence_size
                else:
                    current_group.append(sentence)
                    current_size += sentence_size + 2
            
            if current_group:
                processed_paragraphs.append("\n".join(current_group))
        
        return "\n\n".join(processed_paragraphs)
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using multi-language aware regex.
        
        Handles:
        - English: .!? followed by space or newline
        - Chinese: 。！？（） followed by space, newline, or Chinese character
        - Mixed language content
        
        [ENHANCEMENT] Improved Chinese sentence splitting by handling
        consecutive Chinese punctuation without spaces.
        """
        sentences = []
        
        # First, check if text is primarily Chinese (more than 50% Chinese chars)
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        is_primarily_chinese = chinese_chars > len(text) * 0.5
        
        if is_primarily_chinese:
            # Chinese sentence splitting: split at Chinese punctuation
            # Handle: 。！？ followed by Chinese char or end of text
            # Also handle: ） followed by Chinese char or punctuation
            
            # Pattern: split after Chinese sentence terminators
            # (?<=[。！？]) - positive lookbehind for Chinese terminators
            # (?=[\u4e00-\u9fff]|$) - positive lookahead for Chinese char or end
            # |(?<=[.!?])\s+ - or English pattern with space
            sentence_pattern = r'(?<=[。！？])(?=[\u4e00-\u9fff]|$)|(?<=[.!?])\s+'
            
            raw_sentences = re.split(sentence_pattern, text)
            
            # Merge adjacent short sentences that are likely part of the same sentence
            i = 0
            while i < len(raw_sentences):
                current = raw_sentences[i].strip()
                if not current:
                    i += 1
                    continue
                
                # If current sentence ends with （ or 「, merge with next
                if current.endswith('（') or current.endswith('「') or current.endswith('"'):
                    j = i + 1
                    while j < len(raw_sentences):
                        next_s = raw_sentences[j].strip()
                        if not next_s:
                            j += 1
                            continue
                        current += next_s
                        # Stop when we find a proper terminator
                        if next_s.endswith('。') or next_s.endswith('！') or next_s.endswith('？'):
                            break
                        j += 1
                    i = j
                else:
                    i += 1
                
                sentences.append(current)
        else:
            # English/mixed pattern: .!?。！？ followed by space or newline
            sentence_pattern = r'(?<=[.!?。！？])\s+|(?<=[.!?。！？])$'
            raw_sentences = re.split(sentence_pattern, text)
            sentences = [s.strip() for s in raw_sentences if s.strip()]
        
        # Final cleanup: filter empty and very short sentences
        return [s for s in sentences if s and len(s) > 1]
    
    def _split_large_sentence(self, text: str, max_size: int) -> List[str]:
        """
        Split a single oversized sentence into smaller chunks.
        
        Uses:
        1. Comma/colon/semicolon as primary split points
        2. Space as fallback
        3. Hard split if no natural boundaries found
        """
        chunks = []
        remaining = text
        
        while len(remaining) > max_size:
            # Try to split at natural boundaries within the last 20% of max_size
            search_end = min(max_size, len(remaining))
            search_start = int(search_end * 0.8)
            
            # Priority: comma > colon > semicolon > space
            for boundary in [', ', '，', ': ', '：', '; ', '；', ' ']:
                pos = remaining.rfind(boundary, search_start, search_end)
                if pos > 0:
                    chunks.append(remaining[:pos + len(boundary)].strip())
                    remaining = remaining[pos + len(boundary):].strip()
                    break
            else:
                # No natural boundary found - hard split
                chunks.append(remaining[:max_size].strip())
                remaining = remaining[max_size:].strip()
        
        if remaining:
            chunks.append(remaining)
        
        return chunks
    
    def _final_size_validation(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        [ENHANCEMENT] Final validation pass to ensure no chunks exceed max_size.
        
        This is a safety net that catches any oversized chunks that slipped through
        previous passes, splitting them with sentence-level precision.
        """
        max_size = self.config["max_chunk_size"]
        result = []
        
        for chunk in chunks:
            content = chunk["content"]
            if len(content) <= max_size:
                result.append(chunk)
                continue
            
            logger.warning(f"[FINAL VALIDATION] Oversized chunk detected ({len(content)} chars), splitting...")
            
            # Split by sentences
            sentences = self._split_by_sentences(content)
            
            current_group = []
            current_size = 0
            section = chunk.get("section", "")
            source_file = chunk.get("source_file", "")
            chunk_type = chunk.get("chunk_type", "text")
            
            for sentence in sentences:
                sentence_size = len(sentence)
                
                if sentence_size > max_size:
                    # Single sentence too large - split further
                    if current_group:
                        result.append(self._create_chunk(
                            "\n".join(current_group),
                            section,
                            source_file,
                            chunk_type
                        ))
                        current_group = []
                        current_size = 0
                    
                    sub_chunks = self._split_large_sentence(sentence, max_size)
                    for sub_chunk in sub_chunks:
                        result.append(self._create_chunk(
                            sub_chunk,
                            section,
                            source_file,
                            chunk_type
                        ))
                    continue
                
                if current_size + sentence_size + 2 > max_size and current_group:
                    result.append(self._create_chunk(
                        "\n".join(current_group),
                        section,
                        source_file,
                        chunk_type
                    ))
                    current_group = [sentence]
                    current_size = sentence_size
                else:
                    current_group.append(sentence)
                    current_size += sentence_size + 2
            
            if current_group:
                result.append(self._create_chunk(
                    "\n".join(current_group),
                    section,
                    source_file,
                    chunk_type
                ))
        
        return result

    def _split_oversized_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """[OPTIMIZATION] Split chunks that exceed max_chunk_size.
        
        This is critical because chunk_by_semantic_section does not enforce size limits,
        and a single large section (e.g., a 23000-char paragraph) becomes one huge chunk.
        Large chunks cause:
        1. Prompt > 12000 chars, triggering expensive compression
        2. LLM responses that are too long, causing JSON parse failures
        3. Wasted retry attempts (40+ minutes in BATCH1)
        """
        max_size = self.config["max_chunk_size"]
        result = []
        
        for chunk in chunks:
            if chunk["char_count"] <= max_size:
                result.append(chunk)
                continue
            
            logger.warning(
                f"Oversized chunk detected: {chunk['char_count']} chars in section '{chunk['section']}' "
                f"(max={max_size}), splitting..."
            )
            
            content = chunk["content"]
            section = chunk["section"]
            source_file = chunk.get("source_file", "")
            chunk_type = chunk.get("chunk_type", "text")
            
            # Try paragraph-based splitting first (preserves semantic boundaries)
            paragraphs = re.split(r'\n\s*\n', content)
            
            # If only one paragraph (no natural boundary), use fixed-size splitting
            if len(paragraphs) <= 1:
                sub_chunks = self._fixed_size_split(content, max_size, section, source_file, chunk_type)
                result.extend(sub_chunks)
                continue
            
            # Accumulate paragraphs into sub-chunks under max_size
            current_content = ""
            sub_idx = 0
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                # If single paragraph exceeds max_size, split it by sentences/lines
                if len(para) > max_size:
                    if current_content:
                        result.append(self._create_chunk(current_content, f"{section} (part {sub_idx+1})", source_file, chunk_type))
                        current_content = ""
                        sub_idx += 1
                    sub_chunks = self._fixed_size_split(para, max_size, f"{section} (part {sub_idx+1})", source_file, chunk_type)
                    result.extend(sub_chunks)
                    sub_idx += len(sub_chunks)
                    continue
                
                if len(current_content) + len(para) + 2 <= max_size:
                    current_content += ("\n\n" if current_content else "") + para
                else:
                    if current_content:
                        result.append(self._create_chunk(current_content, f"{section} (part {sub_idx+1})", source_file, chunk_type))
                        sub_idx += 1
                    current_content = para
            
            if current_content:
                result.append(self._create_chunk(current_content, f"{section} (part {sub_idx+1})", source_file, chunk_type))
        
        return result

    def _fixed_size_split(self, text: str, max_size: int, section: str, source_file: str, chunk_type: str) -> List[Dict[str, Any]]:
        """Split text by fixed size with overlap, as a fallback for unbreakable content."""
        overlap = min(self.config["overlap_size"], max_size // 5)
        result = []
        start = 0
        part_idx = 0
        
        while start < len(text):
            end = min(start + max_size, len(text))
            
            # Try to break at sentence or line boundary near the end
            if end < len(text):
                # Look for sentence boundary within last 20% of the chunk
                boundary_search_start = start + int(max_size * 0.8)
                for boundary_char in ['. ', '.\n', '\n\n', '\n']:
                    boundary_pos = text.rfind(boundary_char, boundary_search_start, end)
                    if boundary_pos > 0:
                        end = boundary_pos + len(boundary_char)
                        break
            
            chunk_content = text[start:end].strip()
            if chunk_content:
                result.append(self._create_chunk(
                    chunk_content,
                    f"{section} (part {part_idx+1})" if part_idx > 0 else section,
                    source_file,
                    chunk_type
                ))
                part_idx += 1
            
            if end >= len(text):
                break
            start = end - overlap
        
        return result

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count for text with Chinese character handling."""
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_tokens = len(re.findall(r'\d+|[%$#@&*+=<>/\\|~^]', text))
        
        return int(english_words * 1.3 + chinese_chars * 1.5 + other_tokens * 0.5 + 2)

    @staticmethod
    def count_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count chunks by type."""
        counts: Dict[str, int] = {}
        for chunk in chunks:
            chunk_type = chunk.get("chunk_type", "text")
            counts[chunk_type] = counts.get(chunk_type, 0) + 1
        return counts

    def evaluate_chunking(self, chunks: List[Dict[str, Any]], llm_token_limit: int = 8000) -> Dict[str, Any]:
        """Evaluate chunking quality and provide optimization report."""
        if not chunks:
            return {"error": "No chunks to evaluate"}
        
        token_counts = [chunk.get("token_count", 0) for chunk in chunks]
        char_counts = [chunk.get("char_count", 0) for chunk in chunks]
        
        total_tokens = sum(token_counts)
        total_chars = sum(char_counts)
        avg_tokens = total_tokens / len(chunks) if chunks else 0
        max_tokens = max(token_counts) if token_counts else 0
        min_tokens = min(token_counts) if token_counts else 0
        
        chunks_per_context = llm_token_limit / avg_tokens if avg_tokens > 0 else 0
        
        oversized = sum(1 for t in token_counts if t > llm_token_limit * 0.8)
        undersized = sum(1 for t in token_counts if t < 100)
        
        optimal_range = (llm_token_limit * 0.1, llm_token_limit * 0.5)
        optimal_chunks = sum(1 for t in token_counts if optimal_range[0] <= t <= optimal_range[1])
        quality_score = optimal_chunks / len(chunks) if chunks else 0
        
        recommendations = []
        if oversized > 0:
            recommendations.append(
                f"{oversized} chunks exceed 80% of LLM limit. "
                f"Consider reducing max_chunk_size from {self.config['max_chunk_size']}"
            )
        if undersized > len(chunks) * 0.3:
            recommendations.append(
                f"{undersized} chunks are very small (<100 tokens). "
                f"Consider increasing min_chunk_size from {self.config['min_chunk_size']}"
            )
        if quality_score < 0.7:
            recommendations.append(
                f"Only {quality_score:.1%} chunks are in optimal size range. "
                f"Consider adjusting chunking strategy."
            )
        if chunks_per_context < 2:
            recommendations.append(
                f"Average chunk size ({avg_tokens:.0f} tokens) is large. "
                f"Only ~{chunks_per_context:.1f} chunks fit in LLM context."
            )
        
        return {
            "total_chunks": len(chunks),
            "total_tokens": total_tokens,
            "total_chars": total_chars,
            "avg_tokens_per_chunk": avg_tokens,
            "max_tokens": max_tokens,
            "min_tokens": min_tokens,
            "chunks_per_llm_context": chunks_per_context,
            "oversized_chunks": oversized,
            "undersized_chunks": undersized,
            "quality_score": quality_score,
            "recommendations": recommendations,
            "chunk_type_distribution": self.count_chunks(chunks),
        }