"""Context Budgeting Skill — 上下文浓缩优化

R38-QA Skill #3: 当上下文超过 token 限制时，智能浓缩上下文，优先保留核心实体和关键关系。
"""

import logging
from typing import Any, Dict, List, Tuple

from ..base import Skill, SkillContext, SkillResult, SkillPriority

logger = logging.getLogger(__name__)


class ContextBudgetingSkill(Skill):
    """Optimize context size when it exceeds token limits.
    
    Strategy:
    1. Calculate current context size
    2. Identify priority elements (core entities, key relationships)
    3. Truncate low-priority content while preserving structure
    4. Inject budget header to guide LLM behavior
    
    Evidence: unified_pipeline_service.py context size handling
    Performance: Reduces token usage while maintaining extraction quality.
    """
    
    # Token estimation: ~4 characters per token for Chinese/English mix
    CHARS_PER_TOKEN = 4
    
    # Priority tiers for content elements
    PRIORITY_HIGH = ["core_entity", "canonical_entity", "key_relation", "relationship"]
    PRIORITY_MEDIUM = ["entity", "attribute", "rule", "constraint"]
    PRIORITY_LOW = ["metadata", "evidence", "source_trace", "lineage"]
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        # Target token budget (default 32k context, reserve 8k for response)
        self.target_tokens = self.config.get("target_tokens", 24000)
        self.reserve_tokens = self.config.get("reserve_tokens", 2000)
        self.enable_compression = self.config.get("enable_compression", True)
        self.compression_ratio = self.config.get("compression_ratio", 0.7)
    
    @property
    def priority(self) -> SkillPriority:
        return SkillPriority.HIGH
    
    @property
    def description(self) -> str:
        return f"Condense context to fit {self.target_tokens} token budget"
    
    async def can_apply(self, context: SkillContext) -> bool:
        """Can apply if document content exists."""
        return context.document_content is not None
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(text) // self.CHARS_PER_TOKEN
    
    def _classify_content(self, content: str) -> str:
        """Classify content type by keywords."""
        content_lower = content.lower()
        
        for keyword in self.PRIORITY_HIGH:
            if keyword in content_lower:
                return "high"
        
        for keyword in self.PRIORITY_MEDIUM:
            if keyword in content_lower:
                return "medium"
        
        return "low"
    
    def _build_budget_header(self, current_tokens: int, target_tokens: int) -> str:
        """Build budget header to guide LLM."""
        return f"""## CONTEXT BUDGET NOTICE
Current context size: ~{current_tokens} tokens
Target budget: ~{target_tokens} tokens
Please focus on extracting HIGH priority elements.
"""
    
    def _compress_entities(self, entities: List[Dict[str, Any]], target_count: int) -> List[Dict[str, Any]]:
        """Compress entity list to target count, preserving priority."""
        if len(entities) <= target_count:
            return entities
        
        # Categorize by priority
        high_priority = [e for e in entities if e.get("priority") in ["high", "core", "canonical"]]
        medium_priority = [e for e in entities if e.get("priority") not in ["high", "core", "canonical", "low"]]
        low_priority = [e for e in entities if e.get("priority") == "low"]
        
        # Calculate distribution
        high_count = min(len(high_priority), target_count // 2)
        medium_count = min(len(medium_priority), target_count - high_count)
        low_count = target_count - high_count - medium_count
        
        # Select entities
        selected = high_priority[:high_count]
        selected.extend(medium_priority[:medium_count])
        selected.extend(low_priority[:low_count])
        
        logger.info(f"Compressed {len(entities)} entities to {len(selected)} (high={high_count}, medium={medium_count}, low={low_count})")
        return selected
    
    def _compress_content(self, content: str, target_tokens: int) -> str:
        """Compress content to target token count."""
        current_tokens = self._estimate_tokens(content)
        
        if current_tokens <= target_tokens:
            return content
        
        # Calculate compression ratio needed
        ratio = target_tokens / current_tokens
        
        # Strategy: truncate from end, preserve structure markers
        lines = content.split("\n")
        
        # Always preserve headers and high-priority lines
        priority_lines = []
        regular_lines = []
        
        for i, line in enumerate(lines):
            priority = self._classify_content(line)
            if priority == "high":
                priority_lines.append((i, line, True))  # (original_index, content, keep)
            else:
                regular_lines.append((i, line, False))
        
        # Calculate how many regular lines to keep
        priority_chars = sum(len(line) for _, line, _ in priority_lines)
        available_chars = (target_tokens * self.CHARS_PER_TOKEN) - priority_chars
        
        if available_chars < 0:
            # Even priority lines exceed budget, truncate priority lines
            truncated_priority = []
            remaining_chars = target_tokens * self.CHARS_PER_TOKEN
            for _, line, _ in priority_lines:
                if len(line) <= remaining_chars:
                    truncated_priority.append(line)
                    remaining_chars -= len(line)
                if remaining_chars <= 0:
                    break
            return "\n".join(truncated_priority)
        
        # Keep all priority lines, proportionally keep regular lines
        regular_chars_per_line = available_chars // max(len(regular_lines), 1)
        
        compressed = [line for _, line, _ in priority_lines]
        for _, line, _ in regular_lines:
            if len(line) <= regular_chars_per_line:
                compressed.append(line)
            elif regular_chars_per_line > 50:
                # Truncate long lines
                compressed.append(line[:int(regular_chars_per_line * self.compression_ratio)] + "...")
        
        result = "\n".join(compressed)
        logger.info(f"Compressed {current_tokens} tokens to {self._estimate_tokens(result)} tokens")
        return result
    
    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute context budgeting optimization."""
        start_tokens = self._estimate_tokens(context.document_content or "")
        effective_budget = self.target_tokens - self.reserve_tokens
        
        # Check if optimization needed
        if start_tokens <= effective_budget:
            return SkillResult(
                success=True,
                skill_name=self.name,
                applied=False,
                output={
                    "start_tokens": start_tokens,
                    "target_tokens": effective_budget,
                    "action": "none",
                    "reason": "context within budget"
                }
            )
        
        # Build optimized context
        optimized_content = context.document_content
        
        if self.enable_compression:
            optimized_content = self._compress_content(optimized_content, effective_budget)
        
        # Inject budget header
        budget_header = self._build_budget_header(start_tokens, effective_budget)
        optimized_content = budget_header + "\n\n" + optimized_content
        
        # Update context
        context.document_content = optimized_content
        
        # Also compress entities if present
        original_entity_count = len(context.entities)
        target_entity_count = int(original_entity_count * self.compression_ratio)
        context.entities = self._compress_entities(context.entities, target_entity_count)
        
        end_tokens = self._estimate_tokens(optimized_content)
        compression_ratio = end_tokens / start_tokens if start_tokens > 0 else 1.0
        
        return SkillResult(
            success=True,
            skill_name=self.name,
            applied=True,
            output={
                "start_tokens": start_tokens,
                "end_tokens": end_tokens,
                "target_tokens": effective_budget,
                "compression_ratio": compression_ratio,
                "entity_compression": f"{original_entity_count} -> {len(context.entities)}",
                "action": "compressed"
            }
        )
