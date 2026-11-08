"""Prompt construction machinery extracted from ``TrueUnifiedExtractor``.

Contains ``DocumentAnalyzer`` and ``PromptStrategy`` (formerly defined inside
``true_unified_extractor.py``) plus the ``PromptService`` facade that bundles
the prompt-cluster methods (2026-08-11).
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from ontology_compiler.services.failure_memory import get_failure_memory
except ImportError:
    def get_failure_memory():
        return None

try:
    from core.schema_registry import get_schema_registry
except ImportError:
    get_schema_registry = None

logger = logging.getLogger(__name__)


class DocumentAnalyzer:
    """
    [PIPELINE-IMPROVEMENT]: Analyzes document content to determine optimal extraction strategy.
    
    Features:
    - Detects document type (technical manual, business process, requirements, etc.)
    - Identifies language complexity
    - Estimates entity density
    - Analyzes document structure (outline, sections, key paragraphs)
    - Recommends optimal extraction parameters
    - Selects content intelligently based on document structure
    """
    
    DOCUMENT_TYPES = {
        "technical": ["api", "specification", "technical", "manual", "guide", "reference", "developer"],
        "business": ["business", "process", "workflow", "scenario", "requirement", "policy"],
        "contract": ["contract", "agreement", "terms", "conditions", "legal"],
        "report": ["report", "analysis", "summary", "review", "assessment"],
        "general": []
    }
    
    OUTLINE_KEYWORDS = {
        "chinese": ["摘要", "概述", "简介", "引言", "前言", "总纲", "目录", "章", "节", "条款", "定义"],
        "english": ["abstract", "summary", "introduction", "overview", "preface", "foreword", 
                   "table of contents", "chapter", "section", "article", "clause", "definition"]
    }
    
    def analyze(self, text: str) -> dict:
        """Analyze document and return analysis result."""
        result = {
            "document_type": self._detect_document_type(text),
            "language_complexity": self._assess_language_complexity(text),
            "entity_density": self._estimate_entity_density(text),
            "recommended_l1_count": None,
            "recommended_l2_count": None,
            "recommended_depth": None,
            "prompt_strategy": None,
            "document_structure": self._analyze_document_structure(text),
            "key_topics": self._identify_key_topics(text),
            "language": self._detect_language(text)
        }
        
        self._set_recommendations(result)
        
        return result
    
    def _analyze_document_structure(self, text: str) -> dict:
        """Analyze document structure to identify outline, sections, and key paragraphs."""
        lines = text.split('\n')
        structure = {
            "has_outline": False,
            "outline_position": 0,
            "outline_length": 0,
            "sections": [],
            "key_paragraphs": [],
            "estimated_pages": len(text) // 1500
        }
        
        outline_lines = []
        in_outline = False
        outline_start = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            if not line_stripped:
                continue
            
            if self._is_outline_section(line_stripped):
                if not in_outline:
                    in_outline = True
                    outline_start = i
                outline_lines.append(line_stripped)
            elif in_outline and len(outline_lines) > 2 and len(line_stripped) > 50:
                in_outline = False
                
            if self._is_section_header(line_stripped):
                structure["sections"].append({
                    "position": i,
                    "header": line_stripped[:100],
                    "level": self._get_section_level(line_stripped)
                })
                
            if self._is_key_paragraph(line_stripped):
                structure["key_paragraphs"].append(i)
        
        if outline_lines:
            structure["has_outline"] = True
            structure["outline_position"] = outline_start
            structure["outline_length"] = len(outline_lines)
        
        return structure
    
    def _is_outline_section(self, line: str) -> bool:
        """Check if line is part of document outline."""
        line_lower = line.lower()
        for keywords in self.OUTLINE_KEYWORDS.values():
            if any(kw.lower() in line_lower for kw in keywords):
                return True
        
        if re.match(r'^\s*\d+(\.\d+)*\s+', line):
            return True
        
        return False
    
    def _is_section_header(self, line: str) -> bool:
        """Check if line is a section header."""
        if re.match(r'^#{1,3}\s+', line):
            return True
        
        if re.match(r'^\d+(\.\d+)*\s+[A-Z]', line):
            return True
        
        if len(line) < 80 and line.isupper():
            return True
        
        return False
    
    def _get_section_level(self, line: str) -> int:
        """Determine section level."""
        if line.startswith('###'):
            return 3
        elif line.startswith('##'):
            return 2
        elif line.startswith('#'):
            return 1
        
        parts = line.split('.')
        if len(parts) <= 2:
            return 1
        elif len(parts) <= 3:
            return 2
        else:
            return 3
    
    def _is_key_paragraph(self, line: str) -> bool:
        """Check if paragraph is key content."""
        line_lower = line.lower()
        key_phrases = ["in summary", "in conclusion", "it is important", "the key", 
                       "main objective", "primary goal", "core requirement", "critical"]
        return any(phrase in line_lower for phrase in key_phrases)
    
    DOMAIN_TOPICS = {
        "diagnostic": ["diagnostic", "diagnostics", "fault", "failure", "troubleshooting", 
                      "root cause", "error code", "diagnosis", "defect", "malfunction"],
        "education": ["education", "training", "learning", "academy", "course", 
                      "curriculum", "workshop", "seminar", "tutorial", "lesson"],
        "supply_chain": ["supply", "logistics", "inventory", "warehouse", "shipment", 
                        "procurement", "distribution", "delivery", "vendor", "supplier"],
        "performance": ["performance", "kpi", "metric", "indicator", "efficiency", 
                       "optimization", "benchmark", "score"],
        "manufacturing": ["manufacturing", "production", "assembly", "factory", 
                         "assembly line", "mbom", "bill of materials"],
    }
    
    def _identify_key_topics(self, text: str) -> list:
        """Identify key topics from document with domain awareness."""
        import re
        
        topics = []
        text_lower = text.lower()
        
        uppercase_words = re.findall(r'\b([A-Z]{2,})\b', text)
        topics.extend(uppercase_words[:10])
        
        camel_case = re.findall(r'\b([A-Z][a-z]+[A-Z][a-z]+)\b', text)
        topics.extend(camel_case[:10])
        
        for topic_group, keywords in self.DOMAIN_TOPICS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    topics.append(topic_group)
                    break
        
        return list(set(topics))[:15]
    
    def _detect_language(self, text: str) -> str:
        """Detect document language (simplified)."""
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        if chinese_chars > len(text) * 0.1:
            return "chinese"
        return "english"
    
    def _detect_document_type(self, text: str) -> str:
        """Detect document type based on keywords."""
        text_lower = text.lower()
        
        for doc_type, keywords in self.DOCUMENT_TYPES.items():
            if any(keyword in text_lower for keyword in keywords):
                return doc_type
        
        # Heuristic: check for code-like content
        if "{" in text and "}" in text and "function" in text_lower:
            return "technical"
        
        return "general"
    
    def _assess_language_complexity(self, text: str) -> str:
        """Assess language complexity level."""
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        if avg_word_length > 6:
            return "high"
        elif avg_word_length > 4:
            return "medium"
        else:
            return "low"
    
    def _estimate_entity_density(self, text: str) -> str:
        """Estimate entity density based on proper nouns and capitalization."""
        words = text.split()
        capitalized_words = [w for w in words if w.istitle()]
        
        if len(capitalized_words) / len(words) > 0.15:
            return "high"
        elif len(capitalized_words) / len(words) > 0.08:
            return "medium"
        else:
            return "low"
    
    def _set_recommendations(self, result: dict):
        """Set extraction recommendations based on analysis."""
        doc_type = result["document_type"]
        density = result["entity_density"]
        complexity = result["language_complexity"]
        
        # Base recommendations on document type
        # KEY PRINCIPLE: L2 should ALWAYS have MORE entities than L1 (2-3x more)
        if doc_type == "technical":
            result["recommended_l1_count"] = 5-10  # Abstract subtypes
            result["recommended_l2_count"] = 15-25  # Specific systems, processes - MAIN LAYER
            result["recommended_depth"] = "detailed"
            result["prompt_strategy"] = "technical"
        elif doc_type == "business":
            result["recommended_l1_count"] = 5-8   # Abstract subtypes
            result["recommended_l2_count"] = 12-20  # Business scenarios - MAIN LAYER
            result["recommended_depth"] = "comprehensive"
            result["prompt_strategy"] = "business"
        elif doc_type == "contract":
            result["recommended_l1_count"] = 4-6   # Abstract subtypes
            result["recommended_l2_count"] = 8-15  # Specific obligations, rules - MAIN LAYER
            result["recommended_depth"] = "legal"
            result["prompt_strategy"] = "contract"
        else:
            result["recommended_l1_count"] = 4-8   # Abstract subtypes
            result["recommended_l2_count"] = 10-18 # Specific entities - MAIN LAYER
            result["recommended_depth"] = "standard"
            result["prompt_strategy"] = "general"
        
        # Adjust based on entity density
        if density == "high":
            result["recommended_l1_count"] = result["recommended_l1_count"] + 2
            result["recommended_l2_count"] = result["recommended_l2_count"] + 2
        elif density == "low":
            result["recommended_l1_count"] = max(3, result["recommended_l1_count"] - 2)
            result["recommended_l2_count"] = max(2, result["recommended_l2_count"] - 2)
    
    def select_content_intelligently(self, text: str, max_length: int = 8000) -> str:
        """
        Select content intelligently based on document structure.
        
        Priority:
        1. Full outline/introduction/summary (critical for understanding context)
        2. Key paragraphs (important statements, conclusions)
        3. First paragraph of each main section (representative content)
        4. Remaining content from key sections
        
        Args:
            text: Full document text
            max_length: Maximum length of selected content
            
        Returns:
            Optimized text with key content preserved
        """
        if len(text) <= max_length:
            return text
            
        lines = text.split('\n')
        selected_lines = []
        selected_indices = set()
        
        structure = self._analyze_document_structure(text)
        language = self._detect_language(text)
        
        outline_keywords = self.OUTLINE_KEYWORDS.get(language, self.OUTLINE_KEYWORDS["english"])
        
        outline_start = structure["outline_position"]
        outline_end = outline_start + structure["outline_length"]
        
        # Priority 1: Full outline and introduction section (preserve completely)
        for i in range(min(outline_end, len(lines))):
            if i not in selected_indices:
                selected_lines.append(lines[i])
                selected_indices.add(i)
        
        # Priority 2: Key paragraphs
        for para_idx in structure["key_paragraphs"]:
            if para_idx not in selected_indices and len('\n'.join(selected_lines)) < max_length * 0.5:
                selected_lines.append(lines[para_idx])
                selected_indices.add(para_idx)
        
        # Priority 3: First paragraph of each section
        for section in structure["sections"]:
            sec_pos = section["position"]
            if sec_pos not in selected_indices:
                selected_lines.append(lines[sec_pos])
                selected_indices.add(sec_pos)
                
                # Add 1-2 following paragraphs for context
                for offset in range(1, 3):
                    next_pos = sec_pos + offset
                    if next_pos < len(lines) and next_pos not in selected_indices:
                        if lines[next_pos].strip() and not self._is_section_header(lines[next_pos].strip()):
                            selected_lines.append(lines[next_pos])
                            selected_indices.add(next_pos)
        
        # Priority 4: Sample content from remaining sections (avoid redundancy)
        current_length = len('\n'.join(selected_lines))
        if current_length < max_length:
            remaining_lines = [
                lines[i] for i in range(len(lines)) 
                if i not in selected_indices and lines[i].strip()
            ]
            
            chars_needed = max_length - current_length
            chars_added = 0
            
            for line in remaining_lines:
                if chars_added >= chars_needed:
                    break
                selected_lines.append(line)
                chars_added += len(line)
        
        result = '\n'.join(selected_lines)
        
        if len(result) > max_length:
            result = result[:max_length]
            
        return result


class PromptStrategy:
    """
    [OPTIMIZATION]: Three-layer prompt design implementation with size optimization.
    
    This strategy now composes prompts from smaller reusable fragments:
    - base
    - domain
    - task
    - schema_hint
    - document
    """
    
    MAX_CONTEXT_CHARS = 500
    MAX_SKILL_CHARS = 400
    TOP_K_ENTITIES = 3
    
    EXTRACTION_JSON_SCHEMA = """{"L0_entities":[{"meta_id":"","meta_class":"","name":"","description":"","evidence":"","attributes":[]}],"L1_entities":[{"meta_id":"","meta_class":"","name":"","description":"","parent_class":"","evidence":"","attributes":[]}],"L2_entities":[{"meta_id":"","meta_class":"","name":"","description":"","parent_id":"","evidence":"","attributes":[]}],"L2_5_entities":[{"meta_id":"","meta_class":"","name":"","description":"","evidence":"","attributes":[]}],"L3_entities":[{"meta_id":"","meta_class":"","name":"","description":"","parent_id":"","evidence":"","attributes":[]}],"relationships":[{"from":"","to":"","from_name":"","to_name":"","type":"","evidence_ref":""}]}"""
    ONTOLOGY_PACKET_JSON_SCHEMA = """{"document":{"name":""},"layer_scope":"spec","entities":[{"id":"","code":"","name":"","layer":"L0","layer_scope":"spec","type":"","sub_type":"","status":"","parent_id":"","description":"","domain":"ENTERPRISE","source_file":"","source_span":"","evidence_ref":"","version":"","confidence":1.0,"attributes":{},"aliases":[],"tags":[],"relations_summary":[]}],"relations":[{"source_id":"","target_id":"","relation_type":"","layer_scope":"spec","confidence":1.0,"evidence_ref":"","effective_from":"","effective_to":"","provenance":"llm","description":"","attributes":{}}],"evidence":[],"provenance":{"origin":""},"quality":{},"warnings":[],"assumptions":[]}"""
    
    def __init__(self):
        try:
            from .prompt_loader import get_prompt_loader
            self._prompt_loader = get_prompt_loader()
        except ImportError:
            self._prompt_loader = None
    
    def _load_l0_classification_rules(self) -> str:
        """Load L0 classification rules from external config file.
        
        Returns compact prompt fragment for L0 classification guidance.
        Falls back to inline rules if config file not available.
        """
        try:
            import yaml
            from pathlib import Path
            config_path = Path(__file__).parent.parent.parent.parent / "config" / "ontology" / "l0_classification_rules.yaml"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                return config.get("prompt_fragment", "")
        except Exception:
            pass
        return ""
    
    def build_base_fragment(self, domain: Optional[str] = None) -> str:
        domain_hint = f" for {domain}" if domain else ""
        
        try:
            from core.schema_registry import get_schema_registry
            registry = get_schema_registry()
            main_classes = registry.main_classes
            class_list = ", ".join(main_classes)
        except ImportError:
            main_classes = ["Capability", "System", "Actor", "Asset", "Data", "Process", "Concept"]
            class_list = ", ".join(main_classes)
        
        l0_rule_fragment = self._load_l0_classification_rules().format(class_list=class_list)
        
        return f"""Extract entities, relationships, and attributes{domain_hint}.

RULES:

{l0_rule_fragment}

CRITICAL OUTPUT CONSTRAINTS:
- Output ONLY valid JSON: start with `{{` and end with `}}`. No reasoning, commentary, markdown fences, or prefix text.
- Do NOT wrap output in code fences (no ```json). If you need to think, do it silently.
- Response must contain JSON and nothing else.
- `attributes` in `L1_entities` MUST be an array of objects, never a string array.
- Each attribute object must follow `{{"name":"...","value":"..."}}`.
- Every relationship MUST include `evidence_ref`: a short citation of the source section supporting the link (e.g. "doc:slide-5", "doc:§3.2", or a page/paragraph reference). Never omit it.

RELATION DISCIPLINE (high priority):
- Quality over quantity: extract ONLY relationships explicitly supported by the document text. NEVER invent links to inflate the count.
- Whitelist first: prefer registered relation types (STRUCTURAL / REALIZATION / GOVERNANCE / DEPENDENCY_TRACE / CONFIG_EFFECTIVITY / CHANGE_FLOW families). Do NOT invent ad-hoc relation names.
- Prefer structural (PART_OF, HAS_PART, CONTAINS) and dependency (DEPENDS_ON, USES) relations when the text supports them; use generic/semantic types only when specific evidence exists.
- `RELATED_TO` (or `ASSOCIATED_WITH`) is a LAST-RESORT fallback ONLY when no specific type is justified. Never use it as a default for weakly supported links.
- If two entities are only weakly or not actually connected by the text, DO NOT create a relationship.

CONCISE OUTPUT (save tokens):
- descriptions: max 2 sentences.
- evidence content: max 50 chars, quote key phrase only.
- Omit empty attributes and empty optional fields.
- Use short meta_id values (e.g. L1-001, L2-002).

ATTRIBUTE EXAMPLES:
Valid:
`"attributes": [{{"name": "scope", "value": "inventory control"}}, {{"name": "kpi", "value": "stock accuracy"}}]`
Invalid:
`"attributes": ["scope", "kpi"]`

OUTPUT SCHEMA (respond with JSON matching this structure):

{self._select_output_schema()}"""
    
    @classmethod
    def _select_output_schema(cls) -> str:
        """Select the extraction output schema template.

        Default: legacy task-oriented schema (current mainline).
        Set EXTRACT_OUTPUT_PROTOCOL=packet to emit the OntologyPacket shape
        (schemas/ontology_packet.schema.json). Dual-template coexistence per
        docs/architectures/ENTITY_CLASSIFICATION_ROADMAP.md T1.2; the packet
        output is adapted downstream by extraction_packet_adapter.
        """
        import os
        if os.environ.get("EXTRACT_OUTPUT_PROTOCOL", "").strip().lower() == "packet":
            return cls.ONTOLOGY_PACKET_JSON_SCHEMA
        return cls.EXTRACTION_JSON_SCHEMA
    
    def build_domain_fragment(self, domain: Optional[str], skill_section: str = "", context_section: str = "") -> str:
        compressed_context = self._compress_context(context_section)
        compressed_skills = self._compress_skills(skill_section)
        parts = []
        if compressed_skills:
            parts.append(f"# Skills\n{compressed_skills}")
        if compressed_context:
            parts.append(compressed_context)
        return "\n\n".join(parts)
    
    def build_task_fragment(self, schema_hints: str = "") -> str:
        if not schema_hints:
            return ""
        return f"# Schema Hints\n{schema_hints.strip()}"
    
    def generate_extraction_prompt(self, text: str, domain: Optional[str] = None, 
                                   context_section: str = "", skill_section: str = "",
                                   schema_hints: str = "") -> str:
        base = self.build_base_fragment(domain)
        domain_fragment = self.build_domain_fragment(domain, skill_section, context_section)
        task_fragment = self.build_task_fragment(schema_hints)
        if self._prompt_loader:
            return self._prompt_loader.render_composed(
                base=base,
                domain=domain_fragment,
                task=task_fragment,
                schema_hint=None,
                document=f"## DOCUMENT:\n{text}",
            )
        parts = [base]
        if domain_fragment:
            parts.append(domain_fragment)
        if task_fragment:
            parts.append(task_fragment)
        parts.append(f"## DOCUMENT:\n{text}")
        return "\n\n".join(parts)
    
    def generate_normalization_prompt(self, entities_json: str, domain: Optional[str] = None) -> str:
        """
        Generate normalization prompt for post-extraction processing.
        
        Handles:
        - Naming standardization (Title Case, abbreviations)
        - Hierarchy validation (parent_class/parent_id must exist)
        - Terminology consistency
        - Status normalization
        """
        domain_hint = f" for {domain}" if domain else ""
        
        try:
            from core.schema_registry import get_schema_registry
            registry = get_schema_registry()
            main_classes = registry.main_classes
            class_list = ", ".join(main_classes)
        except ImportError:
            class_list = "Capability, System, Actor, Asset, Data, Process, Concept, BusinessScenario, MappingRule, BusinessRule, ValidationRule"
        
        return f"""Normalize entities{domain_hint}.

RULES:

OUTPUT: ONLY VALID JSON (same structure)

{entities_json}"""
    
    def generate_governance_prompt(self, entities_json: str, conflicts: list = None, 
                                   domain: Optional[str] = None) -> str:
        """
        Generate governance prompt for conflict resolution and quality improvement.
        
        Handles:
        - Entity conflicts (same name, different definitions)
        - Relationship conflicts
        - Low-confidence entities
        - Duplicate detection
        """
        domain_hint = f" for {domain}" if domain else ""
        conflict_section = ""
        if conflicts:
            conflict_section = f"\n## CONFLICTS:\n{json.dumps(conflicts[:5])}\n"
        
        return f"""Resolve conflicts{domain_hint}.

RULES:

{conflict_section}

OUTPUT: ONLY VALID JSON (same structure)

{entities_json}"""
    
    def _compress_context(self, context_section: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
        """[Optimization] Compress context to TOP_K_ENTITIES only."""
        if len(context_section) <= max_chars:
            return context_section
        
        lines = context_section.split("\n")
        filtered_lines = []
        entity_count = 0
        
        for line in lines:
            if "Note:" in line:
                filtered_lines.append(line)
                continue
            
            if entity_count >= self.TOP_K_ENTITIES:
                continue
            
            if ":" in line and ("L0" in line or "L1" in line or "L2" in line):
                filtered_lines.append(line)
                entity_count += 1
        
        return "\n".join(filtered_lines) + "\n" if filtered_lines else ""
    
    def _compress_skills(self, skill_section: str, max_chars: int = MAX_SKILL_CHARS) -> str:
        """[Optimization] Compress skills to trigger keywords only."""
        if len(skill_section) <= max_chars:
            return skill_section
        
        lines = skill_section.split("\n")
        filtered_lines = []
        
        for line in lines:
            if "## Skill:" in line or "- Keywords:" in line:
                if len("\n".join(filtered_lines)) + len(line) + 1 <= max_chars:
                    filtered_lines.append(line)
        
        if not filtered_lines:
            return ""
        
        return "\n".join(filtered_lines) + "\n"
    
    def generate_prompt(self, analysis: dict, text: str, domain: Optional[str] = None, 
                       context_section: str = "", skill_section: str = "",
                       schema_hint_section: str = "", experience_section: str = "") -> str:
        """Backward-compatible wrapper - delegates to generate_extraction_prompt."""
        return self.generate_extraction_prompt(text, domain, context_section, skill_section)


class PromptService:
    """Facade bundling the prompt-cluster methods of ``TrueUnifiedExtractor``.

    Holds the collaborators that the cluster used to reach through ``self``
    (document analyzer, context loader, prompt strategy/loader, skill loader).
    """

    def __init__(
        self, document_analyzer, context_loader, prompt_strategy, prompt_loader, skill_loader,
        budget_policy=None,
    ) -> None:
        self.document_analyzer = document_analyzer
        self.context_loader = context_loader
        self.prompt_strategy = prompt_strategy
        self.prompt_loader = prompt_loader
        self.skill_loader = skill_loader
        # C-2: optional PromptBudgetPolicy for prompt-length and text-limit
        # derivation. Defaults to None to preserve existing behavior.
        self._budget_policy = budget_policy

    def _budget_max_prompt_chars(self) -> Optional[int]:
        """Return the prompt-char ceiling from the budget policy, or None.

        C-2: the policy's ``max_prompt_tokens`` is a token ceiling, but the
        prompt service operates on character counts. We use a conservative
        chars-per-token ratio of 4 (favoring ASCII) to convert, capped at the
        existing env default of 12000. This keeps the prompt budget-aware
        without abruptly shrinking prompts for CJK-heavy extractions.
        """
        if self._budget_policy is None:
            return None
        try:
            budget = self._budget_policy.resolve(task="extraction_technical")
        except Exception:
            return None
        env_default = int(os.environ.get("PROMPT_MAX_LENGTH", "12000"))
        # Conservative chars-per-token estimate (4 = whitespace-light ASCII).
        estimated_chars = int(budget.max_prompt_tokens * 4)
        # Clamp to env default so policy only shrinks, never enlarges.
        return min(estimated_chars, env_default)

    def get_dynamic_text_limit(self, text_length: int) -> int:
        """根据文档大小动态计算截断长度，优化prompt大小。

        [OPTIMIZATION] 大幅降低文本限制，确保总prompt不超过10K字符。
        分档策略（5档）:
        - <2K:  完整保留
        - 2-5K: 3K
        - 5-10K: 4K
        - 10-25K: 5K
        - >25K:  5K

        C-2: when a budget policy is attached, the ceiling is derived from
        the extraction profile's ``layers.document`` budget (chars) so the
        text limit tracks the YAML instead of the hardcoded ladder.
        """
        if text_length < 2000:
            return text_length
        elif text_length < 5000:
            return 3000
        elif text_length < 10000:
            return 4000
        elif text_length < 25000:
            return 5000
        else:
            return 5000

    def _budget_document_char_budget(self) -> Optional[int]:
        """Derive the document-layer char budget from the policy, if any.

        Uses ``extraction_technical``'s ``layers.document`` token budget and
        a chars-per-token ratio of 1.5 (CJK-aware) to compute the char ceiling.
        """
        if self._budget_policy is None:
            return None
        try:
            budget = self._budget_policy.resolve(task="extraction_technical")
        except Exception:
            return None
        tasks_cfg = getattr(self._budget_policy, "tasks", {}) or {}
        profile_cfg = tasks_cfg.get(budget.profile) or {}
        layers = profile_cfg.get("layers") or {}
        if isinstance(layers, dict) and layers.get("document"):
            doc_tokens = int(layers["document"])
            return int(doc_tokens * 1.5)
        return None

    def _resolve_text_limit(self, text_length: int) -> Optional[int]:
        """Return the budget-derived text limit, or None to use the ladder."""
        doc_budget = self._budget_document_char_budget()
        if doc_budget is None:
            return None
        if text_length < doc_budget:
            return text_length
        return doc_budget

    def build_unified_prompt(
        self,
        text: str,
        domain: Optional[str] = None,
        use_context: bool = True,
        skip_truncate: bool = False,
        document_metadata: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build the unified extraction prompt via layered composition.

        ``metrics`` (optional): when provided, truncation/compression outcomes
        are recorded in-place (``truncation_reason`` / ``dropped_chars`` /
        ``coverage_ratio``) for D-5 observability, without changing the str
        return contract.
        """
        MAX_PROMPT_LENGTH = int(os.environ.get("PROMPT_MAX_LENGTH", "12000"))
        # C-2: when a budget policy is attached, the prompt-char ceiling is
        # derived from the extraction profile so the prompt service tracks
        # the YAML budget instead of relying solely on the env default.
        _budget_chars = self._budget_max_prompt_chars()
        if _budget_chars is not None and _budget_chars > 0:
            MAX_PROMPT_LENGTH = _budget_chars
        original_len = len(text)

        if skip_truncate:
            selected_text = text
        else:
            # C-2: prefer the budget-derived text limit when available.
            _budget_text_limit = self._resolve_text_limit(len(text))
            if _budget_text_limit is not None:
                max_text_len = _budget_text_limit
            else:
                max_text_len = self.get_dynamic_text_limit(len(text))
            if len(text) > max_text_len:
                selected_text = self.document_analyzer.select_content_intelligently(text, max_text_len)
                logger.info(f"[Extractor] Smart content selection: {len(text)} chars -> {len(selected_text)} chars")
            else:
                selected_text = text

        truncated_text = selected_text
        if not skip_truncate and len(text) > len(selected_text):
            truncated_text += f"\n[Truncated: {len(text)} chars -> {len(selected_text)} chars]\n"

        schema_hints = self.build_schema_hint_section()

        context_section = ""
        if use_context and self.context_loader:
            try:
                # [IMPROVEMENT] Load comprehensive ontology context for better classification
                # Include: L0 base classes, L1 abstract subtypes, and similar entities
                context_lines = ["# Existing Ontology Context"]
                
                # Load L0 base classes as classification reference
                try:
                    master_context = self.context_loader.load_master_ontology()
                    
                    # Add L0 base classes (for classification reference)
                    l0_classes = master_context.l1_entities  # L0 entities are stored as l1 in loader
                    if not l0_classes:
                        # Try loading directly from master file
                        l0_classes = []
                    
                    # Add L1 abstract subtypes (core concepts)
                    l1_concepts = master_context.l1_entities[:10]
                    if l1_concepts:
                        context_lines.append("")
                        context_lines.append("## L1 Core Concepts (Abstract Subtypes)")
                        for entity in l1_concepts:
                            name = entity.get("name", "")
                            meta_id = entity.get("meta_id", "")
                            meta_class = entity.get("meta_class", "")
                            domain = entity.get("domain", "")
                            context_lines.append(f"- [{meta_id}] {name} ({meta_class}, domain={domain})")
                    
                    # Add similar entities based on relevance
                    similar_entities = self.context_loader.find_similar_entities(text, top_k=5)
                    if similar_entities:
                        context_lines.append("")
                        context_lines.append("## Related Entities (Relevance-based)")
                        for entity in similar_entities:
                            context_lines.append(f"- [{entity['meta_id']}] {entity['name']} ({entity['meta_class']}, {entity.get('domain', '')}, similarity={entity.get('similarity', 0)})")
                    
                    context_lines.append("")
                    context_lines.append("### Instructions:")
                    context_lines.append("- Use existing L1 concepts as parent classes for new entities")
                    context_lines.append("- Link new L2 entities to existing L1 concepts via parent_id")
                    context_lines.append("- Avoid extracting duplicate entities")
                    context_lines.append("- Identify relationships to existing entities")
                    
                    context_section = "\n".join(context_lines) + "\n"
                except Exception as e:
                    logger.warning(f"Failed to load detailed ontology context: {e}")
                    # Fallback to simple format
                    context_section = self.context_loader.format_for_prompt(max_entities=10)
            except Exception as e:
                logger.warning(f"Failed to load ontology context: {e}")
                context_section = ""

        skill_section = self.build_skill_section(text, domain)
        metadata_lines = ["## METADATA"]
        skill_lines = ["## SKILL CONSTRAINTS"]
        agent_lines = ["## AGENT PLAN"]
        if document_metadata:
            for key in ("source_path", "file_suffix", "page_hint", "document_type", "document_type_confidence", "secondary_types"):
                val = document_metadata.get(key)
                if val not in (None, "", [], {}):
                    metadata_lines.append(f"- {key}: {val}")
            matched_skills = document_metadata.get("matched_skills") or []
            if matched_skills:
                metadata_lines.append("- matched_skills:")
                for skill in matched_skills[:8]:
                    if isinstance(skill, dict):
                        sid = skill.get('skill_id', '')
                        sname = skill.get('skill_name', '')
                        metadata_lines.append(f"  - {sid}: {sname}")
                    else:
                        metadata_lines.append(f"  - {skill}")
                doc_type = str(document_metadata.get("document_type") or "").lower()
                if doc_type == "mbom":
                    skill_lines.extend([
                        "- Prioritize delivery unit, occurrence, position, variant, source structure, and provenance.",
                        "- Preserve MBOM-specific names and avoid collapsing them into generic BOM labels.",
                        "- Emit relationships that preserve variant and position semantics.",
                        "- Treat occurrence, source_structure, alignment, position, and variant as mandatory preservation targets.",
                    ])
                elif doc_type == "benchmark":
                    skill_lines.extend([
                        "- Prioritize benchmark questions, scope, KPI, strategy, tools, exclusions, and dimensions.",
                        "- Preserve question structure rather than flattening into generic entities.",
                    ])
                elif doc_type == "process_governance":
                    skill_lines.extend([
                        "- Prioritize rules, owners, gates, workflow steps, and exceptions.",
                        "- Preserve governance chains and audit references.",
                    ])
        agent_lines.extend([
            "- Structure agent: preserve pages/slides, titles, tables, bullet hierarchy, and evidence spans.",
            "- Semantic agent: extract canonical entities and relations while respecting skill constraints.",
            "- Review agent: flag missing mandatory signals and over-normalization.",
            "- MBOM emphasis: preserve occurrence, source structure, alignment, position, and variant as first-class signals.",
        ])
        base = self.prompt_strategy.build_base_fragment(domain)
        domain_fragment = self.prompt_strategy.build_domain_fragment(domain, "\n".join(metadata_lines + [""] + skill_lines + [""] + agent_lines), context_section)
        task_fragment = self.prompt_strategy.build_task_fragment(schema_hints)
        document_fragment = f"## DOCUMENT:\n{truncated_text}"

        if self.prompt_loader:
            prompt = self.prompt_loader.render_composed(
                base=base,
                domain=domain_fragment,
                task=task_fragment,
                schema_hint=None,
                document=document_fragment,
            )
        else:
            parts = [base]
            if domain_fragment:
                parts.append(domain_fragment)
            if task_fragment:
                parts.append(task_fragment)
            parts.append(document_fragment)
            prompt = "\n\n".join(parts)

        if len(prompt) > MAX_PROMPT_LENGTH:
            logger.warning(f"[Extractor] Prompt too long ({len(prompt)} chars), applying compression")
            pre_compress_len = len(prompt)
            prompt = self.compress_prompt_to_limit(prompt, MAX_PROMPT_LENGTH)
            logger.info(f"[Extractor] Prompt compressed to {len(prompt)} chars")
            if isinstance(metrics, dict):
                metrics["truncation_reason"] = "compressed_to_limit"
                metrics["dropped_chars"] = pre_compress_len - len(prompt)
                metrics["coverage_ratio"] = (len(prompt) / original_len) if original_len else 1.0
                metrics["prompt_len_after"] = len(prompt)
        elif isinstance(metrics, dict) and original_len > len(selected_text):
            # Smart content selection truncated the document before composition.
            metrics["truncation_reason"] = "smart_content_selection"
            metrics["dropped_chars"] = original_len - len(selected_text)
            metrics["coverage_ratio"] = (len(selected_text) / original_len) if original_len else 1.0
            metrics["prompt_len_after"] = len(prompt)
        elif isinstance(metrics, dict):
            metrics["truncation_reason"] = "none"
            metrics["dropped_chars"] = 0
            metrics["coverage_ratio"] = 1.0
            metrics["prompt_len_after"] = len(prompt)

        return prompt


    def compress_prompt_to_limit(self, prompt: str, max_length: int) -> str:
        """[CRITICAL OPTIMIZATION] Compress prompt to fit within max_length.
        
        Priority: 
        1. Remove skill_section (non-critical)
        2. Remove context_section (can be regenerated)
        3. Remove experience_section
        4. Truncate document content from the end
        """
        if len(prompt) <= max_length:
            return prompt
        
        lines = prompt.split('\n')
        
        doc_start_idx = None
        skill_start_idx = None
        context_start_idx = None
        experience_start_idx = None
        
        for i, line in enumerate(lines):
            if line.startswith("## DOCUMENT:"):
                doc_start_idx = i
            elif line.startswith("# Related Entities"):
                context_start_idx = i
            elif line.startswith("# Skills"):
                skill_start_idx = i
            elif line.startswith("# Previous Extraction Experience"):
                experience_start_idx = i
        
        new_lines = []
        for i, line in enumerate(lines):
            if skill_start_idx and i >= skill_start_idx and (doc_start_idx is None or i < doc_start_idx):
                continue
            if context_start_idx and i >= context_start_idx and (doc_start_idx is None or i < doc_start_idx):
                continue
            if experience_start_idx and i >= experience_start_idx and (doc_start_idx is None or i < doc_start_idx):
                continue
            new_lines.append(line)
        
        prompt = '\n'.join(new_lines)
        
        if len(prompt) <= max_length:
            return prompt
        
        lines = prompt.split('\n')
        doc_start_idx = None
        
        for i, line in enumerate(lines):
            if line.startswith("## DOCUMENT:"):
                doc_start_idx = i
                break
        
        if doc_start_idx is None:
            return prompt[:max_length]
        
        header = '\n'.join(lines[:doc_start_idx + 1])
        remaining = max_length - len(header) - 30
        content = '\n'.join(lines[doc_start_idx + 1:])
        
        if remaining > 0:
            content = content[:remaining] + "\n[Truncated]"
            return header + '\n' + content
        
        return header[:max_length]


    def build_experience_section(self, domain: Optional[str] = None) -> str:
        """Build experience section from failure memory recommendations.
        
        [P3-EXP-2]: Inject lessons learned from previous failures into prompt
        to improve extraction quality for recurring patterns.
        
        Provides specific prompt adjustments based on failure patterns:
        - JSON parsing failures → stricter format requirements
        - Validation failures → more explicit class definitions
        - Timeout failures → content simplification
        - Extraction failures → alternative strategies
        """
        try:
            memory = get_failure_memory()
            recommendations = memory.get_recommendations(
                task_type="extraction",
                domain=domain or "SUPPLY_CHAIN"
            )
            
            if not recommendations:
                return "# Previous extraction experience: No recurring patterns found.\n"
            
            experience_lines = ["# Previous Extraction Experience (Lessons Learned)\n"]
            format_adjustments = []
            
            for rec in recommendations:
                ft = rec.get("failure_type", "")
                count = rec.get("occurrence_count", 0)
                strategy = rec.get("recommended_strategy", "")
                rate = rec.get("recovery_rate", 0.0)
                severity = rec.get("severity", "medium")
                
                experience_lines.append(
                    f"- **{ft}** (x{count}, recovery: {rate:.0%}, {severity}): "
                    f"{strategy if strategy else 'Monitor closely'}"
                )
                
                if severity in ["high", "critical"] and count >= 2:
                    format_adjustments.append(self.map_failure_to_prompt_adjustment(ft))
            
            if format_adjustments:
                experience_lines.append("\n# FORMAT ADJUSTMENTS (Critical Learnings)\n")
                for adj in format_adjustments:
                    experience_lines.append(f"- {adj}")
            
            return "\n".join(experience_lines) + "\n"
        
        except Exception as e:
            logger.warning(f"Failed to build experience section: {e}")
            return "# Previous extraction experience: Not available.\n"


    def map_failure_to_prompt_adjustment(self, failure_type: str) -> str:
        """Map failure type to specific prompt adjustment instruction.
        
        This translates failure patterns into concrete instructions for the LLM
        to improve output quality.
        """
        adjustment_map = {
            "json_parse_error": "CRITICAL: Always output VALID JSON only. Use strict formatting. Double-check closing braces. Output ONLY the JSON object, no extra text.",
            "validation_failed": "CRITICAL: Ensure ALL entities have required fields: meta_id, meta_class (one of: Capability, System, Actor, Asset, Data, Process), name, description, domain, status. Relationships need: from, to, type.",
            "llm_timeout": "CRITICAL: Be concise. Limit extraction to 5-8 L1 entities and 3-5 L2 scenarios maximum. Focus on most important items only.",
            "extraction_failed": "CRITICAL: If you cannot extract any entities, return empty arrays instead of failing. Use simple, clear naming conventions.",
            "schema_violation": "CRITICAL: Use ONLY valid meta_class values from the schema. Check entity types against the 'Valid Types' section above.",
            "relationship_validation_failed": "CRITICAL: Ensure relationship endpoints (from/to) match actual entity meta_ids in your output.",
            "empty_response": "CRITICAL: Always return a valid JSON object even if no entities are found. Never return null or empty string.",
        }
        
        return adjustment_map.get(failure_type, f"CAUTION: Previous {failure_type} failures detected. Be thorough in your output.")


    def build_schema_hint_section(self) -> str:
        """Build Schema validation hints section to guide LLM output.
        
        [CRITICAL OPTIMIZATION] Keep this section SHORT (<500 chars) to avoid consuming
        valuable prompt space. The EXTRACTION_JSON_SCHEMA already contains type hints.
        """
        try:
            if not get_schema_registry:
                return ""
            
            registry = get_schema_registry()
            if not registry:
                return ""
            
            main_classes = getattr(registry, "main_classes", []) or []
            rel_types = getattr(registry, "relation_types", []) or []
            
            entity_hint = ", ".join(main_classes[:6])
            relation_hint = ", ".join(rel_types[:8])
            
            return (
                "# SCHEMA HINTS\n"
                f"Entities: {entity_hint}\n"
                f"Relations: {relation_hint}\n"
                "Prefer canonical fields and valid capitalization.\n"
                "Do not emit relation types outside the approved list; use RELATED_TO only as a fallback.\n"
            )
        
        except Exception as e:
            logger.warning(f"Failed to build schema hint section: {e}")
            return ""


    # ------------------------------------------------------------------
    # P2-3 (2026-09-09): narrow prompts for the six-role agent skeleton
    # (docs/OPTIMIZATION_PRIORITY_ACTION_LIST.md P2-3 / Gate E).
    # Deliberately small, deterministic, and opt-in: the default extraction
    # path keeps using build_unified_prompt; narrow prompts only take effect
    # when the caller explicitly opts in via build_narrow_prompt().
    # ------------------------------------------------------------------

    def build_narrow_relation_prompt(
        self,
        text: str,
        *,
        domain: Optional[str] = None,
        known_entities: Optional[list] = None,
        max_pairs: int = 25,
    ) -> str:
        """Build a relation-only prompt (P2-3 narrow pass #1).

        Contract: output must be a JSON array of relation objects with
        EXACTLY the fields {from, to, type, evidence}; entity endpoints
        MUST come from ``known_entities`` (no new entities invented).
        """
        ents = ", ".join(str(e) for e in (known_entities or [])[:20])
        entity_hint = ents if ents else "(entities were extracted in a previous pass)"
        return (
            f"# RELATION EXTRACTION (narrow pass)\n"
            f"Domain: {domain or 'auto'}\n\n"
            f"Entities already known (from/entity endpoints MUST be from this list, use exact names):\n"
            f"{entity_hint}\n\n"
            f"TASK: Extract ONLY explicit relations between the entities above, from the document below.\n"
            f"Rules:\n"
            f"- Allowed types: DEPENDS_ON, SUPPORTS, DERIVED_FROM, TRANSFORMS_TO, PART_OF, USES, "
            f"MEASURED_BY, AFFECTS, PROVIDES_INPUT_TO, RECEIVES_INPUT_FROM, TRIGGERS, RELATED_TO (fallback only)\n"
            f"- Related types not in the list: use RELATED_TO\n"
            f"- Each relation MUST cite the supporting sentence in 'evidence' (verbatim substring)\n"
            f"- Do NOT invent new entities; do NOT infer relations from general knowledge\n"
            f"- Output at most {max_pairs} relations; prefer high-confidence explicit statements\n\n"
            f"OUTPUT: ONLY VALID JSON array, each item: {{\"from\": \"<entity name>\", \"to\": \"<entity name>\", "
            f"\"type\": \"<type>\", \"evidence\": \"<verbatim sentence>\"}}\n\n"
            f"DOCUMENT:\n{text}"
        )

    def build_narrow_entity_prompt(
        self,
        text: str,
        *,
        domain: Optional[str] = None,
        max_entities: int = 20,
    ) -> str:
        """Build an entity-only prompt (P2-3 narrow pass #2).

        Contract: no relations in the output; entity fields follow the
        canonical meta_class list; confidence + evidence required for
        borderline items.
        """
        return (
            f"# ENTITY EXTRACTION (narrow pass)\n"
            f"Domain: {domain or 'auto'}\n\n"
            f"TASK: Extract entities ONLY (no relations) from the document below.\n"
            f"Rules:\n"
            f"- meta_class MUST be one of: Capability, System, Actor, Asset, Data, Process, "
            f"Concept, BusinessScenario, MappingRule, BusinessRule, ValidationRule\n"
            f"- Layers: L1 (concept), L2 (business scenario), L2.5 (rule), L3 (instance/data)\n"
            f"- For each entity include: meta_id, meta_class, name, description, domain, status\n"
            f"- Mark uncertain items with \"status\": \"proposed\" and include \"evidence\"\n"
            f"- Do NOT extract relations; do NOT emit relationship fields\n"
            f"- Output at most {max_entities} entities; skip trivially generic items\n\n"
            f"OUTPUT: ONLY VALID JSON object with \"entities\": [ ... ] and \"relationships\": [] (empty)\n\n"
            f"DOCUMENT:\n{text}"
        )

    def build_skill_section(self, text: str, domain: Optional[str] = None) -> str:
        """Build domain-specific extraction rules section from L2.5 Skills.
        
        Returns:
            Formatted string with domain-specific rules or a default message if no skills match.
        """
        try:
            # Find matching skills based on text content and domain
            matching_skills = self.skill_loader.find_matching_skills(text)
            
            # Also filter by domain if specified
            if domain and matching_skills:
                normalized_domain = domain.lower()
                filtered_skills = []
                for skill in matching_skills:
                    skill_domains = getattr(skill, "domain", None) or []
                    if isinstance(skill_domains, str):
                        skill_domains = [skill_domains]
                    if not skill_domains or normalized_domain in [d.lower() for d in skill_domains if isinstance(d, str)]:
                        filtered_skills.append(skill)
                matching_skills = filtered_skills
            
            if not matching_skills:
                return "# No domain-specific rules apply for this document.\n"
            
            skill_rules = []
            for skill in matching_skills:
                skill_name = getattr(skill, "skill_name", getattr(skill, "name", "Unknown Skill"))
                description = getattr(skill, "description", "")
                rule_section = f"## Skill: {skill_name}\n"
                rule_section += f"- Description: {description}\n"
                
                entity_types = getattr(skill, "entity_types", None) or {}
                if isinstance(entity_types, dict) and entity_types:
                    rule_section += "- Entity types to extract:\n"
                    for entity_type, props in entity_types.items():
                        if isinstance(props, dict):
                            desc = props.get('description', '')
                        else:
                            desc = str(props)
                        rule_section += f"  - {entity_type}: {desc}\n"
                elif isinstance(entity_types, list) and entity_types:
                    rule_section += "- Entity types to extract:\n"
                    for entity_type in entity_types:
                        rule_section += f"  - {entity_type}: \n"
                
                triggers = getattr(skill, "triggers", None)
                keywords = getattr(skill, "keywords", None) or []
                regex_patterns = getattr(skill, "regex_patterns", None) or []
                if triggers and (keywords or regex_patterns):
                    rule_section += "- Extraction triggers:\n"
                    if keywords:
                        rule_section += f"  - Keywords: {', '.join(keywords)}\n"
                    if regex_patterns:
                        rule_section += f"  - Patterns: {', '.join(regex_patterns)}\n"
                
                cross_layer_mapping = getattr(skill, "cross_layer_mapping", None) or {}
                if isinstance(cross_layer_mapping, dict) and cross_layer_mapping:
                    rule_section += "- Cross-layer mapping rules:\n"
                    for source, target in cross_layer_mapping.items():
                        rule_section += f"  - {source} → {target}\n"
                
                skill_rules.append(rule_section)
            
            return "\n".join(skill_rules)
        
        except Exception as e:
            logger.warning(f"Failed to build skill section: {e}")
            return "# Domain-specific rules unavailable.\n"


