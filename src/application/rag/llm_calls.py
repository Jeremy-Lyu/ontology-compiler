"""LLM 调用 Mixin - provider 调用、prompt 构建与答案组装。

RAG-07 (§9.7, 2026-09-03): 从 rag_answerer.py 拆出 LLM 调用能力，使
rag_answerer.py 聚焦于统一路由/编排。本模块提供 ``LLMCallMixin``：

1. prompt 构建（system/user/deep-analysis/A-H 框架）
2. provider 调用三级路由（Dify > UnifiedLLMClient > UnifiedLLMProvider）
3. 答案组装（deep-analysis 报告、fallback、自然语言合成）

RAGAnswerer 通过 Mixin 继承获得这些方法；实例级 ``mock.patch.object``
对继承方法同样生效，behavior 由 characterization 测试锁定。
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from .formatting import (
    format_dify_structured_answer,
    format_document_result,
    format_neo4j_result,
    generate_ah_framework_report,
    synthesize_natural_answer,
)

try:
    from ontology_compiler.rigor.llm_provider import UnifiedLLMProvider, LLMRequest
    LLM_PROVIDER_AVAILABLE = True
except ImportError:
    UnifiedLLMProvider = None  # type: ignore[assignment,misc]
    LLMRequest = None  # type: ignore[assignment,misc]
    LLM_PROVIDER_AVAILABLE = False

# Dify client helpers used by ``_call_llm_with_dify`` (module-global lookup).
# Kept in this module's namespace: no test patches these globals directly —
# the Dify wiring tests patch ``rag_answerer.ask_dify_kb_qa`` / ``get_dify_answer_text``
# for ``DifyCallLLMBackend.call_llm`` and ``_rewrite_query``, both of which
# stay in rag_answerer.py and resolve there.
try:
    from .infrastructure.dify_client import (
        ask_dify_kb_qa,
        get_dify_answer_text,
    )
    DIFY_CLIENT_AVAILABLE = True
except ImportError:
    DIFY_CLIENT_AVAILABLE = False
    ask_dify_kb_qa = None  # type: ignore[assignment]
    get_dify_answer_text = None  # type: ignore[assignment]


class LLMCallMixin:
    """LLM 调用能力（prompt 构建 / provider 路由 / 答案组装）。"""

    def _classify_question(self, question: str) -> str:
        q_lower = question.lower()
        scores: Dict[str, int] = {k: 0 for k in self._QUESTION_PATTERNS}
        for q_type, keywords in self._QUESTION_PATTERNS.items():
            scores[q_type] = sum(1 for kw in keywords if kw in q_lower)
        best = max(scores, key=lambda k: scores[k])
        return best if scores[best] > 0 else "GENERAL"

    def _extract_concept(self, question: str) -> str:
        question = question.strip().rstrip("?？")
        for prefix in ["什么是", "是什么", "如何", "怎么", "请介绍", "介绍一下",
                        "说明", "解释", "描述", "what is", "how to", "explain"]:
            question = re.sub(rf"^{prefix}\s*", "", question, flags=re.IGNORECASE)
        return question[:40] if len(question) > 40 else question

    def _format_neo4j_result(self, index: int, result: Dict[str, Any]) -> str:
        """Delegate to `formatting.format_neo4j_result` (RAG-04)."""
        return format_neo4j_result(index, result)

    def _format_document_result(self, index: int, result: Any) -> str:
        """Delegate to `formatting.format_document_result` (RAG-04)."""
        return format_document_result(
            index,
            result,
            max_chunk_length=self.max_chunk_length,
            l2_scenario_types=self._L2_SCENARIO_TYPES,
        )

    def _build_context(
        self,
        results: List[Any],
        neo4j_results: List[Dict] = None,
        max_chunks: Optional[int] = None,
    ) -> str:
        limit = max_chunks if max_chunks is not None else self.max_context_chunks
        context_parts = []

        if neo4j_results:
            context_parts.append("=== 图数据库检索结果 (Neo4j) ===")
            for i, r in enumerate(neo4j_results[:3], 1):
                context_parts.append(self._format_neo4j_result(i, r))
            context_parts.append("")

        if results:
            context_parts.append("=== 文档检索结果 ===")
            for i, r in enumerate(results[:limit], 1):
                context_parts.append(self._format_document_result(i, r))

        return "\n---\n".join(context_parts)

    def _build_llm_system_prompt(self, question: str) -> str:
        q_type = self._classify_question(question)
        concept = self._extract_concept(question)
        sources = "上下文检索结果"

        template = self._SYSTEM_PROMPTS.get(q_type, self._SYSTEM_PROMPTS.get("GENERAL", ""))
        if not template:
            return ""
        return template.format(concept=concept, sources=sources)

    def _build_llm_user_prompt(self, question: str, context: str) -> str:
        return (
            f"上下文信息：\n\n{context}\n\n---\n\n"
            f"用户问题：{question}\n\n"
            f"请基于上下文信息，用自然、友好的语言详细回答用户的问题。\n"
            f"要求：\n"
            f"1. 用中文回答，不要返回JSON格式\n"
            f"2. 深入分析上下文提供的所有相关信息，不要遗漏重要细节\n"
            f"3. 如果上下文包含多个来源，请综合所有信息进行分析\n"
            f"4. 对于涉及流程、系统、数据的问题，请按照类别进行结构化回答\n"
            f"5. 尽可能提供具体的实体名称、系统名称、流程步骤等细节\n"
            f"6. 如果上下文信息不足，请明确指出哪些部分未提及\n"
            f"7. 回答要有深度，分析它们之间的关系和影响\n"
            f"8. 使用清晰的标题和列表结构，使回答易于阅读\n"
            f"9. 如果有相关的业务场景或案例，请举例说明\n"
            f"10. 如果涉及部门或角色信息，请从上下文提取相关的组织实体\n"
        )

    def _call_llm(self, question: str, context: str) -> str:
        # Priority: Dify > UnifiedLLMClient > UnifiedLLMProvider
        if self.dify_config:
            return self._call_llm_with_dify(question, context)
        elif self.llm_client:
            return self._call_llm_with_client(question, context)
        elif self.llm_provider:
            return self._call_llm_with_provider(question, context)
        else:
            raise RuntimeError(
                "LLM provider not available. "
                "Please check your API keys in config/env.env"
            )

    def _format_dify_structured_answer(self, raw_text: str, question: str) -> str:
        """Delegate to `formatting.format_dify_structured_answer` (RAG-04).

        See the delegated function for the two MAINTENANCE NOTES about adding
        new Dify workflow output fields and empty-layer rendering.
        """
        return format_dify_structured_answer(raw_text, question)

    def _fallback_answer_from_context(self, question: str, context: str) -> str:
        """Generate a natural-language answer from retrieval context when Dify
        Workflow returns empty entity/relationship lists.

        Uses the same Dify /chat-messages endpoint but with a direct prompt
        asking the LLM to analyze the retrieved ontology context and answer.
        """
        fallback_prompt = (
            "你是一个本体知识分析助手。请根据以下检索到的本体上下文信息，"
            f"用中文回答用户的问题，并尽可能引用实体的ID、类型、描述和关系。\n\n"
            f"【用户问题】{question}\n\n"
            f"【检索到的本体上下文】\n{context[:8000]}\n\n"
            "请给出结构化的回答，包括：\n"
            "1. 相关实体列表（ID、名称、类型、描述）\n"
            "2. 实体间的关系\n"
            "3. 对问题的综合分析"
        )
        try:
            from .infrastructure.dify_client import ask_dify_kb_qa, get_dify_answer_text
            config = self.dify_config
            raw = ask_dify_kb_qa(
                query=question,
                inputs={
                    "system_prompt": "你是一个专业的本体知识分析师。",
                    "ontology_context": context[:8000],
                    "user_question": question,
                    "full_prompt": fallback_prompt[:16000],
                },
                config=config,
            )
            answer_text = get_dify_answer_text(raw)
            return answer_text or f"（fallback）检索到相关信息但Dify未能生成结构化回答。上下文摘要：{context[:500]}"
        except Exception as e:
            return (
                f"⚠️ Dify Workflow 返回空实体，fallback 生成也失败：{e}\n\n"
                f"以下是检索到的相关上下文（手动摘要）：\n\n"
                f"{context[:3000]}"
            )

    def _call_deepseek_direct(
        self, question: str, context: str, max_context_chars: int = 20000
    ) -> Optional[str]:
        """Call DeepSeek through the unified gateway for deep-analysis mode.

        This keeps the A-H analysis behavior intact while routing the call
        through the shared gateway so budget, pricing, and monitor records are
        applied consistently.

        C-5 enhancements:
        - Context compaction replaces naive truncation: keeps the most recent +
          most relevant context fragments instead of a hard char cut.
        - ``gateway.execute_routed`` picks max_tokens / temperature from the
          cost-aware routing policy (task_kind=RAG_ANSWER) instead of a fixed
          4096 / 0.3, so short answers no longer inherit a heavy token budget.
        """
        # C-5: context compaction (rule-based, no LLM call).
        if len(context) > max_context_chars:
            try:
                from src.application.services.context_compaction import get_context_compactor

                compactor = get_context_compactor()
                # Split context into paragraphs for compaction; each paragraph
                # becomes a ContextItem with uniform score so compaction favors
                # recency (later paragraphs are typically more relevant).
                paragraphs = [p for p in context.split("\n\n") if p.strip()]
                if len(paragraphs) >= 2:
                    from src.application.services.context_compaction import ContextItem

                    items = [
                        ContextItem(text=p, score=0.5, kind="context", index=i)
                        for i, p in enumerate(paragraphs)
                    ]
                    # Override the default max_chars to the caller's budget.
                    from src.application.services.context_compaction import CompactionConfig

                    compactor_local = type(compactor)(CompactionConfig(max_chars=max_context_chars))
                    result = compactor_local.compact(items)
                    context = "\n\n".join(i.text for i in result.kept)
                    if result.budget_exceeded:
                        context += f"\n\n[NOTE: context compacted {len(paragraphs)} -> {len(result.kept)} fragments]"
                else:
                    context = context[:max_context_chars] + (
                        f"\n\n[NOTE: context truncated to {max_context_chars} chars]"
                    )
            except Exception:
                # Fallback to naive truncation if compaction is unavailable.
                context = context[:max_context_chars] + (
                    f"\n\n[NOTE: context truncated to {max_context_chars} chars]"
                )

        system_prompt = (
            "你是一位资深企业数据资产分析师，擅长制造领域的BOM、MRP、生产计划等业务分析。\n"
            "请基于提供的本体知识库上下文，对用户问题进行深度业务分析。\n\n"
            "严格按照以下A-H框架结构输出（用中文），每个部分都要有实质性内容：\n\n"
            "A. 数据对象定义\n"
            "   - 这个数据对象是什么？内部叫法、数据分类\n"
            "   - 核心特征，与相似概念的区别（如EBOM vs MBOM）\n\n"
            "B. 涉及流程与系统\n"
            "   - 在哪些业务流程中使用？关键步骤（详细描述）\n"
            "   - 涉及哪些系统？系统名称和角色（如CMP, OHF, BIS, PLUS, SAP, KDP）\n\n"
            "C. 角色与决策\n"
            "   - 谁创建、谁修改、谁审批？\n"
            "   - 谁使用？用来做什么决策？\n"
            "   - 如果没有它会产生什么影响？\n\n"
            "D. 常见问题与数据标准\n"
            "   - 目前最常见的问题（如同步延迟、断点偏差等）\n"
            "   - 作为数据产品希望有什么标准（更新频率、准确度、关键字段）\n\n"
            "E. 典型业务场景\n"
            "   - 描述一个具体的业务流程场景，展示数据如何流转\n\n"
            "F. 数据来源与权威源\n"
            "   - 数据初始产生自哪里？\n"
            "   - 哪个系统是权威源？冲突如何解决？\n\n"
            "要求：\n"
            "1. 使用中文回答\n"
            "2. 结合业务术语和具体系统名称\n"
            "3. 详细描述流程步骤，不要只罗列\n"
            "4. 充分引用上下文中的信息\n"
            "5. 如果某些信息不足，基于已有信息合理推断\n"
            "6. 使用清晰的小标题和格式\n"
            "7. 不要输出JSON，直接输出自然语言分析报告"
        )

        user_prompt = (
            f"===== 检索到的本体知识库上下文 =====\n\n"
            f"{context}\n\n"
            f"===== 用户问题 =====\n\n"
            f"{question}\n\n"
            f"请按照A-H框架进行深度业务分析："
        )
        gateway_prompt = (
            f"[SYSTEM]\n{system_prompt}\n\n"
            f"[USER]\n{user_prompt}"
        )

        try:
            from src.application.bootstrap.llm_gateway import get_llm_gateway

            gateway = get_llm_gateway()
            # C-5: use execute_routed so max_tokens / temperature are picked
            # by the cost-aware routing policy (task_kind=RAG_ANSWER). The
            # routed model label is preserved for pricing/telemetry even when
            # the gateway falls back to the bootstrap primary provider.
            outcome = gateway.execute_routed(
                gateway_prompt,
                task_kind="rag_answer",
                stage="rag.deep_analysis",
                call_id=f"rag-deep-analysis:{hash(question)}",
            )
            if not outcome.success or not outcome.text.strip():
                return None
            answer = outcome.text.strip()
            if len(answer) < 50:
                return None

            parts = [
                "## 📊 深度业务分析报告（Gateway 托管）",
                f"**分析主题**: {question}",
                f"**分析类型**: 数据资产深度分析 (A-H 框架)",
                f"**模型**: {outcome.model or os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')}",
                f"**费用**: {outcome.cost_usd:.6f} USD",
            ]
            # C-5: surface routing tier when available.
            if outcome.route_decision is not None:
                parts.append(
                    f"**路由**: {outcome.route_decision.tier.value} "
                    f"({outcome.route_decision.reason})"
                )
            parts.extend([
                "",
                "---",
                "",
                answer,
                "",
                "---",
                "",
                f"*报告基于 {len(context)} 字符的本体知识库上下文，由统一 gateway 生成*",
            ])
            return "\n".join(parts)
        except Exception as e:
            print(f"⚠️ DeepSeek gateway 调用失败: {e}，回退到 Dify")
            return None

    def _call_llm_with_dify(self, question: str, context: str) -> str:
        """Call LLM via Dify with deep analysis mode support.

        For deep analysis queries (DEEP_ANALYSIS type), bypasses Dify's JSON
        schema and calls DeepSeek directly to generate natural-language A-H
        framework responses. For other queries, uses Dify's standard
        entity-relationship extraction approach.
        """
        try:
            MAX_CONTEXT_CHARS = max(
                2000, int(os.getenv("DIFY_MAX_CONTEXT_CHARS", "12000"))
            )
        except (TypeError, ValueError):
            MAX_CONTEXT_CHARS = 12000

        question_type = self._classify_question(question)
        is_deep_analysis = (question_type == "DEEP_ANALYSIS")

        if is_deep_analysis:
            deepseek_result = self._call_deepseek_direct(question, context, MAX_CONTEXT_CHARS)
            if deepseek_result:
                return deepseek_result

        truncation_note = ""
        if len(context) > MAX_CONTEXT_CHARS:
            truncation_note = (
                f"\n\n[NOTE: context truncated from {len(context)} chars to "
                f"{MAX_CONTEXT_CHARS}]\n"
            )
            context = context[:MAX_CONTEXT_CHARS] + truncation_note

        system_prompt = self._build_llm_system_prompt(question)

        if is_deep_analysis:
            deep_analysis_prompt = self._build_deep_analysis_prompt(
                question, context
            )
            inputs = {
                "system_prompt": system_prompt[:8000],
                "ontology_context": context,
                "user_question": question,
                "full_prompt": deep_analysis_prompt[:30000],
            }
        else:
            _ = self._build_llm_user_prompt(question, context)
            inputs = {
                "system_prompt": system_prompt[:8000],
                "ontology_context": context,
                "user_question": question,
                "full_prompt": (
                    "===== SYSTEM INSTRUCTIONS (Ontology-RAG) =====\n"
                    f"{system_prompt[:8000]}\n\n"
                    "===== RETRIEVED ONTOLOGY CONTEXT =====\n"
                    f"{context}\n\n"
                    "===== USER QUESTION =====\n"
                    f"{question}\n"
                )[:30000],
            }

        raw = ask_dify_kb_qa(
            query=question,
            inputs=inputs,
            config=self.dify_config,
        )
        answer_text = get_dify_answer_text(raw)

        if is_deep_analysis:
            return self._format_deep_analysis_answer(
                question, answer_text, context
            )

        try:
            formatted = self._format_dify_structured_answer(answer_text, question)

            if "本次分析未从提供的本体上下文中提取到有效实体" in formatted:
                return self._fallback_answer_from_context(question, context)

            has_structured = (
                "### 针对问题" in formatted
                or "实体间关联关系" in formatted
            )
            if has_structured:
                synthesis = self._synthesize_natural_answer(
                    question, answer_text, context
                )
                if synthesis and len(synthesis) > 20:
                    return synthesis + "\n\n---\n\n" + formatted
            return formatted
        except Exception:
            try:
                return self._synthesize_natural_answer(
                    question, answer_text, context
                ) or answer_text
            except Exception:
                return answer_text

    def _build_deep_analysis_prompt(self, question: str, context: str) -> str:
        """Build a comprehensive prompt for deep analysis using A-H framework."""
        return (
            "===== 深度分析指令 =====\n\n"
            "你是一位资深企业知识分析师，擅长数据资产的深度业务分析。\n"
            "请基于下方检索到的本体知识库上下文，对用户提出的问题进行**深度业务分析**。\n\n"
            "请严格按照以下分析框架进行组织和输出：\n\n"
            "A. 【数据对象定义】\n"
            "   - 这个数据对象是什么？内部叫法、数据分类\n"
            "   - 核心特征和与相似概念的区别\n\n"
            "B. 【涉及流程与系统】\n"
            "   - 在哪些业务流程中使用？关键步骤\n"
            "   - 涉及哪些系统？系统名称和角色\n\n"
            "C. 【角色与决策】\n"
            "   - 谁创建、谁修改、谁审批？\n"
            "   - 谁使用？用来做什么决策？\n"
            "   - 如果没有它会产生什么影响？\n\n"
            "D. 【常见问题与数据标准】\n"
            "   - 目前最常见的问题\n"
            "   - 作为数据产品希望有什么标准（更新频率、准确度、关键字段）\n\n"
            "E. 【典型业务场景】\n"
            "   - 描述一个具体的业务流程场景，展示它如何被使用\n\n"
            "F. 【数据来源与权威源】\n"
            "   - 数据初始产生自哪里？\n"
            "   - 哪个系统是权威源？冲突如何解决？\n\n"
            "要求：\n"
            "1. 使用中文回答\n"
            "2. 结合业务术语和具体系统名称（如 CMP, OHF, BIS, PLUS, KDP, SAP 等）\n"
            "3. 详细描述流程步骤，不要只是罗列\n"
            "4. 如果上下文中有相关信息，请充分引用\n"
            "5. 如果某些部分信息不足，可以基于已有信息进行合理推断\n"
            "6. 使用清晰的小标题和格式\n\n"
            "===== 检索到的本体知识库上下文 =====\n\n"
            f"{context}\n\n"
            "===== 用户问题 =====\n\n"
            f"{question}\n\n"
            "请开始深度分析："
        )

    def _format_deep_analysis_answer(
        self, question: str, answer_text: str, context: str
    ) -> str:
        """Format the deep analysis answer, converting JSON entities to A-H report."""
        if not answer_text or len(answer_text.strip()) < 50:
            return self._generate_deep_analysis_fallback(question, context)

        analysis_report = self._generate_ah_framework_report(
            question, answer_text, context
        )

        parts = [
            f"## 📊 深度业务分析报告",
            f"**分析主题**: {question}",
            f"**分析类型**: 数据资产深度分析 (A-H 框架)",
            "",
            "---",
            "",
            analysis_report,
            "",
            "---",
            "",
            f"*报告基于 {len(context)} 字符的本体知识库上下文生成*",
        ]
        return "\n".join(parts)

    def _generate_ah_framework_report(
        self, question: str, answer_text: str, context: str
    ) -> str:
        """Delegate to `formatting.generate_ah_framework_report` (RAG-04)."""
        return generate_ah_framework_report(question, answer_text, context)

    def _generate_deep_analysis_fallback(
        self, question: str, context: str
    ) -> str:
        """Generate a structured fallback analysis when LLM returns empty."""
        entity_extract = self._extract_key_entities(question)
        context_summary = self._summarize_context_business(context)

        parts = [
            f"## 📊 深度业务分析报告 (自动生成版)",
            f"**分析主题**: {question}",
            "",
            "---",
            "",
        ]

        if entity_extract:
            parts.append("### 🔍 识别的关键实体/系统：")
            for entity in entity_extract[:10]:
                parts.append(f"  • **{entity}**")
            parts.append("")

        parts.append("### 📋 基于本体知识库的分析：")
        parts.append("")
        parts.append(context_summary[:2000] if context_summary else context[:2000])
        parts.append("")

        if len(context) > 2000:
            parts.append(
                f"*（上下文长度 {len(context)} 字符，仅展示前 2000 字符摘要）*"
            )

        return "\n".join(parts)

    def _extract_key_entities(self, text: str) -> List[str]:
        """Extract key entity names from text."""
        entities = set()
        patterns = [
            r'\b[A-Z][A-Z]{2,}\b',
            r'(?:L[0-9](?:\.[0-9]+)?-[0-9]+)',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text)
            entities.update(matches)
        return sorted(list(entities))

    def _summarize_context_business(self, context: str) -> str:
        """Summarize context focusing on business-relevant content."""
        lines = context.split("\n")
        relevant_lines = []

        business_keywords = [
            "系统", "流程", "角色", "数据", "BOM", "MBOM", "EBOM", "MRP",
            "CMP", "OHF", "BIS", "PLUS", "KDP", "SAP", "工单", "订单",
            "变更", "C-Note", "Allocation", "断点", "工位", "Station",
            "责任", "审批", "流程", "场景", "标准",
        ]

        for line in lines:
            if any(kw in line for kw in business_keywords):
                relevant_lines.append(line.strip())

        if len(relevant_lines) > 50:
            return "\n".join(relevant_lines[:50])
        return "\n".join(relevant_lines)

    def _synthesize_natural_answer(
        self, question: str, raw_answer: str, context: str
    ) -> str:
        """Generate a natural-language synthesis from Dify's structured output.

        Takes the raw JSON (or already-formatted text) and produces a concise,
        natural Chinese answer that reads like an LLM comprehension — not a
        raw entity dump.
        """
        return synthesize_natural_answer(question, raw_answer, context)

    def _call_llm_with_provider(self, question: str, context: str) -> str:
        """Call LLM using UnifiedLLMProvider (original approach)."""
        llm_request = LLMRequest(
            prompt=self._build_llm_user_prompt(question, context),
            system_prompt=self._build_llm_system_prompt(question),
            temperature=0.3,
            max_tokens=3000,
        )
        llm_response = self.llm_provider.complete(llm_request)
        return llm_response.content

    def _call_llm_with_client(self, question: str, context: str) -> str:
        """Call LLM using UnifiedLLMClient (Pipeline-style with async)."""
        import asyncio
        
        # Build combined prompt (system prompt + user prompt)
        user_prompt = self._build_llm_user_prompt(question, context)
        system_prompt = self._build_llm_system_prompt(question)
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Execute async call. If we are already inside a running event loop
        # (e.g. `_answer_async_impl` runs under `asyncio.run`), running another
        # `asyncio.run()` here would raise RuntimeError. In that case run the
        # coroutine in a dedicated thread so the synchronous signature is kept.
        async def _async_call():
            return await self.llm_client.call(
                prompt=full_prompt,
                temperature=0.3,
                max_tokens=3000,
            )

        def _run_in_thread() -> str:
            return asyncio.run(_async_call())

        try:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                response = _run_in_thread()
            else:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    response = pool.submit(_run_in_thread).result()
            return response
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}") from e
