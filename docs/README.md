# Docs

`docs/` 只保留仍被代码、schema 或 skill 引用的契约说明。可执行规则在 `config/` 与 `schemas/`，本文档不重复它们。

| 文件 | 用途 | 谁在引用 |
|---|---|---|
| `ENTITY_CLASSIFICATION_SPEC.md` | 实体字段、层级、时间与 packet 语义 | 抽取 adapter、质量规则 |
| `CHUNKING_STRATEGY_PROFILES.md` | 按文档类型的切分策略 | `ontology_compiler/services/extraction/text_chunking.py` |
| `UNIFIED_TRACE_SPECIFICATION.md` | stage / LLM / skill / run 的 trace 字段 | `src/application/services/llm_monitor.py` |
| `CQ_SCHEMA_SPEC.md` | Competency Question 结构 | `config/cq/cq_schema.json`、`src/application/cq/cq_models.py` |
| `RELATION_TYPE_PRIORITY_TABLE.md` | 关系类型优先级与改写顺序 | `skills/semantic_organization/relation_rewrite/.ontology-skill/skill.json` |
