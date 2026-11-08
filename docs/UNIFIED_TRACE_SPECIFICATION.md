# Unified Trace Specification

版本：v1.0
日期：2026-07-31

> 本文定义项目统一的 trace 格式标准。
>
> 所有 pipeline 阶段、LLM 调用、skill 执行都必须输出符合本规范的 trace 记录。

---

## 1. Trace 层级

| 层级 | 记录类型 | 产出者 | 存储 |
|------|----------|--------|------|
| Stage | StageTraceRecord | pipeline 每个阶段 | stage_results.json |
| LLM Call | LLMCallRecord | llm_monitor | llm_calls.jsonl |
| Skill | SkillTraceRecord | skill_registry | skill_traces.jsonl |
| Run | RunTraceRecord | pipeline 最终汇总 | run_manifest.json |

---

## 2. Stage Trace Record

每个 pipeline 阶段完成后输出一条。

```json
{
  "stage": "extract",
  "status": "success",
  "duration_ms": 12000,
  "input_summary": {
    "source_path": "data/MBOM_DU_concept_v2.md",
    "chunk_count": 3,
    "input_tokens_estimate": 8500
  },
  "output_summary": {
    "entity_count": 150,
    "relation_count": 100,
    "output_tokens_estimate": 7000
  },
  "model": "deepseek-chat",
  "provider": "deepseek",
  "prompt_tokens": 8500,
  "completion_tokens": 7000,
  "total_tokens": 15500,
  "warnings": ["truncation_repair_applied"],
  "error_type": null,
  "degraded_mode": false
}
```

---

## 3. LLM Call Record

每次 LLM 调用输出一条。

```json
{
  "call_id": "llm-20260731-001",
  "stage": "extract",
  "provider": "deepseek",
  "model": "deepseek-chat",
  "prompt_tokens": 8500,
  "completion_tokens": 7000,
  "total_tokens": 15500,
  "cost_estimate": 0.015,
  "duration_ms": 12000,
  "success": true,
  "truncation_reason": "none",
  "error_message": "",
  "prompt_hash": "a1b2c3d4e5f67890"
}
```

---

## 4. 失败分类枚举

所有 trace 中的 `error_type` 必须使用以下枚举值：

| 枚举值 | 说明 |
|--------|------|
| `none` | 无错误 |
| `timeout` | 超时 |
| `truncation` | 输出截断 |
| `parse_error` | JSON 解析失败 |
| `auth_error` | 认证失败 |
| `rate_limit` | 限流 |
| `network_error` | 网络错误 |
| `schema_validation` | Schema 校验失败 |
| `empty_output` | 空输出 |
| `unknown` | 未知错误 |

---

## 5. Degraded Mode 标记

当系统触发降级模式时，trace 必须记录：

```json
{
  "degraded_mode": true,
  "degraded_reason": "truncation_retry_exhausted",
  "degraded_fallback": "partial_extraction_from_repaired_json"
}
```

---

## 6. 现有实现映射

| 规范字段 | 现有实现 | 状态 |
|----------|----------|------|
| stage | `PipelineStage` enum | ✅ 已存在 |
| prompt_tokens | `LLMCallRecord.prompt_tokens` | ✅ 已存在 |
| completion_tokens | `LLMCallRecord.completion_tokens` | ✅ 已存在 |
| truncation_reason | `TruncationReason` enum | ✅ 已存在 |
| prompt_hash | `hash_text()` | ✅ 已存在 |
| error_type | 需统一到本规范枚举 | ⚠️ 待对齐 |
| degraded_mode | 需新增字段 | ⚠️ 待对齐 |
| input_summary | `StageResult.data` 中部分覆盖 | ⚠️ 待对齐 |
| output_summary | `StageResult.data` 中部分覆盖 | ⚠️ 待对齐 |

---

## 7. 更新规则

- 新增 error_type 值需要更新本文枚举表
- 新增 trace 字段需要更新本文规范
- 每次改动 trace 格式后，必须验证旧 trace 仍可解析（向后兼容）
