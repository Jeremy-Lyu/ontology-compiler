# CQ Schema Spec（M0 建契约）

> 工程契约文档，对应 `CQ_PIPELINE_IMPLEMENTATION_PLAN.md` §6/§9-M0。
> 机器可读 schema：`config/cq/cq_schema.json`；实现：`src/application/cq/cq_models.py`。

## 1. CQ 最小结构

| 字段 | 必填 | 类型 | 说明 |
|---|---|---|---|
| `cq_id` | ✅ | string | 唯一标识（`CQ-0001` 起） |
| `cq_text` | ✅ | string | CQ 化后的业务语言问题（不允许 IT 术语） |
| `entities` | ✅ | string[] | CQ 涉及的实体（≥1） |
| `relations` | ✅ | string[] | CQ 依赖的关系类型（≥1） |
| `query_path` | ✅ | string | 可执行实体查询路径；写不出即模型缺口 |
| `status` | ✅ | enum | `draft` / `reviewed` / `frozen` / `deprecated` |
| `original_text` | — | string | 业务原话，一字不改（Step 1） |
| `pattern_type` | — | string | `attribute_query` / `relationship_trace` / `condition_filter` / `time_slice`（可组合 `+`） |
| `attributes` | — | string[] | 属性/时间维度 |
| `owner` | — | string | 责任人 |
| `version` | — | string | 默认 `v1` |
| `evidence` | — | string[] | 来源引用（`doc://...#L12-L18`） |
| `sample_binding` | — | string | 绑定回归样本（`regression:mbom-du-v2`） |
| `open_questions` | — | string[] | 待业务确认的口径问题 |

## 2. 状态机

```
draft ──► reviewed ──► frozen ──► deprecated
  │           │           │
  └───────────┴───────────┴──► deprecated（可随时废弃）
```

- 仅允许线性前移或进入 `deprecated`；`deprecated` 为终态，不允许再流转。
- **只有 `frozen` 状态的 CQ 进入 M2 验证 / M3 缺口检测 / M5 回归门禁。**
- Python 侧判定：`CQ.can_transition_to(target)` / `allowed_transitions(status)`。

## 3. 存储与加载

- 种子输入（版本控制内）：`config/cq/seed_cq.jsonl`，一行一条 CQ JSON。
- 加载 API：
  - `CQCatalog.from_jsonl(path)` —— 严格加载（坏行抛错，适合测试/校验）。
  - `CQRegistry.load(path)` —— 宽容加载（跳过坏行并记录 `warnings`，适合 pipeline）。
- 目录为空 / 文件不存在 → 视为「无 CQ 输入」，pipeline 跳过所有 CQ 逻辑（M2 零回归约束）。

## 4. M2 产物契约（只读，不阻断）

| 产物 | 路径（运行目录 `artifact_root/cq/`） | 生成 stage |
|---|---|---|
| CQ 目录 | `cq_catalog.json` | preflight |
| 验证报告 | `cq_validation_report.json` | audit（只读可执行性检查） |
| 回译表 | `cq_review_sheet.md` | summarize |

- 三份产物统一登记进 `ArtifactBundle`（stage=`cq`）。
- 无 frozen CQ 或目录为空 → 不写任何文件、不改任何 stage 输出（逐字节一致）。

## 5. 验收（M0）

- `python -c "from src.application.cq.cq_models import CQ"` 可实例化。
- `config/cq/cq_schema.json` 通过 JSON 语法校验，且字段与 `CQ` 模型一致。
- characterization 测试：`tests/cq/test_cq_models_m0.py`。
