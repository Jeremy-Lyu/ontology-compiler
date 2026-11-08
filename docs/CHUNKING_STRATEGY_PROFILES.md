# Chunking Strategy Profiles

版本：v1.0
日期：2026-07-31

> 本文定义不同文档类型的切分策略、预切分条件与统计口径。
>
> 配套实现：`ontology_compiler/services/text_chunking.py`

---

## 1. Strategy Selection

| 文档类型 | 默认策略 | 说明 |
|----------|----------|------|
| `md` | `semantic_section` | Markdown 标题结构清晰，优先保留语义块 |
| `txt` | `context` | 纯文本优先按上下文分组 |
| `docx` | `semantic_section` | 通常有段落与标题结构 |
| `pptx` | `semantic_section` | 优先按幻灯片标题与段落组织 |
| `pdf` | `context` | 结构可能噪声较多，先按上下文切分 |
| `csv` | `fixed` | 结构化表格按固定块切分 |
| `xlsx` | `fixed` | 表格型输入按固定块切分 |

---

## 2. Pre-splitting Rules

### 2.1 Global thresholds

- `auto_split_threshold = 50000` chars
- `max_chunk_size = 7000` chars
- `min_chunk_size = 1500` chars
- `overlap_size = 500` chars
- `max_total_chunks = 80`

### 2.2 When pre-splitting is triggered

- 文档总长度超过 `auto_split_threshold`
- 单段文本超过 `max_chunk_size`
- 单句超过 `max_chunk_size`
- chunk 数量超过 `max_total_chunks`

### 2.3 Pre-splitting behavior

- 先按段落拆
- 再按句子拆
- 再按边界词拆
- 最后硬切兜底

---

## 3. Statistics Report

每次 chunking 应统计以下指标：

| 指标 | 类型 | 说明 |
|------|------|------|
| `total_chunks` | int | 最终 chunk 总数 |
| `strategy_used` | str | 实际采用的策略 |
| `doc_type` | str | 文档类型 |
| `input_chars` | int | 输入字符数 |
| `output_chars` | int | 切分后字符数 |
| `avg_chunk_chars` | float | 平均 chunk 长度 |
| `max_chunk_chars` | int | 最大 chunk 长度 |
| `min_chunk_chars` | int | 最小 chunk 长度 |
| `oversized_chunks` | int | 超阈值 chunk 数 |
| `small_chunks` | int | 小于 `min_chunk_size` 的 chunk 数 |
| `merged_small_chunks` | int | 被合并的小 chunk 数 |
| `semantic_merged_pairs` | int | 语义合并次数 |
| `count_limited_merges` | int | 为控制总数而合并的次数 |
| `pre_split_triggered` | bool | 是否触发预切分 |

---

## 4. Reporting Rules

- chunk 统计必须进入回归报告
- 对每个样本记录 strategy_used 和 total_chunks
- 对超长文档必须记录 pre_split_triggered = true
- 对 chunk 数过大的文档必须记录 count_limited_merges

---

## 5. Relation to Regression

建议在回归报告中追加：
- `chunk_strategy`
- `chunk_count`
- `oversized_chunks`
- `pre_split_triggered`
- `avg_chunk_chars`

这样才能把“切得好不好”与最终质量结果关联起来。
