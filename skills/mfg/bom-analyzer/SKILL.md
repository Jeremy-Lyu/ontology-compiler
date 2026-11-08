# BOM多时间序列分析 Skill

## 概述

本Skill封装了TBS130 BOM分析项目中的核心方法论，支持对制造BOM的多时间序列分析，包括变更审计、配置空间分析和替换关系识别。

## 核心能力

### 1. 多时间序列混合分析 (Multi-TimeSeries Hybrid Analysis)

**方法论**: 脚本层精确统计 + LLM层推理洞察

```
[4+ CSV快照] → [脚本提取统计] → [LLM生成推理] → [可操作建议]
     ↑              ↑                  ↑              ↑
  时间序列      精确数字           模式识别         行动清单
```

**关键指标**:
- POS跨时间变更率: 42.5%
- 变更热点识别: 388个POS
- 版本链验证: DSCIMOD替代ISSUE字段

### 2. BOM变更审计 (BOM Change Audit)

**算法核心**:
```python
def audit_bom_changes(current_df, previous_df):
    """检测两日快照间的BOM变更"""
    changes = []
    
    for pos in all_pos:
        curr_materials = set(current_df[current_df['POS_KEY']==pos]['ISTUK35'])
        prev_materials = set(previous_df[previous_df['POS_KEY']==pos]['ISTUK35'])
        
        # 检测新增/删除/用量变化
        for material in (curr_materials - prev_materials):
            changes.append({'type': 'ADD', 'pos': pos, 'material': material})
        for material in (prev_materials - curr_materials):
            changes.append({'type': 'REMOVE', 'pos': pos, 'material': material})
    
    return changes
```

**输出**:
- 新增物料清单
- 删除物料清单  
- 用量变更清单
- 高频变更POS排行

### 3. 配置空间分析 (Configuration Space Analysis)

**分析维度**:
| 维度 | 计算方法 | 业务含义 |
|------|----------|----------|
| 可配置POS占比 | `configurable_pos / total_pos` | 超级BOM复杂度 |
| 配置组合数 | `物料数 × KLOWER选项数` | 配置空间大小 |
| KLOWER规则 | `GROUP BY KLOWER, ISTUK35` | 配置选择逻辑 |

**关键发现模式**:
- KLOWER=" ": 标准配置
- KLOWER="L": 左侧驾驶配置
- KLOWER="P": 特殊市场配置

### 4. AI增强审计 (AI-Enhanced Audit)

**多字段组合识别**:
| 字段组合 | 识别模式 | 置信度 |
|----------|----------|--------|
| POS_KEY + FINDNUMBER | 版本替换 | HIGH |
| POS_KEY + KLOWER | 配置变体 | MEDIUM |
| ISPECGRP + DSCIMOD | 总成级替换 | MEDIUM |
| POS_KEY + 时间序列 | 替换链 | MEDIUM |

**AI推理逻辑示例**:
```
IF FINDNUMBER相同 AND 物料不同 AND 时间递进:
    THEN 可能是版本替换
    CONFIDENCE = HIGH
    REASON = "工程设计变更通常保持FINDNUMBER不变，替换物料号"
```

### 5. 分层抽样验证 (Stratified Sampling Validation)

**分层标准**:
| 层级 | 物料数 | DSCIMOD数 | 验证优先级 |
|------|--------|-----------|------------|
| 高复杂度 | >10 | >4 | P0 |
| 中复杂度 | 5-10 | 2-4 | P1 |
| 低复杂度 | <5 | 2 | P2 |

**验证标准**:
- VERIFIED: ECO存在 + 物料匹配 + 时间合理
- REJECTED: ECO不存在/取消 或 物料不匹配 或 时间矛盾
- UNCERTAIN: 部分匹配或数据不完整

## 适用场景

### 场景1: 零件替换追溯
**输入**: 4个时间点的TBS130 CSV
**输出**: 1,219个历史替换POS的替换链
**价值**: 识别55.3%的POS变更模式

### 场景2: 配置空间优化
**输入**: 单时间点TBS130快照
**输出**: 42.5%可配置POS的配置规则
**价值**: 理解超级BOM的裁剪逻辑

### 场景3: 变更预警监控
**输入**: 每日TBS130快照
**输出**: 新DSCIMOD告警 + 影响POS清单
**价值**: 24小时内完成影响分析

## 使用方式

### 方式1: 脚本直接调用
```bash
python -m scripts.diagnostics.run_bom_timeseries_analysis \
  --csv-dir Sample_data/TBS130/ \
  --analysis-type full_analysis \
  --output outputs/bom_analysis/
```

### 方式2: SkillRegistry调用
```python
from ontology_compiler.harness import build_default_skill_registry

registry = build_default_skill_registry()
skill = registry.get_skill("bom-timeseries-analysis")

result = skill.execute({
    "csv_files": ["TBS130_202606010000.csv", ...],
    "analysis_type": "full_analysis"
})
```

## 数据要求

### 必需字段
| 字段名 | 说明 | 示例 |
|--------|------|------|
| ISPECGRP | POS组 | 2681 |
| ISPECPOS | POS位 | 10 |
| FINDNUMBER | 查找号 | 0010 |
| ISTUK35 | 物料号 | 32302665 |
| DSCIMOD | 变更单号 | 20262212004 |
| KLOWER | 配置标识 | " ", "L", "P" |

### 可选字段
| 字段名 | 说明 |
|--------|------|
| DUPDATE | 更新时间戳 |
| PART_ADDRESS | 工厂地址 |

## 限制与注意事项

| 限制 | 缓解方案 |
|------|----------|
| ISSUE字段可能失效 | 使用DSCIMOD+时间戳替代版本链 |
| 无法100%确定替换 | HIGH置信度自动采用，MEDIUM/LOW人工验证 |
| 150%BOM结构复杂 | 结合KLOWER编码辅助判断配置vs替换 |

## 相关文件

- **研究总结**: `outputs/deepseek_tools/bom_analysis/TBS130_RESEARCH_SUMMARY.md`
- **实现指南**: `outputs/deepseek_tools/bom_analysis/current/tbs130_implementation_guide.md`
- **验证清单**: `outputs/deepseek_tools/bom_analysis/current/tbs130_validation_checklist.md`
- **AI审计摘要**: `outputs/deepseek_tools/bom_analysis/current/tbs130_ai_audit_summary.md`

## 演进历史

- **v1.0.0** (2026-06-10): 基于TBS130项目提炼，支持4时间点多CSV分析
