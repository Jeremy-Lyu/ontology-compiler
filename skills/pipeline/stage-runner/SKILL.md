# Pipeline Stage Runner Skill

## 概述

统一的 Pipeline Stage 执行器，整合原7个独立的 stage-* skills，提供一致的 stage 编排和管理能力。

## 合并历史

**原 Skills** (2026-06-11 前):
- `stage-aggregate/`: 结果聚合
- `stage-catalog/`: 目录构建
- `stage-guard/`: 质量门禁
- `stage-health/`: 健康检查
- `stage-preflight/`: 预检
- `stage-publish/`: 发布
- `stage-rollback/`: 回滚

**合并原因**:
1. 7个 skills 属于同一业务域 (pipeline orchestration)
2. 调用模式一致 (都是 stage-based execution)
3. 减少 cognitive load，避免目录膨胀

## 支持的 Stages

| Stage | 功能 | 原 Skill |
|-------|------|----------|
| `preflight` | 执行前检查 | `stage-preflight` |
| `catalog` | 构建目录 | `stage-catalog` |
| `aggregate` | 结果聚合 | `stage-aggregate` |
| `guard` | 质量门禁 | `stage-guard` |
| `health` | 健康检查 | `stage-health` |
| `publish` | 结果发布 | `stage-publish` |
| `rollback` | 失败回滚 | `stage-rollback` |

## 使用方式

### 单 Stage 执行

```python
from skills.pipeline.stage_runner import StageRunner

runner = StageRunner()
result = runner.execute_stage("catalog", {
    "run_dir": "outputs/run_001",
    "pipeline_config": {...}
})
```

### Pipeline 链式执行

```python
runner = StageRunner()
results = runner.execute_pipeline([
    "preflight",
    "catalog",
    "aggregate",
    "guard",
    "publish"
], run_dir="outputs/run_001")
```

### 命令行调用

```bash
# 单 stage
python -m scripts.pipeline.stage_runner --stage catalog --run-dir outputs/run_001

# 完整 pipeline
python -m scripts.pipeline.stage_runner --pipeline full --run-dir outputs/run_001

# 跳过健康检查
python -m scripts.pipeline.stage_runner --stage guard --skip-health-check
```

## 配置

### Stage 配置

```json
{
  "stage": "catalog",
  "run_dir": "outputs/run_001",
  "pipeline_config": {
    "catalog_rules": ["rule1", "rule2"],
    "validation_level": "strict"
  }
}
```

### Pipeline 定义

```python
DEFAULT_PIPELINE = [
    "preflight",   # 1. 预检
    "catalog",     # 2. 目录
    "aggregate",   # 3. 聚合
    "guard",       # 4. 门禁
    "health",      # 5. 健康
    "publish"      # 6. 发布
]
```

## 错误处理

| 错误类型 | 行为 | 备注 |
|----------|------|------|
| Stage 失败 | 停止并触发 rollback | 除非配置了 continue_on_error |
| 健康检查失败 | 跳过 stage 并告警 | 需人工干预 |
| 配置缺失 | 预检阶段拦截 | 提前暴露问题 |

## 向后兼容

原 `stage-*` skills 的调用方式仍支持，但内部路由到统一的 `stage-runner`:

```python
# 旧方式 (仍支持，但会发出 DeprecationWarning)
registry.get_skill("stage_catalog")  # 内部映射到 stage-runner with stage="catalog"

# 新方式 (推荐)
registry.get_skill("stage_runner").execute_stage("catalog", {...})
```

## 架构图

```
SkillRegistry
    ↓
StageRunner (统一入口)
    ↓
StageDispatcher
    ├── PreflightStage
    ├── CatalogStage
    ├── AggregateStage
    ├── GuardStage
    ├── HealthStage
    ├── PublishStage
    └── RollbackStage
```

## 迁移指南

### 从旧 skills 迁移

1. **代码调用迁移**
   ```python
   # 旧代码
   from skills.stage_catalog import CatalogStage
   result = CatalogStage().execute(...)
   
   # 新代码
   from skills.pipeline.stage_runner import StageRunner
   result = StageRunner().execute_stage("catalog", ...)
   ```

2. **配置迁移**
   ```json
   // 旧配置 (stage-catalog/skill.json)
   {"catalog_specific": "config"}
   
   // 新配置 (stage-runner/skill.json 中的 sub_skills)
   {"sub_skills": [{"stage": "catalog", "config": {...}}]}
   ```

## 关联文档

- [Skills 构建指南](../README.md)
- [IMPROVEMENT_EXECUTION_TRACKER.md](../../docs/IMPROVEMENT_EXECUTION_TRACKER.md)

---

*合并日期: 2026-06-11*  
*原 7 个 stage-* skills 已归档至 `archived/skills_20260611/`*
