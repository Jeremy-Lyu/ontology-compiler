# Complex RAG Orchestrator Skill

多 Agent 自动任务分解与编排的增强型 RAG 分析技能。

## 核心能力

1. **TaskDecomposer**: LLM 自动分解复杂查询为子任务
2. **AgentCoordinator**: DAG 编排多 Agent 并行执行
3. **ResultSynthesizer**: 多源结果智能整合生成报告

## 使用方式

```bash
# 分析特定 MIX 的系统协作
python scripts/run_complex_rag_analysis.py \
    "分析MIX=544769的系统协作和物料状态" \
    --mix 544769 \
    --save

# C-Note 变更影响分析
python scripts/run_complex_rag_analysis.py \
    "C-Note变更对排产的影响分析" \
    --mix 544769
```

## 任务分解示例

复杂查询 **"分析MIX=544769的系统协作"** 分解为:

| 任务ID | 类型 | 描述 | Agent | 依赖 |
|--------|------|------|-------|------|
| task_1 | RAG_RESEARCH | 查询MBOM系统定义 | RAG_Researcher | - |
| task_2 | DATA_PROFILE | Layer1数据画像 | DataAgent | - |
| task_3 | DATA_JOIN | TBS130×Partcache JOIN | DataAgent | task_2 |
| task_4 | SYS_COLLAB | 系统协作分析 | SystemAnalyst | task_1,task_3 |
| task_5 | SYNTHESIS | 综合报告生成 | Synthesizer | task_4 |

## 数据要求

- 样本数据路径: `Sample_data/`
- 支持表: TBS130, Partcache, TCP700, TCP712

## 输出

生成参考文档级别的 6 章节深度分析报告:
1. 系统全景图
2. 端到端数据流
3. 核心难点分析
4. 现状评分
5. 改进建议
6. 综合结论
