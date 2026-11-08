# Ontology RAG Evaluation Skill

## 概述
执行 ontology RAG 评估管道，用于系统协作分析。评估 RAG 系统在 ontology 数据检索和推理方面的性能。

## 核心能力
- 运行系统协作分析的 RAG 评估
- 测试 ontology 数据检索准确性
- 生成评估报告和指标

## 使用方法

```bash
python scripts/diagnostics/run_systems_rag.py
```

## 输入输出
- **输入**: 自动使用默认配置
- **输出**: RAG 评估报告、性能指标

## 配置
默认从项目配置中读取 RAG 参数和 ontology 数据路径。