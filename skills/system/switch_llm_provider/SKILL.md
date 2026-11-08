# LLM Provider Switcher Skill

## 概述
快速切换本地 LLM 提供商配置，支持多个 LLM 后端的灵活切换。

## 核心能力
- 快速切换 LLM 提供商配置
- 支持多个 LLM 后端（DeepSeek, Dify 等）
- 无需修改代码即可切换提供商

## 使用方法

```bash
python scripts/switch_llm_provider.py <provider>
```

### 参数
- `provider`: LLM 提供商名称，可选值：
  - `deepseek`: 使用 DeepSeek 提供商
  - `dify`: 使用 Dify 提供商

## 输入输出
- **输入**: LLM 提供商名称
- **输出**: 更新后的配置文件

## 示例

```bash
# 切换到 DeepSeek
python scripts/switch_llm_provider.py deepseek

# 切换到 Dify
python scripts/switch_llm_provider.py dify
```