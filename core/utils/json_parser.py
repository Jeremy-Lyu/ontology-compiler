"""Unified JSON Parser - 统一 JSON 解析封装层

目标：解决文件处理过程中的 JSON 解析失败、非标准输入格式、LLM 输出结构不稳定等问题。

提供的能力：
1. normalize_text_to_json_candidate - 从文本中提取 JSON 候选块
2. safe_json_parse - 安全 JSON 解析，带容错和回退
3. extract_json_blocks - 提取多个 JSON 块
4. repair_common_json_issues - 修复常见 JSON 问题（尾逗号、代码块包裹等）
5. safe_json_load - 安全读取 JSON 文件
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union


class ParseResult:
    """解析结果封装"""
    
    def __init__(self, success: bool, data: Any = None, original_text: str = "", 
                 error_message: str = "", repaired_text: str = ""):
        self.success = success
        self.data = data
        self.original_text = original_text
        self.error_message = error_message
        self.repaired_text = repaired_text
    
    def __bool__(self):
        return self.success
    
    def __repr__(self):
        return f"ParseResult(success={self.success}, data={type(self.data).__name__})"


def extract_json_blocks(text: str) -> List[str]:
    """从文本中提取所有 JSON 块"""
    if not text:
        return []
    
    blocks = []
    stack = []
    start_idx = None
    
    for i, char in enumerate(text):
        if char == '{':
            if not stack:
                start_idx = i
            stack.append(char)
        elif char == '}':
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    blocks.append(text[start_idx:i+1])
                    start_idx = None
        elif char == '[' and not stack:
            if not stack:
                start_idx = i
            stack.append(char)
        elif char == ']':
            if stack and stack[-1] == '[':
                stack.pop()
                if not stack and start_idx is not None:
                    blocks.append(text[start_idx:i+1])
                    start_idx = None
    
    return blocks


def normalize_text_to_json_candidate(text: str) -> str:
    """将文本标准化为 JSON 候选
    
    处理：
    1. 移除代码块标记（```json ... ```）
    2. 移除前后解释文本
    3. 移除 BOM
    """
    if not text:
        return ""
    
    text = text.strip()
    
    text = text.lstrip('\ufeff')
    
    json_block_pattern = re.compile(r'```(?:json)?\s*(.*?)\s*```', re.DOTALL)
    match = json_block_pattern.search(text)
    if match:
        return match.group(1).strip()
    
    json_start = text.find('{')
    json_end = text.rfind('}')
    array_start = text.find('[')
    array_end = text.rfind(']')
    
    if json_start != -1 and json_end != -1 and json_start < json_end:
        if array_start != -1 and array_start < json_start:
            return text[array_start:array_end+1].strip()
        return text[json_start:json_end+1].strip()
    
    if array_start != -1 and array_end != -1 and array_start < array_end:
        return text[array_start:array_end+1].strip()
    
    return text


def repair_common_json_issues(text: str) -> str:
    """修复常见的 JSON 格式问题
    
    修复项：
    1. 尾逗号（最后一个元素后的逗号）
    2. 单引号改为双引号
    3. 未加引号的键名
    4. 多行字符串中的换行符
    """
    if not text:
        return ""
    
    text = text.replace('\r\n', '\n')
    
    text = re.sub(r',\s*([}\]])', r'\1', text)
    
    in_string = False
    escaped = False
    result = []
    
    for i, char in enumerate(text):
        if char == '"' and not escaped:
            in_string = not in_string
            result.append(char)
        elif char == "'" and not in_string:
            result.append('"')
        elif char == '\\' and not escaped:
            escaped = True
            result.append(char)
        else:
            escaped = False
            result.append(char)
    
    text = ''.join(result)
    
    text = re.sub(r'([{,]\s*)([a-zA-Z_]\w*)(\s*:)', r'\1"\2"\3', text)
    
    return text


def safe_json_parse(text: str, repair: bool = True, 
                    preserve_original: bool = True) -> ParseResult:
    """安全解析 JSON，带容错机制
    
    步骤：
    1. 标准化文本
    2. 尝试直接解析
    3. 如果失败且 repair=True，尝试修复后解析
    4. 返回解析结果
    """
    original_text = text if preserve_original else ""
    normalized = normalize_text_to_json_candidate(text)
    
    if not normalized:
        return ParseResult(
            success=False,
            original_text=original_text,
            error_message="No JSON content found"
        )
    
    try:
        data = json.loads(normalized)
        return ParseResult(
            success=True,
            data=data,
            original_text=original_text
        )
    except json.JSONDecodeError as e:
        if not repair:
            return ParseResult(
                success=False,
                original_text=original_text,
                error_message=f"JSON parse error: {str(e)}"
            )
    
    repaired = repair_common_json_issues(normalized)
    if repaired != normalized:
        try:
            data = json.loads(repaired)
            return ParseResult(
                success=True,
                data=data,
                original_text=original_text,
                repaired_text=repaired
            )
        except json.JSONDecodeError as e:
            pass
    
    blocks = extract_json_blocks(text)
    for block in blocks:
        try:
            data = json.loads(block)
            return ParseResult(
                success=True,
                data=data,
                original_text=original_text,
                repaired_text=block
            )
        except json.JSONDecodeError:
            repaired_block = repair_common_json_issues(block)
            try:
                data = json.loads(repaired_block)
                return ParseResult(
                    success=True,
                    data=data,
                    original_text=original_text,
                    repaired_text=repaired_block
                )
            except json.JSONDecodeError:
                continue
    
    return ParseResult(
        success=False,
        original_text=original_text,
        error_message="Failed to parse JSON after all attempts"
    )


def safe_json_load(file_path: str, repair: bool = True) -> ParseResult:
    """安全读取 JSON 文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return ParseResult(
            success=False,
            error_message=f"Failed to read file: {str(e)}"
        )
    
    return safe_json_parse(content, repair=repair)


def parse_llm_output(text: str) -> ParseResult:
    """解析 LLM 输出的专用函数"""
    normalized = normalize_text_to_json_candidate(text)
    
    if not normalized:
        return ParseResult(
            success=False,
            original_text=text,
            error_message="No JSON content found in LLM output"
        )
    
    return safe_json_parse(normalized, repair=True)


def validate_structure(data: Any, required_fields: List[str] = None,
                       expected_types: Dict[str, type] = None) -> Tuple[bool, List[str]]:
    """校验数据结构"""
    errors = []
    
    if not isinstance(data, dict):
        return False, ["Data is not a dictionary"]
    
    if required_fields:
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
    
    if expected_types:
        for field, expected_type in expected_types.items():
            if field in data and not isinstance(data[field], expected_type):
                errors.append(
                    f"Field '{field}' has type {type(data[field]).__name__}, "
                    f"expected {expected_type.__name__}"
                )
    
    return len(errors) == 0, errors