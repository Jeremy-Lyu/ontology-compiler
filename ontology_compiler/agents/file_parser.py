"""
file_parser.py - 统一文件解析入口

支持格式:
  .sql   - DDL 语句
  .json  - 结构化数据
  .md    - Markdown 文档
  .txt   - 纯文本
  .pdf   - PDF 文档 (需要 PyPDF2)
  .docx  - Word 文档 (需要 python-docx)
  .pptx  - PowerPoint (需要 python-pptx)
  .xlsx  - Excel 表格 (需要 openpyxl)
  .xls   - Excel 旧格式 (需要 xlrd)

用法:
  from ontology_compiler.agents.file_parser import FileParser
  text = FileParser.extract_text('document.pdf')
  data = FileParser.extract_structured('data.xlsx')
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def markdown_table(rows: List[List[Any]]) -> List[str]:
    """Render a cell grid as Markdown table lines.

    ``list.append`` does not take ``end=``; callers must build each row as one
    string. A table with no rows returns no lines.
    """
    if not rows:
        return []
    width = max(len(row) for row in rows)

    def cell(value: Any) -> str:
        text = "" if value is None else str(value)
        return text.replace("|", "\\|").replace("\n", " ").strip()

    normalized = [
        [cell(row[i]) if i < len(row) else "" for i in range(width)]
        for row in rows
    ]
    header = "| " + " | ".join(normalized[0]) + " |"
    separator = "| " + " | ".join("---" for _ in range(width)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in normalized[1:]]
    return [header, separator, *body, ""]


def _pdf_table_grid(table: Any) -> List[List[Any]]:
    """Read a PyMuPDF table into a cell grid. ``extract()`` is the stable API."""
    extract = getattr(table, "extract", None)
    if callable(extract):
        raw = extract() or []
        return [["" if cell is None else cell for cell in (row or [])] for row in raw]
    cells = getattr(table, "cells", None) or []
    return [[getattr(cell, "text", cell) for cell in row] for row in cells]


class FileParser:
    """
    统一文件解析器。

    所有方法均为 classmethod，无需实例化。
    """

    # 支持的格式分组
    TEXT_FORMATS    = {'.sql', '.txt', '.md'}
    JSON_FORMATS    = {'.json'}
    PDF_FORMATS     = {'.pdf'}
    WORD_FORMATS    = {'.docx', '.doc'}  # 添加 .doc 支持
    PPTX_FORMATS    = {'.pptx', '.ppt'}  # 添加 .ppt 支持
    EXCEL_FORMATS   = {'.xlsx', '.xls'}
    CSV_FORMATS     = {'.csv'}

    ALL_SUPPORTED = (
        TEXT_FORMATS | JSON_FORMATS | PDF_FORMATS |
        WORD_FORMATS | PPTX_FORMATS | EXCEL_FORMATS | CSV_FORMATS
    )

    # ------------------------------------------------------------------
    # 统一文本提取接口
    # ------------------------------------------------------------------

    @classmethod
    def extract_text(cls, file_path: str, page_range: tuple[int, int] | None = None) -> str:
        """
        从任意支持格式提取纯文本。

        Args:
            file_path: 文件路径
            page_range: 可选 (start, end) 1-indexed，仅对PDF生效，提取指定页码范围

        Returns:
            str: 提取的文本内容
        Raises:
            ValueError: 不支持的文件格式
            FileNotFoundError: 文件不存在
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        suffix = path.suffix.lower()
        logger.info(f"[FileParser] 解析文件: {path.name} ({suffix})")

        if suffix in cls.TEXT_FORMATS:
            text = cls._read_text(path)
        elif suffix in cls.JSON_FORMATS:
            text = cls._read_json_as_text(path)
        elif suffix in cls.PDF_FORMATS:
            text = cls._read_pdf(path, page_range=page_range)
        elif suffix == '.docx':
            text = cls._read_docx(path)
        elif suffix == '.doc':
            text = cls._read_doc(path)
        elif suffix in cls.PPTX_FORMATS:
            text = cls._read_pptx(path)
        elif suffix in cls.EXCEL_FORMATS:
            text = cls._read_excel_as_text(path)
        elif suffix in cls.CSV_FORMATS:
            text = cls._read_csv_as_text(path)
        else:
            raise ValueError(
                f"不支持的文件格式: {suffix}\n"
                f"支持的格式: {', '.join(sorted(cls.ALL_SUPPORTED))}"
            )

        # [OPTIMIZATION] 文本去重：移除高频重复行，减少传给LLM的文本量
        # 实测: NOTE-CMP docx 125K chars中有大量重复邮箱(22次)、更新说明(19次)
        # KDP Academy pptx有重复幻灯片内容(25次)
        if text and len(text) > 5000:
            original_len = len(text)
            text = cls._deduplicate_text(text)
            if len(text) < original_len:
                logger.info(f"  [FileParser] 文本去重: {original_len:,} -> {len(text):,} chars (节省{(original_len - len(text)) / original_len * 100:.0f}%)")

        return text

    @classmethod
    def _deduplicate_text(cls, text: str) -> str:
        """[OPTIMIZATION] 移除高频重复行，保留文档结构。

        策略:
        1. 保留结构标记行(如 "--- Slide N ---")
        2. 对出现>2次的非结构行，只保留第一次出现
        3. 移除连续重复的空行
        """
        from collections import Counter

        lines = text.split('\n')

        # 统计每行出现次数（非空、非结构标记）
        line_counter = Counter()
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # 跳过幻灯片结构标记
            if stripped.startswith('--- Slide'):
                continue
            line_counter[stripped] += 1

        # 找出需要去重的高频行（出现>2次且长度>5）
        lines_to_dedupe = {
            line for line, count in line_counter.items()
            if count > 2 and len(line) > 5
        }

        if not lines_to_dedupe:
            return text

        # 重建文本，对高频行只保留第一次出现
        seen = set()
        result = []
        prev_empty = False
        for line in lines:
            stripped = line.strip()
            if stripped in lines_to_dedupe:
                if stripped not in seen:
                    seen.add(stripped)
                    result.append(line)
                    prev_empty = False
                # 跳过重复行
            elif not stripped:
                # 合并连续空行
                if not prev_empty:
                    result.append(line)
                    prev_empty = True
            else:
                result.append(line)
                prev_empty = False

        return '\n'.join(result)

    @classmethod
    def extract_structured(cls, file_path: str) -> Dict[str, Any]:
        """
        从结构化文件提取数据（JSON / Excel）。

        Returns:
            dict: 结构化数据
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        suffix = path.suffix.lower()

        if suffix in cls.JSON_FORMATS:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif suffix in cls.EXCEL_FORMATS:
            return cls._read_excel_structured(path)
        elif suffix in cls.CSV_FORMATS:
            return cls._read_csv_structured(path)
        else:
            # 其他格式返回 {'text': ...}
            return {'text': cls.extract_text(file_path)}

    @classmethod
    def can_parse(cls, file_path: str) -> bool:
        """检查是否支持该文件格式"""
        return Path(file_path).suffix.lower() in cls.ALL_SUPPORTED

    @classmethod
    def extract_markdown(cls, file_path: str) -> str:
        """提取文件内容并渲染为整理好格式的Markdown字符串
        支持: pptx/ppt, docx/doc, pdf, xlsx/xls, csv, md, txt, json
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        if suffix in [".pptx", ".ppt"]:
            return cls._render_pptx_markdown(path)
        elif suffix in [".docx", ".doc"]:
            return cls._render_docx_markdown(path)
        elif suffix == ".pdf":
            return cls._render_pdf_markdown(path)
        elif suffix in [".xlsx", ".xls"]:
            return cls._render_excel_markdown(path)
        elif suffix == ".csv":
            return cls._render_csv_markdown(path)
        elif suffix in [".md", ".markdown", ".txt", ".json"]:
            return cls._render_plain_markdown(path)
        else:
            # 未知类型，返回原始文本
            return cls._read_text(path)
    
    @classmethod
    def _render_pptx_markdown(cls, path: Path) -> str:
        """渲染pptx/ppt为Markdown格式"""
        try:
            from pptx import Presentation
        except ImportError:
            raise RuntimeError("未安装 python-pptx，请运行: pip install python-pptx")
        
        prs = None
        try:
            prs = Presentation(str(path))
        except Exception as e:
            logger.warning(f"PPTX 打开失败: {e}")
            return cls._read_text(path)
        
        md_lines = [f"# 演示文稿: {path.name}", ""]
        
        for slide_idx, slide in enumerate(prs.slides, 1):
            md_lines.append(f"## 幻灯片 {slide_idx}")
            
            # 提取文本形状
            text_shapes = []
            tables = []
            
            for shape in slide.shapes:
                try:
                    if hasattr(shape, "text") and shape.text and shape.text.strip():
                        text_shapes.append(shape)
                    if getattr(shape, "has_table", False):
                        tables.append(shape.table)
                except Exception:
                    continue
            
            # 先添加标题
            for shape in text_shapes:
                if shape.text.strip().startswith("#") or shape.text.strip().startswith("*"):
                    md_lines.append(f"{shape.text.strip()}")
                    text_shapes.remove(shape)
                    break
            
            # 添加普通文本
            for shape in text_shapes:
                text = shape.text.strip()
                if text:
                    md_lines.append(text)
                    md_lines.append("")
            
            # 添加表格。table.columns 是集合，不能拿去 range()。
            for table in tables:
                grid = [[cell.text for cell in row.cells] for row in table.rows]
                md_lines.extend(markdown_table(grid))
            
            # 添加备注
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text:
                notes = slide.notes_slide.notes_text_frame.text.strip()
                if notes:
                    md_lines.append(f"> 备注: {notes}")
                    md_lines.append("")
        
        return "\n".join(md_lines)
    
    @classmethod
    def _render_docx_markdown(cls, path: Path) -> str:
        """渲染docx/doc为Markdown格式"""
        try:
            from docx import Document
        except ImportError:
            raise RuntimeError("未安装 python-docx，请运行: pip install python-docx")
        
        doc = Document(str(path))
        md_lines = [f"# 文档: {path.name}", ""]
        
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            
            # 处理样式
            style = para.style.name.lower()
            if style.startswith("heading 1"):
                md_lines.append(f"# {text}")
            elif style.startswith("heading 2"):
                md_lines.append(f"## {text}")
            elif style.startswith("heading 3"):
                md_lines.append(f"### {text}")
            elif style.startswith("heading"):
                md_lines.append(f"#### {text}")
            elif style.startswith("list bullet"):
                md_lines.append(f"- {text}")
            elif style.startswith("list number"):
                md_lines.append(f"1. {text}")
            else:
                md_lines.append(text)
            md_lines.append("")
        
        for table in doc.tables:
            grid = [[cell.text for cell in row.cells] for row in table.rows]
            md_lines.extend(markdown_table(grid))
        
        return "\n".join(md_lines)
    
    @classmethod
    def _render_pdf_markdown(cls, path: Path) -> str:
        """渲染pdf为Markdown格式"""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise RuntimeError("未安装 PyMuPDF，请运行: pip install PyMuPDF")
        
        doc = fitz.open(str(path))
        try:
            md_lines = [f"# PDF文档: {path.name}", f"页数: {doc.page_count}", ""]
            
            for page_idx, page in enumerate(doc, 1):
                md_lines.append(f"## 第 {page_idx} 页")
                
                # 提取文本
                text = page.get_text("text")
                if text:
                    # 简单的标题识别（基于换行和大写）
                    lines = text.split("\n")
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        # 简单判断标题
                        if len(line) < 30 and (line.isupper() or line.startswith(("1.", "2.", "3.", "一、", "二、", "三、"))):
                            md_lines.append(f"### {line}")
                        else:
                            md_lines.append(line)
                    md_lines.append("")
                
                try:
                    tables = page.find_tables()
                    for table in tables:
                        md_lines.extend(markdown_table(_pdf_table_grid(table)))
                except Exception:
                    pass
            return "\n".join(md_lines)
        finally:
            doc.close()

    @classmethod
    def _read_text(cls, path: Path) -> str:
        """读取 .sql / .txt / .md"""
        text = path.read_text(encoding='utf-8', errors='ignore')
        logger.debug(f"  读取完成: {len(text):,} 字符")
        return text

    @classmethod
    def _read_json_as_text(cls, path: Path) -> str:
        """读取 .json 并转为格式化文本"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        text = json.dumps(data, ensure_ascii=False, indent=2)
        logger.debug(f"  JSON 读取完成: {len(text):,} 字符")
        return text

    # ------------------------------------------------------------------
    # 内部实现：PDF
    # ------------------------------------------------------------------

    @classmethod
    def _read_pdf(cls, path: Path, page_range: tuple[int, int] | None = None) -> str:
        """读取 .pdf，可选指定页码范围 (1-indexed, inclusive)

        输出尽量保留页级结构，并用清晰的 page 标记包裹每页文本，
        方便后续 provenance / evidence 对齐。
        """
        def _clean_pdf_text(text: str) -> str:
            if not text:
                return ""
            lines = [line.rstrip() for line in text.splitlines()]
            cleaned = []
            prev_blank = False
            for line in lines:
                s = line.strip()
                if not s:
                    if not prev_blank:
                        cleaned.append("")
                    prev_blank = True
                    continue
                prev_blank = False
                cleaned.append(s)
            return "\n".join(cleaned).strip()

        # 优先使用 pdfplumber（质量更好）
        try:
            import pdfplumber
            with pdfplumber.open(str(path)) as pdf:
                total = len(pdf.pages)
                if page_range:
                    start, end = page_range
                    start = max(1, start) - 1  # 0-indexed
                    end = min(total, end)
                    selected = pdf.pages[start:end]
                    page_items = []
                    for idx, page in enumerate(selected, start=start + 1):
                        raw = page.extract_text() or ''
                        cleaned = _clean_pdf_text(raw)
                        page_items.append(f'--- Page {idx} ---\n{cleaned}')
                    logger.info(f"  PDF 读取完成(pdfplumber): 页 {page_range[0]}-{end} / 共{total}页, {sum(len(p) for p in page_items):,} 字符")
                else:
                    page_items = []
                    for idx, page in enumerate(pdf.pages, start=1):
                        raw = page.extract_text() or ''
                        cleaned = _clean_pdf_text(raw)
                        page_items.append(f'--- Page {idx} ---\n{cleaned}')
                    logger.info(f"  PDF 读取完成(pdfplumber): {total} 页, {sum(len(p) for p in page_items):,} 字符")
            text = '\n\n'.join(page_items)
            return text
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"  pdfplumber 失败: {e}, 尝试 PyPDF2")

        # 备用：PyPDF2
        try:
            import PyPDF2
            with open(str(path), 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                total = len(reader.pages)
                if page_range:
                    start, end = page_range
                    start = max(1, start) - 1
                    end = min(total, end)
                    selected = reader.pages[start:end]
                    pages = [page.extract_text() or '' for page in selected]
                    logger.info(f"  PDF 读取完成(PyPDF2): 页 {page_range[0]}-{end} / 共{total}页, {sum(len(p) for p in pages):,} 字符")
                else:
                    pages = [page.extract_text() or '' for page in reader.pages]
                    logger.info(f"  PDF 读取完成(PyPDF2): {total} 页, {sum(len(p) for p in pages):,} 字符")
            text = '\n\n'.join(pages)
            return text
        except ImportError:
            raise RuntimeError(
                "未安装 PDF 解析库，请运行:\n"
                "  pip install pdfplumber   # 推荐\n"
                "  # 或\n"
                "  pip install PyPDF2"
            )

    # ------------------------------------------------------------------
    # 内部实现：Word
    # ------------------------------------------------------------------

    @classmethod
    def _read_docx(cls, path: Path) -> str:
        """读取 .docx
        
        [FIX] 添加超时处理和错误恢复，防止大文件或复杂格式导致卡住
        """
        try:
            from docx import Document
        except ImportError:
            raise RuntimeError(
                "未安装 python-docx，请运行: pip install python-docx"
            )

        import time
        start_time = time.time()
        MAX_PARSE_TIME = 120  # 2分钟超时
        
        doc = Document(str(path))
        parts = []

        # 段落（添加超时保护）
        try:
            for i, para in enumerate(doc.paragraphs):
                if time.time() - start_time > MAX_PARSE_TIME:
                    logger.warning(f"  DOCX 段落解析超时，已处理 {i} 个段落")
                    break
                t = para.text.strip()
                if t:
                    parts.append(t)
        except Exception as e:
            logger.warning(f"  DOCX 段落解析失败: {e}")

        # 表格（添加超时保护）
        try:
            for table in doc.tables:
                if time.time() - start_time > MAX_PARSE_TIME:
                    logger.warning(f"  DOCX 表格解析超时")
                    break
                for row in table.rows:
                    row_texts = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_texts:
                        parts.append(' | '.join(row_texts))
        except Exception as e:
            logger.warning(f"  DOCX 表格解析失败: {e}")

        # 页眉/页脚（可选，跳过不影响核心内容）
        try:
            for section in doc.sections:
                for para in section.header.paragraphs:
                    t = para.text.strip()
                    if t:
                        parts.append(t)
        except Exception as e:
            logger.warning(f"Failed to extract DOCX header text: {e}")

        text = '\n'.join(parts)
        elapsed = time.time() - start_time
        logger.info(f"  DOCX 读取完成: {len(parts)} 段落, {len(text):,} 字符, 耗时 {elapsed:.1f}s")
        
        if elapsed > 60:
            logger.warning(f"  DOCX 解析耗时较长 ({elapsed:.1f}s)，建议检查文件大小")
        
        return text

    @classmethod
    def _read_doc(cls, path: Path) -> str:
        """读取旧 .doc 格式 (Word 97-2003)"""
        # 尝试方法1: 使用 pywin32 (Windows 原生 COM 支持)
        try:
            import win32com.client
            word = win32com.client.Dispatch("Word.Application")
            word.Visible = False
            doc = word.Documents.Open(str(path))
            text = doc.Content.Text
            doc.Close()
            word.Quit()
            logger.info(f"  DOC 读取完成(pywin32): {len(text):,} 字符")
            return text
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"pywin32 读取失败: {e}")

        # 尝试方法2: 使用 textract (通用文本提取)
        try:
            import textract
            text = textract.process(str(path)).decode('utf-8', errors='ignore')
            logger.info(f"  DOC 读取完成(textract): {len(text):,} 字符")
            return text
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"textract 读取失败: {e}")

        # 尝试方法3: 使用 docx2txt (某些版本支持 .doc)
        try:
            import docx2txt
            text = docx2txt.process(str(path))
            logger.info(f"  DOC 读取完成(docx2txt): {len(text):,} 字符")
            return text
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"docx2txt 读取失败: {e}")

        # 尝试方法4: 使用 antiword (Linux/macOS 常用工具)
        try:
            import subprocess
            result = subprocess.run(
                ['antiword', str(path)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )
            if result.returncode == 0:
                text = result.stdout
                logger.info(f"  DOC 读取完成(antiword): {len(text):,} 字符")
                return text
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"antiword 读取失败: {e}")

        # 所有方法都失败，给出友好提示
        raise RuntimeError(
            f"无法读取 .doc 文件: {path.name}\n"
            "请安装文本提取库之一:\n"
            "  pip install pywin32 (Windows)\n"
            "  pip install textract\n"
            "  pip install docx2txt\n"
            "或在 Linux/macOS 上安装 antiword:\n"
            "  sudo apt-get install antiword"
        )

    # ------------------------------------------------------------------
    # 内部实现：PowerPoint
    # ------------------------------------------------------------------

    @classmethod
    def _read_pptx(cls, path: Path) -> str:
        """读取 .pptx / .ppt"""
        is_legacy_ppt = path.suffix.lower() == ".ppt"

        if is_legacy_ppt:
            # 方法1: 使用 PowerPoint COM 对象 (Windows + Office)
            try:
                import subprocess
                ps_script = (
                    "$ErrorActionPreference='Stop';"
                    "$pp=New-Object -ComObject PowerPoint.Application;"
                    "$pres=$pp.Presentations.Open([System.IO.Path]::GetFullPath('" + str(path).replace("'", "''") + "'),$true,$false,$false);"
                    "$parts=@();"
                    "foreach($slide in $pres.Slides){"
                    "$slideParts=@();"
                    "foreach($shape in $slide.Shapes){"
                    "if($shape.HasTextFrame -and $shape.TextFrame.HasText){"
                    "$t=$shape.TextFrame.TextRange.Text.Trim();"
                    "if($t){$slideParts+=$t}"
                    "}"
                    "}"
                    "if($slideParts){$parts+='--- Slide '+$slide.SlideIndex+' ---';$parts+=$slideParts}"
                    "}"
                    "$pres.Close();$pp.Quit();"
                    "$parts -join \"`n\""
                )
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", ps_script],
                    capture_output=True, text=True, timeout=60,
                    encoding='utf-8', errors='ignore'
                )
                if result.returncode == 0 and result.stdout.strip():
                    text = result.stdout.strip()
                    logger.info(f"  PPT 读取完成(PowerPoint COM): {len(text):,} 字符")
                    return text
            except Exception as e:
                logger.debug(f"  PowerPoint COM 读取失败: {e}")

            # 方法2: 使用 antiword (Linux/macOS)
            try:
                import subprocess
                result = subprocess.run(
                    ["antiword", str(path)],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
            except Exception:
                pass

            # 方法3: 使用 textract
            try:
                import textract
                text = textract.process(str(path)).decode('utf-8')
                if text and text.strip():
                    return text.strip()
            except Exception:
                pass

            logger.warning(f"  .ppt (legacy) 文件无法解析（需要 PowerPoint 或 textract），跳过: {path.name}")
            return ""

        try:
            from pptx import Presentation
        except ImportError:
            raise RuntimeError(
                "未安装 python-pptx，请运行: pip install python-pptx"
            )

        prs = None
        try:
            prs = Presentation(str(path))
        except Exception as e:
            logger.warning(f"  PPTX 打开失败，尝试使用绝对路径: {e}")
            try:
                prs = Presentation(path.resolve())
            except Exception as e2:
                logger.error(f"  PPTX 文件损坏或无法解析，跳过: {path.name}, 错误: {e2}")
                return ""

        def _shape_text(shape) -> str:
            try:
                if hasattr(shape, "text") and shape.text and shape.text.strip():
                    return shape.text.strip()
            except Exception:
                pass
            return ""

        def _shape_kind(shape) -> str:
            try:
                if getattr(shape, "has_table", False):
                    return "table"
                if getattr(shape, "shape_type", None) is not None and getattr(shape, "is_placeholder", False):
                    return "placeholder"
                if getattr(shape, "text", ""):
                    text = getattr(shape, "text", "").strip()
                    if text and "\n" in text:
                        return "bullet_or_multiline"
                    return "text"
            except Exception:
                pass
            return "shape"

        parts = []
        try:
            for slide_idx, slide in enumerate(prs.slides, 1):
                slide_parts = []
                title_text = ""
                for shape in slide.shapes:
                    text = _shape_text(shape)
                    if not text:
                        continue
                    kind = _shape_kind(shape)
                    if not title_text and kind == "placeholder":
                        title_text = text.splitlines()[0].strip()
                        continue
                    if getattr(shape, "has_table", False):
                        try:
                            table_rows = []
                            for row in shape.table.rows:
                                row_vals = [cell.text.strip() for cell in row.cells if cell.text and cell.text.strip()]
                                if row_vals:
                                    table_rows.append(" | ".join(row_vals))
                            if table_rows:
                                slide_parts.append("[Table]\n" + "\n".join(table_rows))
                            continue
                        except Exception as e:
                            logger.debug(f"  Slide {slide_idx} table extraction skipped: {e}")
                    if "\n" in text:
                        bullets = [line.strip() for line in text.splitlines() if line.strip()]
                        if bullets:
                            slide_parts.append("[Bullet]\n" + "\n".join(bullets))
                    else:
                        slide_parts.append(text)
                try:
                    if slide.has_notes_slide:
                        notes_tf = slide.notes_slide.notes_text_frame
                        notes_text = notes_tf.text.strip() if notes_tf else ''
                        if notes_text:
                            slide_parts.append(f'[Notes] {notes_text}')
                except Exception as _e:
                    logger.debug(f'  Slide {slide_idx} notes extraction skipped: {_e}')
                if title_text:
                    parts.append(f'--- Slide {slide_idx}: {title_text} ---')
                elif slide_parts:
                    parts.append(f'--- Slide {slide_idx} ---')
                parts.extend(slide_parts)
        except Exception as e3:
            logger.error(f"  PPTX 解析过程中出错，返回已提取内容: {path.name}, 错误: {e3}")

        text = '\n'.join(parts)
        if text:
            logger.info(f"  PPTX 读取完成: {len(prs.slides)} 页, {len(text):,} 字符 (含备注)")
        else:
            logger.warning(f"  PPTX 文件为空或无法提取内容: {path.name}")
        return text

    # ------------------------------------------------------------------
    # 内部实现：Excel
    # ------------------------------------------------------------------

    @classmethod
    def _read_excel_as_text(cls, path: Path) -> str:
        """读取 Excel 并转为文本（每行用 tab 分隔）
        
        [OPTIMIZATION]: 对大型xlsx文件进行智能采样，避免生成过多文本导致chunk爆炸
        - 移除全None列
        - 限制最大行数（50行）
        - 限制最大字符数（15000字符）
        """
        data = cls._read_excel_structured(path)
        lines = []
        MAX_ROWS_PER_SHEET = 50  # 每个sheet最多保留50行
        MAX_TOTAL_CHARS = 15000  # 总字符数限制
        
        for sheet_name, rows in data.get('sheets', {}).items():
            if not rows:
                lines.append(f'=== Sheet: {sheet_name} (empty) ===')
                continue
            
            # 找出全None列并移除
            all_keys = set()
            for row in rows[:10]:  # 检查前10行确定非空列
                for k, v in row.items():
                    if v is not None and str(v).strip() != '':
                        all_keys.add(k)
            
            # 如果没有非空列，保留所有列
            if not all_keys:
                all_keys = set(rows[0].keys()) if rows else set()
            
            # 按原始顺序排列列
            if rows:
                original_keys = list(rows[0].keys())
                kept_keys = [k for k in original_keys if k in all_keys]
            else:
                kept_keys = list(all_keys)
            
            lines.append(f'=== Sheet: {sheet_name} ({len(rows)} rows, showing first {min(len(rows), MAX_ROWS_PER_SHEET)}) ===')
            
            # 添加表头
            lines.append('\t'.join(str(k) for k in kept_keys))
            
            # 添加数据行（限制数量）
            for row in rows[:MAX_ROWS_PER_SHEET]:
                values = [str(row.get(k, '')) if row.get(k) is not None else '' for k in kept_keys]
                lines.append('\t'.join(values))
            
            # 检查总字符数
            current_chars = sum(len(l) for l in lines)
            if current_chars > MAX_TOTAL_CHARS:
                # 截断到限制内
                truncated_lines = []
                char_count = 0
                for line in lines:
                    if char_count + len(line) > MAX_TOTAL_CHARS:
                        truncated_lines.append(f'... [truncated, total {len(lines)} lines]')
                        break
                    truncated_lines.append(line)
                    char_count += len(line)
                lines = truncated_lines
                break
        
        return '\n'.join(lines)

    @classmethod
    def _read_excel_structured(cls, path: Path) -> Dict[str, Any]:
        """读取 Excel 返回结构化数据"""
        try:
            import openpyxl
        except ImportError:
            raise RuntimeError(
                "未安装 openpyxl，请运行: pip install openpyxl"
            )

        suffix = path.suffix.lower()

        if suffix == '.xlsx':
            wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
            result = {'file': path.name, 'sheets': {}}

            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                rows = []
                headers = None
                for row in ws.iter_rows(values_only=True):
                    if headers is None:
                        headers = [str(h) if h is not None else f'col_{i}'
                                   for i, h in enumerate(row)]
                        continue
                    row_dict = dict(zip(headers, row))
                    # 跳过全空行
                    if any(v is not None for v in row_dict.values()):
                        rows.append(row_dict)
                result['sheets'][sheet_name] = rows
                logger.debug(f"  Sheet '{sheet_name}': {len(rows)} 行")

            wb.close()
            total_rows = sum(len(r) for r in result['sheets'].values())
            logger.info(f"  Excel 读取完成: {len(wb.sheetnames)} sheets, {total_rows} 行")
            return result

        elif suffix == '.xls':
            try:
                import xlrd
            except ImportError:
                raise RuntimeError(
                    "读取 .xls 需要 xlrd，请运行: pip install xlrd==1.2.0"
                )
            wb = xlrd.open_workbook(str(path))
            result = {'file': path.name, 'sheets': {}}
            for sheet in wb.sheets():
                headers = [str(sheet.cell_value(0, c)) for c in range(sheet.ncols)]
                rows = []
                for r in range(1, sheet.nrows):
                    row_dict = {headers[c]: sheet.cell_value(r, c)
                                for c in range(sheet.ncols)}
                    rows.append(row_dict)
                result['sheets'][sheet.name] = rows
            return result

        raise ValueError(f"不支持的 Excel 格式: {suffix}")

    # ------------------------------------------------------------------
    # 内部实现：CSV
    # ------------------------------------------------------------------

    @classmethod
    def _read_csv_as_text(cls, path: Path) -> str:
        """读取 CSV 并转为文本（每行用 tab 分隔）"""
        import csv
        lines = []
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                for row in reader:
                    lines.append('\t'.join(row))
        except UnicodeDecodeError:
            with open(path, 'r', encoding='latin-1', errors='ignore') as f:
                reader = csv.reader(f)
                for row in reader:
                    lines.append('\t'.join(row))
        text = '\n'.join(lines)
        logger.info(f"  CSV 读取完成: {len(lines)} 行, {len(text):,} 字符")
        return text

    @classmethod
    def _read_csv_structured(cls, path: Path) -> Dict[str, Any]:
        """读取 CSV 返回结构化数据"""
        import csv
        rows = []
        headers = None
        
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                for row in reader:
                    if headers is None:
                        headers = [str(h) if h is not None else f'col_{i}'
                                   for i, h in enumerate(row)]
                        continue
                    row_dict = dict(zip(headers, row))
                    if any(v for v in row_dict.values()):
                        rows.append(row_dict)
        except UnicodeDecodeError:
            with open(path, 'r', encoding='latin-1', errors='ignore') as f:
                reader = csv.reader(f)
                for row in reader:
                    if headers is None:
                        headers = [str(h) if h is not None else f'col_{i}'
                                   for i, h in enumerate(row)]
                        continue
                    row_dict = dict(zip(headers, row))
                    if any(v for v in row_dict.values()):
                        rows.append(row_dict)
        
        result = {'file': path.name, 'rows': rows, 'headers': headers}
        logger.info(f"  CSV 读取完成: {len(rows)} 行, {len(headers)} 列")
        return result

