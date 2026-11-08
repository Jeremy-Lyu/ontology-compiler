"""Phase 3: Structured Data Extractor with BOM Analysis.

Consolidated:
- Original StructuredDataExtractor (CSV/Excel extraction + BOM time-series)
- excel_adapter.py core logic (field catalog mapping, quality profiling)

R3 Consolidation (2026-06-29):
- Merged _load_catalog_aliases, normalize_columns, profile_workbook from excel_adapter.py
- Retained backward-compatible API while adding enhanced features
- excel_adapter.py shim created for backward compatibility
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import json
import re
import functools
import time

from ontology_compiler.agents.extractor_plugin import (
    ExtractorPlugin,
    ExtractorMetadata,
    ExtractorResult,
)


DICT_SHEET_KEYWORDS = {"dict", "definition", "field", "schema", "legend", "glossary", "reference", "ref"}
SKIP_SHEET_KEYWORDS = {"cover", "readme", "toc", "contents", "change log", "history", "template"}

# IbP inventory definitions are the only shipped field catalog. manufacturing /
# supplier copies in ontology_2026 overlapped it by 227/228 fields, and the
# production default domain is SUPPLY_CHAIN, which had no catalog file at all.
_CATALOG_DOMAIN_ALIASES = {
    "supply_chain": "inventory",
    "manufacturing": "inventory",
    "supplier": "inventory",
}
_SYNONYM_BASELINE = "excel_column_synonyms_baseline_v1.json"


def _project_root() -> Path:
    """Repo root. This module lives at ontology_compiler/services/extraction/."""
    return Path(__file__).resolve().parents[3]


def _catalog_path_for_domain(domain: str) -> Path:
    """Prefer a domain-specific catalog; otherwise the shared inventory catalog."""
    root = _project_root()
    key = domain.lower().strip()
    direct = root / "catalogs" / "domains" / key / f"field_catalog_{key}_v1.json"
    if direct.exists():
        return direct
    aliased = _CATALOG_DOMAIN_ALIASES.get(key)
    if aliased:
        return root / "catalogs" / "domains" / aliased / f"field_catalog_{aliased}_v1.json"
    return direct


def _read_alias_payload(path: Path) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    if not path.exists():
        return alias_map
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return alias_map
    for fld in payload.get("fields") or []:
        canonical = str(fld.get("canonical_name") or "").strip()
        if not canonical:
            continue
        for alias in fld.get("aliases") or []:
            alias_key = str(alias).strip().lower()
            if alias_key:
                alias_map[alias_key] = canonical
        alias_map.setdefault(canonical.lower(), canonical)
    return alias_map


def _read_synonym_baseline() -> dict[str, str]:
    path = _project_root() / "catalogs" / "governance" / _SYNONYM_BASELINE
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    synonyms = payload.get("synonyms") or {}
    return {
        str(alias).strip().lower(): str(canonical).strip()
        for alias, canonical in synonyms.items()
        if str(alias).strip() and str(canonical).strip()
    }


@functools.lru_cache(maxsize=16)
def _load_catalog_aliases(domain: str) -> dict[str, str]:
    """Load field alias→canonical mapping for a domain; cached per domain.

    Catalog aliases win over the governance synonym baseline. The baseline
    covers generic inventory column names (part number, safety stock) that
    the IbP field catalog does not list.
    """
    alias_map = _read_alias_payload(_catalog_path_for_domain(domain))
    for alias, canonical in _read_synonym_baseline().items():
        alias_map.setdefault(alias, canonical)
    return alias_map


def _normalize_col(name: str) -> str:
    """Lowercase + strip + collapse spaces for synonym lookup."""
    return re.sub(r"\s+", " ", str(name).lower().strip())


def normalize_columns(
    df: "pd.DataFrame",
    domain: str | None = None,
) -> "pd.DataFrame":
    """Rename DataFrame columns using field catalog mapping.
    
    Args:
        df: Input DataFrame
        domain: Domain for catalog lookup (e.g., 'inventory', 'supply_chain').
                If None, no normalization is performed.
    
    Returns:
        DataFrame with normalized column names
    """
    if not domain:
        return df
    
    alias_map = _load_catalog_aliases(domain.lower())
    mapping = {}
    for col in df.columns:
        norm = _normalize_col(col)
        if norm in alias_map:
            mapping[col] = alias_map[norm]
    return df.rename(columns=mapping)


def apply_catalog_canonical_mapping(
    columns: List[str],
    domain: str | None = None,
) -> Dict[str, Any]:
    mapped: Dict[str, str] = {}
    recognized = 0
    unrecognized: List[str] = []

    for col in columns:
        canonical = _normalize_col(col)
        if domain:
            alias_map = _load_catalog_aliases(domain.lower())
            canonical = alias_map.get(canonical, canonical)
        mapped[col] = canonical
        if canonical != _normalize_col(col):
            recognized += 1
        else:
            unrecognized.append(col)

    total = len(columns)
    unrecognized_rate = round(len(unrecognized) / total, 4) if total else 0.0
    canonical_hit_rate = round(recognized / total, 4) if total else 0.0

    return {
        "mapping": mapped,
        "canonical_hit_rate": canonical_hit_rate,
        "unrecognized_fields": unrecognized,
        "unrecognized_field_rate": unrecognized_rate,
    }


def _col_quality(series: "pd.Series") -> Dict[str, Any]:
    total = len(series)
    null_count = int(series.isna().sum())
    unique_count = int(series.nunique(dropna=True))
    return {
        "total": total,
        "null_count": null_count,
        "null_rate": round(null_count / total, 4) if total else 0.0,
        "unique_count": unique_count,
        "duplicate_rate": round(1 - unique_count / max(total - null_count, 1), 4),
    }


def generate_field_definition_md(
    path: str,
    output_path: str | None = None,
    field_definitions: dict[str, str] | None = None,
) -> str:
    """Generate a human-readable Markdown field definition document from Excel headers."""
    profile = profile_workbook(path)
    ext_defs = field_definitions or {}
    lines = [
        f"# Field Definitions — {Path(path).stem}",
        "",
        f"> Auto-generated from: `{path}`",
        "",
    ]

    for sheet in profile["sheets"]:
        if sheet.get("skip") or sheet.get("error"):
            continue
        lines += [
            f"## Sheet: {sheet['name']}",
            "",
            f"Rows: {sheet.get('rows', '?')} | Columns: {sheet.get('cols', '?')}",
            "",
            "| Field Name | Canonical Name | Description | Null Rate |",
            "|------------|---------------|-------------|----------|",
        ]
        for col in sheet.get("columns", []):
            norm = _normalize_col(col)
            canonical = norm
            desc = ext_defs.get(col, ext_defs.get(norm, ""))
            cq = sheet.get("col_quality", {}).get(col, {})
            null_rate = cq.get("null_rate", 0)
            lines.append(f"| {col} | {canonical} | {desc} | {null_rate:.0%} |")
        lines.append("")

    md = "\n".join(lines)
    if output_path:
        Path(output_path).write_text(md, encoding="utf-8")
    return md


def _is_dict_sheet(name: str) -> bool:
    n = name.lower()
    return any(kw in n for kw in DICT_SHEET_KEYWORDS)


def _is_skip_sheet(name: str) -> bool:
    n = name.lower()
    return any(kw in n for kw in SKIP_SHEET_KEYWORDS)


def profile_workbook(path: str) -> Dict[str, Any]:
    """Return a lightweight profile of an Excel workbook."""
    try:
        import openpyxl
    except ImportError:
        raise ImportError("pandas and openpyxl are required: pip install pandas openpyxl")

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")

    xl = pd.ExcelFile(p, engine="openpyxl")
    sheets_info = []

    for sheet_name in xl.sheet_names:
        skip = _is_skip_sheet(sheet_name)
        try:
            df = xl.parse(sheet_name, nrows=500)
        except Exception:
            sheets_info.append({"name": sheet_name, "skip": True, "error": "parse_failed"})
            continue

        df = normalize_columns(df)
        col_quality = {str(c): _col_quality(df[c]) for c in df.columns}

        sheets_info.append({
            "name": sheet_name,
            "rows": len(df),
            "cols": len(df.columns),
            "columns": list(df.columns.astype(str)),
            "is_dict_sheet": _is_dict_sheet(sheet_name),
            "skip": skip,
            "col_quality": col_quality,
        })

    return {"path": str(p), "sheets": sheets_info}


class StructuredDataExtractor(ExtractorPlugin):
    """Enhanced CSV/Excel extractor with BOM time-series support.
    
    **ExtractorPlugin Interface**: Now implements the standard ExtractorPlugin interface
    for hot-pluggable integration.
    
    **Backward Compatibility**: The `extract_to_dict()` method provides the original
    dictionary-based API for legacy code.
    """
    
    FIELD_MAPPINGS = {
        "ISPECGRP": "spec_group", "ISPECPOS": "spec_pos",
        "FINDNUMBER": "find_number", "ISTUK35": "material",
        "KLOWER": "quantity", "DSCIMOD": "dscimod",
    }
    
    def __init__(self, domain: str = "SUPPLY_CHAIN", config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.domain = domain or (config.get("domain", "SUPPLY_CHAIN") if config else "SUPPLY_CHAIN")
    
    def get_metadata(self) -> ExtractorMetadata:
        """Return metadata about this extractor (ExtractorPlugin interface)."""
        return ExtractorMetadata(
            name="structured_data_extractor",
            version="2.0.0",
            description="CSV/Excel structured data extractor with BOM analysis",
            author="Ontology Compiler Team",
            supported_formats=[".csv", ".xlsx", ".xls"],
            capabilities=[
                "column_extraction",
                "row_to_instance",
                "data_type_inference",
                "schema_detection",
                "bom_analysis",
                "temporal_analysis",
                "quality_report",
            ],
            tags=["structured", "csv", "excel", "tabular", "unified"],
        )
    
    def can_extract(self, source: str | Path, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if this extractor can handle the given source (ExtractorPlugin interface)."""
        source_str = str(source)
        return any(source_str.lower().endswith(ext) for ext in (".csv", ".xlsx", ".xls"))
    
    def extract(self, source: str | Path, context: Optional[Dict[str, Any]] = None) -> ExtractorResult:
        """Extract ontology from CSV/Excel source (ExtractorPlugin interface).
        
        Returns:
            ExtractorResult with standardized Fragment containing nodes, instances, relationships
        """
        start_time = time.time()
        source_file = str(source) if isinstance(source, Path) else source
        ctx = context or {}
        effective_domain = ctx.get("domain", self.domain)
        
        try:
            result_data = self.extract_to_dict(source_file, ctx)
            
            # Inject domain and source_file at node level for consistent FragmentNode creation
            nodes_with_domain = [
                {**n, "domain": effective_domain, "source_file": source_file}
                for n in result_data.get("nodes", [])
            ]
            instances_with_domain = [
                {**i, "domain": effective_domain, "source_file": source_file}
                for i in result_data.get("instances", [])
            ]
            relationships_with_domain = [
                {**r, "domain": effective_domain, "source_file": source_file}
                for r in result_data.get("relationships", result_data.get("edges", []))
            ]
            
            metadata = {
                "source_file": source_file,
                "domain": effective_domain,
                "extractor": "structured_data_extractor",
            }
            if "bom" in result_data:
                metadata["bom_analysis"] = result_data["bom"]
            
            fragment = self._create_fragment(
                nodes=nodes_with_domain,
                instances=instances_with_domain,
                relationships=relationships_with_domain,
                context={**ctx, "source_file": source_file, "domain": effective_domain}
            )
            fragment.metadata.update(metadata)
            
            return ExtractorResult(
                success=True,
                fragment=fragment,
                execution_time_ms=(time.time() - start_time) * 1000,
                extractor_name="structured_data_extractor",
            )
        
        except Exception as e:
            return ExtractorResult(
                success=False,
                error_message=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
                extractor_name="structured_data_extractor",
            )
    
    def extract_to_dict(self, source: Union[str, Path], context: Optional[Dict] = None) -> Dict:
        """Extract from CSV/Excel with optional BOM analysis.
        
        **Backward Compatibility**: This method provides the original dictionary-based API.
        For new code, use `extract()` which returns standard ExtractorResult.
        
        Returns:
            Dict with nodes, instances, metadata, and optional bom analysis
        """
        context = context or {}
        df = self._load(Path(source))
        df = self._normalize(df)
        
        result = {
            "nodes": self._l1_entities(df),
            "instances": [],
            "metadata": {"source": str(source), "rows": len(df)}
        }
        
        if context.get("bom_analysis"):
            result["bom"] = self._bom_analysis(df)
        
        return result
    
    def extract_temporal(self, files: List[Path], timestamps: List[str]) -> Dict:
        """Multi-temporal BOM analysis."""
        snapshots = []
        for f, ts in zip(files, timestamps):
            df = self._load(f)
            df = self._normalize(df)
            df["_ts"] = ts
            snapshots.append(df)
        
        return {"snapshots": len(snapshots), "analysis": self._temporal_analysis(snapshots)}
    
    def extract_semantic(self, path: str, topic_id: str = "", field_definitions: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Extract L1 candidate objects from Excel headers and data-dictionary sheets."""
        try:
            import openpyxl
        except ImportError:
            raise ImportError("pandas and openpyxl are required")

        xl = pd.ExcelFile(Path(path), engine="openpyxl")
        slug_prefix = re.sub(r"[^a-z0-9]+", "-", topic_id.lower()).strip("-") or "excel"
        ext_defs: Dict[str, str] = field_definitions or {}

        objects: List[Dict[str, Any]] = []
        seen_ids: set = set()

        all_columns: List[str] = []
        for sheet_name in xl.sheet_names:
            if _is_skip_sheet(sheet_name) or _is_dict_sheet(sheet_name):
                continue
            try:
                df_preview = xl.parse(sheet_name, nrows=1)
            except Exception:
                continue
            all_columns.extend([str(c).strip() for c in df_preview.columns if str(c).strip()])

        catalog_mapping = apply_catalog_canonical_mapping(all_columns, domain=self.domain)

        def _add_object(field_name: str, description: str, source_sheet: str) -> None:
            norm = _normalize_col(field_name)
            canonical = catalog_mapping.get("mapping", {}).get(field_name) or norm
            obj_id = "DAT-{}-{}".format(
                slug_prefix,
                re.sub(r"[^a-z0-9]+", "-", canonical).strip("-").upper(),
            )
            if obj_id in seen_ids:
                return
            seen_ids.add(obj_id)
            objects.append({
                "meta_id": obj_id,
                "meta_name": field_name,
                "meta_class": "Data",
                "meta_subclass": "Excel_Field",
                "meta_description": description or f"{field_name} — Excel field from {source_sheet}",
                "source_file": path,
                "source_sheet": source_sheet,
                "sandbox_domain": self.domain,
                "meta_status": "active",
                "evidence": path,
            })

        for sheet_name in xl.sheet_names:
            if _is_skip_sheet(sheet_name) or not _is_dict_sheet(sheet_name):
                continue
            try:
                df = xl.parse(sheet_name)
            except Exception:
                continue
            df = normalize_columns(df, domain=self.domain)
            name_col = next(
                (c for c in df.columns if any(k in str(c).lower() for k in ("field", "name", "column"))),
                None,
            )
            desc_col = next(
                (c for c in df.columns if any(k in str(c).lower() for k in ("desc", "definition", "meaning", "remark", "note"))),
                None,
            )
            if name_col is None:
                continue
            for _, row in df.iterrows():
                fname = str(row.get(name_col, "")).strip()
                fdesc = str(row.get(desc_col, "")).strip() if desc_col else ""
                if fname and fname.lower() not in ("nan", "", "field", "name"):
                    _add_object(fname, fdesc, sheet_name)

        for sheet_name in xl.sheet_names:
            if _is_skip_sheet(sheet_name) or _is_dict_sheet(sheet_name):
                continue
            try:
                df = xl.parse(sheet_name, nrows=1)
            except Exception:
                continue
            df = normalize_columns(df, domain=self.domain)
            for col in df.columns:
                col_str = str(col).strip()
                if col_str.lower() in ("nan", "", "unnamed"):
                    continue
                desc = ext_defs.get(col_str, ext_defs.get(_normalize_col(col_str), ""))
                _add_object(col_str, desc, sheet_name)

        for fname, fdesc in ext_defs.items():
            _add_object(fname, fdesc, "external_definition")

        return {
            "objects": objects,
            "rules": [],
            "relationships": [],
            "mapping_stats": {
                "canonical_hit_rate": catalog_mapping.get("canonical_hit_rate", 0.0),
                "unrecognized_field_rate": catalog_mapping.get("unrecognized_field_rate", 0.0),
                "unrecognized_fields": catalog_mapping.get("unrecognized_fields", []),
            },
        }
    
    def extract_instances(self, path: str, topic_id: str = "", id_columns: Optional[List[str]] = None, max_rows_per_sheet: int = 1000) -> Dict[str, Any]:
        """Extract L2 operational instances from data rows in Excel."""
        try:
            import openpyxl
        except ImportError:
            raise ImportError("pandas and openpyxl are required")

        xl = pd.ExcelFile(Path(path), engine="openpyxl")
        slug_prefix = re.sub(r"[^a-z0-9]+", "-", topic_id.lower()).strip("-") or "excel"
        default_id_cols = id_columns or ["pn", "part_number", "material", "sku", "item_code"]

        instances: List[Dict[str, Any]] = []
        total_rows = 0
        all_cols: List[str] = []

        for sheet_name in xl.sheet_names:
            if _is_skip_sheet(sheet_name) or _is_dict_sheet(sheet_name):
                continue
            try:
                df = xl.parse(sheet_name, nrows=max_rows_per_sheet)
            except Exception:
                continue

            df = normalize_columns(df, domain=self.domain)
            all_cols.extend([str(c) for c in df.columns])
            df = df.dropna(how="all")
            if df.empty:
                continue

            id_col = next((c for c in default_id_cols if c in df.columns), None)
            sheet_slug = re.sub(r"[^a-z0-9]+", "-", sheet_name.lower()).strip("-")
            instance_of = "DAT-{}-{}".format(slug_prefix, sheet_slug).upper()

            for idx, row in df.iterrows():
                row_dict = {k: v for k, v in row.items() if pd.notna(v)}
                if not row_dict:
                    continue

                row_id = str(row_dict[id_col]) if id_col and id_col in row_dict else str(idx)
                inst_id = "INST-{}-{}-{}".format(
                    slug_prefix,
                    sheet_slug,
                    re.sub(r"[^a-z0-9]+", "-", row_id.lower()),
                ).upper()

                instances.append({
                    "meta_id": inst_id,
                    "meta_class": "Data",
                    "meta_subclass": "Excel_Instance",
                    "meta_name": f"{sheet_name} row {idx}",
                    "instance_of": instance_of,
                    "instance_source": f"{path}#{sheet_name}",
                    "instance_last_sync": None,
                    "instance_refresh": "on_demand",
                    "instance_values": {str(k): str(v) for k, v in row_dict.items()},
                    "sandbox_domain": self.domain,
                    "meta_status": "active",
                })
                total_rows += 1

        mapping = apply_catalog_canonical_mapping(all_cols, domain="inventory")
        return {
            "instances": instances,
            "stats": {
                "total_instances": total_rows,
                "sheets_processed": len(xl.sheet_names),
                "canonical_hit_rate": mapping.get("canonical_hit_rate", 0.0),
                "unrecognized_field_rate": mapping.get("unrecognized_field_rate", 0.0),
            },
        }
    
    def build_quality_report(self, path: str) -> Dict[str, Any]:
        """Return a quality report dict for all sheets in the workbook."""
        profile = profile_workbook(path)
        sheet_reports = []
        for sheet in profile["sheets"]:
            if sheet.get("skip") or sheet.get("error"):
                continue
            cq = sheet.get("col_quality") or {}
            high_null = [c for c, q in cq.items() if q.get("null_rate", 0) > 0.5]
            high_dup = [c for c, q in cq.items() if q.get("duplicate_rate", 0) > 0.9]
            sheet_reports.append({
                "sheet": sheet["name"],
                "rows": sheet.get("rows", 0),
                "cols": sheet.get("cols", 0),
                "high_null_columns": high_null,
                "high_duplicate_columns": high_dup,
                "warnings": (
                    ([f"high null rate in: {', '.join(high_null)}"] if high_null else []) +
                    ([f"high duplicate rate in: {', '.join(high_dup)}"] if high_dup else [])
                ),
            })
        return {"path": path, "sheets": sheet_reports}
    
    def _load(self, path: Path) -> pd.DataFrame:
        if path.suffix == ".csv":
            return pd.read_csv(path, dtype=str)
        return pd.read_excel(path, dtype=str, engine="openpyxl")
    
    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        mapping = {c: self.FIELD_MAPPINGS.get(c.upper().strip(), c) for c in df.columns}
        df = df.rename(columns=mapping)
        df = normalize_columns(df, domain=self.domain)
        return df
    
    def _l1_entities(self, df: pd.DataFrame) -> List[Dict]:
        return [{"meta_id": f"F-{c.upper()}", "meta_name": c, "meta_class": "DataField"}
                for c in df.columns if not c.startswith("_")]
    
    def _bom_analysis(self, df: pd.DataFrame) -> Dict:
        if "pos_key" not in df.columns and "spec_group" in df.columns:
            df["pos_key"] = df["spec_group"] + "_" + df["spec_pos"]

        # Build aggregation dict dynamically based on available columns
        agg_dict = {
            "material": "nunique",
            "dscimod": "nunique",
        }
        if "_ts" in df.columns:
            agg_dict["_ts"] = "nunique"

        pos_stats = df.groupby("pos_key").agg(agg_dict).reset_index()

        # Rename columns
        col_names = ["pos_key", "material_count", "dscimod_count"]
        if "_ts" in df.columns:
            col_names.append("timestamp_count")
        else:
            # Add default timestamp_count = 1 for single snapshot
            pos_stats["timestamp_count"] = 1
            col_names.append("timestamp_count")
        pos_stats.columns = col_names

        def classify(row):
            if row["timestamp_count"] > 1 and row["material_count"] > 1:
                return "HISTORICAL_REPLACEMENT"
            elif row["material_count"] > 1:
                return "CONFIG_OPTION"
            return "SINGLE"

        pos_stats["category"] = pos_stats.apply(classify, axis=1)
        
        return {
            "pos_count": len(pos_stats),
            "config_options": int((pos_stats["category"] == "CONFIG_OPTION").sum()),
            "historical_replacements": int((pos_stats["category"] == "HISTORICAL_REPLACEMENT").sum()),
            "details": pos_stats.to_dict("records")
        }
    
    def _temporal_analysis(self, snapshots: List[pd.DataFrame]) -> Dict:
        if not snapshots:
            return {}
        
        all_pos = set()
        for df in snapshots:
            if "pos_key" in df.columns:
                all_pos.update(df["pos_key"].unique())
        
        changes = []
        for pos in all_pos:
            pos_history = []
            for df in snapshots:
                pos_data = df[df["pos_key"] == pos] if "pos_key" in df.columns else pd.DataFrame()
                if not pos_data.empty:
                    pos_history.append({
                        "ts": df["_ts"].iloc[0],
                        "materials": set(pos_data["material"].dropna().unique()) if "material" in pos_data.columns else set(),
                        "dscimod": pos_data["dscimod"].iloc[0] if "dscimod" in pos_data.columns else None
                    })
            
            for i in range(1, len(pos_history)):
                prev, curr = pos_history[i-1], pos_history[i]
                if prev["materials"] != curr["materials"]:
                    added = curr["materials"] - prev["materials"]
                    removed = prev["materials"] - curr["materials"]
                    changes.append({
                        "pos": pos,
                        "from_ts": prev["ts"],
                        "to_ts": curr["ts"],
                        "added": list(added),
                        "removed": list(removed),
                        "dscimod": curr["dscimod"]
                    })
        
        return {
            "total_changes": len(changes),
            "change_rate": len(changes) / len(all_pos) if all_pos else 0,
            "changes": changes[:100]
        }