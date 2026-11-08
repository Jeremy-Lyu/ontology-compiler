# Scene Analysis Skill

## Description

Performs complex BOM scenario analysis using a 3-layer analytical framework:

1. **Layer 1: Field Profiling** - Explore data fields, identify potential join keys, and detect data quality issues
2. **Layer 2: Precision JOIN Analysis** - Validate hypotheses with full data, calculate match rates, and identify 1:N relationships
3. **Layer 3: RAG Synthesis** - Combine quantitative findings with business context to generate actionable insights

## When to Use

Use this skill when:
- Analyzing multi-table data relationships (e.g., MBOM, MRP, inventory systems)
- Investigating complex business scenarios involving multiple systems (OHF, CMP, CIP, BIS, HERCULES, etc.)
- Needing precise quantitative analysis combined with business context
- Performing root cause analysis for production or material management issues

Do NOT use for:
- Simple single-table queries (use RAG directly)
- Basic field documentation (use RAG or field catalogs)

## Workflow Reference

See `.windsurf/workflows/complex-bom-scenario-analysis.md` for detailed methodology and analysis templates.

## Parameters

- `--scenario <scenario>`: Required. Scenario identifier (e.g., `bom_mrp`, `c_note_changes`)
- `--layer <layer>`: Optional. Analysis layer to run (`1`, `2`, `3`, or `all`). Default: `all`

## Examples

```bash
# Run full 3-layer analysis for BOM/MRP scenario
python scripts/diagnostics/run_scene_analysis.py --scenario bom_mrp --layer all

# Run only field profiling (Layer 1)
python scripts/diagnostics/run_scene_analysis.py --scenario bom_mrp --layer 1

# Run only precision JOIN analysis (Layer 2)
python scripts/diagnostics/run_scene_analysis.py --scenario bom_mrp --layer 2

# Run only RAG synthesis (Layer 3)
python scripts/diagnostics/run_scene_analysis.py --scenario bom_mrp --layer 3
```

## Output

- Layer 1: Field profile report (columns, data types, null rates, unique values)
- Layer 2: JOIN analysis report (match rates, 1:N distributions, consistency checks)
- Layer 3: Synthesis report combining quantitative findings with business context

## Related Skills

- `run_ontology_rag_eval`: For systems collaboration analysis
- `run_ontology_rag_validate`: For operations deep validation
- `build_inventory_field_catalog`: For building field catalogs from source data
