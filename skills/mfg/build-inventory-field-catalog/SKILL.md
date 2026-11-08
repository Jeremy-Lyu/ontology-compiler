# build-inventory-field-catalog

Build authoritative field catalog(s) from source CSV files.

## Description
This skill builds field catalogs from CSV source files for different domains. It processes domain-specific CSV files to generate:
- Domain-specific field catalogs: catalogs/domains/<domain>/field_catalog_<domain>_v1.json
- Domain-specific conflict reports: catalogs/domains/<domain>/conflict_report_<domain>_v1.json
- Global field catalog: catalogs/global/field_catalog_global_v1.json
- Global conflict report: catalogs/global/global_conflict_report_v1.json

## Supported Domains
- inventory (enabled)
- supplier (bootstrap)
- manufacturing (bootstrap)

## Parameters
- `--profile`: Domain source profile to use (default: from config or environment)
- `--domains`: Comma-separated list of domains to process (default: all configured domains)

## Heuristics
- Use this skill when CSV source files are updated
- Prefer building catalogs before ontology compilation
- Use for CI/CD pipeline to ensure catalogs are up-to-date
