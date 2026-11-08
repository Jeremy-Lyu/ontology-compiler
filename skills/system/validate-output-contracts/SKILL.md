# validate-output-contracts

Validate pipeline output contracts against JSON schemas and format requirements.

## Description
This skill validates various pipeline output artifacts including:
- Analysis manifest JSON files
- L1/L2 domain sandbox JSON files
- Pipeline summary JSON files
- Inventory aggregate JSON files
- Business health view JSON files
- Published changelog Markdown files
- Replay event payloads
- RAG feedback responses

## Parameters
- `--analysis-manifest`: Path to analysis_manifest.json file
- `--l1-domain-sandbox`: Path to L1_DOMAIN_SANDBOX.json file
- `--l2-operational-instances`: Path to L2_OPERATIONAL_INSTANCES.json file
- `--pipeline-summary`: Path to PIPELINE_RUN_SUMMARY json file
- `--inventory-aggregate`: Path to INVENTORY_AGGREGATE_VALIDATE json file
- `--business-health-view`: Path to BUSINESS_HEALTH_VIEW json file
- `--replay-exec-envelope`: Path to replay exec envelope json file
- `--replay-sse-event`: Path to session replay SSE event payload json file
- `--replay-summary`: Path to session replay summary payload json file
- `--live-analysis-progress`: Path to live analysis progress payload json file
- `--live-analysis-completion`: Path to live analysis completion payload json file
- `--live-analysis-error`: Path to live analysis error payload json file
- `--published-audit-log`: Path to published AUDIT_LOG.json file
- `--published-changelog`: Path to published CHANGELOG.md file
- `--observability-openapi`: Path to observability OpenAPI json file
- `--replay-event-error`: Path to replay event error envelope json file
- `--rag-feedback-response`: Path to rag feedback success response json file
- `--rag-feedback-error-response`: Path to rag feedback error response json file
- `--exec-json`: Suppress logging and output only JSON

## Heuristics
- Use this skill for contract validation after pipeline stages complete
- Prefer contract validation before publish or patch apply
- Use for CI/CD gatekeeping to ensure output contract compliance
