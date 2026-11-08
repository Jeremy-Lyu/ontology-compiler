# entity-classification

Classify extracted entities into the ontology layer system with rule-based matching and optional LLM fallback.

## Purpose
- Normalize entity type and layer assignments after extraction
- Fill missing classification fields in a deterministic way
- Mark low-confidence items for review instead of dropping them
- Keep the output compatible with pipeline, verifier, and merge steps

## Inputs
- `--input`: Path to JSON file containing entities to classify
- `--output`: Output path for classified entities
- `--config`: Path to classification config, default `config/ontology/entity_classification_unified.yaml`
- `--use-llm`: Enable LLM fallback for uncertain classifications, default `true`
- `--confidence-threshold`: Auto-accept threshold, default `0.85`
- `--dry-run`: Run classification without writing output, default `false`

## When To Use
- After entity extraction
- Before verifier and merge
- When consolidating entities from multiple sources
- When you need consistent `layer` / `entity_type` assignments across files

## Classification Flow
1. Rule Matching: apply ordered classification rules
2. Conflict Resolution: resolve multiple matches by priority and specificity
3. Field Writeback: always write `layer`, `entity_type`, and `classification_confidence`
4. Review Marking: set `needs_review=true` for low-confidence results
5. LLM Fallback: use LLM only when the rule engine cannot resolve the case confidently
6. Validation: check evidence presence and layer/type consistency

## Output Contract
The skill must preserve a canonical classification contract.

### Required Fields
- `meta_id`
- `meta_name`
- `meta_class`
- `layer`
- `entity_type`
- `classification_confidence`
- `needs_review`

### Optional Fields
- `sub_type`
- `evidence_ref`
- `matched_rules`
- `reason`
- `fallback_decision`

## Canonical Output Example
```json
{
  "meta_id": "L1-001",
  "meta_name": "Supply Chain Planning",
  "meta_class": "Capability",
  "layer": "L1",
  "entity_type": "Capability",
  "classification_confidence": 0.92,
  "needs_review": false,
  "sub_type": "PlanningCapability",
  "evidence_ref": "section_2.1",
  "matched_rules": ["rule_12", "rule_18"],
  "reason": "Matched planning-related capability rules",
  "fallback_decision": "auto_accept"
}
```

## Supported Layers
- `L0`: core concepts and root classes
- `L1`: standard domain classes and refinements
- `L2`: business scenarios and use cases
- `L2.5`: mapping and validation rules
- `L3`: instances and concrete realizations

## Review Policy
- Auto-accept: confidence `>= 0.85`
- Manual review: confidence `0.60-0.85`
- Reject: confidence `< 0.60` or invalid contract
- Missing evidence never gets silently accepted

## Quality Gates
- `missing_name`
- `missing_evidence`
- `layer_conflict`
- `type_conflict`
- `confidence_below_threshold`

## Notes
- This skill should not invent semantic content
- Deterministic rule output takes precedence over LLM suggestions
- Low-confidence results should still be written back with `needs_review=true`
