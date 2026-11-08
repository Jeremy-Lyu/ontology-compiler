#!/usr/bin/env bash
# Run the 5-sample isolated regression with the LLM-free deterministic
# safety net. LLM candidate generation is attempted but degrades gracefully
# when the provider is unreachable; the offline safety net still runs.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

export EXTRACT_ENABLE_CANDIDATE_GEN=1

python3 scripts/run_regression_samples.py \
  --production \
  --html \
  --sample-id mbom-du-v2 \
  --sample-id ebom-mbom-2023 \
  --sample-id production-inventory-pptx \
  --sample-id production-inventory-md \
  --sample-id doc-folder-all

status=$?
echo "exit_code=$status"
exit $status
