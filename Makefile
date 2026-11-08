# 2026-10-16: 默认走项目 venv（`.venv`）；系统 python3 缺 jieba/sentence-transformers
# 会让 `make check` 出现 2 个环境类 error。仍可用 `make check PYTHON=python3` 覆盖。
PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
REGRESSION_MANIFEST ?= config/regression_samples.json
REGRESSION_OUTPUT_DIR ?= outputs/runs/ontology/pipeline
REGRESSION_BASELINE ?= outputs/runs/ontology/baselines/regression_summary_baseline.json
RETRIEVAL_EVAL ?= outputs/runs/ontology/rag/eval/retrieval_eval_latest.json
REGRESSION_DOMAIN ?= SUPPLY_CHAIN

# Shared flags for production regression (7-sample LLM path + RAG gate)
REGRESSION_FLAGS = --production --html --skip-governed-rebuild \
	--retrieval-eval $(RETRIEVAL_EVAL) --enforce-rag-gate

.PHONY: bootstrap-env regression regression-baseline regression-compare regression-smoke test-unit check check-code-size check-import-boundary check-docs check-compat check-compat-gate retention-prune retention-weekly governed-publish check-governed-publish


bootstrap-env:
	@if [ ! -f config/env.env ]; then \
	  cp config/env.example.env config/env.env; \
	  echo "Created config/env.env from env.example.env (edit keys before live LLM runs)"; \
	fi

test-unit:
	$(PYTHON) -m unittest discover -s tests -t . -v

check-code-size:
	$(PYTHON) scripts/check_code_size.py

check-import-boundary:
	$(PYTHON) scripts/check_import_boundary.py

check-docs:
	$(PYTHON) scripts/check_docs_links.py

check-compat:
	$(PYTHON) scripts/compat_usage_stats.py --json

check-compat-gate:
	$(PYTHON) scripts/check_compat_usage_gate.py

# P1-4 (2026-09-09): weekly outputs hygiene — keeps newest 12 pipeline runs,
# 12 retrieval packet dirs, newest 24 regression reports, newest 5 master
# backups/audits each. Dry-run first to review, then --apply.
retention-prune:
	$(PYTHON) scripts/cleanup_outputs_retention.py --keep 12

retention-weekly: retention-prune
	$(PYTHON) scripts/cleanup_outputs_retention.py --keep 12 --apply

# P2-4 (2026-09-09): CI publishing convention — the regression matrix runs
# with --skip-governed-rebuild on every rank; ONLY the last rank (or a
# dedicated publish step) refreshes the governed packet, then verifies
# freshness + rebuild health. Local equivalent of that rank-last publish.
governed-publish:
	$(PYTHON) scripts/build_governed_packet.py

check-governed-publish:
	$(PYTHON) scripts/check_governed_publish.py
# Prefer a strict publish rank locally / in CI after rebuild:
#   $(PYTHON) scripts/check_governed_publish.py --require-receipt --require-publishable

# Single verification command (execution plan Phase 0): unit + compile + smoke + gates.
# Fresh clones: seed a local env.env from the example (gitignored) so imports do not crash.
check: bootstrap-env
	$(PYTHON) -m unittest discover -s tests -t .
	$(PYTHON) -m compileall -q ontology_compiler src core tests
	$(PYTHON) scripts/regression_manifest_smoke.py
	$(PYTHON) scripts/check_code_size.py
	$(PYTHON) scripts/check_import_boundary.py
	$(PYTHON) scripts/check_docs_links.py
	$(PYTHON) scripts/check_compat_usage_gate.py

regression:
	$(PYTHON) scripts/run_regression_samples.py \
		--manifest $(REGRESSION_MANIFEST) \
		--output-dir $(REGRESSION_OUTPUT_DIR) \
		--domain $(REGRESSION_DOMAIN) \
		$(REGRESSION_FLAGS)

regression-baseline:
	$(PYTHON) scripts/run_regression_samples.py \
		--manifest $(REGRESSION_MANIFEST) \
		--output-dir $(REGRESSION_OUTPUT_DIR) \
		--domain $(REGRESSION_DOMAIN) \
		$(REGRESSION_FLAGS) \
		--write-baseline \
		--baseline-output $(REGRESSION_BASELINE)

regression-compare:
	$(PYTHON) scripts/run_regression_samples.py \
		--manifest $(REGRESSION_MANIFEST) \
		--output-dir $(REGRESSION_OUTPUT_DIR) \
		--domain $(REGRESSION_DOMAIN) \
		--compare-only \
		--before $(REGRESSION_BASELINE) \
		--after $(REGRESSION_OUTPUT_DIR)/regression_reports/latest_regression_summary.json

regression-smoke:
	$(PYTHON) scripts/regression_manifest_smoke.py
