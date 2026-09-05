.PHONY: test audit-configs audit-params smoke reproduce-headline reproduce-training verify-release

SEED ?=
RELEASE ?=

# --- test ---------------------------------------------------------------
# Runs the full CPU-compatible unit + integration suite. Never runs the
# 25-run experiment matrix or full pretraining (see reproduce-training).
test:
	python -m pytest tests -q

# --- audit-configs --------------------------------------------------------
# Wraps Yasin's real config contract (src.models.config.ARM_ALLOWLIST) --
# does not reimplement it. Add FULL=1 to also cross-check every arm
# against Ibrahim's real training_data_contract.json via
# tests/test_config_contract.py.
audit-configs:
	python scripts/audit_configs.py --repo-root . $(if $(FULL),--full,)

# --- audit-params ---------------------------------------------------------
# Wraps Yasin's real src.models.params.fairness_report -- does not
# reimplement it. Fails loudly (exit 2) if Yasin's src.models isn't
# importable, never silently skips the check.
audit-params:
	python scripts/audit_parameters.py --repo-root . $(if $(SEED),--seed $(SEED),)

# --- smoke ------------------------------------------------------------
# CPU-compatible smoke check: config audit + parameter audit + fast tests.
# Does NOT launch any GPU training.
smoke: audit-configs audit-params
	python -m pytest tests -q -k "not slow"

# --- reproduce-headline -----------------------------------------------
# Reconstructs headline results (tables/figures) from ALREADY-RELEASED
# checkpoints/raw metrics. Verifies required artifacts and checksums
# first; fails clearly if they're missing. Never starts training.
reproduce-headline:
	@if [ -z "$(RELEASE)" ]; then \
		echo "[reproduce-headline] ERROR: RELEASE=<tag> is required, e.g."; \
		echo "  make reproduce-headline RELEASE=v1.0-final"; \
		exit 1; \
	fi
	python scripts/verify_release.py --repo-root . --out release_verification.json
	@echo "[reproduce-headline] artifact/config/param verification complete."
	@echo "[reproduce-headline] TODO(Fidan/Nihat-integration): wire scripts/evaluate.py + "
	@echo "  scripts/analyze.py + scripts/make_figures.py here once Fidan/Nihat land."

# --- reproduce-training (separate, compute-warned, not wired by default) --
reproduce-training:
	@echo "[reproduce-training] WARNING: this launches FULL pretraining runs."
	@echo "[reproduce-training] This is compute-intensive (booked A100 slot"
	@echo "[reproduce-training] required) and is intentionally NOT part of"
	@echo "[reproduce-training] 'make test' / CI / 'make smoke'."
	@if [ -z "$(MATRIX)" ]; then \
		echo "[reproduce-training] ERROR: MATRIX=<smoke|core> is required."; \
		exit 1; \
	fi
	@echo "[reproduce-training] BLOCKED: scripts/m3_make_run_plan.py requires"
	@echo "  --micro-batch-sequences, a value only produced by Fidan's A100"
	@echo "  benchmark (not yet run). Wiring this target before that value"
	@echo "  exists would mean guessing it, which this project's own rule"
	@echo "  forbids. Re-check this target once the benchmark lands."
	@exit 1

# --- verify-release -----------------------------------------------------
verify-release:
	python scripts/verify_release.py --repo-root . $(if $(SEED),--seed $(SEED),) --out release_verification.json
