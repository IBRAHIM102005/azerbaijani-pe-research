.PHONY: test audit-configs audit-params smoke reproduce-headline reproduce-training verify-release

SEED ?=
RELEASE ?=
MATRIX ?=
MICRO_BATCH_SEQUENCES ?=
GPUS ?=

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

# --- reproduce-training --------------------------------------------------
# Explicitly separated from CI/smoke. MATRIX=smoke runs a real CUDA smoke.
# MATRIX=core regenerates the strict 25-run plan from the measured A100
# microbatch, performs the final production preflight, then launches M3.
reproduce-training:
	@echo "[reproduce-training] WARNING: this may launch GPU training."
	@echo "[reproduce-training] This target is intentionally NOT part of CI."
	@if [ -z "$(MATRIX)" ]; then \
		echo "[reproduce-training] ERROR: MATRIX=<smoke|core> is required."; \
		exit 1; \
	fi
	@if [ "$(MATRIX)" = "smoke" ]; then \
		python scripts/m3_model_smoke.py --pe all --device cuda; \
	elif [ "$(MATRIX)" = "core" ]; then \
		if [ -z "$(MICRO_BATCH_SEQUENCES)" ]; then \
			echo "[reproduce-training] ERROR: MICRO_BATCH_SEQUENCES=<measured A100 value> is required."; \
			exit 1; \
		fi; \
		python scripts/m3_make_run_plan.py --micro-batch-sequences "$(MICRO_BATCH_SEQUENCES)"; \
		python scripts/m3_server_preflight.py --require-cuda --require-bf16 --require-cache --require-environment-lock --require-headline-plan; \
		python scripts/m3_launch_matrix.py $(if $(GPUS),--gpus $(GPUS),); \
	else \
		echo "[reproduce-training] ERROR: MATRIX must be smoke or core."; \
		exit 1; \
	fi

# --- verify-release -----------------------------------------------------
verify-release:
	python scripts/verify_release.py --repo-root . $(if $(SEED),--seed $(SEED),) --out release_verification.json
