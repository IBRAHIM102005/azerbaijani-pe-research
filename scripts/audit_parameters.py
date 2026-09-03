#!/usr/bin/env python3
"""
Parameter-count fairness auditor -- thin wrapper around Yasin's
real fairness report, not a reimplementation of it.

Yasin already builds all five arms and checks core/embedding parameter parity
*and* bit-identical shared-weight initialization
(`src.models.params.fairness_report`) -- more rigorous than an earlier
placeholder, which only checked parameter counts. This script's job is to run it as
a standalone, CI/Makefile-friendly script with a clear exit code -- not to
re-derive what counts as "fair". (Yasin also ships its own equivalent script,
`scripts/count_params.py`; this one exists so `make audit-params` has a
stable, dedicated entry point and JSON report shape, and so verify_release.py
has one thing to call regardless of which scripts Yasin happens to add later.)

Usage:
    python scripts/audit_parameters.py --repo-root .
    python scripts/audit_parameters.py --repo-root . --seed 42
    python scripts/audit_parameters.py --repo-root . --json out.json

Exit codes:
    0  all fairness checks passed (parameter parity + shared-init parity)
    1  a fairness check failed
    2  usage / interface error (Yasin not importable, config missing, bad seed)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.reproducibility.adapters import MissingInterfaceError, get_models_module  # noqa: E402


def run_audit(repo_root: Path, base_config_path: Path | None = None, seed: int | None = None) -> dict:
    try:
        models = get_models_module()
    except MissingInterfaceError as exc:
        raise SystemExit(f"[audit_parameters] ERROR: {exc}") from exc

    config_path = base_config_path or (repo_root / "configs" / "model_base.json")
    if not config_path.is_file():
        raise SystemExit(f"[audit_parameters] ERROR: {config_path} not found")

    run_matrix = models.load_run_matrix(repo_root / "configs" / "run_matrix.json")
    seeds = run_matrix["run_seeds"]
    resolved_seed = seed if seed is not None else seeds[0]
    if resolved_seed not in seeds:
        raise SystemExit(
            f"[audit_parameters] ERROR: {resolved_seed} is not a preregistered "
            f"run seed {seeds}"
        )

    # Parameter counts don't depend on the seed, but ModelConfig refuses to
    # build a model from the template sentinel (init_seed=-1) -- a seed must
    # be resolved before a model can be constructed at all.
    base_config = models.ModelConfig.from_json(config_path).with_seed(resolved_seed)
    report = models.fairness_report(base_config, models.PE_TYPES)
    report["resolved_seed"] = resolved_seed
    report["base_config_path"] = str(config_path)
    return report


def print_report(report: dict) -> None:
    from importlib import import_module

    models = import_module("src.models")
    print(
        f"[audit_parameters] base config: {report['base_config_path']} "
        f"(resolved with init_seed={report['resolved_seed']})"
    )
    print(models.format_fairness_table(report))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=None, help="defaults to configs/model_base.json")
    parser.add_argument("--seed", type=int, default=None, help="defaults to the first preregistered run seed")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    report = run_audit(args.repo_root, args.config, args.seed)
    print_report(report)

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=str))

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
