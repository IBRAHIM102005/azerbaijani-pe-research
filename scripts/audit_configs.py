from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.reproducibility.adapters import MissingInterfaceError, get_models_module  # noqa: E402


def arm_payload(repo_root: Path, pe_type: str) -> dict:
    path = repo_root / "configs" / "pe" / f"{pe_type}.json"
    if not path.is_file():
        raise SystemExit(f"[audit_configs] ERROR: missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_audit(repo_root: Path) -> dict:
    """Fast, in-process check using other's real ARM_ALLOWLIST/PE_TYPES: every
    shipped arm config must have the same field set, and only allowlisted
    fields may differ in value."""
    try:
        models = get_models_module()
    except MissingInterfaceError as exc:
        raise SystemExit(f"[audit_configs] ERROR: {exc}") from exc

    payloads = {pe: arm_payload(repo_root, pe) for pe in models.PE_TYPES}

    key_sets = {frozenset(p) for p in payloads.values()}
    if len(key_sets) != 1:
        raise SystemExit(
            "[audit_configs] ERROR: arm configs do not even share the same "
            f"field set: {[sorted(p) for p in payloads.values()]}"
        )

    differing: dict[str, dict[str, object]] = {}
    for key in next(iter(key_sets)):
        values = {pe: payloads[pe][key] for pe in payloads}
        serialized = {json.dumps(v, sort_keys=True) for v in values.values()}
        if len(serialized) > 1:
            differing[key] = values

    allowlist = set(models.ARM_ALLOWLIST)
    forbidden = {k: v for k, v in differing.items() if k not in allowlist}
    allowed = {k: v for k, v in differing.items() if k in allowlist}

    return {
        "compared": sorted(payloads),
        "allowlist": sorted(allowlist),
        "allowed_differences": allowed,
        "forbidden_differences": forbidden,
        "forbidden_total": len(forbidden),
    }


def run_full_pytest_check(repo_root: Path) -> dict:
    """Run other's own tests/test_config_contract.py, which additionally
    cross-checks every arm against real training_data_contract.json
    (tokenizer/manifest hashes, vocab size, data seed, run-id template,
    etc.) -- checks this script's fast in-process diff does not attempt to
    duplicate."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_config_contract.py", "-q"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=600,
    )
    return {
        "pass": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-40:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-40:]),
    }


def print_report(report: dict, full_report: dict | None) -> None:
    print(f"[audit_configs] Compared PE variants: {report['compared']}")
    print(f"[audit_configs] Allowlisted fields (from other's ARM_ALLOWLIST): {report['allowlist']}")
    if report["allowed_differences"]:
        print("  allowed differences:")
        for k, v in report["allowed_differences"].items():
            print(f"    {k}: {v}")
    if report["forbidden_differences"]:
        print("  FORBIDDEN differences:")
        for k, v in report["forbidden_differences"].items():
            print(f"    {k}: {v}")
    print()
    if report["forbidden_total"] == 0:
        print("[audit_configs] PASS (fast check): no forbidden configuration drift detected.")
    else:
        print(f"[audit_configs] FAIL (fast check): {report['forbidden_total']} forbidden difference(s).")

    if full_report is not None:
        print()
        if full_report["pass"]:
            print("[audit_configs] PASS (--full): tests/test_config_contract.py passed, "
                  "including other cross-checks.")
        else:
            print("[audit_configs] FAIL (--full): tests/test_config_contract.py failed.")
            print(full_report["stdout_tail"])
            print(full_report["stderr_tail"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also run tests/test_config_contract.py (other cross-checks), "
        "not just the fast in-process allowlist diff.",
    )
    args = parser.parse_args()

    report = run_audit(args.repo_root)
    full_report = run_full_pytest_check(args.repo_root) if args.full else None
    print_report(report, full_report)

    if args.json:
        payload = {"fast_check": report, "full_pytest_check": full_report}
        args.json.write_text(json.dumps(payload, indent=2, default=str))

    ok = report["forbidden_total"] == 0 and (full_report is None or full_report["pass"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
