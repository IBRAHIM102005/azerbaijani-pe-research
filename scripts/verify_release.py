#!/usr/bin/env python3
"""
Clean-checkout release verification.

Verifies a clean checkout has everything required for reproduction, then
runs the config audit, parameter audit, and test suite, and writes
release_verification.json.

Usage:
    python scripts/verify_release.py --repo-root .

Exit codes:
    0  all release-blocking checks passed
    1  a release-blocking check failed
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_configs  # noqa: E402
import audit_parameters  # noqa: E402
from src.reproducibility.metadata import git_info  # noqa: E402

# Relative to repo root. Each entry: (path, required, description)
REQUIRED_ARTIFACTS = [
    ("README.md", True, "top-level README"),
    ("Makefile", True, "reproducibility commands"),
    ("requirements-reproducibility.txt", False, "pinned dependency lock (or pyproject.toml)"),
    ("pyproject.toml", False, "dependency declaration"),
    ("configs/pe", True, "per-arm PE configuration templates (Yasin)"),
    ("configs/model_base.json", True, "shared base model config (Yasin)"),
    ("configs/run_matrix.json", True, "preregistered run/seed matrix (Yasin)"),
    ("data/README.md", True, "data access documentation"),
    ("data/metadata/source_registry.yaml", True, "source license/access registry"),
    ("data/metadata/training_data_contract.json", True, "Ibrahim data/tokenizer handoff contract"),
    ("tokenizer/tokenizer_hashes.json", True, "tokenizer artifact hashes"),
    ("scripts", True, "reproducibility scripts"),
    ("tests", True, "test suite"),
    ("experiments/manifests", True, "run metadata manifests"),
    ("docs/reproducibility.md", False, "reproducibility documentation"),
    ("docs/ai_use.md", True, "AI-use disclosure"),
    ("docs/failure_log.md", False, "failure log"),
]


def check_artifacts(repo_root: Path) -> dict:
    results = []
    ok = True
    for rel_path, required, desc in REQUIRED_ARTIFACTS:
        full = repo_root / rel_path
        present = full.exists()
        if required and not present:
            ok = False
        results.append(
            {"path": rel_path, "required": required, "description": desc, "present": present}
        )
    return {"pass": ok, "items": results}


def run_config_audit(repo_root: Path) -> dict:
    try:
        report = audit_configs.run_audit(repo_root)
    except SystemExit as exc:
        return {"pass": False, "error": str(exc)}
    return {"pass": report["forbidden_total"] == 0, "report": report}


def run_parameter_audit(repo_root: Path, seed: int | None = None) -> dict:
    try:
        report = audit_parameters.run_audit(repo_root, seed=seed)
    except SystemExit as exc:
        return {"pass": False, "error": str(exc)}
    return {"pass": report["passed"], "report": report}


def run_tests(repo_root: Path) -> dict:
    """Run the full suite (pytest.ini's testpaths = tests), not just tests/unit."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    return {
        "pass": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-40:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-40:]),
    }


def _hash_file(repo_root: Path, path: Path) -> str:
    """Prefer Ibrahim's real sha256_file (confirms this tool is hashing the exact same
    way Ibrahim did); fall back to an equivalent local implementation if Ibrahim's
    module isn't importable from this repo_root."""
    try:
        import sys as _sys

        if str(repo_root) not in _sys.path:
            _sys.path.insert(0, str(repo_root))
        from src.reproducibility.adapters import manifest_hash_fn  # noqa: PLC0415

        return manifest_hash_fn()(path)
    except Exception:  # noqa: BLE001
        import hashlib

        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


def run_training_data_contract_check(repo_root: Path) -> dict:
    """Verify Ibrahim's data/tokenizer handoff contract: required fields present,
    m1_status == 'complete', and best-effort on-disk hash verification for
    whichever listed artifacts actually exist in this checkout (large
    Parquet files are commonly not checked out / gitignored, so a missing
    file is reported as 'skipped', not a failure)."""
    from src.reproducibility.adapters import (  # noqa: PLC0415
        MissingInterfaceError,
        data_seed_from_contract,
        load_training_data_contract,
        manifest_hashes_from_contract,
        tokenizer_hashes_from_contract,
        tokenizer_vocab_size_from_contract,
        training_subset_hash_from_contract,
    )

    try:
        contract = load_training_data_contract(repo_root)
    except MissingInterfaceError as exc:
        return {"pass": False, "error": str(exc)}

    problems: list[str] = []
    if contract.get("m1_status") != "complete":
        problems.append(f"m1_status is {contract.get('m1_status')!r}, expected 'complete'")

    try:
        manifest_hashes = manifest_hashes_from_contract(contract)
    except MissingInterfaceError as exc:
        problems.append(str(exc))
        manifest_hashes = {}

    try:
        tokenizer_hashes = tokenizer_hashes_from_contract(contract)
    except MissingInterfaceError as exc:
        problems.append(str(exc))
        tokenizer_hashes = {}

    try:
        vocab_size = tokenizer_vocab_size_from_contract(contract)
    except MissingInterfaceError as exc:
        problems.append(str(exc))
        vocab_size = None

    try:
        data_seed = data_seed_from_contract(contract)
    except MissingInterfaceError as exc:
        problems.append(str(exc))
        data_seed = None

    try:
        training_subset_hash = training_subset_hash_from_contract(contract)
    except MissingInterfaceError as exc:
        problems.append(str(exc))
        training_subset_hash = None

    verified: list[str] = []
    skipped_not_present: list[str] = []
    mismatched: list[dict] = []

    for split, expected_hash in manifest_hashes.items():
        rel_path = contract["artifacts"]["manifests"][split]["path"]
        full = repo_root / rel_path
        if not full.is_file():
            skipped_not_present.append(rel_path)
            continue
        actual_hash = _hash_file(repo_root, full)
        if actual_hash == expected_hash:
            verified.append(rel_path)
        else:
            mismatched.append({"path": rel_path, "expected": expected_hash, "actual": actual_hash})

    for filename, expected_hash in tokenizer_hashes.items():
        full = repo_root / "tokenizer" / filename
        rel_path = f"tokenizer/{filename}"
        if not full.is_file():
            skipped_not_present.append(rel_path)
            continue
        actual_hash = _hash_file(repo_root, full)
        if actual_hash == expected_hash:
            verified.append(rel_path)
        else:
            mismatched.append({"path": rel_path, "expected": expected_hash, "actual": actual_hash})

    if training_subset_hash is not None:
        # The 50M-token training subset (train_50m.parquet).
        rel_path = contract.get("training_subset", {}).get("manifest_path", "data/manifests/train_50m.parquet")
        full = repo_root / rel_path
        if not full.is_file():
            skipped_not_present.append(rel_path)
        else:
            actual_hash = _hash_file(repo_root, full)
            if actual_hash == training_subset_hash:
                verified.append(rel_path)
            else:
                mismatched.append({"path": rel_path, "expected": training_subset_hash, "actual": actual_hash})

    if mismatched:
        problems.append(f"hash mismatch for: {[m['path'] for m in mismatched]}")

    return {
        "pass": not problems,
        "m1_status": contract.get("m1_status"),
        "vocab_size": vocab_size,
        "data_seed": data_seed,
        "manifest_hashes": manifest_hashes,
        "tokenizer_hashes": tokenizer_hashes,
        "training_subset_hash": training_subset_hash,
        "verified_on_disk": verified,
        "skipped_not_present": skipped_not_present,
        "mismatched": mismatched,
        "problems": problems,
    }


def check_checkpoint_interface(repo_root: Path) -> dict:
    """Report whether checkpoint tests ran against Fidan's real
    training.checkpoint module or the SYNTHETIC_FIXTURE fallback."""
    import sys as _sys

    if str(repo_root) not in _sys.path:
        _sys.path.insert(0, str(repo_root))
    from src.reproducibility.adapters import MissingInterfaceError, checkpoint_adapter  # noqa: PLC0415

    try:
        checkpoint_adapter()
        return {
            "using_real_m3_checkpoint": True,
            "note": "tests/integration/test_checkpoint_integration.py ran against Fidan's real training.checkpoint.",
        }
    except MissingInterfaceError as exc:
        return {
            "using_real_m3_checkpoint": False,
            "note": (
                "Fidan's training.checkpoint module is not present; ran against the "
                f"SYNTHETIC_FIXTURE reference implementation instead. ({exc})"
            ),
        }


def run_verification(repo_root: Path, seed: int | None, skip_tests: bool) -> dict:
    result: dict = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        **git_info(repo_root),
    }

    artifacts = check_artifacts(repo_root)
    result["artifact_verification"] = artifacts
    result["checkpoint_interface"] = check_checkpoint_interface(repo_root)

    config_dir = repo_root / "configs" / "pe"
    if config_dir.exists():
        result["config_audit"] = run_config_audit(repo_root)
        result["parameter_audit"] = run_parameter_audit(repo_root, seed)
    else:
        result["config_audit"] = {"pass": False, "error": f"{config_dir} does not exist"}
        result["parameter_audit"] = {"pass": False, "error": f"{config_dir} does not exist"}

    result["training_data_contract"] = run_training_data_contract_check(repo_root)

    if skip_tests:
        result["test_result"] = {"pass": None, "skipped": True}
    else:
        result["test_result"] = run_tests(repo_root)

    blocking = [
        result["artifact_verification"]["pass"],
        result["config_audit"]["pass"],
        result["parameter_audit"]["pass"],
        result["training_data_contract"]["pass"],
    ]
    if not skip_tests:
        blocking.append(bool(result["test_result"]["pass"]))

    result["overall_pass"] = all(blocking)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Preregistered run seed to resolve for the parameter audit; "
        "defaults to the first seed in configs/run_matrix.json.",
    )
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("release_verification.json"))
    args = parser.parse_args()

    result = run_verification(args.repo_root, args.seed, args.skip_tests)

    args.out.write_text(json.dumps(result, indent=2, default=str))
    print(f"[verify_release] wrote {args.out}")
    print(f"[verify_release] overall_pass = {result['overall_pass']}")

    if not result["artifact_verification"]["pass"]:
        missing = [i["path"] for i in result["artifact_verification"]["items"] if i["required"] and not i["present"]]
        print(f"[verify_release] MISSING required artifacts: {missing}")

    if not result["checkpoint_interface"]["using_real_m3_checkpoint"]:
        print(f"[verify_release] NOTE: {result['checkpoint_interface']['note']}")

    return 0 if result["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
