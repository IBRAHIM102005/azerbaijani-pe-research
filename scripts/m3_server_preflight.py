#!/usr/bin/env python3
"""M3 server preflight.

Run this BEFORE:
    - building the frozen 50M token cache
    - A100 benchmarking
    - formal 5-PE smoke runs
    - full scientific training

Checks:
    1. Python / PyTorch / CUDA environment
    2. GPU visibility and bf16 support
    3. frozen M1 data contract
    4. tokenizer artifact + hash
    5. train_50m manifest + hash
    6. processed train corpus + hash
    7. cache presence/size/SHA-256 identity if already built
    8. frozen runtime environment lock if available
    9. filesystem free space
    10. M3 run-plan presence and strict headline validation

This script does not modify any scientific artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


# ============================================================
# Repository
# ============================================================

REPO_ROOT = Path(
    __file__
).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )


from src.models.data_contract import (
    load_contract,
)

from src.reproducibility.metadata import (
    environment_fingerprint,
)

from src.training.production_guards import (
    validate_cache_artifact,
    validate_headline_plan,
)


# ============================================================
# Result type
# ============================================================


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    @property
    def failed(self) -> bool:
        return self.status == "FAIL"


# ============================================================
# CLI
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-contract",
        type=Path,
        default=Path(
            "data/metadata/"
            "training_data_contract.json"
        ),
    )

    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "data/cache/"
            "train_50m.uint16.bin"
        ),
    )

    parser.add_argument(
        "--cache-metadata",
        type=Path,
        default=Path(
            "data/cache/"
            "train_50m.uint16.json"
        ),
    )

    parser.add_argument(
        "--environment-lock",
        type=Path,
        default=Path(
            "configs/hardware/"
            "a100_environment.json"
        ),
    )

    parser.add_argument(
        "--plan",
        type=Path,
        default=Path(
            "results/manifests/"
            "m3_run_plan.json"
        ),
    )

    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=5.0,
        help=(
            "Minimum free disk space required "
            "on the filesystem containing the cache."
        ),
    )

    parser.add_argument(
        "--skip-large-hashes",
        action="store_true",
        help=(
            "Skip SHA-256 of large corpus artifacts. "
            "Useful for a fast first inspection."
        ),
    )

    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help=(
            "Fail if CUDA is unavailable."
        ),
    )

    parser.add_argument(
        "--require-bf16",
        action="store_true",
        help=(
            "Fail if CUDA bf16 is unavailable."
        ),
    )

    parser.add_argument(
        "--require-cache",
        action="store_true",
        help=(
            "Fail if the frozen cache or its "
            "SHA-256 metadata is unavailable."
        ),
    )

    parser.add_argument(
        "--require-environment-lock",
        action="store_true",
        help=(
            "Fail unless the current runtime exactly "
            "matches the frozen A100 environment lock."
        ),
    )

    parser.add_argument(
        "--require-headline-plan",
        action="store_true",
        help=(
            "Fail unless the run plan passes the "
            "strict 25-run headline validator."
        ),
    )

    return parser.parse_args()


# ============================================================
# Helpers
# ============================================================


def resolve_path(
    path: Path,
) -> Path:
    if path.is_absolute():
        return path.resolve()

    return (
        REPO_ROOT
        / path
    ).resolve()


def sha256_file(
    path: Path,
    *,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

        while True:
            block = handle.read(
                chunk_size
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def human_bytes(
    value: int,
) -> str:
    size = float(
        value
    )

    units = [
        "B",
        "KiB",
        "MiB",
        "GiB",
        "TiB",
    ]

    for unit in units:

        if (
            size < 1024.0
            or unit == units[-1]
        ):
            return (
                f"{size:.2f} {unit}"
            )

        size /= 1024.0

    return f"{size:.2f} TiB"


def load_json(
    path: Path,
) -> dict[str, Any]:

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def artifact_entry(
    contract,
    *keys: str,
) -> dict[str, Any]:
    value: Any = (
        contract.raw["artifacts"]
    )

    for key in keys:
        value = value[key]

    if not isinstance(
        value,
        dict,
    ):
        raise TypeError(
            "Artifact entry must be "
            "a dictionary."
        )

    return value


def check_file(
    *,
    name: str,
    path: Path,
    expected_bytes: int | None = None,
) -> CheckResult:

    if not path.is_file():
        return CheckResult(
            name=name,
            status="FAIL",
            detail=(
                f"missing: {path}"
            ),
        )

    actual_bytes = (
        path.stat().st_size
    )

    if (
        expected_bytes is not None
        and actual_bytes
        != expected_bytes
    ):
        return CheckResult(
            name=name,
            status="FAIL",
            detail=(
                f"size mismatch: "
                f"expected "
                f"{human_bytes(expected_bytes)}, "
                f"got "
                f"{human_bytes(actual_bytes)} "
                f"({path})"
            ),
        )

    return CheckResult(
        name=name,
        status="PASS",
        detail=(
            f"{human_bytes(actual_bytes)} "
            f"— {path}"
        ),
    )


def check_hash(
    *,
    name: str,
    path: Path,
    expected_sha256: str,
) -> CheckResult:

    if not path.is_file():
        return CheckResult(
            name=name,
            status="FAIL",
            detail=(
                f"cannot hash missing file: "
                f"{path}"
            ),
        )

    actual = sha256_file(
        path
    )

    if actual != expected_sha256:
        return CheckResult(
            name=name,
            status="FAIL",
            detail=(
                "SHA-256 mismatch\n"
                f"expected={expected_sha256}\n"
                f"actual  ={actual}"
            ),
        )

    return CheckResult(
        name=name,
        status="PASS",
        detail=(
            f"sha256={actual}"
        ),
    )


def add_result(
    results: list[CheckResult],
    result: CheckResult,
) -> None:
    results.append(
        result
    )

    print(
        f"[{result.status:<4}] "
        f"{result.name}"
    )

    print(
        f"       {result.detail}"
    )


# ============================================================
# Main
# ============================================================


def main():
    args = parse_args()

    if args.min_free_gb <= 0:
        raise ValueError(
            "--min-free-gb must be positive."
        )

    contract_path = resolve_path(
        args.data_contract
    )

    cache_path = resolve_path(
        args.cache
    )

    cache_metadata_path = resolve_path(
        args.cache_metadata
    )

    environment_lock_path = resolve_path(
        args.environment_lock
    )

    plan_path = resolve_path(
        args.plan
    )

    results: list[
        CheckResult
    ] = []

    print()
    print("=" * 78)
    print("M3 SERVER PREFLIGHT")
    print("=" * 78)

    print(
        f"repository: "
        f"{REPO_ROOT}"
    )

    print(
        f"python:     "
        f"{platform.python_version()}"
    )

    print(
        f"torch:      "
        f"{torch.__version__}"
    )

    print()

    # ========================================================
    # Contract
    # ========================================================

    contract_file_result = check_file(
        name="M1 data contract",
        path=contract_path,
    )

    add_result(
        results,
        contract_file_result,
    )

    if contract_file_result.failed:
        print()
        print(
            "Cannot continue without "
            "the frozen M1 contract."
        )

        raise SystemExit(
            1
        )

    contract = load_contract(
        contract_path
    )

    add_result(
        results,
        CheckResult(
            name="M1 frozen identity",
            status="PASS",
            detail=(
                f"target_tokens="
                f"{contract.target_tokens:,}, "
                f"selected_tokens="
                f"{contract.selected_tokens:,}, "
                f"data_seed="
                f"{contract.data_seed}, "
                f"vocab="
                f"{contract.vocab_size:,}, "
                f"eod_id="
                f"{contract.eod_id}"
            ),
        ),
    )

    if (
        contract.model_seed_affects_order
    ):
        add_result(
            results,
            CheckResult(
                name="paired data order",
                status="FAIL",
                detail=(
                    "contract says model seed "
                    "affects data order"
                ),
            ),
        )

    else:
        add_result(
            results,
            CheckResult(
                name="paired data order",
                status="PASS",
                detail=(
                    "model_seed_affects_order=False"
                ),
            ),
        )

    # ========================================================
    # CUDA
    # ========================================================

    cuda_available = (
        torch.cuda.is_available()
    )

    cuda_status = (
        "PASS"
        if cuda_available
        else (
            "FAIL"
            if args.require_cuda
            else "WARN"
        )
    )

    add_result(
        results,
        CheckResult(
            name="CUDA",
            status=cuda_status,
            detail=(
                f"available="
                f"{cuda_available}, "
                f"torch_cuda="
                f"{torch.version.cuda}"
            ),
        ),
    )

    bf16_supported = False

    if cuda_available:

        device_count = (
            torch.cuda.device_count()
        )

        gpu_lines = []

        for index in range(
            device_count
        ):
            props = (
                torch.cuda.get_device_properties(
                    index
                )
            )

            gpu_lines.append(
                f"cuda:{index} "
                f"{props.name}, "
                f"{props.total_memory / (1024**3):.2f} GiB"
            )

        add_result(
            results,
            CheckResult(
                name="visible GPUs",
                status="PASS",
                detail=(
                    "; ".join(
                        gpu_lines
                    )
                ),
            ),
        )

        bf16_supported = (
            torch.cuda.is_bf16_supported()
        )

        bf16_status = (
            "PASS"
            if bf16_supported
            else (
                "FAIL"
                if args.require_bf16
                else "WARN"
            )
        )

        add_result(
            results,
            CheckResult(
                name="CUDA bf16",
                status=bf16_status,
                detail=(
                    f"supported="
                    f"{bf16_supported}"
                ),
            ),
        )

    elif args.require_bf16:

        add_result(
            results,
            CheckResult(
                name="CUDA bf16",
                status="FAIL",
                detail=(
                    "CUDA unavailable, therefore "
                    "bf16 CUDA training unavailable"
                ),
            ),
        )

    # ========================================================
    # Frozen runtime environment
    # ========================================================

    if environment_lock_path.is_file():

        try:
            environment_lock = load_json(
                environment_lock_path
            )

            expected_environment = (
                environment_lock.get(
                    "environment"
                )
            )

            if not isinstance(
                expected_environment,
                dict,
            ):
                raise ValueError(
                    "environment lock is missing "
                    "an 'environment' object"
                )

            actual_environment = (
                environment_fingerprint()
            )

            mismatches = {
                key: {
                    "expected": expected_environment.get(
                        key
                    ),
                    "actual": actual_environment.get(
                        key
                    ),
                }
                for key in expected_environment
                if (
                    actual_environment.get(
                        key
                    )
                    != expected_environment.get(
                        key
                    )
                )
            }

            if mismatches:
                add_result(
                    results,
                    CheckResult(
                        name="A100 environment lock",
                        status="FAIL",
                        detail=(
                            "runtime mismatch: "
                            f"{json.dumps(mismatches, sort_keys=True)}"
                        ),
                    ),
                )

            else:
                add_result(
                    results,
                    CheckResult(
                        name="A100 environment lock",
                        status="PASS",
                        detail=(
                            f"exact match — "
                            f"{environment_lock_path}"
                        ),
                    ),
                )

        except Exception as exc:
            add_result(
                results,
                CheckResult(
                    name="A100 environment lock",
                    status="FAIL",
                    detail=(
                        f"invalid lock: {exc}"
                    ),
                ),
            )

    else:

        add_result(
            results,
            CheckResult(
                name="A100 environment lock",
                status=(
                    "FAIL"
                    if args.require_environment_lock
                    else "WARN"
                ),
                detail=(
                    f"not frozen yet: "
                    f"{environment_lock_path}"
                ),
            ),
        )

    # ========================================================
    # Tokenizer
    # ========================================================

    tokenizer_entry = artifact_entry(
        contract,
        "tokenizer",
        "tokenizer.model",
    )

    tokenizer_path = (
        REPO_ROOT
        / tokenizer_entry["path"]
    ).resolve()

    add_result(
        results,
        check_file(
            name="tokenizer.model",
            path=tokenizer_path,
            expected_bytes=int(
                tokenizer_entry["bytes"]
            ),
        ),
    )

    if tokenizer_path.is_file():

        add_result(
            results,
            check_hash(
                name="tokenizer.model hash",
                path=tokenizer_path,
                expected_sha256=(
                    tokenizer_entry[
                        "sha256"
                    ]
                ),
            ),
        )

    # ========================================================
    # 50M subset manifest
    # ========================================================

    subset_entry = artifact_entry(
        contract,
        "training_subset_manifest",
    )

    subset_path = (
        REPO_ROOT
        / subset_entry["path"]
    ).resolve()

    add_result(
        results,
        check_file(
            name="train_50m manifest",
            path=subset_path,
            expected_bytes=int(
                subset_entry["bytes"]
            ),
        ),
    )

    if (
        subset_path.is_file()
        and not args.skip_large_hashes
    ):
        add_result(
            results,
            check_hash(
                name="train_50m manifest hash",
                path=subset_path,
                expected_sha256=(
                    subset_entry[
                        "sha256"
                    ]
                ),
            ),
        )

    elif (
        subset_path.is_file()
        and args.skip_large_hashes
    ):
        add_result(
            results,
            CheckResult(
                name="train_50m manifest hash",
                status="WARN",
                detail=(
                    "skipped by "
                    "--skip-large-hashes"
                ),
            ),
        )

    # ========================================================
    # Processed training corpus
    # ========================================================

    processed_entry = artifact_entry(
        contract,
        "processed_corpus",
        "train",
    )

    processed_path = (
        REPO_ROOT
        / processed_entry["path"]
    ).resolve()

    add_result(
        results,
        check_file(
            name="processed train corpus",
            path=processed_path,
            expected_bytes=int(
                processed_entry["bytes"]
            ),
        ),
    )

    if (
        processed_path.is_file()
        and not args.skip_large_hashes
    ):
        print(
            "       hashing ~2 GB processed "
            "corpus; this may take a while..."
        )

        add_result(
            results,
            check_hash(
                name="processed train hash",
                path=processed_path,
                expected_sha256=(
                    processed_entry[
                        "sha256"
                    ]
                ),
            ),
        )

    elif (
        processed_path.is_file()
        and args.skip_large_hashes
    ):
        add_result(
            results,
            CheckResult(
                name="processed train hash",
                status="WARN",
                detail=(
                    "skipped by "
                    "--skip-large-hashes"
                ),
            ),
        )

    # ========================================================
    # Cache
    # ========================================================

    expected_cache_bytes = (
        contract.target_tokens
        * 2
    )

    if cache_path.is_file():

        add_result(
            results,
            check_file(
                name="50M uint16 cache",
                path=cache_path,
                expected_bytes=(
                    expected_cache_bytes
                ),
            ),
        )

        try:
            cache_sha256 = validate_cache_artifact(
                cache_path,
                cache_metadata_path,
                expected_tokens=(
                    contract.target_tokens
                ),
            )

            add_result(
                results,
                CheckResult(
                    name="50M cache SHA-256 identity",
                    status="PASS",
                    detail=(
                        f"sha256={cache_sha256}; "
                        f"metadata={cache_metadata_path}"
                    ),
                ),
            )

        except Exception as exc:
            add_result(
                results,
                CheckResult(
                    name="50M cache SHA-256 identity",
                    status="FAIL",
                    detail=str(
                        exc
                    ),
                ),
            )

    else:

        add_result(
            results,
            CheckResult(
                name="50M uint16 cache",
                status=(
                    "FAIL"
                    if args.require_cache
                    else "WARN"
                ),
                detail=(
                    "not built yet — expected "
                    f"{human_bytes(expected_cache_bytes)} "
                    f"at {cache_path}"
                ),
            ),
        )

        if (
            args.require_cache
            and not cache_metadata_path.is_file()
        ):
            add_result(
                results,
                CheckResult(
                    name="50M cache SHA-256 identity",
                    status="FAIL",
                    detail=(
                        "cache metadata missing: "
                        f"{cache_metadata_path}"
                    ),
                ),
            )

    # ========================================================
    # Disk space
    # ========================================================

    cache_parent = (
        cache_path.parent
    )

    cache_parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    disk = shutil.disk_usage(
        cache_parent
    )

    free_gb = (
        disk.free
        / (1024 ** 3)
    )

    disk_status = (
        "PASS"
        if free_gb
        >= args.min_free_gb
        else "FAIL"
    )

    add_result(
        results,
        CheckResult(
            name="cache filesystem free space",
            status=disk_status,
            detail=(
                f"free={free_gb:.2f} GiB, "
                f"required>="
                f"{args.min_free_gb:.2f} GiB"
            ),
        ),
    )

    # ========================================================
    # M3 run plan
    # ========================================================

    if plan_path.is_file():

        try:
            plan = load_json(
                plan_path
            )

            num_runs = len(
                plan.get(
                    "runs",
                    [],
                )
            )

            detail = (
                f"runs={num_runs}, "
                f"seq_len="
                f"{plan.get('seq_len')}, "
                f"microbatch="
                f"{plan.get('micro_batch_sequences')}, "
                f"GAS="
                f"{plan.get('grad_accum_steps')}, "
                f"global_batch="
                f"{plan.get('global_batch_tokens')}, "
                f"precision="
                f"{plan.get('precision')}"
            )

            try:
                validate_headline_plan(
                    plan
                )

                add_result(
                    results,
                    CheckResult(
                        name="M3 headline run plan",
                        status="PASS",
                        detail=detail,
                    ),
                )

            except Exception as exc:
                add_result(
                    results,
                    CheckResult(
                        name="M3 headline run plan",
                        status=(
                            "FAIL"
                            if args.require_headline_plan
                            else "WARN"
                        ),
                        detail=(
                            f"{detail}; "
                            f"not frozen: {exc}"
                        ),
                    ),
                )

        except Exception as exc:
            add_result(
                results,
                CheckResult(
                    name="M3 headline run plan",
                    status=(
                        "FAIL"
                        if args.require_headline_plan
                        else "WARN"
                    ),
                    detail=(
                        f"invalid plan: {exc}"
                    ),
                ),
            )

    else:

        add_result(
            results,
            CheckResult(
                name="M3 headline run plan",
                status=(
                    "FAIL"
                    if args.require_headline_plan
                    else "WARN"
                ),
                detail=(
                    f"not found: {plan_path}"
                ),
            ),
        )

    # ========================================================
    # Summary
    # ========================================================

    failures = [
        result
        for result in results
        if result.failed
    ]

    warnings = [
        result
        for result in results
        if result.status == "WARN"
    ]

    passes = [
        result
        for result in results
        if result.passed
    ]

    print()
    print("=" * 78)
    print("PREFLIGHT SUMMARY")
    print("=" * 78)

    print(
        f"PASS: {len(passes)}"
    )

    print(
        f"WARN: {len(warnings)}"
    )

    print(
        f"FAIL: {len(failures)}"
    )

    if failures:

        print()
        print(
            "SERVER IS NOT READY "
            "FOR SCIENTIFIC TRAINING."
        )

        print()
        print(
            "Fix the FAIL items above "
            "before continuing."
        )

        raise SystemExit(
            1
        )

    print()

    if warnings:

        print(
            "PREFLIGHT PASSED WITH WARNINGS."
        )

        print(
            "Warnings may be expected before "
            "cache build / benchmark."
        )

    else:

        print(
            "PREFLIGHT PASSED."
        )


if __name__ == "__main__":
    main()
