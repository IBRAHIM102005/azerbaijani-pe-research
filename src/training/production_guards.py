"""Production safety checks for M3 training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.training.cache_builder import sha256_file


HEADLINE_PE_TYPES = (
    "learned",
    "sinusoidal",
    "rope",
    "alibi",
    "nope",
)

HEADLINE_INIT_SEEDS = (
    17,
    42,
    1234,
    2027,
    5003,
)

HEADLINE_DATA_SEED = 2026
HEADLINE_TOTAL_TOKENS = 50_000_000
HEADLINE_SEQ_LEN = 512
HEADLINE_GLOBAL_BATCH_TOKENS = 65_536
HEADLINE_PRECISION = "bf16"
HEADLINE_NUM_RUNS = 25


def load_json_object(
    path: str | Path,
) -> dict[str, Any]:
    """Load one JSON object from disk."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {path}"
        )

    payload = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )

    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected JSON object in {path}"
        )

    return payload


def validate_cache_artifact(
    cache_path: str | Path,
    metadata_path: str | Path,
    *,
    expected_tokens: int,
) -> str:
    """Verify training cache size and SHA-256 identity."""

    cache_path = Path(cache_path)
    metadata_path = Path(metadata_path)

    if expected_tokens <= 0:
        raise ValueError(
            "expected_tokens must be positive."
        )

    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Training cache not found: {cache_path}"
        )

    metadata = load_json_object(
        metadata_path
    )

    if metadata.get("dtype") != "uint16":
        raise ValueError(
            "Training cache metadata must declare "
            "dtype='uint16'."
        )

    metadata_tokens = int(
        metadata.get(
            "target_tokens",
            -1,
        )
    )

    if metadata_tokens != expected_tokens:
        raise ValueError(
            "Training cache token count mismatch: "
            f"metadata={metadata_tokens:,}, "
            f"expected={expected_tokens:,}"
        )

    expected_bytes = (
        expected_tokens * 2
    )

    metadata_bytes = int(
        metadata.get(
            "bytes",
            -1,
        )
    )

    if metadata_bytes != expected_bytes:
        raise ValueError(
            "Training cache metadata byte-size mismatch: "
            f"metadata={metadata_bytes:,}, "
            f"expected={expected_bytes:,}"
        )

    actual_bytes = (
        cache_path.stat().st_size
    )

    if actual_bytes != expected_bytes:
        raise ValueError(
            "Training cache file byte-size mismatch: "
            f"actual={actual_bytes:,}, "
            f"expected={expected_bytes:,}"
        )

    expected_sha256 = str(
        metadata.get(
            "cache_sha256",
            "",
        )
    ).lower()

    if (
        len(expected_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_sha256
        )
    ):
        raise ValueError(
            "Training cache metadata contains "
            "an invalid cache_sha256."
        )

    actual_sha256 = sha256_file(
        cache_path
    )

    if actual_sha256 != expected_sha256:
        raise ValueError(
            "Training cache SHA-256 mismatch: "
            f"expected={expected_sha256}, "
            f"actual={actual_sha256}"
        )

    return actual_sha256


def validate_headline_plan(
    payload: dict[str, Any],
) -> None:
    """Validate the frozen 5 PE × 5 seed headline experiment."""

    runs = payload.get(
        "runs"
    )

    if not isinstance(runs, list):
        raise ValueError(
            "Headline plan must contain a runs list."
        )

    if len(runs) != HEADLINE_NUM_RUNS:
        raise ValueError(
            "Headline plan must contain exactly "
            f"{HEADLINE_NUM_RUNS} runs, "
            f"got {len(runs)}."
        )

    expected_pairs = {
        (
            pe_type,
            seed,
        )
        for seed in HEADLINE_INIT_SEEDS
        for pe_type in HEADLINE_PE_TYPES
    }

    observed_pairs: set[
        tuple[str, int]
    ] = set()

    observed_microbatch: set[int] = set()
    observed_gas: set[int] = set()

    for run in runs:
        if not isinstance(run, dict):
            raise TypeError(
                "Each headline run must be a dictionary."
            )

        run_id = str(
            run.get(
                "run_id",
                "<unknown>",
            )
        )

        pe_type = str(
            run.get(
                "pe_type",
                "",
            )
        )

        init_seed = int(
            run.get(
                "init_seed",
                -1,
            )
        )

        observed_pairs.add(
            (
                pe_type,
                init_seed,
            )
        )

        checks = {
            "data_seed": (
                run.get("data_seed"),
                HEADLINE_DATA_SEED,
            ),
            "total_tokens": (
                run.get("total_tokens"),
                HEADLINE_TOTAL_TOKENS,
            ),
            "seq_len": (
                run.get("seq_len"),
                HEADLINE_SEQ_LEN,
            ),
            "global_batch_tokens": (
                run.get("global_batch_tokens"),
                HEADLINE_GLOBAL_BATCH_TOKENS,
            ),
            "precision": (
                run.get("precision"),
                HEADLINE_PRECISION,
            ),
        }

        for key, (
            actual,
            expected,
        ) in checks.items():
            if actual != expected:
                raise ValueError(
                    "Headline plan freeze violation "
                    f"for {run_id}: "
                    f"{key}={actual!r}, "
                    f"expected={expected!r}"
                )

        micro_batch_sequences = int(
            run.get(
                "micro_batch_sequences",
                0,
            )
        )

        grad_accum_steps = int(
            run.get(
                "grad_accum_steps",
                0,
            )
        )

        micro_batch_tokens = int(
            run.get(
                "micro_batch_tokens",
                0,
            )
        )

        if micro_batch_sequences <= 0:
            raise ValueError(
                f"Invalid microbatch for {run_id}."
            )

        if grad_accum_steps <= 0:
            raise ValueError(
                f"Invalid GAS for {run_id}."
            )

        expected_micro_batch_tokens = (
            micro_batch_sequences
            * HEADLINE_SEQ_LEN
        )

        if (
            micro_batch_tokens
            != expected_micro_batch_tokens
        ):
            raise ValueError(
                "Microbatch token count mismatch "
                f"for {run_id}: "
                f"{micro_batch_tokens} != "
                f"{expected_micro_batch_tokens}"
            )

        effective_global_batch = (
            micro_batch_tokens
            * grad_accum_steps
        )

        if (
            effective_global_batch
            != HEADLINE_GLOBAL_BATCH_TOKENS
        ):
            raise ValueError(
                "Effective global batch mismatch "
                f"for {run_id}: "
                f"{effective_global_batch} != "
                f"{HEADLINE_GLOBAL_BATCH_TOKENS}"
            )

        observed_microbatch.add(
            micro_batch_sequences
        )

        observed_gas.add(
            grad_accum_steps
        )

    if observed_pairs != expected_pairs:
        missing = (
            expected_pairs
            - observed_pairs
        )

        extra = (
            observed_pairs
            - expected_pairs
        )

        raise ValueError(
            "Headline PE/seed matrix mismatch. "
            f"missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    if len(observed_microbatch) != 1:
        raise ValueError(
            "All headline runs must use the same "
            "microbatch configuration."
        )

    if len(observed_gas) != 1:
        raise ValueError(
            "All headline runs must use the same "
            "gradient accumulation setting."
        )