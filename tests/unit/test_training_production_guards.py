import hashlib
import json

import pytest

from src.training.production_guards import (
    HEADLINE_DATA_SEED,
    HEADLINE_GLOBAL_BATCH_TOKENS,
    HEADLINE_INIT_SEEDS,
    HEADLINE_PE_TYPES,
    HEADLINE_PRECISION,
    HEADLINE_SEQ_LEN,
    HEADLINE_TOTAL_TOKENS,
    validate_cache_artifact,
    validate_headline_plan,
)


def write_metadata(
    path,
    *,
    token_count,
    cache_bytes,
    cache_sha256,
):
    path.write_text(
        json.dumps(
            {
                "dtype": "uint16",
                "target_tokens": token_count,
                "bytes": len(cache_bytes),
                "cache_sha256": cache_sha256,
            }
        ),
        encoding="utf-8",
    )


def make_headline_plan(
    *,
    micro_batch_sequences=8,
    grad_accum_steps=16,
):
    runs = []

    for seed in HEADLINE_INIT_SEEDS:
        for pe_type in HEADLINE_PE_TYPES:
            micro_batch_tokens = (
                micro_batch_sequences
                * HEADLINE_SEQ_LEN
            )

            runs.append(
                {
                    "run_id": (
                        f"test-{pe_type}-s{seed}"
                    ),
                    "pe_type": pe_type,
                    "init_seed": seed,
                    "data_seed": (
                        HEADLINE_DATA_SEED
                    ),
                    "total_tokens": (
                        HEADLINE_TOTAL_TOKENS
                    ),
                    "seq_len": (
                        HEADLINE_SEQ_LEN
                    ),
                    "global_batch_tokens": (
                        HEADLINE_GLOBAL_BATCH_TOKENS
                    ),
                    "precision": (
                        HEADLINE_PRECISION
                    ),
                    "micro_batch_sequences": (
                        micro_batch_sequences
                    ),
                    "micro_batch_tokens": (
                        micro_batch_tokens
                    ),
                    "grad_accum_steps": (
                        grad_accum_steps
                    ),
                }
            )

    return {
        "runs": runs,
    }


def test_cache_sha256_validation_passes(
    tmp_path,
):
    cache_path = (
        tmp_path / "tokens.uint16.bin"
    )

    metadata_path = (
        tmp_path / "tokens.uint16.json"
    )

    cache_bytes = bytes(
        [
            1,
            0,
            2,
            0,
            3,
            0,
            4,
            0,
        ]
    )

    cache_path.write_bytes(
        cache_bytes
    )

    cache_sha256 = (
        hashlib.sha256(
            cache_bytes
        ).hexdigest()
    )

    write_metadata(
        metadata_path,
        token_count=4,
        cache_bytes=cache_bytes,
        cache_sha256=cache_sha256,
    )

    actual = validate_cache_artifact(
        cache_path,
        metadata_path,
        expected_tokens=4,
    )

    assert actual == cache_sha256


def test_cache_sha256_corruption_is_rejected(
    tmp_path,
):
    cache_path = (
        tmp_path / "tokens.uint16.bin"
    )

    metadata_path = (
        tmp_path / "tokens.uint16.json"
    )

    original_bytes = bytes(
        [
            1,
            0,
            2,
            0,
            3,
            0,
            4,
            0,
        ]
    )

    cache_path.write_bytes(
        original_bytes
    )

    original_sha256 = (
        hashlib.sha256(
            original_bytes
        ).hexdigest()
    )

    write_metadata(
        metadata_path,
        token_count=4,
        cache_bytes=original_bytes,
        cache_sha256=original_sha256,
    )

    # Same file size, different contents.
    corrupted_bytes = bytes(
        [
            1,
            0,
            2,
            0,
            9,
            0,
            4,
            0,
        ]
    )

    cache_path.write_bytes(
        corrupted_bytes
    )

    with pytest.raises(
        ValueError,
        match="SHA-256 mismatch",
    ):
        validate_cache_artifact(
            cache_path,
            metadata_path,
            expected_tokens=4,
        )


def test_cache_metadata_token_mismatch_is_rejected(
    tmp_path,
):
    cache_path = (
        tmp_path / "tokens.uint16.bin"
    )

    metadata_path = (
        tmp_path / "tokens.uint16.json"
    )

    cache_bytes = bytes(
        [
            1,
            0,
            2,
            0,
            3,
            0,
            4,
            0,
        ]
    )

    cache_path.write_bytes(
        cache_bytes
    )

    cache_sha256 = (
        hashlib.sha256(
            cache_bytes
        ).hexdigest()
    )

    write_metadata(
        metadata_path,
        token_count=4,
        cache_bytes=cache_bytes,
        cache_sha256=cache_sha256,
    )

    with pytest.raises(
        ValueError,
        match="token count mismatch",
    ):
        validate_cache_artifact(
            cache_path,
            metadata_path,
            expected_tokens=5,
        )


def test_cache_physical_size_mismatch_is_rejected(
    tmp_path,
):
    cache_path = (
        tmp_path / "tokens.uint16.bin"
    )

    metadata_path = (
        tmp_path / "tokens.uint16.json"
    )

    expected_bytes = bytes(
        [
            1,
            0,
            2,
            0,
            3,
            0,
            4,
            0,
        ]
    )

    cache_sha256 = (
        hashlib.sha256(
            expected_bytes
        ).hexdigest()
    )

    write_metadata(
        metadata_path,
        token_count=4,
        cache_bytes=expected_bytes,
        cache_sha256=cache_sha256,
    )

    # Metadata declares the correct 8-byte uint16 cache,
    # but the physical file is truncated to 6 bytes.
    cache_path.write_bytes(
        expected_bytes[:-2]
    )

    with pytest.raises(
        ValueError,
        match="file byte-size mismatch",
    ):
        validate_cache_artifact(
            cache_path,
            metadata_path,
            expected_tokens=4,
        )


def test_cache_invalid_sha256_is_rejected(
    tmp_path,
):
    cache_path = (
        tmp_path / "tokens.uint16.bin"
    )

    metadata_path = (
        tmp_path / "tokens.uint16.json"
    )

    cache_bytes = bytes(
        [
            1,
            0,
            2,
            0,
            3,
            0,
            4,
            0,
        ]
    )

    cache_path.write_bytes(
        cache_bytes
    )

    write_metadata(
        metadata_path,
        token_count=4,
        cache_bytes=cache_bytes,
        cache_sha256="not-a-valid-sha256",
    )

    with pytest.raises(
        ValueError,
        match="invalid cache_sha256",
    ):
        validate_cache_artifact(
            cache_path,
            metadata_path,
            expected_tokens=4,
        )


def test_headline_plan_validation_passes():
    payload = make_headline_plan()

    validate_headline_plan(
        payload
    )


def test_headline_plan_requires_exact_run_count():
    payload = make_headline_plan()

    payload["runs"].pop()

    with pytest.raises(
        ValueError,
        match="exactly 25 runs",
    ):
        validate_headline_plan(
            payload
        )


def test_headline_plan_rejects_wrong_pe_seed_matrix():
    payload = make_headline_plan()

    payload["runs"][0][
        "init_seed"
    ] = 999_999

    with pytest.raises(
        ValueError,
        match="PE/seed matrix mismatch",
    ):
        validate_headline_plan(
            payload
        )


@pytest.mark.parametrize(
    (
        "field",
        "bad_value",
    ),
    [
        (
            "data_seed",
            HEADLINE_DATA_SEED + 1,
        ),
        (
            "total_tokens",
            HEADLINE_TOTAL_TOKENS - 1,
        ),
        (
            "seq_len",
            HEADLINE_SEQ_LEN // 2,
        ),
        (
            "global_batch_tokens",
            HEADLINE_GLOBAL_BATCH_TOKENS // 2,
        ),
        (
            "precision",
            "fp32",
        ),
    ],
)
def test_headline_plan_rejects_frozen_field_drift(
    field,
    bad_value,
):
    payload = make_headline_plan()

    payload["runs"][0][
        field
    ] = bad_value

    with pytest.raises(
        ValueError,
        match="freeze violation",
    ):
        validate_headline_plan(
            payload
        )


def test_headline_plan_rejects_microbatch_token_mismatch():
    payload = make_headline_plan()

    payload["runs"][0][
        "micro_batch_tokens"
    ] += HEADLINE_SEQ_LEN

    with pytest.raises(
        ValueError,
        match="Microbatch token count mismatch",
    ):
        validate_headline_plan(
            payload
        )


def test_headline_plan_rejects_effective_global_batch_mismatch():
    payload = make_headline_plan()

    payload["runs"][0][
        "grad_accum_steps"
    ] -= 1

    with pytest.raises(
        ValueError,
        match="Effective global batch mismatch",
    ):
        validate_headline_plan(
            payload
        )


def test_headline_plan_rejects_inconsistent_microbatch_configuration():
    payload = make_headline_plan()

    payload["runs"][0][
        "micro_batch_sequences"
    ] = 16

    payload["runs"][0][
        "micro_batch_tokens"
    ] = (
        16
        * HEADLINE_SEQ_LEN
    )

    payload["runs"][0][
        "grad_accum_steps"
    ] = 8

    with pytest.raises(
        ValueError,
        match="same microbatch configuration",
    ):
        validate_headline_plan(
            payload
        )


def test_headline_plan_rejects_nonpositive_microbatch():
    payload = make_headline_plan()

    payload["runs"][0][
        "micro_batch_sequences"
    ] = 0

    payload["runs"][0][
        "micro_batch_tokens"
    ] = 0

    with pytest.raises(
        ValueError,
        match="Invalid microbatch",
    ):
        validate_headline_plan(
            payload
        )


def test_headline_plan_rejects_nonpositive_gas():
    payload = make_headline_plan()

    payload["runs"][0][
        "grad_accum_steps"
    ] = 0

    with pytest.raises(
        ValueError,
        match="Invalid GAS",
    ):
        validate_headline_plan(
            payload
        )
