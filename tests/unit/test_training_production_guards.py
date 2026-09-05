import hashlib
import json

import pytest

from src.training.production_guards import (
    validate_cache_artifact,
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