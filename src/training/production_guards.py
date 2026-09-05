"""Production safety checks for M3 training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.training.cache_builder import sha256_file


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

    # uint16 = exactly 2 bytes per token.
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
            character
            not in "0123456789abcdef"
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