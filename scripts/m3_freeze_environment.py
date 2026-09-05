#!/usr/bin/env python3
"""Freeze the exact runtime environment used for A100 headline training.

Run this on the target A100 server AFTER the benchmark environment is chosen
and BEFORE the 25-run headline matrix is launched.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(
    __file__
).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )

from src.reproducibility.metadata import (  # noqa: E402
    UNAVAILABLE,
    environment_fingerprint,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "configs/hardware/"
            "a100_environment.json"
        ),
    )

    parser.add_argument(
        "--require-cuda",
        action="store_true",
    )

    parser.add_argument(
        "--require-bf16",
        action="store_true",
    )

    return parser.parse_args()


def resolve_path(
    path: Path,
) -> Path:
    if path.is_absolute():
        return path.resolve()

    return (
        REPO_ROOT
        / path
    ).resolve()


def main():
    args = parse_args()

    environment = (
        environment_fingerprint()
    )

    if (
        args.require_cuda
        and environment[
            "cuda_version"
        ] == UNAVAILABLE
    ):
        raise RuntimeError(
            "CUDA is required before freezing "
            "the A100 training environment."
        )

    if (
        args.require_bf16
        and environment[
            "bf16_supported"
        ] is not True
    ):
        raise RuntimeError(
            "CUDA bf16 support is required before "
            "freezing the A100 training environment."
        )

    out_path = resolve_path(
        args.out
    )

    payload = {
        "schema_version": 1,
        "captured_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "environment": environment,
    }

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"[m3_freeze_environment] wrote "
        f"{out_path}"
    )

    for key, value in environment.items():
        print(
            f"{key}: {value}"
        )


if __name__ == "__main__":
    main()
