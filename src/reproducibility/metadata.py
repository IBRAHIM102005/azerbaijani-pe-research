"""Run metadata collection.

Collects everything listed in the project plan for a single experiment
run into a stable, JSON-serializable dict.

Must work on CPU-only machines and must never crash because CUDA,
Git, or another optional runtime facility is unavailable. Every field
that cannot be determined is set to an explicit sentinel string rather
than omitted or guessed.
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from src.reproducibility.config_utils import (
    config_hash,
)


UNAVAILABLE = "unavailable"

_SHA256_HEX_RE = re.compile(
    r"^[0-9a-f]{64}$"
)

_SAFE_RUN_ID_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
)


# ============================================================
# Subprocess helper
# ============================================================


def _run(
    cmd: list[str],
) -> str | None:
    """Run a small metadata command safely.

    Any failure means the requested information is unavailable in
    the current environment.
    """

    try:

        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )

        return (
            out.stdout.strip()
        )

    except Exception:
        return None


# ============================================================
# Git metadata
# ============================================================


def git_info(
    repo_dir: str | Path = ".",
) -> dict:
    """Return Git commit and dirty status for the requested repository.

    repo_dir is expected to be the repository root.

    We deliberately require a .git marker directly inside repo_dir
    before invoking Git. This prevents Git's parent-directory discovery
    or inherited Git environment variables from accidentally reporting
    metadata for a different repository.

    A normal Git checkout has a .git directory.
    A Git worktree may have a .git file.
    Both are accepted by Path.exists().
    """

    repo_path = Path(
        repo_dir
    ).resolve()

    unavailable = {
        "git_commit": UNAVAILABLE,
        "dirty_repository": UNAVAILABLE,
    }

    # --------------------------------------------------------
    # The supplied path must exist and be a directory.
    # --------------------------------------------------------

    if (
        not repo_path.exists()
        or not repo_path.is_dir()
    ):
        return unavailable

    # --------------------------------------------------------
    # Critical isolation check.
    #
    # Do NOT ask Git to discover a parent repository.
    # repo_dir is supposed to be the repository root itself.
    #
    # Normal checkout:
    #     repo/.git/        directory
    #
    # Git worktree:
    #     repo/.git         file
    # --------------------------------------------------------

    git_marker = (
        repo_path
        / ".git"
    )

    if not git_marker.exists():
        return unavailable

    # --------------------------------------------------------
    # Only now is it safe to query Git.
    # --------------------------------------------------------

    commit = _run(
        [
            "git",
            "-C",
            str(
                repo_path
            ),
            "rev-parse",
            "HEAD",
        ]
    )

    status = _run(
        [
            "git",
            "-C",
            str(
                repo_path
            ),
            "status",
            "--porcelain",
        ]
    )

    if commit is None:
        return unavailable

    return {
        "git_commit": commit,
        "dirty_repository": (
            status != ""
            if status is not None
            else UNAVAILABLE
        ),
    }


# ============================================================
# Timestamps
# ============================================================


def timestamps() -> dict:
    """Return UTC and Baku timestamps."""

    now_utc = dt.datetime.now(
        dt.timezone.utc
    )

    baku = dt.timezone(
        dt.timedelta(
            hours=4
        )
    )

    now_baku = (
        now_utc.astimezone(
            baku
        )
    )

    return {
        "timestamp_utc": (
            now_utc.isoformat()
        ),
        "timestamp_baku": (
            now_baku.isoformat()
        ),
    }


# ============================================================
# Device metadata
# ============================================================


def device_info() -> dict:
    """Return CPU/CUDA/PyTorch metadata safely.

    Never raises merely because CUDA or PyTorch is unavailable.
    """

    info: dict[
        str,
        Any,
    ] = {
        "device_name": (
            UNAVAILABLE
        ),
        "device_uuid": (
            UNAVAILABLE
        ),
        "cuda_version": (
            UNAVAILABLE
        ),
        "pytorch_version": (
            UNAVAILABLE
        ),
        "python_version": platform.python_version(),
        "numpy_version": UNAVAILABLE,
        "peak_allocated_vram_bytes": (
            UNAVAILABLE
        ),
        "peak_reserved_vram_bytes": (
            UNAVAILABLE
        ),
    }

    try:
        import numpy

        info["numpy_version"] = numpy.__version__
    except ModuleNotFoundError:
        pass

    try:

        import torch

        info[
            "pytorch_version"
        ] = torch.__version__

        if torch.cuda.is_available():

            idx = (
                torch.cuda.current_device()
            )

            info[
                "device_name"
            ] = (
                torch.cuda
                .get_device_name(
                    idx
                )
            )

            try:

                info[
                    "device_uuid"
                ] = str(
                    torch.cuda
                    .get_device_properties(
                        idx
                    )
                    .uuid
                )

            except Exception:

                info[
                    "device_uuid"
                ] = UNAVAILABLE

            info[
                "cuda_version"
            ] = (
                torch.version.cuda
                or UNAVAILABLE
            )

            info[
                "peak_allocated_vram_bytes"
            ] = (
                torch.cuda
                .max_memory_allocated(
                    idx
                )
            )

            info[
                "peak_reserved_vram_bytes"
            ] = (
                torch.cuda
                .max_memory_reserved(
                    idx
                )
            )

        else:

            info[
                "device_name"
            ] = (
                f"cpu "
                f"("
                f"{platform.processor() or platform.machine()}"
                f")"
            )

    except ModuleNotFoundError:
        pass

    return info


# ============================================================
# Metadata identity validation
# ============================================================


def _validate_run_metadata_identity(
    run_id: str,
    resolved_config_hash: str,
    repo_dir: str | Path,
) -> None:
    """Validate run ID and frozen resolved-config identity."""

    if (
        not run_id
        or not _SAFE_RUN_ID_RE.match(
            run_id
        )
    ):
        raise ValueError(
            f"run_id {run_id!r} is empty or "
            "contains characters unsafe for a "
            "filename/path component; expected "
            "letters, digits, '.', '_', '-' only"
        )

    if not _SHA256_HEX_RE.match(
        resolved_config_hash
    ):
        raise ValueError(
            f"resolved_config_hash "
            f"{resolved_config_hash!r} "
            "is not a 64-character lowercase "
            "hex sha256 digest"
        )

    resolved_path = (
        Path(
            repo_dir
        )
        / "configs"
        / "frozen"
        / "resolved"
        / f"{run_id}.yaml"
    )

    if resolved_path.is_file():

        with resolved_path.open(
            "r",
            encoding="utf-8",
        ) as handle:

            resolved_config = (
                yaml.safe_load(
                    handle
                )
            )

        actual_hash = (
            config_hash(
                resolved_config
            )
        )

        if (
            actual_hash
            != resolved_config_hash
        ):
            raise ValueError(
                f"resolved_config_hash "
                f"{resolved_config_hash!r} "
                f"does not match the hash of "
                f"{resolved_path} "
                f"({actual_hash!r})"
            )


# ============================================================
# Main metadata collector
# ============================================================


def collect_metadata(
    *,
    run_id: str,
    pe_method: str,
    model_seed: int,
    data_seed: int,
    resolved_config_hash: str,
    tokenizer_hash: str | None = None,
    train_manifest_hash: str | None = None,
    validation_manifest_hash: str | None = None,
    test_manifest_hash: str | None = None,
    training_subset_manifest_hash: str | None = None,
    training_cache_hash: str | None = None,
    dataset_source_revision: str | None = None,
    precision: str | None = None,
    tokens_seen: int | None = None,
    checkpoint_hashes: dict[
        str,
        str,
    ]
    | None = None,
    exit_code: int | None = None,
    metrics_path: str | None = None,
    repo_dir: str | Path = ".",
    peak_allocated_vram_bytes: int | None = None,
    peak_reserved_vram_bytes: int | None = None,
) -> dict:
    """Build one experiment's stable metadata dictionary.

    Hash/identifier arguments default to UNAVAILABLE when they are not
    supplied by the responsible project module.
    """

    _validate_run_metadata_identity(
        run_id,
        resolved_config_hash,
        repo_dir,
    )

    meta: dict[
        str,
        Any,
    ] = {}

    meta[
        "run_id"
    ] = run_id

    meta.update(
        timestamps()
    )

    meta.update(
        git_info(
            repo_dir
        )
    )

    meta[
        "resolved_config_hash"
    ] = (
        resolved_config_hash
    )

    meta[
        "pe_method"
    ] = (
        pe_method
    )

    meta[
        "model_seed"
    ] = (
        model_seed
    )

    meta[
        "data_seed"
    ] = (
        data_seed
    )

    meta[
        "tokenizer_hash"
    ] = (
        tokenizer_hash
        or UNAVAILABLE
    )

    meta[
        "train_manifest_hash"
    ] = (
        train_manifest_hash
        or UNAVAILABLE
    )

    meta[
        "validation_manifest_hash"
    ] = (
        validation_manifest_hash
        or UNAVAILABLE
    )

    meta[
        "test_manifest_hash"
    ] = (
        test_manifest_hash
        or UNAVAILABLE
    )

    meta[
        "training_subset_manifest_hash"
    ] = (
        training_subset_manifest_hash
        or UNAVAILABLE
    )
    # sha256 of the .bin token cache scripts/m3_train.py trains against.
    # Only Fidan's training run can compute this; not derivable here.
    meta["training_cache_hash"] = training_cache_hash or UNAVAILABLE

    meta[
        "dataset_source_revision"
    ] = (
        dataset_source_revision
        or UNAVAILABLE
    )

    meta.update(
        device_info()
    )

    # Peak-memory counters are per-process.
    # Prefer measurements explicitly passed by
    # the actual training process.
    if (
        peak_allocated_vram_bytes
        is not None
    ):
        meta[
            "peak_allocated_vram_bytes"
        ] = (
            peak_allocated_vram_bytes
        )

    if (
        peak_reserved_vram_bytes
        is not None
    ):
        meta[
            "peak_reserved_vram_bytes"
        ] = (
            peak_reserved_vram_bytes
        )

    meta[
        "precision"
    ] = (
        precision
        or UNAVAILABLE
    )

    meta[
        "tokens_seen"
    ] = (
        tokens_seen
        if tokens_seen is not None
        else UNAVAILABLE
    )

    meta[
        "checkpoint_hashes"
    ] = (
        checkpoint_hashes
        or {}
    )

    meta[
        "exit_code"
    ] = (
        exit_code
        if exit_code is not None
        else UNAVAILABLE
    )

    meta[
        "metrics_path"
    ] = (
        metrics_path
        or UNAVAILABLE
    )

    return meta


# ============================================================
# Writer
# ============================================================


def write_metadata(
    meta: dict,
    out_path: str | Path,
) -> None:
    """Write metadata and verify that it round-trips as JSON."""

    out_path = Path(
        out_path
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path.write_text(
        json.dumps(
            meta,
            indent=2,
            sort_keys=False,
        )
        + "\n",
        encoding="utf-8",
    )

    # Validate that what was written is valid JSON.
    json.loads(
        out_path.read_text(
            encoding="utf-8"
        )
    )