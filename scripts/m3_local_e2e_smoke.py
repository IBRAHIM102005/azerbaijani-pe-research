#!/usr/bin/env python3
"""Local end-to-end smoke test for scripts/m3_train.py.

This test intentionally uses:

    - real M2 PELanguageModel
    - real M3 Trainer
    - real SequentialTokenBatcher
    - real m3_train.py entrypoint
    - synthetic uint16 token cache
    - tiny token budget
    - CPU

It verifies that the production training entrypoint can:

    model
      -> train
      -> milestone checkpoint
      -> rolling latest checkpoint
      -> final checkpoint
      -> completed.json

This is NOT a scientific experiment.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


# ============================================================
# Repository path
# ============================================================

REPO_ROOT = Path(
    __file__
).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )


from src.models.run_config import (
    resolve_run_config,
)


TRAIN_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "m3_train.py"
)


# ============================================================
# Constants
# ============================================================

PE_TYPE = "rope"

INIT_SEED = 17

TOTAL_TOKENS = 256

SEQ_LEN = 32

MICRO_BATCH_SEQUENCES = 1

MICRO_BATCH_TOKENS = (
    SEQ_LEN
    * MICRO_BATCH_SEQUENCES
)

GRAD_ACCUM_STEPS = 2

GLOBAL_BATCH_TOKENS = (
    MICRO_BATCH_TOKENS
    * GRAD_ACCUM_STEPS
)


# ============================================================
# Helpers
# ============================================================


def write_json(
    path: Path,
    payload: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def create_synthetic_cache(
    path: Path,
    *,
    vocab_size: int,
) -> None:
    """Create deterministic valid uint16 token IDs."""

    rng = np.random.default_rng(
        2026
    )

    tokens = rng.integers(
        low=2,
        high=vocab_size,
        size=TOTAL_TOKENS,
        dtype=np.uint16,
    )

    tokens.tofile(
        path
    )

    expected_bytes = (
        TOTAL_TOKENS
        * 2
    )

    actual_bytes = (
        path.stat().st_size
    )

    if (
        actual_bytes
        != expected_bytes
    ):
        raise AssertionError(
            "Synthetic cache size mismatch: "
            f"{actual_bytes} "
            f"!= {expected_bytes}"
        )


def create_synthetic_cache_metadata(
    cache_path: Path,
    metadata_path: Path,
) -> str:
    """Create metadata required by the production cache guard."""

    cache_bytes = (
        cache_path.read_bytes()
    )

    cache_sha256 = hashlib.sha256(
        cache_bytes
    ).hexdigest()

    write_json(
        metadata_path,
        {
            "dtype": "uint16",
            "target_tokens": TOTAL_TOKENS,
            "bytes": len(
                cache_bytes
            ),
            "cache_sha256": (
                cache_sha256
            ),
        },
    )

    return cache_sha256


# ============================================================
# Main
# ============================================================


def main():
    if not TRAIN_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Training script missing: "
            f"{TRAIN_SCRIPT}"
        )

    resolved = resolve_run_config(
        PE_TYPE,
        INIT_SEED,
    )

    print()
    print("=" * 72)
    print("M3 LOCAL END-TO-END SMOKE")
    print("=" * 72)

    print(
        f"run_id:       "
        f"{resolved.run_id}"
    )

    print(
        f"PE:           "
        f"{PE_TYPE}"
    )

    print(
        f"seed:         "
        f"{INIT_SEED}"
    )

    print(
        f"tokens:       "
        f"{TOTAL_TOKENS}"
    )

    print(
        f"seq_len:      "
        f"{SEQ_LEN}"
    )

    print(
        f"microbatch:   "
        f"{MICRO_BATCH_SEQUENCES}"
    )

    print(
        f"GAS:          "
        f"{GRAD_ACCUM_STEPS}"
    )

    print(
        f"global batch: "
        f"{GLOBAL_BATCH_TOKENS}"
    )

    # 256 / 64 = exactly 4 optimizer updates.
    expected_optimizer_steps = (
        TOTAL_TOKENS
        // GLOBAL_BATCH_TOKENS
    )

    if expected_optimizer_steps != 4:
        raise AssertionError(
            "Unexpected local smoke "
            "optimizer-step count."
        )

    with tempfile.TemporaryDirectory(
        prefix="m3-local-e2e-"
    ) as tmp:

        tmp_dir = Path(
            tmp
        )

        cache_path = (
            tmp_dir
            / "tokens.uint16.bin"
        )

        cache_metadata_path = (
            tmp_dir
            / "tokens.uint16.json"
        )

        plan_path = (
            tmp_dir
            / "plan.json"
        )

        run_dir = (
            tmp_dir
            / "run"
        )

        # ----------------------------------------------------
        # Synthetic token cache
        # ----------------------------------------------------

        create_synthetic_cache(
            cache_path,
            vocab_size=(
                resolved.config.vocab_size
            ),
        )

        cache_sha256 = (
            create_synthetic_cache_metadata(
                cache_path,
                cache_metadata_path,
            )
        )

        # ----------------------------------------------------
        # Tiny run plan
        #
        # Milestones are intentionally exact optimizer
        # boundaries:
        #
        #   128 tokens = 2 updates
        #   256 tokens = 4 updates
        # ----------------------------------------------------

        plan = {
            "schema_version": 1,
            "num_runs": 1,
            "pe_types": [
                PE_TYPE,
            ],
            "init_seeds": [
                INIT_SEED,
            ],
            "data_seed": (
                resolved.config.data_seed
            ),
            "total_tokens_per_run": (
                TOTAL_TOKENS
            ),
            "seq_len": (
                SEQ_LEN
            ),
            "micro_batch_sequences": (
                MICRO_BATCH_SEQUENCES
            ),
            "micro_batch_tokens": (
                MICRO_BATCH_TOKENS
            ),
            "grad_accum_steps": (
                GRAD_ACCUM_STEPS
            ),
            "global_batch_tokens": (
                GLOBAL_BATCH_TOKENS
            ),
            "precision": "auto",
            "runs": [
                {
                    "run_id": (
                        resolved.run_id
                    ),
                    "pe_type": (
                        PE_TYPE
                    ),
                    "init_seed": (
                        INIT_SEED
                    ),
                    "data_seed": (
                        resolved.config.data_seed
                    ),
                    "config_sha256": (
                        resolved.config_sha256
                    ),
                    "total_tokens": (
                        TOTAL_TOKENS
                    ),
                    "seq_len": (
                        SEQ_LEN
                    ),
                    "micro_batch_sequences": (
                        MICRO_BATCH_SEQUENCES
                    ),
                    "micro_batch_tokens": (
                        MICRO_BATCH_TOKENS
                    ),
                    "grad_accum_steps": (
                        GRAD_ACCUM_STEPS
                    ),
                    "global_batch_tokens": (
                        GLOBAL_BATCH_TOKENS
                    ),
                    "precision": "auto",
                    "run_dir": str(
                        run_dir
                    ),
                    "checkpoints": [
                        {
                            "label": "128t",
                            "nominal_tokens": 128,
                            "actual_tokens": 128,
                            "overshoot_tokens": 0,
                            "is_final": False,
                        },
                        {
                            "label": "256t",
                            "nominal_tokens": 256,
                            "actual_tokens": 256,
                            "overshoot_tokens": 0,
                            "is_final": True,
                        },
                    ],
                }
            ],
        }

        write_json(
            plan_path,
            plan,
        )

        # ----------------------------------------------------
        # Production m3_train.py command
        # ----------------------------------------------------

        command = [
            sys.executable,
            str(
                TRAIN_SCRIPT
            ),
            "--plan",
            str(
                plan_path
            ),
            "--run-id",
            resolved.run_id,
            "--cache",
            str(
                cache_path
            ),
            "--cache-metadata",
            str(
                cache_metadata_path
            ),
            "--device",
            "cpu",
            "--log-every",
            "1",
        ]

        print()
        print("COMMAND")
        print(
            " ".join(
                command
            )
        )

        print()
        print("=" * 72)
        print("TRAINING OUTPUT")
        print("=" * 72)
        print()

        result = subprocess.run(
            command,
            cwd=str(
                REPO_ROOT
            ),
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "m3_train.py failed during "
                "local end-to-end smoke. "
                f"exit={result.returncode}"
            )

        # ====================================================
        # Validate artifacts
        # ====================================================

        checkpoint_dir = (
            run_dir
            / "checkpoints"
        )

        milestone_dir = (
            checkpoint_dir
            / "milestones"
        )

        latest_checkpoint = (
            checkpoint_dir
            / "latest.pt"
        )

        milestone_128 = (
            milestone_dir
            / "128t_model.pt"
        )

        milestone_256 = (
            milestone_dir
            / "256t_model.pt"
        )

        completed_path = (
            run_dir
            / "completed.json"
        )

        status_path = (
            run_dir
            / "status.json"
        )

        optimizer_groups_path = (
            run_dir
            / "optimizer_groups.json"
        )

        metadata_path = (
            run_dir
            / "metadata.json"
        )

        required_files = [
            cache_metadata_path,
            metadata_path,
            latest_checkpoint,
            milestone_128,
            milestone_256,
            completed_path,
            status_path,
            optimizer_groups_path,
        ]

        for path in required_files:
            if not path.is_file():
                raise AssertionError(
                    "Expected artifact missing: "
                    f"{path}"
                )

        # ----------------------------------------------------
        # completed.json validation
        # ----------------------------------------------------

        completed = json.loads(
            completed_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            completed["status"]
            != "completed"
        ):
            raise AssertionError(
                "completed.json status "
                "is not 'completed'."
            )

        if (
            completed["tokens_seen"]
            != TOTAL_TOKENS
        ):
            raise AssertionError(
                "Final tokens_seen mismatch: "
                f"{completed['tokens_seen']} "
                f"!= {TOTAL_TOKENS}"
            )

        if (
            completed["optimizer_steps"]
            != expected_optimizer_steps
        ):
            raise AssertionError(
                "Final optimizer-step mismatch: "
                f"{completed['optimizer_steps']} "
                f"!= "
                f"{expected_optimizer_steps}"
            )

        completed_data_identity = (
            completed.get(
                "data_identity",
                {}
            )
        )

        if (
            completed_data_identity.get(
                "cache_sha256"
            )
            != cache_sha256
        ):
            raise AssertionError(
                "completed.json cache SHA-256 "
                "does not match the synthetic "
                "cache metadata."
            )

        # ----------------------------------------------------
        # metadata.json provenance validation
        # ----------------------------------------------------

        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            metadata["run_id"]
            != resolved.run_id
        ):
            raise AssertionError(
                "metadata.json run_id mismatch."
            )

        if (
            metadata["training_cache_hash"]
            != cache_sha256
        ):
            raise AssertionError(
                "metadata.json cache SHA-256 mismatch."
            )

        if (
            metadata["tokens_seen"]
            != TOTAL_TOKENS
        ):
            raise AssertionError(
                "metadata.json tokens_seen mismatch."
            )

        if metadata["exit_code"] != 0:
            raise AssertionError(
                "metadata.json exit_code is not 0."
            )

        checkpoint_hashes = (
            metadata.get(
                "checkpoint_hashes",
                {}
            )
        )

        for checkpoint_key in (
            "latest",
            "128t_model",
            "256t_model",
        ):
            if checkpoint_key not in checkpoint_hashes:
                raise AssertionError(
                    "metadata.json missing checkpoint hash: "
                    f"{checkpoint_key}"
                )

        # ----------------------------------------------------
        # status.json validation
        # ----------------------------------------------------

        status = json.loads(
            status_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            status["status"]
            != "completed"
        ):
            raise AssertionError(
                "status.json did not reach "
                "'completed'."
            )

        # ----------------------------------------------------
        # Optimizer audit exists and has groups
        # ----------------------------------------------------

        optimizer_audit = json.loads(
            optimizer_groups_path.read_text(
                encoding="utf-8"
            )
        )

        if not optimizer_audit.get(
            "groups"
        ):
            raise AssertionError(
                "Optimizer audit has no groups."
            )

        print()
        print("=" * 72)
        print("ARTIFACT VALIDATION")
        print("=" * 72)

        print(
            "cache metadata:        OK"
        )

        print(
            "cache SHA-256:         OK"
        )

        print(
            "latest.pt:             OK"
        )

        print(
            "128t_model.pt:         OK"
        )

        print(
            "256t_model.pt:         OK"
        )

        print(
            "optimizer_groups.json: OK"
        )

        print(
            "metadata.json:         OK"
        )

        print(
            "checkpoint hashes:     OK"
        )

        print(
            "status.json:           OK"
        )

        print(
            "completed.json:        OK"
        )

        print(
            f"tokens_seen:           "
            f"{completed['tokens_seen']}"
        )

        print(
            f"optimizer_steps:       "
            f"{completed['optimizer_steps']}"
        )

        print()
        print("=" * 72)
        print(
            "PASS: M3 LOCAL END-TO-END "
            "TRAINING SMOKE"
        )
        print("=" * 72)


if __name__ == "__main__":
    main()