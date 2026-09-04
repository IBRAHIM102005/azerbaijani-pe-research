#!/usr/bin/env python3
"""Run one frozen M3 training experiment.

This script connects:

    frozen M1 data contract
        -> frozen M3 run plan
        -> real M2 PELanguageModel
        -> frozen uint16 token cache
        -> M3 optimizer / Trainer / TrainingRunner
        -> rolling latest checkpoint
        -> milestone model checkpoints
        -> completion manifest

The production training path reads EOD/tokenizer/data-seed identity
from M1's frozen training_data_contract.json. These values are not
re-stated as training constants here.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ============================================================
# Repository import path
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

from src.models.run_config import (
    resolve_run_config,
)

from src.models.transformer import (
    PELanguageModel,
)

from src.training.batching import (
    SequentialTokenBatcher,
)

from src.training.optimizer import (
    build_optimizer,
    describe_optimizer_groups,
)

from src.training.runner import (
    TrainingRunner,
)

from src.training.trainer import (
    Trainer,
)


# ============================================================
# Generic helpers
# ============================================================


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def atomic_write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Atomically write one JSON artifact."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = Path(
        str(path) + ".tmp"
    )

    temp.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        temp,
        path,
    )


def set_seed(
    seed: int,
) -> None:
    """Seed Python, NumPy and PyTorch."""

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def resolve_device(
    requested: str,
) -> torch.device:
    """Resolve auto/cpu/cuda."""

    if requested == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if (
        requested == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    return torch.device(
        requested
    )


def resolve_path(
    path: Path,
) -> Path:
    """Resolve a repo-relative or absolute path."""

    if path.is_absolute():
        return path.resolve()

    return (
        REPO_ROOT
        / path
    ).resolve()


def load_json(
    path: Path,
) -> dict[str, Any]:
    """Read one JSON object."""

    if not path.is_file():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def find_run_plan(
    payload: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """Select one run from the launch manifest."""

    runs = payload.get(
        "runs"
    )

    if not isinstance(
        runs,
        list,
    ):
        raise ValueError(
            "Run plan is missing a valid "
            "'runs' list."
        )

    for run in runs:
        if run.get(
            "run_id"
        ) == run_id:
            return run

    raise ValueError(
        f"run_id not found in plan: "
        f"{run_id}"
    )


# ============================================================
# Model-only milestone checkpoint
# ============================================================


def save_model_only_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    run_id: str,
    pe_type: str,
    init_seed: int,
    nominal_tokens: int,
    actual_tokens: int,
) -> None:
    """Save model weights for sample-efficiency evaluation.

    Optimizer/RNG/data-cursor state belongs in rolling latest.pt.
    Milestone files intentionally contain model state only.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "model_state_dict": (
            model.state_dict()
        ),
        "run_id": (
            run_id
        ),
        "pe_type": (
            pe_type
        ),
        "init_seed": (
            init_seed
        ),
        "nominal_tokens": (
            nominal_tokens
        ),
        "actual_tokens": (
            actual_tokens
        ),
        "created_at_utc": (
            utc_now()
        ),
    }

    temp = Path(
        str(path) + ".tmp"
    )

    torch.save(
        payload,
        temp,
    )

    os.replace(
        temp,
        path,
    )


# ============================================================
# CLI
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--plan",
        type=Path,
        default=Path(
            "results/manifests/"
            "m3_run_plan.json"
        ),
    )

    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
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
        "--data-contract",
        type=Path,
        default=Path(
            "data/metadata/"
            "training_data_contract.json"
        ),
        help=(
            "Frozen M1 training data contract."
        ),
    )

    parser.add_argument(
        "--device",
        choices=(
            "auto",
            "cpu",
            "cuda",
        ),
        default="auto",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from "
            "run_dir/checkpoints/latest.pt "
            "if it exists."
        ),
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help=(
            "Print every N optimizer steps."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================


def main():
    args = parse_args()

    if args.log_every <= 0:
        raise ValueError(
            "--log-every must be positive."
        )

    plan_path = resolve_path(
        args.plan
    )

    cache_path = resolve_path(
        args.cache
    )

    contract_path = resolve_path(
        args.data_contract
    )

    # ========================================================
    # Load M3 plan
    # ========================================================

    plan_payload = load_json(
        plan_path
    )

    run_plan = find_run_plan(
        plan_payload,
        args.run_id,
    )

    # ========================================================
    # Load frozen M1 data identity
    # ========================================================

    if not contract_path.is_file():
        raise FileNotFoundError(
            "Frozen M1 data contract "
            f"not found: {contract_path}"
        )

    contract = load_contract(
        contract_path
    )

    eod_id = int(
        contract.eod_id
    )

    contract_vocab_size = int(
        contract.vocab_size
    )

    contract_data_seed = int(
        contract.data_seed
    )

    # ========================================================
    # Resolve M2 configuration
    # ========================================================

    pe_type = str(
        run_plan["pe_type"]
    )

    init_seed = int(
        run_plan["init_seed"]
    )

    set_seed(
        init_seed
    )

    resolved = resolve_run_config(
        pe_type,
        init_seed,
    )

    # --------------------------------------------------------
    # Cross-module identity checks
    # --------------------------------------------------------

    if (
        resolved.run_id
        != run_plan["run_id"]
    ):
        raise ValueError(
            "Resolved M2 run_id does not "
            "match M3 run plan."
        )

    plan_config_hash = (
        run_plan.get(
            "config_sha256"
        )
    )

    if (
        plan_config_hash is not None
        and plan_config_hash
        != resolved.config_sha256
    ):
        raise ValueError(
            "Resolved configuration hash "
            "does not match M3 run plan."
        )

    if (
        int(
            run_plan["data_seed"]
        )
        != contract_data_seed
    ):
        raise ValueError(
            "M3 run-plan data seed does not "
            "match frozen M1 data contract: "
            f"{run_plan['data_seed']} "
            f"!= {contract_data_seed}"
        )

    if (
        int(
            resolved.config.data_seed
        )
        != contract_data_seed
    ):
        raise ValueError(
            "M2 configuration data seed does "
            "not match frozen M1 contract."
        )

    if (
        int(
            resolved.config.vocab_size
        )
        != contract_vocab_size
    ):
        raise ValueError(
            "M2 vocabulary size does not "
            "match frozen M1 tokenizer: "
            f"{resolved.config.vocab_size} "
            f"!= {contract_vocab_size}"
        )

    if not (
        0
        <= eod_id
        < contract_vocab_size
    ):
        raise ValueError(
            "Frozen M1 EOD id is outside "
            "the tokenizer vocabulary."
        )

    # Model seed must not change frozen stream ordering.
    if contract.model_seed_affects_order:
        raise ValueError(
            "Frozen M1 contract unexpectedly "
            "states that model seed affects "
            "training-data order."
        )

    # ========================================================
    # Device
    # ========================================================

    device = resolve_device(
        args.device
    )

    # ========================================================
    # Run paths
    # ========================================================

    run_dir = Path(
        run_plan["run_dir"]
    )

    if not run_dir.is_absolute():
        run_dir = (
            REPO_ROOT
            / run_dir
        )

    run_dir = (
        run_dir.resolve()
    )

    checkpoint_dir = (
        run_dir
        / "checkpoints"
    )

    milestone_dir = (
        checkpoint_dir
        / "milestones"
    )

    logs_dir = (
        run_dir
        / "logs"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    milestone_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    latest_checkpoint = (
        checkpoint_dir
        / "latest.pt"
    )

    status_path = (
        run_dir
        / "status.json"
    )

    completion_path = (
        run_dir
        / "completed.json"
    )

    optimizer_audit_path = (
        run_dir
        / "optimizer_groups.json"
    )

    # ========================================================
    # Frozen cache validation
    # ========================================================

    if not cache_path.is_file():
        raise FileNotFoundError(
            "Frozen token cache not found: "
            f"{cache_path}"
        )

    total_tokens = int(
        run_plan["total_tokens"]
    )

    if total_tokens <= 0:
        raise ValueError(
            "total_tokens must be positive."
        )

    # uint16 = exactly 2 bytes per token.
    expected_cache_bytes = (
        total_tokens
        * np.dtype(
            np.uint16
        ).itemsize
    )

    actual_cache_bytes = (
        cache_path.stat().st_size
    )

    if (
        actual_cache_bytes
        != expected_cache_bytes
    ):
        raise ValueError(
            "Frozen token cache byte-size "
            "mismatch: "
            f"expected={expected_cache_bytes:,}, "
            f"actual={actual_cache_bytes:,}"
        )

    # ========================================================
    # Model
    # ========================================================

    model = PELanguageModel(
        resolved.config
    )

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = build_optimizer(
        model,
        peak_lr=6e-4,
        betas=(
            0.9,
            0.95,
        ),
        eps=1e-8,
        weight_decay=0.1,
    )

    optimizer_audit = (
        describe_optimizer_groups(
            model,
            optimizer,
        )
    )

    atomic_write_json(
        optimizer_audit_path,
        {
            "run_id": (
                resolved.run_id
            ),
            "pe_type": (
                pe_type
            ),
            "created_at_utc": (
                utc_now()
            ),
            "groups": (
                optimizer_audit
            ),
        },
    )

    # ========================================================
    # Trainer
    # ========================================================

    global_batch_tokens = int(
        run_plan[
            "global_batch_tokens"
        ]
    )

    if global_batch_tokens <= 0:
        raise ValueError(
            "global_batch_tokens "
            "must be positive."
        )

    # Exact 50M budget can end in a final
    # partial optimizer accumulation cycle.
    total_optimizer_steps = (
        total_tokens
        + global_batch_tokens
        - 1
    ) // global_batch_tokens

    trainer = Trainer(
        model,
        optimizer,
        total_steps=(
            total_optimizer_steps
        ),
        grad_accum_steps=int(
            run_plan[
                "grad_accum_steps"
            ]
        ),
        grad_clip=1.0,
        peak_lr=6e-4,
        warmup_ratio=0.01,
        min_lr_ratio=0.10,
        device=device,
        precision=(
            run_plan[
                "precision"
            ]
        ),
    )

    # ========================================================
    # Frozen token stream
    # ========================================================

    batcher = SequentialTokenBatcher(
        cache_path,
        total_tokens=(
            total_tokens
        ),
        seq_len=int(
            run_plan[
                "seq_len"
            ]
        ),
        micro_batch_sequences=int(
            run_plan[
                "micro_batch_sequences"
            ]
        ),

        # Important:
        # Do not hardcode this value.
        # M1's frozen data contract is
        # the source of truth.
        eod_id=eod_id,
    )

    runner = TrainingRunner(
        trainer,
        batcher,
    )

    # ========================================================
    # Resume
    # ========================================================

    resumed = False

    if (
        args.resume
        and latest_checkpoint.is_file()
    ):
        runner.load_checkpoint(
            latest_checkpoint
        )

        resumed = True

    elif (
        args.resume
        and not latest_checkpoint.is_file()
    ):
        print(
            "Resume requested, but no "
            "latest.pt exists. "
            "Starting from zero."
        )

    # Throughput must count only tokens processed
    # by this process invocation.
    process_start_tokens = int(
        trainer.state.tokens_seen
    )

    # ========================================================
    # Environment / metadata
    # ========================================================

    environment = {
        "torch_version": (
            torch.__version__
        ),
        "cuda_available": (
            torch.cuda.is_available()
        ),
        "cuda_version": (
            str(
                torch.version.cuda
            )
        ),
        "device": (
            str(device)
        ),
        "precision": (
            trainer.precision
        ),
        "gpu_name": (
            torch.cuda.get_device_name(
                device
            )
            if device.type == "cuda"
            else None
        ),
    }

    data_identity = {
        "contract_path": (
            str(
                contract_path
            )
        ),
        "contract_target_tokens": (
            contract.target_tokens
        ),
        "contract_selected_tokens": (
            contract.selected_tokens
        ),
        "data_seed": (
            contract_data_seed
        ),
        "vocab_size": (
            contract_vocab_size
        ),
        "eod_id": (
            eod_id
        ),
        "model_seed_affects_order": (
            contract.model_seed_affects_order
        ),
    }

    atomic_write_json(
        status_path,
        {
            "status": (
                "running"
            ),
            "started_at_utc": (
                utc_now()
            ),
            "resumed": (
                resumed
            ),
            "run": (
                run_plan
            ),
            "environment": (
                environment
            ),
            "data_identity": (
                data_identity
            ),
            "tokens_seen": (
                trainer.state.tokens_seen
            ),
            "optimizer_step": (
                trainer.state.optimizer_step
            ),
        },
    )

    # ========================================================
    # Console summary
    # ========================================================

    print()
    print("=" * 78)
    print("M3 TRAINING RUN")
    print("=" * 78)

    print(
        f"run_id:       "
        f"{resolved.run_id}"
    )

    print(
        f"PE:           "
        f"{pe_type}"
    )

    print(
        f"seed:         "
        f"{init_seed}"
    )

    print(
        f"data seed:    "
        f"{contract_data_seed}"
    )

    print(
        f"EOD id:       "
        f"{eod_id}"
    )

    print(
        f"parameters:   "
        f"{model.num_parameters():,}"
    )

    print(
        f"device:       "
        f"{device}"
    )

    print(
        f"precision:    "
        f"{trainer.precision}"
    )

    print(
        f"microbatch:   "
        f"{run_plan['micro_batch_sequences']}"
    )

    print(
        f"GAS:          "
        f"{run_plan['grad_accum_steps']}"
    )

    print(
        f"global batch: "
        f"{global_batch_tokens:,}"
    )

    print(
        f"start tokens: "
        f"{trainer.state.tokens_seen:,}"
    )

    print(
        f"target:       "
        f"{total_tokens:,}"
    )

    # ========================================================
    # Determine remaining milestones
    # ========================================================

    milestones = list(
        run_plan[
            "checkpoints"
        ]
    )

    milestones.sort(
        key=lambda item: int(
            item[
                "actual_tokens"
            ]
        )
    )

    if not milestones:
        raise ValueError(
            "Run plan contains no "
            "checkpoint milestones."
        )

    milestone_index = 0

    while (
        milestone_index
        < len(milestones)
        and int(
            milestones[
                milestone_index
            ]["actual_tokens"]
        )
        <= trainer.state.tokens_seen
    ):
        milestone_index += 1

    # ========================================================
    # Training
    # ========================================================

    started = (
        time.perf_counter()
    )

    last_printed_step = (
        trainer.state.optimizer_step
    )

    def on_step(
        result,
    ):
        nonlocal last_printed_step

        if not (
            result.did_optimizer_step
        ):
            return

        if (
            result.optimizer_step
            - last_printed_step
            < args.log_every
        ):
            return

        last_printed_step = (
            result.optimizer_step
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        process_tokens = (
            result.tokens_seen
            - process_start_tokens
        )

        tokens_per_second = (
            process_tokens
            / max(
                elapsed,
                1e-9,
            )
        )

        print(
            f"step="
            f"{result.optimizer_step:<5} "
            f"tokens="
            f"{result.tokens_seen:>10,} "
            f"loss="
            f"{result.loss:.5f} "
            f"lr="
            f"{result.lr:.8f} "
            f"tok/s≈"
            f"{tokens_per_second:,.0f}"
        )

    # ========================================================
    # Run milestone by milestone
    # ========================================================

    while not batcher.exhausted:

        if (
            milestone_index
            < len(
                milestones
            )
        ):
            target = int(
                milestones[
                    milestone_index
                ][
                    "actual_tokens"
                ]
            )

        else:
            target = (
                total_tokens
            )

        remaining = (
            target
            - trainer.state.tokens_seen
        )

        if remaining <= 0:
            milestone_index += 1
            continue

        micro_batch_capacity = int(
            run_plan[
                "micro_batch_tokens"
            ]
        )

        max_microbatches = (
            remaining
            // micro_batch_capacity
        )

        if (
            remaining
            % micro_batch_capacity
            != 0
        ):
            # Expected for the exact final
            # 50M tail only.
            max_microbatches += 1

        runner.run(
            max_microbatches=(
                max_microbatches
            ),
            on_step=on_step,
        )

        # ====================================================
        # Exact stream end
        # ====================================================

        if batcher.exhausted:

            final_milestone = (
                milestones[-1]
            )

            final_path = (
                milestone_dir
                / (
                    f"{final_milestone['label']}"
                    "_model.pt"
                )
            )

            if not final_path.is_file():
                save_model_only_checkpoint(
                    final_path,
                    model=model,
                    run_id=(
                        resolved.run_id
                    ),
                    pe_type=(
                        pe_type
                    ),
                    init_seed=(
                        init_seed
                    ),
                    nominal_tokens=int(
                        final_milestone[
                            "nominal_tokens"
                        ]
                    ),
                    actual_tokens=(
                        trainer.state.tokens_seen
                    ),
                )

            if not runner.can_checkpoint:
                raise RuntimeError(
                    "Final state is not "
                    "checkpointable."
                )

            runner.save_checkpoint(
                latest_checkpoint,
                extra={
                    "run_id": (
                        resolved.run_id
                    ),
                    "reason": (
                        "final"
                    ),
                    "data_seed": (
                        contract_data_seed
                    ),
                    "eod_id": (
                        eod_id
                    ),
                },
            )

            break

        # ====================================================
        # Non-final milestone
        # ====================================================

        if not runner.can_checkpoint:
            raise RuntimeError(
                "Milestone reached but runner "
                "is not checkpointable."
            )

        milestone = (
            milestones[
                milestone_index
            ]
        )

        actual_expected = int(
            milestone[
                "actual_tokens"
            ]
        )

        if (
            trainer.state.tokens_seen
            != actual_expected
        ):
            raise RuntimeError(
                "Milestone token mismatch: "
                f"expected="
                f"{actual_expected:,}, "
                f"actual="
                f"{trainer.state.tokens_seen:,}"
            )

        milestone_path = (
            milestone_dir
            / (
                f"{milestone['label']}"
                "_model.pt"
            )
        )

        save_model_only_checkpoint(
            milestone_path,
            model=model,
            run_id=(
                resolved.run_id
            ),
            pe_type=(
                pe_type
            ),
            init_seed=(
                init_seed
            ),
            nominal_tokens=int(
                milestone[
                    "nominal_tokens"
                ]
            ),
            actual_tokens=(
                trainer.state.tokens_seen
            ),
        )

        # Rolling full state:
        # model + optimizer + RNG + trainer + cursor.
        runner.save_checkpoint(
            latest_checkpoint,
            extra={
                "run_id": (
                    resolved.run_id
                ),
                "reason": (
                    f"milestone_"
                    f"{milestone['label']}"
                ),
                "nominal_tokens": int(
                    milestone[
                        "nominal_tokens"
                    ]
                ),
                "actual_tokens": int(
                    milestone[
                        "actual_tokens"
                    ]
                ),
                "data_seed": (
                    contract_data_seed
                ),
                "eod_id": (
                    eod_id
                ),
            },
        )

        print()
        print(
            f"checkpoint "
            f"{milestone['label']} "
            f"saved at "
            f"{trainer.state.tokens_seen:,} "
            f"tokens"
        )
        print()

        atomic_write_json(
            status_path,
            {
                "status": (
                    "running"
                ),
                "updated_at_utc": (
                    utc_now()
                ),
                "resumed": (
                    resumed
                ),
                "run": (
                    run_plan
                ),
                "environment": (
                    environment
                ),
                "data_identity": (
                    data_identity
                ),
                "tokens_seen": (
                    trainer.state.tokens_seen
                ),
                "optimizer_step": (
                    trainer.state.optimizer_step
                ),
                "last_checkpoint": (
                    milestone[
                        "label"
                    ]
                ),
            },
        )

        milestone_index += 1

    # ========================================================
    # Completion validation
    # ========================================================

    if (
        trainer.state.tokens_seen
        != total_tokens
    ):
        raise RuntimeError(
            "Training ended with incorrect "
            "token count: "
            f"{trainer.state.tokens_seen:,} "
            f"!= {total_tokens:,}"
        )

    if not batcher.exhausted:
        raise RuntimeError(
            "Training finished but token "
            "stream is not exhausted."
        )

    if not runner.can_checkpoint:
        raise RuntimeError(
            "Final training state is not "
            "checkpointable."
        )

    elapsed = (
        time.perf_counter()
        - started
    )

    process_tokens = (
        trainer.state.tokens_seen
        - process_start_tokens
    )

    average_tokens_per_second = (
        process_tokens
        / max(
            elapsed,
            1e-9,
        )
    )

    completion = {
        "status": (
            "completed"
        ),
        "completed_at_utc": (
            utc_now()
        ),
        "run_id": (
            resolved.run_id
        ),
        "pe_type": (
            pe_type
        ),
        "init_seed": (
            init_seed
        ),
        "data_seed": (
            contract_data_seed
        ),
        "eod_id": (
            eod_id
        ),
        "tokens_seen": (
            trainer.state.tokens_seen
        ),
        "optimizer_steps": (
            trainer.state.optimizer_step
        ),
        "process_start_tokens": (
            process_start_tokens
        ),
        "tokens_processed_this_process": (
            process_tokens
        ),
        "elapsed_seconds_this_process": (
            elapsed
        ),
        "average_tokens_per_second_this_process": (
            average_tokens_per_second
        ),
        "environment": (
            environment
        ),
        "data_identity": (
            data_identity
        ),
        "latest_checkpoint": (
            str(
                latest_checkpoint
            )
        ),
        "milestone_directory": (
            str(
                milestone_dir
            )
        ),
    }

    atomic_write_json(
        completion_path,
        completion,
    )

    atomic_write_json(
        status_path,
        completion,
    )

    # ========================================================
    # Final console output
    # ========================================================

    print()
    print("=" * 78)
    print("RUN COMPLETE")
    print("=" * 78)

    print(
        f"tokens_seen:     "
        f"{trainer.state.tokens_seen:,}"
    )

    print(
        f"optimizer_steps: "
        f"{trainer.state.optimizer_step:,}"
    )

    print(
        f"process tokens:  "
        f"{process_tokens:,}"
    )

    print(
        f"avg tok/s:       "
        f"{average_tokens_per_second:,.0f}"
    )

    print(
        f"elapsed:         "
        f"{elapsed / 60:.2f} min"
    )

    print(
        f"run_dir:         "
        f"{run_dir}"
    )


if __name__ == "__main__":
    main()