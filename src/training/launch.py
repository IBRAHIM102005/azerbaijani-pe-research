"""Run planning and manifest utilities for M3."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.models.run_config import (
    iter_runs,
)


DEFAULT_TOTAL_TOKENS = 50_000_000

DEFAULT_GLOBAL_BATCH_TOKENS = 65_536

DEFAULT_SEQ_LEN = 512

DEFAULT_CHECKPOINT_MILESTONES = (
    5_000_000,
    10_000_000,
    20_000_000,
    50_000_000,
)


@dataclass(frozen=True)
class CheckpointMilestone:
    """One nominal research checkpoint and its safe actual boundary."""

    label: str

    nominal_tokens: int

    actual_tokens: int

    is_final: bool

    @property
    def overshoot_tokens(self) -> int:
        return (
            self.actual_tokens
            - self.nominal_tokens
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "nominal_tokens": (
                self.nominal_tokens
            ),
            "actual_tokens": (
                self.actual_tokens
            ),
            "overshoot_tokens": (
                self.overshoot_tokens
            ),
            "is_final": (
                self.is_final
            ),
        }


@dataclass(frozen=True)
class RunPlan:
    """One fully resolved M3 training run."""

    run_id: str

    pe_type: str

    init_seed: int

    data_seed: int

    config_sha256: str

    total_tokens: int

    seq_len: int

    micro_batch_sequences: int

    micro_batch_tokens: int

    grad_accum_steps: int

    global_batch_tokens: int

    precision: str

    run_dir: str

    checkpoints: tuple[
        CheckpointMilestone,
        ...,
    ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": (
                self.run_id
            ),
            "pe_type": (
                self.pe_type
            ),
            "init_seed": (
                self.init_seed
            ),
            "data_seed": (
                self.data_seed
            ),
            "config_sha256": (
                self.config_sha256
            ),
            "total_tokens": (
                self.total_tokens
            ),
            "seq_len": (
                self.seq_len
            ),
            "micro_batch_sequences": (
                self.micro_batch_sequences
            ),
            "micro_batch_tokens": (
                self.micro_batch_tokens
            ),
            "grad_accum_steps": (
                self.grad_accum_steps
            ),
            "global_batch_tokens": (
                self.global_batch_tokens
            ),
            "precision": (
                self.precision
            ),
            "run_dir": (
                self.run_dir
            ),
            "checkpoints": [
                checkpoint.to_dict()
                for checkpoint
                in self.checkpoints
            ],
        }


def milestone_label(
    tokens: int,
) -> str:
    """Convert common million-token budgets into stable labels."""

    if (
        tokens >= 1_000_000
        and tokens % 1_000_000 == 0
    ):
        return (
            f"{tokens // 1_000_000}m"
        )

    return str(
        tokens
    )


def resolve_checkpoint_milestones(
    *,
    total_tokens: int = DEFAULT_TOTAL_TOKENS,
    global_batch_tokens: int = DEFAULT_GLOBAL_BATCH_TOKENS,
    nominal_milestones: Iterable[int] = (
        DEFAULT_CHECKPOINT_MILESTONES
    ),
) -> tuple[
    CheckpointMilestone,
    ...,
]:
    """Map nominal research budgets to safe optimizer boundaries.

    Scientific milestone labels remain:

        5M / 10M / 20M / 50M

    but non-final checkpoints are written only after a complete
    optimizer update.

    The manifest records both nominal and actual consumed tokens.
    """

    if total_tokens <= 0:
        raise ValueError(
            "total_tokens must be positive"
        )

    if global_batch_tokens <= 0:
        raise ValueError(
            "global_batch_tokens must be positive"
        )

    requested = sorted(
        set(
            int(value)
            for value
            in nominal_milestones
        )
    )

    if not requested:
        raise ValueError(
            "At least one checkpoint milestone "
            "is required."
        )

    checkpoints: list[
        CheckpointMilestone
    ] = []

    seen_actual_tokens: set[int] = set()

    for nominal_tokens in requested:

        if nominal_tokens <= 0:
            raise ValueError(
                "Checkpoint milestones "
                "must be positive."
            )

        if nominal_tokens > total_tokens:
            raise ValueError(
                "Checkpoint milestone exceeds "
                f"the total token budget: "
                f"{nominal_tokens:,} > "
                f"{total_tokens:,}"
            )

        # Final training boundary may be a partial
        # global batch, but must remain exactly the
        # frozen total token budget.
        if nominal_tokens == total_tokens:

            actual_tokens = (
                total_tokens
            )

        else:

            completed_steps = math.ceil(
                nominal_tokens
                / global_batch_tokens
            )

            actual_tokens = (
                completed_steps
                * global_batch_tokens
            )

            actual_tokens = min(
                actual_tokens,
                total_tokens,
            )

        if (
            actual_tokens
            in seen_actual_tokens
        ):
            raise ValueError(
                "Two nominal milestones map to "
                "the same optimizer boundary: "
                f"{actual_tokens:,} tokens."
            )

        seen_actual_tokens.add(
            actual_tokens
        )

        checkpoints.append(
            CheckpointMilestone(
                label=milestone_label(
                    nominal_tokens
                ),
                nominal_tokens=(
                    nominal_tokens
                ),
                actual_tokens=(
                    actual_tokens
                ),
                is_final=(
                    actual_tokens
                    == total_tokens
                ),
            )
        )

    return tuple(
        checkpoints
    )


def make_run_plans(
    *,
    micro_batch_sequences: int,
    seq_len: int = DEFAULT_SEQ_LEN,
    global_batch_tokens: int = (
        DEFAULT_GLOBAL_BATCH_TOKENS
    ),
    total_tokens: int = (
        DEFAULT_TOTAL_TOKENS
    ),
    precision: str = "auto",
    run_root: str | Path = (
        "results/runs"
    ),
    config_dir: str | Path | None = None,
) -> list[RunPlan]:
    """Resolve the complete preregistered run matrix."""

    if micro_batch_sequences <= 0:
        raise ValueError(
            "micro_batch_sequences "
            "must be positive"
        )

    if seq_len <= 1:
        raise ValueError(
            "seq_len must be "
            "greater than 1"
        )

    micro_batch_tokens = (
        micro_batch_sequences
        * seq_len
    )

    if (
        global_batch_tokens
        % micro_batch_tokens
        != 0
    ):
        raise ValueError(
            "micro_batch_sequences × seq_len "
            "must divide global_batch_tokens "
            "exactly."
        )

    grad_accum_steps = (
        global_batch_tokens
        // micro_batch_tokens
    )

    checkpoints = (
        resolve_checkpoint_milestones(
            total_tokens=(
                total_tokens
            ),
            global_batch_tokens=(
                global_batch_tokens
            ),
        )
    )

    if config_dir is None:
        resolved_runs = list(
            iter_runs()
        )
    else:
        resolved_runs = list(
            iter_runs(
                config_dir
            )
        )

    if not resolved_runs:
        raise RuntimeError(
            "Run matrix contains no runs."
        )

    plans: list[
        RunPlan
    ] = []

    run_root = Path(
        run_root
    )

    for resolved in resolved_runs:

        run_dir = (
            run_root
            / resolved.run_id
        )

        plans.append(
            RunPlan(
                run_id=(
                    resolved.run_id
                ),
                pe_type=(
                    resolved.pe_type
                ),
                init_seed=(
                    resolved.init_seed
                ),
                data_seed=(
                    resolved.config
                    .data_seed
                ),
                config_sha256=(
                    resolved.config_sha256
                ),
                total_tokens=(
                    total_tokens
                ),
                seq_len=(
                    seq_len
                ),
                micro_batch_sequences=(
                    micro_batch_sequences
                ),
                micro_batch_tokens=(
                    micro_batch_tokens
                ),
                grad_accum_steps=(
                    grad_accum_steps
                ),
                global_batch_tokens=(
                    global_batch_tokens
                ),
                precision=(
                    precision
                ),
                run_dir=str(
                    run_dir
                ),
                checkpoints=(
                    checkpoints
                ),
            )
        )

    return plans


def build_plan_manifest(
    plans: list[RunPlan],
) -> dict[str, Any]:
    """Build a machine-readable experiment launch manifest."""

    if not plans:
        raise ValueError(
            "plans cannot be empty"
        )

    first = plans[0]

    pe_types = sorted(
        {
            plan.pe_type
            for plan
            in plans
        }
    )

    seeds = sorted(
        {
            plan.init_seed
            for plan
            in plans
        }
    )

    data_seeds = {
        plan.data_seed
        for plan
        in plans
    }

    if len(
        data_seeds
    ) != 1:
        raise ValueError(
            "All runs must share "
            "one frozen data seed."
        )

    return {
        "schema_version": 1,
        "num_runs": len(
            plans
        ),
        "pe_types": (
            pe_types
        ),
        "init_seeds": (
            seeds
        ),
        "data_seed": next(
            iter(
                data_seeds
            )
        ),
        "total_tokens_per_run": (
            first.total_tokens
        ),
        "seq_len": (
            first.seq_len
        ),
        "micro_batch_sequences": (
            first.micro_batch_sequences
        ),
        "micro_batch_tokens": (
            first.micro_batch_tokens
        ),
        "grad_accum_steps": (
            first.grad_accum_steps
        ),
        "global_batch_tokens": (
            first.global_batch_tokens
        ),
        "precision": (
            first.precision
        ),
        "checkpoint_policy": (
            "first_complete_optimizer_boundary_"
            "at_or_after_nominal_budget"
        ),
        "runs": [
            plan.to_dict()
            for plan
            in plans
        ],
    }


def write_plan_manifest(
    path: str | Path,
    plans: list[RunPlan],
) -> Path:
    """Write experiment launch manifest atomically."""

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = (
        build_plan_manifest(
            plans
        )
    )

    temp_path = Path(
        str(path) + ".tmp"
    )

    temp_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temp_path.replace(
        path
    )

    return path