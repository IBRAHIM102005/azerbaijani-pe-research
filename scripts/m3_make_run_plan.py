#!/usr/bin/env python3
"""Generate the frozen M3 25-run launch manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(
    __file__
).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )


from src.training.launch import (
    make_run_plans,
    write_plan_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--micro-batch-sequences",
        type=int,
        required=True,
        help=(
            "Measured microbatch sequence count "
            "selected from the A100 benchmark."
        ),
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--global-batch-tokens",
        type=int,
        default=65_536,
    )

    parser.add_argument(
        "--total-tokens",
        type=int,
        default=50_000_000,
    )

    parser.add_argument(
        "--precision",
        choices=(
            "auto",
            "bf16",
            "fp16",
            "fp32",
        ),
        default="auto",
    )

    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "results/runs"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/manifests/"
            "m3_run_plan.json"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_root = (
        args.run_root
        if args.run_root.is_absolute()
        else REPO_ROOT
        / args.run_root
    )

    output = (
        args.output
        if args.output.is_absolute()
        else REPO_ROOT
        / args.output
    )

    plans = make_run_plans(
        micro_batch_sequences=(
            args.micro_batch_sequences
        ),
        seq_len=args.seq_len,
        global_batch_tokens=(
            args.global_batch_tokens
        ),
        total_tokens=(
            args.total_tokens
        ),
        precision=(
            args.precision
        ),
        run_root=(
            run_root
        ),
    )

    write_plan_manifest(
        output,
        plans,
    )

    first = plans[0]

    print()
    print("=" * 72)
    print("M3 RUN PLAN")
    print("=" * 72)

    print(
        f"runs:              "
        f"{len(plans)}"
    )

    print(
        f"sequence length:   "
        f"{first.seq_len}"
    )

    print(
        f"microbatch seqs:   "
        f"{first.micro_batch_sequences}"
    )

    print(
        f"microbatch tokens: "
        f"{first.micro_batch_tokens:,}"
    )

    print(
        f"GAS:               "
        f"{first.grad_accum_steps}"
    )

    print(
        f"global batch:      "
        f"{first.global_batch_tokens:,}"
    )

    print(
        f"tokens/run:        "
        f"{first.total_tokens:,}"
    )

    print(
        f"precision:         "
        f"{first.precision}"
    )

    print()
    print("CHECKPOINT BOUNDARIES")

    for checkpoint in (
        first.checkpoints
    ):
        print(
            f"  {checkpoint.label:>4} "
            f"nominal="
            f"{checkpoint.nominal_tokens:,} "
            f"actual="
            f"{checkpoint.actual_tokens:,} "
            f"overshoot="
            f"{checkpoint.overshoot_tokens:,}"
        )

    print()
    print(
        f"manifest: "
        f"{output}"
    )

    print()
    print("RUN MATRIX")

    for index, plan in enumerate(
        plans,
        start=1,
    ):
        print(
            f"{index:02d}. "
            f"{plan.pe_type:<10} "
            f"seed={plan.init_seed:<4} "
            f"{plan.run_id}"
        )


if __name__ == "__main__":
    main()