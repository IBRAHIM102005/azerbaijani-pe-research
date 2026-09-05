#!/usr/bin/env python3
"""Generate the frozen M3 25-run launch manifest."""

from __future__ import annotations

import argparse
import json
import os
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

from src.training.production_guards import (
    validate_headline_plan,
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
        default="bf16",
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

    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Allow non-headline settings for "
            "local/debug experiments."
        ),
    )

    return parser.parse_args()


def load_json_object(
    path: Path,
) -> dict:
    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(
            "Run-plan manifest must contain "
            "one JSON object."
        )

    return payload


def main():
    args = parse_args()

    if args.micro_batch_sequences <= 0:
        raise ValueError(
            "--micro-batch-sequences "
            "must be positive."
        )

    run_root = (
        args.run_root
        if args.run_root.is_absolute()
        else REPO_ROOT
        / args.run_root
    ).resolve()

    output = (
        args.output
        if args.output.is_absolute()
        else REPO_ROOT
        / args.output
    ).resolve()

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

    if not plans:
        raise RuntimeError(
            "Run-plan generation produced "
            "no runs."
        )

    # Build the candidate manifest separately.
    # Production/headline validation must pass
    # before the canonical output path is replaced.
    candidate_output = Path(
        str(output)
        + ".candidate"
    )

    try:
        write_plan_manifest(
            candidate_output,
            plans,
        )

        payload = load_json_object(
            candidate_output
        )

        if not args.debug:
            validate_headline_plan(
                payload
            )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        os.replace(
            candidate_output,
            output,
        )

    finally:
        if candidate_output.exists():
            candidate_output.unlink()

    first = plans[0]

    print()
    print("=" * 72)
    print("M3 RUN PLAN")
    print("=" * 72)

    print(
        f"mode:              "
        f"{'debug' if args.debug else 'headline'}"
    )

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
