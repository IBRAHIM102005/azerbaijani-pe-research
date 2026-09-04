#!/usr/bin/env python3
"""M3 CUDA training throughput / VRAM benchmark.

This benchmark selects the microbatch/GAS configuration for the
frozen 65,536-token global batch.

It uses the real M2 PELanguageModel and the real M3 Trainer.

Example:

    python scripts/m3_benchmark.py --pe rope --seed 17

The scientific experiment must use a frozen configuration selected
from this measured benchmark rather than an assumed microbatch size.
"""

from __future__ import annotations

import argparse
import csv
import gc
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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


from src.models.run_config import (
    resolve_run_config,
)

from src.models.transformer import (
    PELanguageModel,
)

from src.training.benchmark import (
    BenchmarkCandidate,
    make_benchmark_candidates,
)

from src.training.optimizer import (
    build_optimizer,
)

from src.training.trainer import (
    Trainer,
)


# ============================================================
# CLI
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pe",
        choices=(
            "learned",
            "sinusoidal",
            "rope",
            "alibi",
            "nope",
        ),
        default="rope",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=17,
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
        "--warmup-steps",
        type=int,
        default=1,
        help=(
            "Optimizer steps performed before "
            "measurement for each candidate."
        ),
    )

    parser.add_argument(
        "--measure-steps",
        type=int,
        default=3,
        help=(
            "Measured optimizer steps per candidate."
        ),
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
        "--microbatches",
        type=int,
        nargs="+",
        default=[
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
        ],
        help=(
            "Candidate microbatch sequence counts."
        ),
    )

    parser.add_argument(
        "--max-vram-gb",
        type=float,
        default=12.0,
        help=(
            "Mark candidates above this allocated "
            "VRAM budget as over_budget."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/benchmarks/"
            "m3_training_benchmark.csv"
        ),
    )

    return parser.parse_args()


# ============================================================
# Helpers
# ============================================================


def synchronize():
    torch.cuda.synchronize()


def make_synthetic_batch(
    *,
    batch_sequences: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Create one deterministic CUDA batch."""

    generator = torch.Generator(
        device=device
    )

    generator.manual_seed(
        seed
    )

    return torch.randint(
        low=2,
        high=vocab_size,
        size=(
            batch_sequences,
            seq_len,
        ),
        dtype=torch.long,
        device=device,
        generator=generator,
    )


def cleanup():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def environment_info() -> dict[str, str]:
    return {
        "timestamp_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "hostname": (
            socket.gethostname()
        ),
        "platform": (
            platform.platform()
        ),
        "python_version": (
            platform.python_version()
        ),
        "torch_version": (
            torch.__version__
        ),
        "cuda_version": (
            str(
                torch.version.cuda
            )
        ),
        "gpu_name": (
            torch.cuda.get_device_name(0)
        ),
    }


# ============================================================
# Benchmark one candidate
# ============================================================


def benchmark_candidate(
    candidate: BenchmarkCandidate,
    *,
    pe_type: str,
    seed: int,
    precision: str,
    warmup_steps: int,
    measure_steps: int,
    max_vram_gb: float,
) -> dict:
    """Benchmark one exact microbatch/GAS candidate."""

    device = torch.device(
        "cuda"
    )

    row = {
        **environment_info(),
        "pe_type": pe_type,
        "seed": seed,
        "precision_requested": precision,
        "precision_resolved": "",
        "seq_len": candidate.seq_len,
        "micro_batch_sequences": (
            candidate.micro_batch_sequences
        ),
        "micro_batch_tokens": (
            candidate.micro_batch_tokens
        ),
        "grad_accum_steps": (
            candidate.grad_accum_steps
        ),
        "global_batch_tokens": (
            candidate.global_batch_tokens
        ),
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "elapsed_seconds": "",
        "seconds_per_optimizer_step": "",
        "tokens_per_second": "",
        "peak_allocated_gb": "",
        "peak_reserved_gb": "",
        "status": "pending",
        "error": "",
    }

    try:
        resolved = resolve_run_config(
            pe_type,
            seed,
        )

        model = PELanguageModel(
            resolved.config
        )

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

        trainer = Trainer(
            model,
            optimizer,
            # Benchmark only needs a valid
            # schedule horizon.
            total_steps=(
                warmup_steps
                + measure_steps
                + 10
            ),
            grad_accum_steps=(
                candidate.grad_accum_steps
            ),
            grad_clip=1.0,
            peak_lr=6e-4,
            warmup_ratio=0.01,
            min_lr_ratio=0.10,
            device=device,
            precision=precision,
        )

        row[
            "precision_resolved"
        ] = trainer.precision

        batch = make_synthetic_batch(
            batch_sequences=(
                candidate
                .micro_batch_sequences
            ),
            seq_len=candidate.seq_len,
            vocab_size=(
                resolved.config.vocab_size
            ),
            device=device,
            seed=2026,
        )

        # ----------------------------------------------
        # Warmup optimizer steps
        # ----------------------------------------------

        for _ in range(
            warmup_steps
        ):
            for _ in range(
                candidate.grad_accum_steps
            ):
                trainer.train_microbatch(
                    batch
                )

        synchronize()

        # Reset peak after model init + warmup.
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()

        start_optimizer_step = (
            trainer.state.optimizer_step
        )

        # ----------------------------------------------
        # Measured optimizer steps
        # ----------------------------------------------

        for _ in range(
            measure_steps
        ):
            for _ in range(
                candidate.grad_accum_steps
            ):
                trainer.train_microbatch(
                    batch
                )

        synchronize()

        elapsed = (
            time.perf_counter()
            - start
        )

        measured_optimizer_steps = (
            trainer.state.optimizer_step
            - start_optimizer_step
        )

        if (
            measured_optimizer_steps
            != measure_steps
        ):
            raise RuntimeError(
                "Unexpected measured optimizer "
                "step count."
            )

        measured_tokens = (
            candidate.global_batch_tokens
            * measure_steps
        )

        tokens_per_second = (
            measured_tokens
            / elapsed
        )

        seconds_per_step = (
            elapsed
            / measure_steps
        )

        peak_allocated_gb = (
            torch.cuda
            .max_memory_allocated()
            / (1024 ** 3)
        )

        peak_reserved_gb = (
            torch.cuda
            .max_memory_reserved()
            / (1024 ** 3)
        )

        row[
            "elapsed_seconds"
        ] = f"{elapsed:.6f}"

        row[
            "seconds_per_optimizer_step"
        ] = f"{seconds_per_step:.6f}"

        row[
            "tokens_per_second"
        ] = f"{tokens_per_second:.2f}"

        row[
            "peak_allocated_gb"
        ] = f"{peak_allocated_gb:.4f}"

        row[
            "peak_reserved_gb"
        ] = f"{peak_reserved_gb:.4f}"

        if (
            peak_allocated_gb
            > max_vram_gb
        ):
            row["status"] = (
                "over_budget"
            )
        else:
            row["status"] = "ok"

        del batch
        del trainer
        del optimizer
        del model

        cleanup()

        return row

    except torch.cuda.OutOfMemoryError as exc:
        row["status"] = "oom"

        row["error"] = (
            str(exc)
            .replace("\n", " ")
        )

        cleanup()

        return row

    except Exception as exc:
        row["status"] = "error"

        row["error"] = (
            f"{type(exc).__name__}: "
            f"{exc}"
        ).replace(
            "\n",
            " ",
        )

        cleanup()

        return row


# ============================================================
# CSV
# ============================================================


CSV_FIELDS = [
    "timestamp_utc",
    "hostname",
    "platform",
    "python_version",
    "torch_version",
    "cuda_version",
    "gpu_name",
    "pe_type",
    "seed",
    "precision_requested",
    "precision_resolved",
    "seq_len",
    "micro_batch_sequences",
    "micro_batch_tokens",
    "grad_accum_steps",
    "global_batch_tokens",
    "warmup_steps",
    "measure_steps",
    "elapsed_seconds",
    "seconds_per_optimizer_step",
    "tokens_per_second",
    "peak_allocated_gb",
    "peak_reserved_gb",
    "status",
    "error",
]


def write_csv(
    rows: list[dict],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# Main
# ============================================================


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "M3 benchmark requires CUDA. "
            "Your current PyTorch build reports "
            "torch.cuda.is_available() == False. "
            "Run this script on the A100 server "
            "or another CUDA-enabled environment."
        )

    if args.warmup_steps < 0:
        raise ValueError(
            "--warmup-steps cannot be negative"
        )

    if args.measure_steps <= 0:
        raise ValueError(
            "--measure-steps must be positive"
        )

    if args.max_vram_gb <= 0:
        raise ValueError(
            "--max-vram-gb must be positive"
        )

    candidates = (
        make_benchmark_candidates(
            global_batch_tokens=(
                args.global_batch_tokens
            ),
            seq_len=args.seq_len,
            micro_batch_sequence_candidates=(
                tuple(args.microbatches)
            ),
        )
    )

    print()
    print("=" * 78)
    print("M3 TRAINING BENCHMARK")
    print("=" * 78)

    print(
        "GPU:             ",
        torch.cuda.get_device_name(0),
    )

    print(
        "CUDA version:    ",
        torch.version.cuda,
    )

    print(
        "bf16 supported:  ",
        torch.cuda.is_bf16_supported(),
    )

    print(
        "PE:              ",
        args.pe,
    )

    print(
        "seed:            ",
        args.seed,
    )

    print(
        "sequence length: ",
        args.seq_len,
    )

    print(
        "global batch:    ",
        f"{args.global_batch_tokens:,} tokens",
    )

    print(
        "precision:       ",
        args.precision,
    )

    print(
        "VRAM budget:     ",
        f"{args.max_vram_gb:.1f} GB",
    )

    print()

    rows: list[
        dict
    ] = []

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):
        print(
            f"[{index}/{len(candidates)}] "
            f"microbatch="
            f"{candidate.micro_batch_sequences} "
            f"sequences "
            f"({candidate.micro_batch_tokens:,} tokens), "
            f"GAS={candidate.grad_accum_steps}"
        )

        row = benchmark_candidate(
            candidate,
            pe_type=args.pe,
            seed=args.seed,
            precision=args.precision,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
            max_vram_gb=args.max_vram_gb,
        )

        rows.append(
            row
        )

        status = row[
            "status"
        ]

        if status in {
            "ok",
            "over_budget",
        }:
            print(
                "    "
                f"status={status}, "
                f"{row['tokens_per_second']} tok/s, "
                f"{row['seconds_per_optimizer_step']} s/step, "
                f"{row['peak_allocated_gb']} GB allocated"
            )

        else:
            print(
                "    "
                f"status={status}: "
                f"{row['error']}"
            )

    output_path = (
        args.output
        if args.output.is_absolute()
        else REPO_ROOT
        / args.output
    )

    write_csv(
        rows,
        output_path,
    )

    valid = [
        row
        for row in rows
        if row["status"] == "ok"
    ]

    print()
    print("=" * 78)
    print("BENCHMARK SUMMARY")
    print("=" * 78)

    if not valid:
        print(
            "No candidate satisfied the VRAM "
            "budget successfully."
        )

    else:
        best = max(
            valid,
            key=lambda row: float(
                row["tokens_per_second"]
            ),
        )

        print(
            "Best valid candidate:"
        )

        print(
            "  microbatch sequences:",
            best[
                "micro_batch_sequences"
            ],
        )

        print(
            "  GAS:",
            best[
                "grad_accum_steps"
            ],
        )

        print(
            "  tokens/sec:",
            best[
                "tokens_per_second"
            ],
        )

        print(
            "  seconds/step:",
            best[
                "seconds_per_optimizer_step"
            ],
        )

        print(
            "  peak allocated GB:",
            best[
                "peak_allocated_gb"
            ],
        )

        print(
            "  precision:",
            best[
                "precision_resolved"
            ],
        )

    print()
    print(
        f"CSV written to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()