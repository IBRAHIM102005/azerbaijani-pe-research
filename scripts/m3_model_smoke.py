#!/usr/bin/env python3
"""M3 smoke test using M2's real PELanguageModel.

This is a plumbing/integration smoke test, NOT a scientific run.

It verifies:

    M2 run config
        -> real PELanguageModel
        -> M3 optimizer
        -> M3 Trainer
        -> deterministic synthetic token stream
        -> forward/backward
        -> optimizer update
        -> checkpoint
        -> fresh model construction
        -> resume
        -> exact token-cursor continuation

The synthetic loss produced here must not be reported as a research result.
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
import tempfile
import time
from pathlib import Path


# ============================================================
# Repository import path
# ============================================================
#
# Running:
#
#     python scripts/m3_model_smoke.py
#
# makes Python treat "scripts/" as the initial import directory.
# Therefore the repository root is added explicitly so imports
# such as:
#
#     from src.models...
#
# work on Windows, Linux, the server, Kaggle, etc.
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )


import numpy as np
import torch

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
)

from src.training.runner import (
    TrainingRunner,
)

from src.training.trainer import (
    Trainer,
)


# ============================================================
# Frozen experiment arms
# ============================================================

PE_TYPES = (
    "learned",
    "sinusoidal",
    "rope",
    "alibi",
    "nope",
)


# ============================================================
# CLI
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test the real M2 language model "
            "with the M3 training stack."
        )
    )

    parser.add_argument(
        "--pe",
        choices=(*PE_TYPES, "all"),
        default="rope",
        help=(
            "Positional-encoding arm to smoke-test, "
            "or 'all' for all five arms."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help=(
            "Preregistered model initialization seed."
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
        "--precision",
        choices=(
            "auto",
            "fp32",
            "bf16",
            "fp16",
        ),
        default="auto",
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=32,
        help=(
            "Short context length for plumbing smoke. "
            "The frozen model max context remains 512."
        ),
    )

    parser.add_argument(
        "--micro-batch-sequences",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--optimizer-cycles",
        type=int,
        default=2,
        help=(
            "Total optimizer updates represented by "
            "the synthetic token stream. Must be >= 2 "
            "so checkpoint + resume are both exercised."
        ),
    )

    return parser.parse_args()


# ============================================================
# Device helpers
# ============================================================


def resolve_device(
    requested: str,
) -> torch.device:
    """Resolve auto/cpu/cuda into a torch.device."""

    if requested == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested, "
                "but CUDA is not available."
            )

    return torch.device(
        requested
    )


def synchronize_if_cuda(
    device: torch.device,
) -> None:
    """Synchronize CUDA so smoke timings are meaningful."""

    if device.type == "cuda":
        torch.cuda.synchronize(
            device
        )


# ============================================================
# Synthetic frozen token cache
# ============================================================


def write_synthetic_cache(
    path: Path,
    *,
    total_tokens: int,
    vocab_size: int,
    seed: int,
) -> None:
    """Create a deterministic stream of valid synthetic token IDs."""

    if total_tokens <= 0:
        raise ValueError(
            "total_tokens must be positive"
        )

    if vocab_size <= 2:
        raise ValueError(
            "vocab_size must be greater than 2"
        )

    if vocab_size > np.iinfo(
        np.uint16
    ).max + 1:
        raise ValueError(
            "vocab_size does not fit in uint16"
        )

    rng = np.random.default_rng(
        seed
    )

    # Keep 0/1 free from random generation because
    # project special tokens may occupy low IDs.
    tokens = rng.integers(
        low=2,
        high=vocab_size,
        size=total_tokens,
        dtype=np.uint16,
    )

    tokens.tofile(
        path
    )


# ============================================================
# Construct real M2 + M3 stack
# ============================================================


def make_runner(
    *,
    pe_type: str,
    seed: int,
    cache_path: Path,
    total_tokens: int,
    seq_len: int,
    micro_batch_sequences: int,
    grad_accum_steps: int,
    device: torch.device,
    precision: str,
):
    """Construct a fresh M2 model and M3 training runner."""

    # --------------------------------------------------------
    # Resolve frozen M2 configuration.
    # --------------------------------------------------------

    resolved = resolve_run_config(
        pe_type,
        seed,
    )

    # --------------------------------------------------------
    # Real M2 model.
    # --------------------------------------------------------

    model = PELanguageModel(
        resolved.config
    )

    # --------------------------------------------------------
    # Frozen M3 optimizer settings.
    # --------------------------------------------------------

    optimizer = build_optimizer(
        model,
        peak_lr=6e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )

    # --------------------------------------------------------
    # Compute optimizer-step count represented by this
    # synthetic smoke stream.
    # --------------------------------------------------------

    micro_batch_tokens = (
        micro_batch_sequences
        * seq_len
    )

    global_batch_tokens = (
        micro_batch_tokens
        * grad_accum_steps
    )

    total_steps = math.ceil(
        total_tokens
        / global_batch_tokens
    )

    # --------------------------------------------------------
    # M3 trainer.
    # --------------------------------------------------------

    trainer = Trainer(
        model,
        optimizer,
        total_steps=total_steps,
        grad_accum_steps=(
            grad_accum_steps
        ),
        grad_clip=1.0,
        peak_lr=6e-4,
        warmup_ratio=0.01,
        min_lr_ratio=0.10,
        device=device,
        precision=precision,
    )

    # --------------------------------------------------------
    # Deterministic token stream.
    # --------------------------------------------------------

    batcher = SequentialTokenBatcher(
        cache_path,
        total_tokens=total_tokens,
        seq_len=seq_len,
        micro_batch_sequences=(
            micro_batch_sequences
        ),
        eod_id=1,
    )

    # --------------------------------------------------------
    # End-to-end M3 runner.
    # --------------------------------------------------------

    runner = TrainingRunner(
        trainer,
        batcher,
    )

    return (
        resolved,
        runner,
    )


# ============================================================
# One PE smoke
# ============================================================


def run_one_arm(
    *,
    pe_type: str,
    seed: int,
    device: torch.device,
    precision: str,
    seq_len: int,
    micro_batch_sequences: int,
    grad_accum_steps: int,
    optimizer_cycles: int,
) -> None:
    """Run checkpoint/resume smoke for one PE arm."""

    print()
    print("=" * 72)
    print(
        f"M3 REAL MODEL SMOKE: "
        f"PE={pe_type} seed={seed}"
    )
    print("=" * 72)

    # Resolve once here as well so the synthetic
    # cache uses the actual frozen vocabulary size.
    preview = resolve_run_config(
        pe_type,
        seed,
    )

    vocab_size = (
        preview.config.vocab_size
    )

    micro_batch_tokens = (
        seq_len
        * micro_batch_sequences
    )

    total_microbatches = (
        grad_accum_steps
        * optimizer_cycles
    )

    total_tokens = (
        micro_batch_tokens
        * total_microbatches
    )

    with tempfile.TemporaryDirectory(
        prefix=(
            f"m3-smoke-{pe_type}-"
        )
    ) as tmp:

        tmp_dir = Path(
            tmp
        )

        cache_path = (
            tmp_dir
            / "tokens.uint16.bin"
        )

        checkpoint_path = (
            tmp_dir
            / "checkpoint.pt"
        )

        # ----------------------------------------------------
        # Synthetic token stream.
        # ----------------------------------------------------

        write_synthetic_cache(
            cache_path,
            total_tokens=total_tokens,
            vocab_size=vocab_size,
            seed=2026,
        )

        # ====================================================
        # FIRST PROCESS
        # ====================================================

        resolved, runner = make_runner(
            pe_type=pe_type,
            seed=seed,
            cache_path=cache_path,
            total_tokens=total_tokens,
            seq_len=seq_len,
            micro_batch_sequences=(
                micro_batch_sequences
            ),
            grad_accum_steps=(
                grad_accum_steps
            ),
            device=device,
            precision=precision,
        )

        model = (
            runner.trainer.model
        )

        print(
            f"run_id:       "
            f"{resolved.run_id}"
        )

        print(
            f"PE:           "
            f"{resolved.pe_type}"
        )

        print(
            f"init seed:    "
            f"{resolved.init_seed}"
        )

        print(
            f"parameters:   "
            f"{model.num_parameters():,}"
        )

        print(
            f"vocab:        "
            f"{resolved.config.vocab_size:,}"
        )

        print(
            f"model max ctx:"
            f" {resolved.config.max_seq_len}"
        )

        print(
            f"smoke seq_len:"
            f" {seq_len}"
        )

        print(
            f"device:       "
            f"{device}"
        )

        print(
            f"precision:    "
            f"{runner.trainer.precision}"
        )

        print(
            f"microbatch:   "
            f"{micro_batch_sequences} sequence(s)"
        )

        print(
            f"GAS:          "
            f"{grad_accum_steps}"
        )

        print(
            f"smoke tokens: "
            f"{total_tokens:,}"
        )

        # ----------------------------------------------------
        # Snapshot a real trainable parameter.
        # ----------------------------------------------------

        before = (
            model.lm_head.weight
            .detach()
            .cpu()
            .clone()
        )

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(
                device
            )

        synchronize_if_cuda(
            device
        )

        start_time = (
            time.perf_counter()
        )

        # Exactly one complete gradient-accumulation cycle.
        first = runner.run(
            max_microbatches=(
                grad_accum_steps
            )
        )

        synchronize_if_cuda(
            device
        )

        elapsed_first = (
            time.perf_counter()
            - start_time
        )

        after = (
            model.lm_head.weight
            .detach()
            .cpu()
            .clone()
        )

        # ----------------------------------------------------
        # First-segment assertions.
        # ----------------------------------------------------

        if torch.equal(
            before,
            after,
        ):
            raise AssertionError(
                "Model parameters did not change "
                "after the first optimizer update."
            )

        if (
            runner.trainer
            .state.optimizer_step
            != 1
        ):
            raise AssertionError(
                "Expected exactly one optimizer "
                "step before checkpoint."
            )

        if not runner.can_checkpoint:
            raise AssertionError(
                "Runner should be checkpointable "
                "after a complete accumulation cycle."
            )

        if first.last_loss is None:
            raise AssertionError(
                "Smoke run produced no loss."
            )

        if not math.isfinite(
            first.last_loss
        ):
            raise AssertionError(
                "Smoke run produced "
                "non-finite loss."
            )

        expected_first_tokens = (
            micro_batch_tokens
            * grad_accum_steps
        )

        if (
            first.end_tokens
            != expected_first_tokens
        ):
            raise AssertionError(
                "Unexpected first-segment "
                "token count: "
                f"{first.end_tokens} "
                f"!= {expected_first_tokens}"
            )

        print()
        print("FIRST SEGMENT")
        print(
            f"loss:         "
            f"{first.last_loss:.6f}"
        )
        print(
            f"tokens_seen:  "
            f"{first.end_tokens:,}"
        )
        print(
            f"opt_steps:    "
            f"{runner.trainer.state.optimizer_step}"
        )
        print(
            f"time:         "
            f"{elapsed_first:.3f}s"
        )

        # ====================================================
        # SAVE CHECKPOINT
        # ====================================================

        runner.save_checkpoint(
            checkpoint_path,
            extra={
                "smoke_test": True,
                "pe_type": pe_type,
                "seed": seed,
                "run_id": (
                    resolved.run_id
                ),
            },
        )

        if not checkpoint_path.is_file():
            raise AssertionError(
                "Checkpoint file "
                "was not created."
            )

        saved_tokens = (
            runner.trainer
            .state.tokens_seen
        )

        saved_optimizer_steps = (
            runner.trainer
            .state.optimizer_step
        )

        checkpoint_size_mb = (
            checkpoint_path
            .stat()
            .st_size
            / (1024 ** 2)
        )

        print()
        print("CHECKPOINT")
        print(
            f"path:         "
            f"{checkpoint_path.name}"
        )
        print(
            f"size:         "
            f"{checkpoint_size_mb:.2f} MiB"
        )
        print(
            f"saved tokens: "
            f"{saved_tokens:,}"
        )

        # ----------------------------------------------------
        # Destroy first runner to simulate process death.
        # ----------------------------------------------------

        del runner
        del model
        del before
        del after

        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

        # ====================================================
        # FRESH PROCESS / RESUME
        # ====================================================

        (
            resolved_again,
            resumed,
        ) = make_runner(
            pe_type=pe_type,
            seed=seed,
            cache_path=cache_path,
            total_tokens=total_tokens,
            seq_len=seq_len,
            micro_batch_sequences=(
                micro_batch_sequences
            ),
            grad_accum_steps=(
                grad_accum_steps
            ),
            device=device,
            precision=precision,
        )

        restored_extra = (
            resumed.load_checkpoint(
                checkpoint_path
            )
        )

        # ----------------------------------------------------
        # Resume assertions before continuing.
        # ----------------------------------------------------

        if (
            resolved_again.run_id
            != resolved.run_id
        ):
            raise AssertionError(
                "Freshly resolved run_id "
                "does not match original run."
            )

        if (
            resumed.trainer
            .state.tokens_seen
            != saved_tokens
        ):
            raise AssertionError(
                "tokens_seen was not "
                "restored correctly."
            )

        if (
            resumed.batcher
            .token_offset
            != saved_tokens
        ):
            raise AssertionError(
                "Batcher cursor was not "
                "restored correctly."
            )

        if (
            resumed.trainer
            .state.optimizer_step
            != saved_optimizer_steps
        ):
            raise AssertionError(
                "optimizer_step was not "
                "restored correctly."
            )

        if (
            restored_extra.get(
                "run_id"
            )
            != resolved.run_id
        ):
            raise AssertionError(
                "Checkpoint run_id mismatch."
            )

        print()
        print("RESUME RESTORED")
        print(
            f"tokens_seen:  "
            f"{resumed.trainer.state.tokens_seen:,}"
        )
        print(
            f"data cursor:  "
            f"{resumed.batcher.token_offset:,}"
        )
        print(
            f"opt_steps:    "
            f"{resumed.trainer.state.optimizer_step}"
        )

        # ====================================================
        # CONTINUE AFTER RESUME
        # ====================================================

        synchronize_if_cuda(
            device
        )

        start_time = (
            time.perf_counter()
        )

        second = resumed.run()

        synchronize_if_cuda(
            device
        )

        elapsed_second = (
            time.perf_counter()
            - start_time
        )

        # ----------------------------------------------------
        # Final assertions.
        # ----------------------------------------------------

        if not second.exhausted:
            raise AssertionError(
                "Resumed run did not consume "
                "the entire synthetic stream."
            )

        if (
            resumed.trainer
            .state.tokens_seen
            != total_tokens
        ):
            raise AssertionError(
                "Final tokens_seen mismatch: "
                f"{resumed.trainer.state.tokens_seen} "
                f"!= {total_tokens}"
            )

        if (
            resumed.batcher
            .token_offset
            != total_tokens
        ):
            raise AssertionError(
                "Final data-cursor mismatch: "
                f"{resumed.batcher.token_offset} "
                f"!= {total_tokens}"
            )

        if (
            resumed.trainer
            .state.optimizer_step
            != optimizer_cycles
        ):
            raise AssertionError(
                "Unexpected final optimizer "
                "step count: "
                f"{resumed.trainer.state.optimizer_step} "
                f"!= {optimizer_cycles}"
            )

        if second.last_loss is None:
            raise AssertionError(
                "Resumed segment produced "
                "no loss."
            )

        if not math.isfinite(
            second.last_loss
        ):
            raise AssertionError(
                "Resumed segment produced "
                "non-finite loss."
            )

        if not resumed.can_checkpoint:
            raise AssertionError(
                "Final runner state should "
                "be checkpointable."
            )

        print()
        print("RESUMED SEGMENT")
        print(
            f"loss:         "
            f"{second.last_loss:.6f}"
        )
        print(
            f"tokens_seen:  "
            f"{second.end_tokens:,}"
        )
        print(
            f"opt_steps:    "
            f"{resumed.trainer.state.optimizer_step}"
        )
        print(
            f"time:         "
            f"{elapsed_second:.3f}s"
        )

        if device.type == "cuda":
            synchronize_if_cuda(
                device
            )

            peak_mb = (
                torch.cuda
                .max_memory_allocated(
                    device
                )
                / (1024 ** 2)
            )

            print(
                f"peak CUDA:    "
                f"{peak_mb:.1f} MiB"
            )

        print()
        print(
            f"PASS: {pe_type}"
        )

        # ----------------------------------------------------
        # Cleanup before next PE.
        # ----------------------------------------------------

        del resumed

        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()


# ============================================================
# Main
# ============================================================


def main():
    args = parse_args()

    device = resolve_device(
        args.device
    )

    # --------------------------------------------------------
    # CLI validation.
    # --------------------------------------------------------

    if (
        args.seq_len <= 1
        or args.seq_len > 512
    ):
        raise ValueError(
            "--seq-len must be "
            "between 2 and 512."
        )

    if (
        args.micro_batch_sequences
        <= 0
    ):
        raise ValueError(
            "--micro-batch-sequences "
            "must be positive."
        )

    if (
        args.grad_accum_steps
        <= 0
    ):
        raise ValueError(
            "--grad-accum-steps "
            "must be positive."
        )

    if (
        args.optimizer_cycles
        < 2
    ):
        raise ValueError(
            "--optimizer-cycles must be >= 2 "
            "so checkpoint and resume are "
            "both tested."
        )

    # --------------------------------------------------------
    # Environment summary.
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("M3 SMOKE ENVIRONMENT")
    print("=" * 72)

    print(
        f"repository:  "
        f"{REPO_ROOT}"
    )

    print(
        f"torch:       "
        f"{torch.__version__}"
    )

    print(
        f"CUDA:        "
        f"{torch.cuda.is_available()}"
    )

    if device.type == "cuda":

        print(
            f"GPU:         "
            f"{torch.cuda.get_device_name(device)}"
        )

        print(
            f"bf16:        "
            f"{torch.cuda.is_bf16_supported()}"
        )

    else:

        print(
            "GPU:         CPU smoke"
        )

    print(
        f"device:      "
        f"{device}"
    )

    print(
        f"precision:   "
        f"{args.precision} (requested)"
    )

    # --------------------------------------------------------
    # PE arms.
    # --------------------------------------------------------

    if args.pe == "all":
        arms = PE_TYPES
    else:
        arms = (
            args.pe,
        )

    for pe_type in arms:

        run_one_arm(
            pe_type=pe_type,
            seed=args.seed,
            device=device,
            precision=args.precision,
            seq_len=args.seq_len,
            micro_batch_sequences=(
                args.micro_batch_sequences
            ),
            grad_accum_steps=(
                args.grad_accum_steps
            ),
            optimizer_cycles=(
                args.optimizer_cycles
            ),
        )

    print()
    print("=" * 72)
    print(
        "ALL REQUESTED M3 MODEL "
        f"SMOKES PASSED ({len(arms)} arm(s))"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()