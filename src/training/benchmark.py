"""Benchmark-planning utilities for M3."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkCandidate:
    """One microbatch/GAS configuration."""

    micro_batch_sequences: int
    seq_len: int
    grad_accum_steps: int
    global_batch_tokens: int

    @property
    def micro_batch_tokens(self) -> int:
        return (
            self.micro_batch_sequences
            * self.seq_len
        )


def make_benchmark_candidates(
    *,
    global_batch_tokens: int = 65_536,
    seq_len: int = 512,
    micro_batch_sequence_candidates: tuple[int, ...] = (
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
    ),
) -> list[BenchmarkCandidate]:
    """Create exact global-batch benchmark candidates.

    Every candidate must satisfy:

        micro_batch_sequences
        × seq_len
        × grad_accum_steps
        == global_batch_tokens
    """

    if global_batch_tokens <= 0:
        raise ValueError(
            "global_batch_tokens must be positive"
        )

    if seq_len <= 1:
        raise ValueError(
            "seq_len must be greater than 1"
        )

    candidates: list[
        BenchmarkCandidate
    ] = []

    for micro_batch_sequences in (
        micro_batch_sequence_candidates
    ):
        if micro_batch_sequences <= 0:
            raise ValueError(
                "microbatch candidates "
                "must be positive"
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
            continue

        grad_accum_steps = (
            global_batch_tokens
            // micro_batch_tokens
        )

        candidates.append(
            BenchmarkCandidate(
                micro_batch_sequences=(
                    micro_batch_sequences
                ),
                seq_len=seq_len,
                grad_accum_steps=(
                    grad_accum_steps
                ),
                global_batch_tokens=(
                    global_batch_tokens
                ),
            )
        )

    if not candidates:
        raise ValueError(
            "No exact benchmark candidates "
            "could be constructed."
        )

    return candidates