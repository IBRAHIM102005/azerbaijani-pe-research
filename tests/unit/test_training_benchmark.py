import pytest

from src.training.benchmark import (
    make_benchmark_candidates,
)


def test_default_candidates_keep_global_batch_exact():
    candidates = (
        make_benchmark_candidates()
    )

    assert candidates

    for candidate in candidates:
        assert (
            candidate.micro_batch_sequences
            * candidate.seq_len
            * candidate.grad_accum_steps
            == 65_536
        )


def test_expected_512_context_candidates():
    candidates = (
        make_benchmark_candidates()
    )

    pairs = [
        (
            candidate.micro_batch_sequences,
            candidate.grad_accum_steps,
        )
        for candidate in candidates
    ]

    assert pairs == [
        (1, 128),
        (2, 64),
        (4, 32),
        (8, 16),
        (16, 8),
        (32, 4),
        (64, 2),
        (128, 1),
    ]


def test_non_divisible_candidate_is_skipped():
    candidates = (
        make_benchmark_candidates(
            global_batch_tokens=1000,
            seq_len=100,
            micro_batch_sequence_candidates=(
                1,
                2,
                3,
                5,
            ),
        )
    )

    pairs = [
        (
            candidate.micro_batch_sequences,
            candidate.grad_accum_steps,
        )
        for candidate in candidates
    ]

    assert pairs == [
        (1, 10),
        (2, 5),
        (5, 2),
    ]


def test_invalid_global_batch_raises():
    with pytest.raises(
        ValueError,
        match="global_batch_tokens",
    ):
        make_benchmark_candidates(
            global_batch_tokens=0
        )