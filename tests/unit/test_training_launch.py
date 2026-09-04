import json

import pytest

from src.training.launch import (
    make_run_plans,
    resolve_checkpoint_milestones,
    write_plan_manifest,
)


def test_default_checkpoint_boundaries():
    checkpoints = (
        resolve_checkpoint_milestones()
    )

    assert [
        checkpoint.label
        for checkpoint
        in checkpoints
    ] == [
        "5m",
        "10m",
        "20m",
        "50m",
    ]

    assert [
        checkpoint.nominal_tokens
        for checkpoint
        in checkpoints
    ] == [
        5_000_000,
        10_000_000,
        20_000_000,
        50_000_000,
    ]

    assert [
        checkpoint.actual_tokens
        for checkpoint
        in checkpoints
    ] == [
        5_046_272,
        10_027_008,
        20_054_016,
        50_000_000,
    ]

    assert [
        checkpoint.overshoot_tokens
        for checkpoint
        in checkpoints
    ] == [
        46_272,
        27_008,
        54_016,
        0,
    ]


def test_run_plan_matches_current_25_run_matrix():
    plans = make_run_plans(
        micro_batch_sequences=16
    )

    assert len(plans) == 25

    assert {
        plan.pe_type
        for plan
        in plans
    } == {
        "learned",
        "sinusoidal",
        "rope",
        "alibi",
        "nope",
    }

    assert {
        plan.init_seed
        for plan
        in plans
    } == {
        17,
        42,
        1234,
        2027,
        5003,
    }

    assert {
        plan.data_seed
        for plan
        in plans
    } == {
        2026,
    }


def test_microbatch_and_gas_keep_global_batch_exact():
    plans = make_run_plans(
        micro_batch_sequences=16,
        seq_len=512,
        global_batch_tokens=65_536,
    )

    first = plans[0]

    assert (
        first.micro_batch_tokens
        == 8_192
    )

    assert (
        first.grad_accum_steps
        == 8
    )

    assert (
        first.micro_batch_tokens
        * first.grad_accum_steps
        == 65_536
    )


def test_invalid_microbatch_is_rejected():
    with pytest.raises(
        ValueError,
        match="divide",
    ):
        make_run_plans(
            micro_batch_sequences=3,
            seq_len=512,
            global_batch_tokens=65_536,
        )


def test_manifest_is_written(
    tmp_path,
):
    plans = make_run_plans(
        micro_batch_sequences=16
    )

    output = (
        tmp_path
        / "run_plan.json"
    )

    written = write_plan_manifest(
        output,
        plans,
    )

    assert written == output
    assert output.is_file()

    payload = json.loads(
        output.read_text(
            encoding="utf-8"
        )
    )

    assert (
        payload["num_runs"]
        == 25
    )

    assert (
        payload[
            "global_batch_tokens"
        ]
        == 65_536
    )

    assert (
        payload[
            "micro_batch_sequences"
        ]
        == 16
    )

    assert (
        payload[
            "grad_accum_steps"
        ]
        == 8
    )

    assert (
        len(
            payload["runs"]
        )
        == 25
    )