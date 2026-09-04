import json

import pytest

from src.training.queue import (
    MatrixJob,
    build_matrix_jobs,
)


def make_plan(
    tmp_path,
):
    return {
        "num_runs": 4,
        "runs": [
            {
                "run_id": "learned-s17",
                "pe_type": "learned",
                "init_seed": 17,
                "run_dir": str(
                    tmp_path
                    / "learned-s17"
                ),
            },
            {
                "run_id": "rope-s17",
                "pe_type": "rope",
                "init_seed": 17,
                "run_dir": str(
                    tmp_path
                    / "rope-s17"
                ),
            },
            {
                "run_id": "learned-s42",
                "pe_type": "learned",
                "init_seed": 42,
                "run_dir": str(
                    tmp_path
                    / "learned-s42"
                ),
            },
            {
                "run_id": "rope-s42",
                "pe_type": "rope",
                "init_seed": 42,
                "run_dir": str(
                    tmp_path
                    / "rope-s42"
                ),
            },
        ],
    }


def test_build_matrix_jobs_preserves_order(
    tmp_path,
):
    payload = make_plan(
        tmp_path
    )

    jobs = build_matrix_jobs(
        payload,
        repo_root=tmp_path,
    )

    assert [
        job.run_id
        for job in jobs
    ] == [
        "learned-s17",
        "rope-s17",
        "learned-s42",
        "rope-s42",
    ]


def test_filter_by_pe(
    tmp_path,
):
    payload = make_plan(
        tmp_path
    )

    jobs = build_matrix_jobs(
        payload,
        repo_root=tmp_path,
        pe_types=[
            "rope",
        ],
    )

    assert [
        job.run_id
        for job in jobs
    ] == [
        "rope-s17",
        "rope-s42",
    ]


def test_filter_by_seed(
    tmp_path,
):
    payload = make_plan(
        tmp_path
    )

    jobs = build_matrix_jobs(
        payload,
        repo_root=tmp_path,
        seeds=[
            42,
        ],
    )

    assert [
        job.init_seed
        for job in jobs
    ] == [
        42,
        42,
    ]


def test_completed_runs_are_skipped(
    tmp_path,
):
    payload = make_plan(
        tmp_path
    )

    completed_dir = (
        tmp_path
        / "learned-s17"
    )

    completed_dir.mkdir(
        parents=True
    )

    (
        completed_dir
        / "completed.json"
    ).write_text(
        json.dumps(
            {
                "status": "completed"
            }
        ),
        encoding="utf-8",
    )

    jobs = build_matrix_jobs(
        payload,
        repo_root=tmp_path,
    )

    assert [
        job.run_id
        for job in jobs
    ] == [
        "rope-s17",
        "learned-s42",
        "rope-s42",
    ]


def test_completed_runs_can_be_included(
    tmp_path,
):
    payload = make_plan(
        tmp_path
    )

    completed_dir = (
        tmp_path
        / "learned-s17"
    )

    completed_dir.mkdir(
        parents=True
    )

    (
        completed_dir
        / "completed.json"
    ).write_text(
        "{}",
        encoding="utf-8",
    )

    jobs = build_matrix_jobs(
        payload,
        repo_root=tmp_path,
        include_completed=True,
    )

    assert len(jobs) == 4


def test_resume_detection(
    tmp_path,
):
    run_dir = (
        tmp_path
        / "rope-s17"
    )

    checkpoint_dir = (
        run_dir
        / "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True
    )

    (
        checkpoint_dir
        / "latest.pt"
    ).write_bytes(
        b"checkpoint"
    )

    job = MatrixJob(
        index=1,
        run_id="rope-s17",
        pe_type="rope",
        init_seed=17,
        run_dir=run_dir,
    )

    assert job.can_resume
    assert not job.is_completed


def test_duplicate_run_id_is_rejected(
    tmp_path,
):
    payload = {
        "runs": [
            {
                "run_id": "same",
                "pe_type": "rope",
                "init_seed": 17,
                "run_dir": "a",
            },
            {
                "run_id": "same",
                "pe_type": "rope",
                "init_seed": 42,
                "run_dir": "b",
            },
        ]
    }

    with pytest.raises(
        ValueError,
        match="Duplicate run_id",
    ):
        build_matrix_jobs(
            payload,
            repo_root=tmp_path,
        )