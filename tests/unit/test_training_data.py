import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.training.data import (
    TokenBlockDataset,
    build_consumption_plan,
    expected_full_blocks,
)


def write_manifest(
    path,
    token_counts,
):
    rows = []

    for index, token_count in enumerate(
        token_counts
    ):
        rows.append(
            {
                "sampling_order": index,
                "document_id": (
                    f"doc-{index}"
                ),
                "token_count": (
                    token_count
                ),
                "processed_file": (
                    "data/processed/"
                    "corpus/train.parquet"
                ),
                "processed_row": index,
            }
        )

    table = pa.Table.from_pylist(
        rows
    )

    pq.write_table(
        table,
        path,
    )


def test_consumption_plan_stops_exactly_at_target(
    tmp_path,
):
    manifest = (
        tmp_path / "manifest.parquet"
    )

    write_manifest(
        manifest,
        [4, 5, 6],
    )

    plan = build_consumption_plan(
        manifest,
        target_tokens=10,
    )

    assert len(plan) == 3

    assert plan[0].output_start == 0
    assert plan[0].tokens_to_write == 4

    assert plan[1].output_start == 4
    assert plan[1].tokens_to_write == 5

    # Only one token is needed from
    # the third document:
    # 4 + 5 + 1 = 10.
    assert plan[2].output_start == 9
    assert plan[2].tokens_to_write == 1

    assert sum(
        record.tokens_to_write
        for record in plan
    ) == 10


def test_consumption_plan_uses_frozen_sampling_order(
    tmp_path,
):
    manifest = (
        tmp_path / "manifest.parquet"
    )

    rows = [
        {
            "sampling_order": 2,
            "document_id": "c",
            "token_count": 5,
            "processed_file": "train.parquet",
            "processed_row": 2,
        },
        {
            "sampling_order": 0,
            "document_id": "a",
            "token_count": 5,
            "processed_file": "train.parquet",
            "processed_row": 0,
        },
        {
            "sampling_order": 1,
            "document_id": "b",
            "token_count": 5,
            "processed_file": "train.parquet",
            "processed_row": 1,
        },
    ]

    pq.write_table(
        pa.Table.from_pylist(rows),
        manifest,
    )

    plan = build_consumption_plan(
        manifest,
        target_tokens=12,
    )

    assert [
        record.document_id
        for record in plan
    ] == [
        "a",
        "b",
        "c",
    ]

    assert [
        record.tokens_to_write
        for record in plan
    ] == [
        5,
        5,
        2,
    ]


def test_token_block_dataset_reads_full_blocks_and_tail(
    tmp_path,
):
    path = (
        tmp_path / "tokens.bin"
    )

    tokens = np.arange(
        13,
        dtype=np.uint16,
    )

    tokens.tofile(
        path
    )

    dataset = TokenBlockDataset(
        path,
        total_tokens=13,
        seq_len=5,
    )

    assert len(dataset) == 2
    assert dataset.full_blocks == 2
    assert dataset.tail_tokens == 3

    assert torch.equal(
        dataset[0],
        torch.tensor(
            [0, 1, 2, 3, 4]
        ),
    )

    assert torch.equal(
        dataset[1],
        torch.tensor(
            [5, 6, 7, 8, 9]
        ),
    )

    tail = dataset.tail()

    assert tail is not None

    assert torch.equal(
        tail,
        torch.tensor(
            [10, 11, 12]
        ),
    )


def test_token_block_dataset_without_tail(
    tmp_path,
):
    path = (
        tmp_path / "tokens.bin"
    )

    np.arange(
        10,
        dtype=np.uint16,
    ).tofile(path)

    dataset = TokenBlockDataset(
        path,
        total_tokens=10,
        seq_len=5,
    )

    assert len(dataset) == 2
    assert dataset.tail() is None


def test_50m_block_math():
    full_blocks, tail = (
        expected_full_blocks(
            50_000_000,
            512,
        )
    )

    assert full_blocks == 97_656
    assert tail == 128

    assert (
        full_blocks * 512
        + tail
        == 50_000_000
    )