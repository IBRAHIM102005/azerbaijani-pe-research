import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.training.cache_builder import (
    group_documents_by_processed_file,
    load_cache_documents,
)


MANIFEST_SCHEMA = pa.schema(
    [
        (
            "sampling_order",
            pa.int64(),
        ),
        (
            "document_id",
            pa.string(),
        ),
        (
            "token_count",
            pa.int64(),
        ),
        (
            "processed_file",
            pa.string(),
        ),
        (
            "processed_row",
            pa.int64(),
        ),
    ]
)


def write_manifest(
    path,
    rows,
):
    table = pa.Table.from_pylist(
        rows,
        schema=MANIFEST_SCHEMA,
    )

    pq.write_table(
        table,
        path,
    )


def test_exact_budget_consumption(
    tmp_path,
):
    path = (
        tmp_path
        / "subset.parquet"
    )

    write_manifest(
        path,
        [
            {
                "sampling_order": 0,
                "document_id": "a",
                "token_count": 5,
                "processed_file": (
                    "train.parquet"
                ),
                "processed_row": 10,
            },
            {
                "sampling_order": 1,
                "document_id": "b",
                "token_count": 4,
                "processed_file": (
                    "train.parquet"
                ),
                "processed_row": 2,
            },
            {
                "sampling_order": 2,
                "document_id": "c",
                "token_count": 8,
                "processed_file": (
                    "train.parquet"
                ),
                "processed_row": 7,
            },
        ],
    )

    documents = (
        load_cache_documents(
            path,
            target_tokens=12,
        )
    )

    assert len(
        documents
    ) == 3

    assert [
        document.stream_offset
        for document in documents
    ] == [
        0,
        5,
        9,
    ]

    assert [
        document.take_tokens
        for document in documents
    ] == [
        5,
        4,
        3,
    ]

    assert (
        documents[-1]
        .is_partial
    )

    assert (
        documents[-1]
        .stream_end
        == 12
    )


def test_exact_document_boundary(
    tmp_path,
):
    path = (
        tmp_path
        / "subset.parquet"
    )

    write_manifest(
        path,
        [
            {
                "sampling_order": 0,
                "document_id": "a",
                "token_count": 5,
                "processed_file": "x",
                "processed_row": 0,
            },
            {
                "sampling_order": 1,
                "document_id": "b",
                "token_count": 7,
                "processed_file": "x",
                "processed_row": 1,
            },
            {
                "sampling_order": 2,
                "document_id": "c",
                "token_count": 9,
                "processed_file": "x",
                "processed_row": 2,
            },
        ],
    )

    documents = (
        load_cache_documents(
            path,
            target_tokens=12,
        )
    )

    assert len(
        documents
    ) == 2

    assert not (
        documents[-1]
        .is_partial
    )

    assert (
        documents[-1]
        .stream_end
        == 12
    )


def test_grouping_sorts_processed_rows(
    tmp_path,
):
    path = (
        tmp_path
        / "subset.parquet"
    )

    write_manifest(
        path,
        [
            {
                "sampling_order": 0,
                "document_id": "a",
                "token_count": 5,
                "processed_file": "train.parquet",
                "processed_row": 10,
            },
            {
                "sampling_order": 1,
                "document_id": "b",
                "token_count": 5,
                "processed_file": "train.parquet",
                "processed_row": 2,
            },
        ],
    )

    documents = (
        load_cache_documents(
            path,
            target_tokens=10,
        )
    )

    groups = (
        group_documents_by_processed_file(
            documents
        )
    )

    assert [
        document.processed_row
        for document
        in groups[
            "train.parquet"
        ]
    ] == [
        2,
        10,
    ]

    # Sampling-order stream positions
    # remain unchanged.
    assert [
        document.stream_offset
        for document
        in groups[
            "train.parquet"
        ]
    ] == [
        5,
        0,
    ]


def test_sampling_order_must_be_contiguous(
    tmp_path,
):
    path = (
        tmp_path
        / "subset.parquet"
    )

    write_manifest(
        path,
        [
            {
                "sampling_order": 0,
                "document_id": "a",
                "token_count": 5,
                "processed_file": "x",
                "processed_row": 0,
            },
            {
                "sampling_order": 2,
                "document_id": "b",
                "token_count": 5,
                "processed_file": "x",
                "processed_row": 1,
            },
        ],
    )

    with pytest.raises(
        ValueError,
        match="contiguous",
    ):
        load_cache_documents(
            path,
            target_tokens=10,
        )


def test_duplicate_processed_reference_rejected(
    tmp_path,
):
    path = (
        tmp_path
        / "subset.parquet"
    )

    write_manifest(
        path,
        [
            {
                "sampling_order": 0,
                "document_id": "a",
                "token_count": 5,
                "processed_file": "x",
                "processed_row": 7,
            },
            {
                "sampling_order": 1,
                "document_id": "b",
                "token_count": 5,
                "processed_file": "x",
                "processed_row": 7,
            },
        ],
    )

    with pytest.raises(
        ValueError,
        match="Duplicate processed reference",
    ):
        load_cache_documents(
            path,
            target_tokens=10,
        )


def test_insufficient_tokens_rejected(
    tmp_path,
):
    path = (
        tmp_path
        / "subset.parquet"
    )

    write_manifest(
        path,
        [
            {
                "sampling_order": 0,
                "document_id": "a",
                "token_count": 5,
                "processed_file": "x",
                "processed_row": 0,
            },
        ],
    )

    with pytest.raises(
        ValueError,
        match="enough tokens",
    ):
        load_cache_documents(
            path,
            target_tokens=10,
        )