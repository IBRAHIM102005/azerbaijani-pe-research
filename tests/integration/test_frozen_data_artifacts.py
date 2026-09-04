import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import sentencepiece as spm

from src.data.hashing import sha256_file


ROOT = Path(__file__).resolve().parents[2]


def test_frozen_tokenizer_is_valid():
    """Tokenizer artifacts are small/tracked and must always validate."""

    tokenizer_dir = ROOT / "tokenizer"

    tokenizer_model = (
        tokenizer_dir
        / "tokenizer.model"
    )

    tokenizer_hashes = (
        tokenizer_dir
        / "tokenizer_hashes.json"
    )

    assert tokenizer_model.is_file()
    assert tokenizer_hashes.is_file()

    processor = (
        spm.SentencePieceProcessor(
            model_file=str(
                tokenizer_model
            )
        )
    )

    hashes = json.loads(
        tokenizer_hashes.read_text(
            encoding="utf-8"
        )
    )

    assert (
        processor.vocab_size()
        == 16_000
    )

    assert (
        processor.unk_id(),
        processor.eos_id(),
        processor.bos_id(),
        processor.pad_id(),
    ) == (
        0,
        1,
        -1,
        -1,
    )

    assert (
        processor.id_to_piece(1)
        == "<eod>"
    )

    sample_text = (
        "Azərbaycan şəhərləri"
    )

    assert (
        processor.decode(
            processor.encode(
                sample_text
            )
        )
        == sample_text
    )

    for (
        filename,
        expected,
    ) in hashes.items():

        artifact_path = (
            tokenizer_dir
            / filename
        )

        assert (
            artifact_path.is_file()
        )

        assert (
            sha256_file(
                artifact_path
            )
            == expected
        )


def test_frozen_training_sample_is_valid_when_available():
    """Validate the large tokenizer-training sample when materialized.

    Parquet artifacts are intentionally excluded from a clean Git checkout.
    Their absence in CI is therefore not a data-integrity failure.
    When the artifact is available locally/server-side, validate it fully.
    """

    sample_path = (
        ROOT
        / "tokenizer"
        / "training_sample_manifest.parquet"
    )

    if not sample_path.is_file():
        pytest.skip(
            "Large frozen artifact "
            "tokenizer/training_sample_manifest.parquet "
            "is not present in this clean checkout."
        )

    sample = pq.ParquetFile(
        sample_path
    )

    assert (
        sample.metadata.num_rows
        == 1_000_000
    )

    for batch in sample.iter_batches(
        batch_size=50_000,
        columns=[
            "split",
        ],
    ):
        assert (
            set(
                batch
                .column(0)
                .to_pylist()
            )
            == {
                "train",
            }
        )


def test_frozen_50m_manifest_is_unique_train_only_and_hash_frozen():
    """Validate the frozen 50M subset whenever the large manifest exists."""

    subset_path = (
        ROOT
        / "data"
        / "manifests"
        / "train_50m.parquet"
    )

    summary_path = (
        ROOT
        / "data"
        / "metadata"
        / "training_subset_summary.json"
    )

    assert (
        summary_path.is_file()
    )

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    if not subset_path.is_file():
        pytest.skip(
            "Large frozen artifact "
            "data/manifests/train_50m.parquet "
            "is not present in this clean checkout."
        )

    rows = pq.read_table(
        subset_path,
        columns=[
            "sampling_order",
            "document_id",
            "token_count",
            "processed_file",
            "processed_row",
        ],
    ).to_pylist()

    assert (
        len(rows)
        == summary[
            "selected_documents"
        ]
    )

    assert [
        row[
            "sampling_order"
        ]
        for row in rows
    ] == list(
        range(
            len(rows)
        )
    )

    assert (
        len(
            {
                row[
                    "document_id"
                ]
                for row in rows
            }
        )
        == len(rows)
    )

    assert (
        sum(
            row[
                "token_count"
            ]
            for row in rows
        )
        == summary[
            "selected_unique_tokens"
        ]
    )

    assert all(
        Path(
            row[
                "processed_file"
            ]
        ).name
        == "train.parquet"
        for row in rows
    )

    assert all(
        row[
            "processed_row"
        ]
        >= 0
        for row in rows
    )

    assert (
        sha256_file(
            subset_path
        )
        == summary[
            "manifest_sha256"
        ]
    )