import json
from pathlib import Path

import pyarrow.parquet as pq
import sentencepiece as spm

from src.data.hashing import sha256_file


ROOT = Path(__file__).resolve().parents[2]


def test_frozen_tokenizer_and_training_sample_are_valid():
    tokenizer_dir = ROOT / "tokenizer"
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_dir / "tokenizer.model"))
    hashes = json.loads((tokenizer_dir / "tokenizer_hashes.json").read_text(encoding="utf-8"))
    assert processor.vocab_size() == 16_000
    assert (processor.unk_id(), processor.eos_id(), processor.bos_id(), processor.pad_id()) == (0, 1, -1, -1)
    assert processor.id_to_piece(1) == "<eod>"
    assert processor.decode(processor.encode("Azərbaycan şəhərləri")) == "Azərbaycan şəhərləri"
    for filename, expected in hashes.items():
        assert sha256_file(tokenizer_dir / filename) == expected

    sample = pq.ParquetFile(tokenizer_dir / "training_sample_manifest.parquet")
    assert sample.metadata.num_rows == 1_000_000
    for batch in sample.iter_batches(batch_size=50_000, columns=["split"]):
        assert set(batch.column(0).to_pylist()) == {"train"}


def test_frozen_50m_manifest_is_unique_train_only_and_hash_frozen():
    subset_path = ROOT / "data" / "manifests" / "train_50m.parquet"
    summary = json.loads(
        (ROOT / "data" / "metadata" / "training_subset_summary.json").read_text(encoding="utf-8")
    )
    rows = pq.read_table(
        subset_path,
        columns=["sampling_order", "document_id", "token_count", "processed_file", "processed_row"],
    ).to_pylist()
    assert len(rows) == summary["selected_documents"]
    assert [row["sampling_order"] for row in rows] == list(range(len(rows)))
    assert len({row["document_id"] for row in rows}) == len(rows)
    assert sum(row["token_count"] for row in rows) == summary["selected_unique_tokens"]
    assert all(Path(row["processed_file"]).name == "train.parquet" for row in rows)
    assert all(row["processed_row"] >= 0 for row in rows)
    assert sha256_file(subset_path) == summary["manifest_sha256"]
