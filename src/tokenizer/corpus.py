"""Build a deterministic train-only SentencePiece corpus."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.hashing import sha256_file


SAMPLE_SCHEMA = pa.schema(
    [
        ("document_id", pa.string()),
        ("source", pa.string()),
        ("source_group", pa.string()),
        ("split", pa.string()),
        ("selection_order", pa.int64()),
    ]
)


def tokenizer_text(text: str) -> str:
    """Project canonical text onto SentencePiece's one-line document input."""

    return " ".join(text.splitlines())


def build_training_corpus(
    train_parquet: Path,
    output_text: Path,
    output_manifest: Path,
    max_documents: int,
    batch_size: int = 4096,
) -> dict[str, Any]:
    """Write the first stable document-ID-ranked train records for tokenizer training."""

    output_text.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_text = output_text.with_suffix(output_text.suffix + ".tmp")
    temporary_manifest = output_manifest.with_suffix(output_manifest.suffix + ".tmp")
    writer = pq.ParquetWriter(temporary_manifest, SAMPLE_SCHEMA, compression="zstd")
    count = 0
    characters = 0
    manifest_rows = []
    parquet = pq.ParquetFile(train_parquet)
    with temporary_text.open("w", encoding="utf-8", newline="\n") as text_handle:
        try:
            for batch in parquet.iter_batches(
                batch_size=batch_size,
                columns=["document_id", "source", "source_group", "text", "split"],
            ):
                for document_id, source, group, text, split in zip(*[column.to_pylist() for column in batch.columns]):
                    if count >= max_documents:
                        break
                    if split != "train":
                        raise RuntimeError("Tokenizer corpus received a non-train document")
                    line = tokenizer_text(text)
                    text_handle.write(line)
                    text_handle.write("\n")
                    characters += len(line)
                    manifest_rows.append(
                        {
                            "document_id": document_id,
                            "source": source,
                            "source_group": group,
                            "split": split,
                            "selection_order": count,
                        }
                    )
                    count += 1
                    if len(manifest_rows) >= batch_size:
                        writer.write_table(pa.Table.from_pylist(manifest_rows, schema=SAMPLE_SCHEMA))
                        manifest_rows.clear()
                if count >= max_documents:
                    break
            if manifest_rows:
                writer.write_table(pa.Table.from_pylist(manifest_rows, schema=SAMPLE_SCHEMA))
            text_handle.flush()
            os.fsync(text_handle.fileno())
        finally:
            writer.close()
    if pq.ParquetFile(temporary_manifest).metadata.num_rows != count:
        raise RuntimeError("Incomplete tokenizer-training sample manifest")
    os.replace(temporary_text, output_text)
    os.replace(temporary_manifest, output_manifest)
    return {
        "documents": count,
        "characters": characters,
        "training_corpus_sha256": sha256_file(output_text),
        "training_sample_manifest_sha256": sha256_file(output_manifest),
        "selection_rule": "First documents in the processed train parquet, which is sorted by SHA-256 document_id.",
        "newline_policy": "Internal line breaks are replaced with spaces only in the SentencePiece trainer input.",
    }
