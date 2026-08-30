"""Write processed parquet files and compact frozen manifests."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .hashing import sha256_file
from .paths import repository_relative
from .split import assign_split


PROCESSED_SCHEMA = pa.schema(
    [
        ("document_id", pa.string()),
        ("source", pa.string()),
        ("source_group", pa.string()),
        ("text", pa.string()),
        ("character_count", pa.int64()),
        ("approximate_word_count", pa.int64()),
        ("split", pa.string()),
        ("duplicate_cluster_id", pa.string()),
        ("canonical_text_hash", pa.string()),
        ("raw_record_id", pa.string()),
        ("raw_shard", pa.string()),
        ("raw_row_index", pa.int64()),
        ("quality_flags", pa.string()),
    ]
)

MANIFEST_SCHEMA = pa.schema(
    [
        ("document_id", pa.string()),
        ("source", pa.string()),
        ("source_group", pa.string()),
        ("duplicate_cluster_id", pa.string()),
        ("canonical_text_hash", pa.string()),
        ("processed_file", pa.string()),
        ("processed_row", pa.int64()),
        ("raw_record_id", pa.string()),
        ("raw_shard", pa.string()),
        ("raw_row_index", pa.int64()),
    ]
)


def _write_batch(writer: pq.ParquetWriter, rows: list[dict], schema: pa.Schema) -> None:
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=schema))
        rows.clear()


def write_processed_and_manifests(
    connection: sqlite3.Connection,
    processed_dir: Path,
    manifest_dir: Path,
    source_settings: dict[str, dict],
    split_settings: dict[str, int],
    split_seed: int,
    batch_size: int = 10_000,
    repo_root: Path | None = None,
    reference_processed_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Write retained documents and metadata in stable document-ID order."""

    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    splits = ("train", "validation", "test")
    processed_paths = {split: processed_dir / f"{split}.parquet" for split in splits}
    manifest_paths = {split: manifest_dir / f"{split}.parquet" for split in splits}
    processed_temporaries = {
        split: path.with_name(f".{path.name}.partial") for split, path in processed_paths.items()
    }
    manifest_temporaries = {
        split: path.with_name(f".{path.name}.partial") for split, path in manifest_paths.items()
    }
    for path in (*processed_temporaries.values(), *manifest_temporaries.values()):
        if path.exists():
            path.unlink()
    processed_writers = {
        split: pq.ParquetWriter(processed_temporaries[split], PROCESSED_SCHEMA, compression="zstd")
        for split in splits
    }
    manifest_writers = {
        split: pq.ParquetWriter(manifest_temporaries[split], MANIFEST_SCHEMA, compression="zstd")
        for split in splits
    }
    processed_buffers = {split: [] for split in splits}
    manifest_buffers = {split: [] for split in splits}
    row_offsets = {split: 0 for split in splits}
    summary: dict[str, Any] = {
        "splits": {split: {"documents": 0, "characters": 0, "approximate_words": 0, "sources": {}} for split in splits}
    }
    query = """
        SELECT d.rowid, d.document_id, d.source, d.text, d.character_count,
               d.word_count, d.canonical_hash, d.raw_record_id, d.shard,
               d.row_index, d.quality_flags, m.cluster_id
        FROM documents d
        LEFT JOIN near_members m ON m.rowid = d.rowid
        WHERE COALESCE(m.removed, 0) = 0
        ORDER BY d.document_id
    """
    root = (repo_root or manifest_dir.parent).resolve()
    reference_dir = reference_processed_dir or processed_dir
    processed_references = {
        split: repository_relative(reference_dir / f"{split}.parquet", root)
        for split in splits
    }
    failed = False
    try:
        for row in connection.execute(query):
            (
                rowid, doc_id, source, text, character_count, word_count,
                canonical_hash, record_id, shard, row_index, flags, cluster_id,
            ) = row
            cluster_id = cluster_id or doc_id
            split = assign_split(
                cluster_id,
                split_seed,
                split_settings["train_upper"],
                split_settings["validation_upper"],
                split_settings["modulus"],
            )
            group = source_settings[source]["group"]
            processed_row = row_offsets[split]
            processed_buffers[split].append(
                {
                    "document_id": doc_id,
                    "source": source,
                    "source_group": group,
                    "text": text,
                    "character_count": character_count,
                    "approximate_word_count": word_count,
                    "split": split,
                    "duplicate_cluster_id": cluster_id,
                    "canonical_text_hash": canonical_hash,
                    "raw_record_id": record_id,
                    "raw_shard": shard,
                    "raw_row_index": row_index,
                    "quality_flags": flags,
                }
            )
            manifest_buffers[split].append(
                {
                    "document_id": doc_id,
                    "source": source,
                    "source_group": group,
                    "duplicate_cluster_id": cluster_id,
                    "canonical_text_hash": canonical_hash,
                    "processed_file": processed_references[split],
                    "processed_row": processed_row,
                    "raw_record_id": record_id,
                    "raw_shard": shard,
                    "raw_row_index": row_index,
                }
            )
            row_offsets[split] += 1
            split_summary = summary["splits"][split]
            split_summary["documents"] += 1
            split_summary["characters"] += character_count
            split_summary["approximate_words"] += word_count
            source_summary = split_summary["sources"].setdefault(
                source, {"documents": 0, "characters": 0, "approximate_words": 0}
            )
            source_summary["documents"] += 1
            source_summary["characters"] += character_count
            source_summary["approximate_words"] += word_count
            if len(processed_buffers[split]) >= batch_size:
                _write_batch(processed_writers[split], processed_buffers[split], PROCESSED_SCHEMA)
                _write_batch(manifest_writers[split], manifest_buffers[split], MANIFEST_SCHEMA)
    except BaseException:
        failed = True
        raise
    finally:
        for split in splits:
            if not failed:
                _write_batch(processed_writers[split], processed_buffers[split], PROCESSED_SCHEMA)
                _write_batch(manifest_writers[split], manifest_buffers[split], MANIFEST_SCHEMA)
            processed_writers[split].close()
            manifest_writers[split].close()

    for split in splits:
        processed_rows = pq.ParquetFile(processed_temporaries[split]).metadata.num_rows
        manifest_rows = pq.ParquetFile(manifest_temporaries[split]).metadata.num_rows
        if processed_rows != row_offsets[split] or manifest_rows != row_offsets[split]:
            raise RuntimeError(f"Incomplete staged {split} corpus or manifest")
    for split in splits:
        os.replace(processed_temporaries[split], processed_paths[split])
        os.replace(manifest_temporaries[split], manifest_paths[split])

    hashes = {repository_relative(path, root): sha256_file(path) for path in manifest_paths.values()}
    return summary, hashes
