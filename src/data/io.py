"""Stream DOLLMA rows in a stable order."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq


@dataclass(frozen=True)
class RawRecord:
    source: str
    shard: str
    row_index: int
    text: str | None


def source_shards(raw_root: Path, source: str) -> list[Path]:
    shards = sorted((raw_root / source).glob("*.parquet"), key=lambda item: item.name)
    if not shards:
        raise FileNotFoundError(f"No parquet shards found for {source} under {raw_root}")
    return shards


def stream_source(
    raw_root: Path,
    source: str,
    text_column: str,
    batch_size: int = 4096,
) -> Iterator[RawRecord]:
    """Stream one source without loading the full corpus into memory."""

    for shard in source_shards(raw_root, source):
        parquet = pq.ParquetFile(shard)
        if text_column not in parquet.schema_arrow.names:
            raise ValueError(f"{shard} has no configured text column {text_column!r}")
        row_index = 0
        for batch in parquet.iter_batches(batch_size=batch_size, columns=[text_column]):
            for text in batch.column(0).to_pylist():
                yield RawRecord(source, shard.name, row_index, text)
                row_index += 1


def stream_core(raw_root: Path, source_settings: dict[str, dict]) -> Iterator[RawRecord]:
    """Stream included sources according to the frozen configuration order."""

    for source, settings in source_settings.items():
        if settings["included_in_core"]:
            yield from stream_source(raw_root, source, settings["text_column"])
