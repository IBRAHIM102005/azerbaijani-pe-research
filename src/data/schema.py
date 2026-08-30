"""Inspect parquet metadata and enforce source-specific text columns."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .hashing import canonical_json_hash, sha256_file


def inspect_parquet(path: Path, source: str, text_column: str) -> dict[str, Any]:
    """Inspect one shard without reading its full text into memory."""

    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    if text_column not in schema.names:
        raise ValueError(f"{path} has no configured text column {text_column!r}")
    field = schema.field(text_column)
    if not (pa.types.is_string(field.type) or pa.types.is_large_string(field.type)):
        raise TypeError(f"{path}:{text_column} is {field.type}, not a string")

    column_index = parquet.schema.names.index(text_column)
    null_count = 0
    null_count_complete = True
    for row_group_index in range(parquet.metadata.num_row_groups):
        statistics = parquet.metadata.row_group(row_group_index).column(column_index).statistics
        if statistics is None or statistics.null_count is None:
            null_count_complete = False
            break
        null_count += statistics.null_count

    schema_fields = [{"name": item.name, "type": str(item.type)} for item in schema]
    return {
        "source": source,
        "filename": path.name,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
        "schema": schema_fields,
        "schema_signature": canonical_json_hash(schema_fields),
        "selected_text_column": text_column,
        "null_text_count": null_count if null_count_complete else None,
    }
