"""Stream final SentencePiece token counts for every retained document."""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import sentencepiece as spm

from .corpus import tokenizer_text


TOKEN_COUNT_SCHEMA = pa.schema(
    [
        ("document_id", pa.string()),
        ("source", pa.string()),
        ("source_group", pa.string()),
        ("split", pa.string()),
        ("token_count", pa.int64()),
        ("includes_eod", pa.bool_()),
    ]
)


def count_processed_tokens(
    model_path: Path,
    processed_dir: Path,
    output_path: Path,
    batch_size: int = 4096,
    num_threads: int = 16,
) -> dict[str, Any]:
    """Count final tokenizer IDs in bounded batches, including one EOD per document."""

    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(temporary, TOKEN_COUNT_SCHEMA, compression="zstd")
    totals: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "documents": 0,
                "tokens": 0,
                "sentencepiece_tokens_excluding_eod": 0,
                "unknown_tokens": 0,
                "documents_with_unknown_tokens": 0,
            }
        )
    )
    rows = []
    processed_documents = 0
    started = time.perf_counter()
    try:
        for split in ("train", "validation", "test"):
            parquet = pq.ParquetFile(processed_dir / f"{split}.parquet")
            for batch in parquet.iter_batches(
                batch_size=batch_size,
                columns=["document_id", "source", "source_group", "text", "split"],
            ):
                columns = [column.to_pylist() for column in batch.columns]
                prepared_texts = [tokenizer_text(text) for text in columns[3]]
                encoded = processor.encode(prepared_texts, out_type=int, num_threads=num_threads)
                for document_id, source, group, stored_split, ids in zip(
                    columns[0], columns[1], columns[2], columns[4], encoded
                ):
                    if stored_split != split:
                        raise RuntimeError(f"Processed split mismatch for {document_id}")
                    token_count = len(ids) + 1
                    unknowns = sum(token_id == processor.unk_id() for token_id in ids)
                    rows.append(
                        {
                            "document_id": document_id,
                            "source": source,
                            "source_group": group,
                            "split": split,
                            "token_count": token_count,
                            "includes_eod": True,
                        }
                    )
                    totals[split][source]["documents"] += 1
                    totals[split][source]["tokens"] += token_count
                    totals[split][source]["sentencepiece_tokens_excluding_eod"] += len(ids)
                    totals[split][source]["unknown_tokens"] += unknowns
                    totals[split][source]["documents_with_unknown_tokens"] += bool(unknowns)
                    processed_documents += 1
                    if len(rows) >= batch_size:
                        writer.write_table(pa.Table.from_pylist(rows, schema=TOKEN_COUNT_SCHEMA))
                        rows.clear()
                if processed_documents and processed_documents % (batch_size * 64) < batch_size:
                    logging.info(
                        "stage=token_count_progress split=%s documents=%d runtime_seconds=%.1f",
                        split,
                        processed_documents,
                        time.perf_counter() - started,
                    )
            if rows:
                writer.write_table(pa.Table.from_pylist(rows, schema=TOKEN_COUNT_SCHEMA))
                rows.clear()
    finally:
        writer.close()
    os.replace(temporary, output_path)
    output = {}
    for split, sources in totals.items():
        output[split] = {}
        for source, values in sorted(sources.items()):
            source_values = dict(values)
            denominator = source_values["sentencepiece_tokens_excluding_eod"]
            source_values["unknown_token_rate"] = (
                source_values["unknown_tokens"] / denominator if denominator else 0.0
            )
            source_values["unknown_rate_denominator_policy"] = (
                "SentencePiece tokens excluding the one appended EOD token per document."
            )
            output[split][source] = source_values
    return output
