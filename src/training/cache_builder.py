"""Fast deterministic M1 -> M3 token-cache builder.

The frozen M1 training subset stores, for each selected document:

    sampling_order
    document_id
    token_count
    processed_file
    processed_row

The naive implementation may repeatedly search parquet batches for
selected rows. That is too expensive for the real 5.57M-row train
corpus.

This implementation instead:

1. reads the small frozen 50M subset manifest once;
2. computes every selected document's final token-stream offset;
3. groups references by processed parquet file;
4. scans each processed parquet file exactly once;
5. batch-tokenizes only selected rows;
6. writes token IDs directly into their final positions in a
   preallocated uint16 memory map.

Because final stream offsets are known from M1's frozen token counts,
documents may be discovered in processed-row order while still being
written in frozen sampling order.

The final document may be truncated so the cache contains exactly the
frozen target token budget.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import sentencepiece as spm

from src.tokenizer.corpus import (
    tokenizer_text,
)


REQUIRED_MANIFEST_COLUMNS = (
    "sampling_order",
    "document_id",
    "token_count",
    "processed_file",
    "processed_row",
)


# ============================================================
# Records
# ============================================================


@dataclass(frozen=True)
class CacheDocument:
    """One document that contributes tokens to the final cache."""

    sampling_order: int

    document_id: str

    token_count: int

    processed_file: str

    processed_row: int

    stream_offset: int

    take_tokens: int

    @property
    def stream_end(self) -> int:
        return (
            self.stream_offset
            + self.take_tokens
        )

    @property
    def is_partial(self) -> bool:
        return (
            self.take_tokens
            < self.token_count
        )


# ============================================================
# Generic helpers
# ============================================================


def sha256_file(
    path: str | Path,
    *,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    """SHA-256 a file without loading it into memory."""

    digest = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as handle:

        while True:

            block = handle.read(
                chunk_size
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def atomic_write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Atomically write JSON metadata."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = Path(
        str(path) + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def resolve_processed_path(
    processed_file: str,
    *,
    repo_root: Path,
) -> Path:
    """Resolve M1 repository-relative processed parquet path."""

    path = Path(
        processed_file
    )

    if path.is_absolute():
        return path.resolve()

    return (
        repo_root
        / path
    ).resolve()


# ============================================================
# Frozen manifest
# ============================================================


def load_cache_documents(
    manifest_path: str | Path,
    *,
    target_tokens: int,
) -> list[CacheDocument]:
    """Convert M1 subset manifest into exact cache-write records."""

    manifest_path = Path(
        manifest_path
    )

    if target_tokens <= 0:
        raise ValueError(
            "target_tokens must be positive"
        )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Training subset manifest missing: "
            f"{manifest_path}"
        )

    parquet = pq.ParquetFile(
        manifest_path
    )

    available_columns = set(
        parquet.schema_arrow.names
    )

    missing = [
        column
        for column in REQUIRED_MANIFEST_COLUMNS
        if column not in available_columns
    ]

    if missing:
        raise ValueError(
            "Training subset manifest is missing "
            f"required columns: {missing}. "
            f"Available columns: "
            f"{sorted(available_columns)}"
        )

    table = pq.read_table(
        manifest_path,
        columns=list(
            REQUIRED_MANIFEST_COLUMNS
        ),
    )

    rows = table.to_pylist()

    if not rows:
        raise ValueError(
            "Training subset manifest is empty."
        )

    rows.sort(
        key=lambda row: int(
            row["sampling_order"]
        )
    )

    documents: list[
        CacheDocument
    ] = []

    cumulative_tokens = 0

    previous_processed_keys: set[
        tuple[str, int]
    ] = set()

    for expected_order, row in enumerate(
        rows
    ):

        sampling_order = int(
            row["sampling_order"]
        )

        if (
            sampling_order
            != expected_order
        ):
            raise ValueError(
                "sampling_order must be contiguous "
                "and zero-based: "
                f"expected {expected_order}, "
                f"got {sampling_order}"
            )

        document_id = str(
            row["document_id"]
        )

        token_count = int(
            row["token_count"]
        )

        processed_file = str(
            row["processed_file"]
        )

        processed_row = int(
            row["processed_row"]
        )

        if token_count <= 0:
            raise ValueError(
                f"Invalid token_count for "
                f"{document_id}: {token_count}"
            )

        if processed_row < 0:
            raise ValueError(
                f"Invalid processed_row for "
                f"{document_id}: {processed_row}"
            )

        processed_key = (
            processed_file,
            processed_row,
        )

        if (
            processed_key
            in previous_processed_keys
        ):
            raise ValueError(
                "Duplicate processed reference: "
                f"{processed_file} "
                f"row={processed_row}"
            )

        previous_processed_keys.add(
            processed_key
        )

        if (
            cumulative_tokens
            >= target_tokens
        ):
            break

        remaining = (
            target_tokens
            - cumulative_tokens
        )

        take_tokens = min(
            token_count,
            remaining,
        )

        documents.append(
            CacheDocument(
                sampling_order=(
                    sampling_order
                ),
                document_id=(
                    document_id
                ),
                token_count=(
                    token_count
                ),
                processed_file=(
                    processed_file
                ),
                processed_row=(
                    processed_row
                ),
                stream_offset=(
                    cumulative_tokens
                ),
                take_tokens=(
                    take_tokens
                ),
            )
        )

        cumulative_tokens += (
            take_tokens
        )

        if (
            cumulative_tokens
            == target_tokens
        ):
            break

    if (
        cumulative_tokens
        != target_tokens
    ):
        raise ValueError(
            "Frozen subset does not contain "
            "enough tokens: "
            f"{cumulative_tokens:,} "
            f"< {target_tokens:,}"
        )

    partial_documents = [
        document
        for document in documents
        if document.is_partial
    ]

    if len(
        partial_documents
    ) > 1:
        raise RuntimeError(
            "Only the final consumed document "
            "may be partial."
        )

    if (
        partial_documents
        and partial_documents[-1]
        is not documents[-1]
    ):
        raise RuntimeError(
            "Partial document must be the "
            "final consumed document."
        )

    return documents


# ============================================================
# Group by processed parquet
# ============================================================


def group_documents_by_processed_file(
    documents: list[CacheDocument],
) -> dict[
    str,
    list[CacheDocument],
]:
    """Group selected references and sort them by processed row."""

    groups: dict[
        str,
        list[CacheDocument],
    ] = {}

    for document in documents:

        groups.setdefault(
            document.processed_file,
            [],
        ).append(
            document
        )

    for group in groups.values():

        group.sort(
            key=lambda document: (
                document.processed_row
            )
        )

    return groups


# ============================================================
# Processed parquet scan
# ============================================================


def write_processed_file_to_cache(
    *,
    processed_path: Path,
    documents: list[CacheDocument],
    output: np.memmap,
    processor: Any,
    eod_id: int,
    vocab_size: int,
    batch_size: int,
    num_threads: int,
) -> int:
    """Scan one processed parquet once and fill final cache positions."""

    if not processed_path.is_file():
        raise FileNotFoundError(
            f"Processed parquet missing: "
            f"{processed_path}"
        )

    if not documents:
        return 0

    parquet = pq.ParquetFile(
        processed_path
    )

    schema_names = set(
        parquet.schema_arrow.names
    )

    required = {
        "document_id",
        "text",
    }

    if not required.issubset(
        schema_names
    ):
        raise ValueError(
            f"Processed parquet {processed_path} "
            f"must contain {sorted(required)}; "
            f"has {sorted(schema_names)}"
        )

    total_rows = (
        parquet.metadata.num_rows
    )

    if (
        documents[-1].processed_row
        >= total_rows
    ):
        raise ValueError(
            "Manifest processed_row exceeds "
            f"parquet row count for "
            f"{processed_path}"
        )

    pointer = 0

    global_row_offset = 0

    written_tokens = 0

    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=[
            "document_id",
            "text",
        ],
    ):

        batch_rows = (
            batch.num_rows
        )

        batch_start = (
            global_row_offset
        )

        batch_end = (
            batch_start
            + batch_rows
        )

        # Skip until the next selected row falls
        # inside this parquet batch.
        if (
            pointer
            >= len(
                documents
            )
        ):
            break

        next_selected_row = (
            documents[
                pointer
            ].processed_row
        )

        if (
            next_selected_row
            >= batch_end
        ):
            global_row_offset = (
                batch_end
            )
            continue

        document_ids = (
            batch.column(0)
            .to_pylist()
        )

        texts = (
            batch.column(1)
            .to_pylist()
        )

        selected_documents: list[
            CacheDocument
        ] = []

        selected_texts: list[
            str
        ] = []

        while (
            pointer
            < len(documents)
            and documents[
                pointer
            ].processed_row
            < batch_end
        ):

            document = (
                documents[
                    pointer
                ]
            )

            if (
                document.processed_row
                < batch_start
            ):
                raise RuntimeError(
                    "Processed-row scan passed "
                    f"required row "
                    f"{document.processed_row}."
                )

            local_row = (
                document.processed_row
                - batch_start
            )

            actual_document_id = str(
                document_ids[
                    local_row
                ]
            )

            if (
                actual_document_id
                != document.document_id
            ):
                raise RuntimeError(
                    "M1 processed reference mismatch: "
                    f"row={document.processed_row}, "
                    f"manifest="
                    f"{document.document_id}, "
                    f"processed="
                    f"{actual_document_id}"
                )

            text = texts[
                local_row
            ]

            if not isinstance(
                text,
                str,
            ):
                raise RuntimeError(
                    "Processed document text is "
                    "not a string for "
                    f"{document.document_id}"
                )

            selected_documents.append(
                document
            )

            selected_texts.append(
                tokenizer_text(
                    text
                )
            )

            pointer += 1

        if selected_texts:

            encoded_batch = (
                processor.encode(
                    selected_texts,
                    out_type=int,
                    num_threads=num_threads,
                )
            )

            if (
                len(encoded_batch)
                != len(
                    selected_documents
                )
            ):
                raise RuntimeError(
                    "SentencePiece returned an "
                    "unexpected batch size."
                )

            for document, ids in zip(
                selected_documents,
                encoded_batch,
            ):

                actual_token_count = (
                    len(ids)
                    + 1
                )

                if (
                    actual_token_count
                    != document.token_count
                ):
                    raise RuntimeError(
                        "Frozen M1 token-count mismatch "
                        f"for {document.document_id}: "
                        f"manifest="
                        f"{document.token_count}, "
                        f"recomputed="
                        f"{actual_token_count}"
                    )

                if ids:

                    minimum_id = min(
                        ids
                    )

                    maximum_id = max(
                        ids
                    )

                    if minimum_id < 0:
                        raise RuntimeError(
                            "Negative SentencePiece "
                            f"token ID for "
                            f"{document.document_id}"
                        )

                    if maximum_id >= vocab_size:
                        raise RuntimeError(
                            "SentencePiece token ID "
                            "outside frozen vocabulary: "
                            f"{maximum_id} "
                            f">= {vocab_size}"
                        )

                # Full M1 document sequence:
                #
                # SentencePiece IDs + one EOD.
                #
                # The exact 50M boundary may truncate
                # the final document before its EOD.
                full_ids = np.empty(
                    document.token_count,
                    dtype=np.uint16,
                )

                if ids:

                    full_ids[
                        : len(ids)
                    ] = np.asarray(
                        ids,
                        dtype=np.uint16,
                    )

                full_ids[-1] = np.uint16(
                    eod_id
                )

                take = (
                    document.take_tokens
                )

                start = (
                    document.stream_offset
                )

                end = (
                    document.stream_end
                )

                output[
                    start:end
                ] = full_ids[
                    :take
                ]

                written_tokens += (
                    take
                )

        global_row_offset = (
            batch_end
        )

    if (
        pointer
        != len(
            documents
        )
    ):
        missing = (
            documents[
                pointer
            ]
        )

        raise RuntimeError(
            "Processed scan ended before "
            "all selected documents were found. "
            f"Next missing document="
            f"{missing.document_id}, "
            f"processed_row="
            f"{missing.processed_row}"
        )

    return written_tokens


# ============================================================
# Main builder
# ============================================================


def build_fast_token_cache(
    *,
    repo_root: str | Path,
    manifest_path: str | Path,
    tokenizer_path: str | Path,
    output_path: str | Path,
    target_tokens: int,
    eod_id: int,
    vocab_size: int,
    batch_size: int = 16_384,
    num_threads: int = 16,
) -> dict[str, Any]:
    """Build exact deterministic uint16 cache.

    Returns metadata suitable for JSON serialization.
    """

    repo_root = Path(
        repo_root
    ).resolve()

    manifest_path = Path(
        manifest_path
    ).resolve()

    tokenizer_path = Path(
        tokenizer_path
    ).resolve()

    output_path = Path(
        output_path
    ).resolve()

    if target_tokens <= 0:
        raise ValueError(
            "target_tokens must be positive"
        )

    if (
        vocab_size <= 0
        or vocab_size
        > np.iinfo(
            np.uint16
        ).max + 1
    ):
        raise ValueError(
            "vocab_size must fit uint16."
        )

    if not (
        0
        <= eod_id
        < vocab_size
    ):
        raise ValueError(
            "eod_id must be inside vocabulary."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    if num_threads <= 0:
        raise ValueError(
            "num_threads must be positive."
        )

    if not tokenizer_path.is_file():
        raise FileNotFoundError(
            f"Tokenizer missing: "
            f"{tokenizer_path}"
        )

    documents = load_cache_documents(
        manifest_path,
        target_tokens=(
            target_tokens
        ),
    )

    groups = (
        group_documents_by_processed_file(
            documents
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = Path(
        str(output_path)
        + ".partial"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    # Exactly target_tokens uint16 values.
    output = np.memmap(
        temporary_path,
        mode="w+",
        dtype=np.uint16,
        shape=(
            target_tokens,
        ),
    )

    processor = (
        spm.SentencePieceProcessor(
            model_file=str(
                tokenizer_path
            )
        )
    )

    written_tokens = 0

    try:

        for (
            processed_file,
            selected_documents,
        ) in groups.items():

            processed_path = (
                resolve_processed_path(
                    processed_file,
                    repo_root=(
                        repo_root
                    ),
                )
            )

            written_tokens += (
                write_processed_file_to_cache(
                    processed_path=(
                        processed_path
                    ),
                    documents=(
                        selected_documents
                    ),
                    output=(
                        output
                    ),
                    processor=(
                        processor
                    ),
                    eod_id=(
                        eod_id
                    ),
                    vocab_size=(
                        vocab_size
                    ),
                    batch_size=(
                        batch_size
                    ),
                    num_threads=(
                        num_threads
                    ),
                )
            )

        if (
            written_tokens
            != target_tokens
        ):
            raise RuntimeError(
                "Final cache token count mismatch: "
                f"{written_tokens:,} "
                f"!= {target_tokens:,}"
            )

        output.flush()

        del output

        expected_bytes = (
            target_tokens
            * np.dtype(
                np.uint16
            ).itemsize
        )

        actual_bytes = (
            temporary_path
            .stat()
            .st_size
        )

        if (
            actual_bytes
            != expected_bytes
        ):
            raise RuntimeError(
                "Cache byte-size mismatch: "
                f"{actual_bytes:,} "
                f"!= {expected_bytes:,}"
            )

        # Hash before promotion so a partially
        # written artifact can never appear at
        # the final cache path.
        cache_sha256 = (
            sha256_file(
                temporary_path
            )
        )

        os.replace(
            temporary_path,
            output_path,
        )

    except BaseException:

        try:
            del output
        except Exception:
            pass

        if temporary_path.exists():
            temporary_path.unlink()

        raise

    final_document = (
        documents[-1]
    )

    metadata = {
        "schema_version": 1,
        "dtype": "uint16",
        "target_tokens": (
            target_tokens
        ),
        "bytes": (
            target_tokens
            * 2
        ),
        "documents_consumed": (
            len(
                documents
            )
        ),
        "manifest_selected_documents": (
            pq.ParquetFile(
                manifest_path
            ).metadata.num_rows
        ),
        "manifest_path": (
            str(
                manifest_path
            )
        ),
        "manifest_sha256": (
            sha256_file(
                manifest_path
            )
        ),
        "tokenizer_path": (
            str(
                tokenizer_path
            )
        ),
        "tokenizer_sha256": (
            sha256_file(
                tokenizer_path
            )
        ),
        "vocab_size": (
            vocab_size
        ),
        "eod_id": (
            eod_id
        ),
        "processed_files": (
            sorted(
                groups.keys()
            )
        ),
        "cache_path": (
            str(
                output_path
            )
        ),
        "cache_sha256": (
            cache_sha256
        ),
        "final_document": {
            "sampling_order": (
                final_document
                .sampling_order
            ),
            "document_id": (
                final_document
                .document_id
            ),
            "full_token_count": (
                final_document
                .token_count
            ),
            "tokens_consumed": (
                final_document
                .take_tokens
            ),
            "partial": (
                final_document
                .is_partial
            ),
            "processed_file": (
                final_document
                .processed_file
            ),
            "processed_row": (
                final_document
                .processed_row
            ),
        },
    }

    return metadata