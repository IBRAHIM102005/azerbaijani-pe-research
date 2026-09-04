"""Frozen M1 token-stream utilities for M3 training."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import sentencepiece as spm
import torch
from torch.utils.data import Dataset

from src.tokenizer.corpus import tokenizer_text


UINT16_MAX = np.iinfo(np.uint16).max


def sha256_file(
    path: str | Path,
    chunk_size: int = 1 << 20,
) -> str:
    """Return SHA-256 for a file."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(chunk_size),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


@dataclass(frozen=True)
class ConsumptionRecord:
    """One document's contribution to the exact token stream."""

    sampling_order: int
    document_id: str
    processed_file: str
    processed_row: int

    # Full token count from M1, including EOD.
    token_count: int

    # Position inside output token stream.
    output_start: int

    # May be smaller than token_count for final document.
    tokens_to_write: int


def build_consumption_plan(
    manifest_path: str | Path,
    target_tokens: int,
) -> list[ConsumptionRecord]:
    """Build exact M1 consumption plan.

    Documents are consumed in frozen sampling_order.

    The final document may be truncated so that the resulting stream
    contains exactly ``target_tokens`` token IDs.
    """

    if target_tokens <= 0:
        raise ValueError(
            "target_tokens must be positive"
        )

    manifest_path = Path(manifest_path)

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Training subset manifest not found: "
            f"{manifest_path}"
        )

    table = pq.read_table(
        manifest_path,
        columns=[
            "sampling_order",
            "document_id",
            "token_count",
            "processed_file",
            "processed_row",
        ],
    )

    rows = table.to_pylist()

    rows.sort(
        key=lambda row: row["sampling_order"]
    )

    plan: list[ConsumptionRecord] = []

    output_start = 0

    expected_order = 0

    for row in rows:
        if output_start >= target_tokens:
            break

        sampling_order = int(
            row["sampling_order"]
        )

        if sampling_order != expected_order:
            raise ValueError(
                "Non-contiguous sampling_order: "
                f"expected {expected_order}, "
                f"got {sampling_order}"
            )

        expected_order += 1

        token_count = int(
            row["token_count"]
        )

        if token_count <= 0:
            raise ValueError(
                f"Invalid token_count={token_count} "
                f"for {row['document_id']}"
            )

        processed_row = int(
            row["processed_row"]
        )

        if processed_row < 0:
            raise ValueError(
                "processed_row cannot be negative"
            )

        remaining = (
            target_tokens - output_start
        )

        tokens_to_write = min(
            token_count,
            remaining,
        )

        plan.append(
            ConsumptionRecord(
                sampling_order=sampling_order,
                document_id=str(
                    row["document_id"]
                ),
                processed_file=str(
                    row["processed_file"]
                ),
                processed_row=processed_row,
                token_count=token_count,
                output_start=output_start,
                tokens_to_write=tokens_to_write,
            )
        )

        output_start += tokens_to_write

    if output_start != target_tokens:
        raise ValueError(
            "Manifest does not contain enough tokens: "
            f"planned={output_start:,}, "
            f"target={target_tokens:,}"
        )

    return plan


def _group_plan_by_processed_file(
    plan: list[ConsumptionRecord],
) -> dict[str, dict[int, ConsumptionRecord]]:
    """Group selected rows by processed parquet path."""

    grouped: dict[
        str,
        dict[int, ConsumptionRecord],
    ] = {}

    for record in plan:
        rows = grouped.setdefault(
            record.processed_file,
            {},
        )

        if record.processed_row in rows:
            raise ValueError(
                "Duplicate processed-row reference: "
                f"{record.processed_file}:"
                f"{record.processed_row}"
            )

        rows[record.processed_row] = record

    return grouped


def build_token_cache(
    *,
    repo_root: str | Path,
    manifest_path: str | Path,
    tokenizer_path: str | Path,
    output_path: str | Path,
    target_tokens: int,
    eod_id: int,
    vocab_size: int,
    expected_manifest_sha256: str | None = None,
    expected_tokenizer_sha256: str | None = None,
    metadata_path: str | Path | None = None,
    batch_size: int = 8192,
    num_threads: int = 8,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build exact frozen token cache from M1 artifacts.

    The processed parquet is scanned sequentially.

    Selected documents are written directly into their final
    sampling-order offsets in a uint16 memory-mapped file.
    """

    repo_root = Path(repo_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    tokenizer_path = Path(tokenizer_path).resolve()
    output_path = Path(output_path).resolve()

    if metadata_path is None:
        metadata_path = output_path.with_suffix(
            output_path.suffix + ".json"
        )
    else:
        metadata_path = Path(
            metadata_path
        ).resolve()

    if vocab_size <= 0:
        raise ValueError(
            "vocab_size must be positive"
        )

    if vocab_size - 1 > UINT16_MAX:
        raise ValueError(
            f"vocab_size={vocab_size} does not fit "
            "inside uint16 token cache."
        )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            "M1 training subset artifact is missing: "
            f"{manifest_path}"
        )

    if not tokenizer_path.is_file():
        raise FileNotFoundError(
            "Frozen tokenizer is missing: "
            f"{tokenizer_path}"
        )

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Token cache already exists: "
            f"{output_path}. "
            "Use overwrite=True to rebuild it."
        )

    manifest_sha256 = sha256_file(
        manifest_path
    )

    tokenizer_sha256 = sha256_file(
        tokenizer_path
    )

    if (
        expected_manifest_sha256 is not None
        and manifest_sha256
        != expected_manifest_sha256
    ):
        raise ValueError(
            "M1 manifest SHA-256 mismatch."
        )

    if (
        expected_tokenizer_sha256 is not None
        and tokenizer_sha256
        != expected_tokenizer_sha256
    ):
        raise ValueError(
            "Tokenizer SHA-256 mismatch."
        )

    processor = spm.SentencePieceProcessor(
        model_file=str(tokenizer_path)
    )

    if processor.vocab_size() != vocab_size:
        raise ValueError(
            "Tokenizer vocabulary mismatch: "
            f"expected={vocab_size}, "
            f"actual={processor.vocab_size()}"
        )

    if processor.id_to_piece(eod_id) != "<eod>":
        raise ValueError(
            f"Token id {eod_id} is not <eod>."
        )

    plan = build_consumption_plan(
        manifest_path,
        target_tokens,
    )

    grouped = _group_plan_by_processed_file(
        plan
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(metadata_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    if temp_path.exists():
        temp_path.unlink()

    # Pre-allocate exactly target_tokens uint16 values.
    cache = np.memmap(
        temp_path,
        dtype=np.uint16,
        mode="w+",
        shape=(target_tokens,),
    )

    found_documents = 0
    written_tokens = 0

    try:
        for (
            processed_relative_path,
            selected_rows,
        ) in grouped.items():

            processed_path = (
                repo_root
                / processed_relative_path
            ).resolve()

            if not processed_path.is_file():
                raise FileNotFoundError(
                    "Processed M1 corpus artifact "
                    f"is missing: {processed_path}"
                )

            parquet = pq.ParquetFile(
                processed_path
            )

            row_offset = 0

            for batch in parquet.iter_batches(
                batch_size=batch_size,
                columns=[
                    "document_id",
                    "text",
                ],
            ):
                batch_size_actual = len(batch)

                batch_end = (
                    row_offset
                    + batch_size_actual
                )

                local_indices: list[int] = []
                records: list[
                    ConsumptionRecord
                ] = []

                # Find selected M1 rows that live
                # inside this parquet batch.
                for processed_row, record in (
                    selected_rows.items()
                ):
                    if (
                        row_offset
                        <= processed_row
                        < batch_end
                    ):
                        local_indices.append(
                            processed_row
                            - row_offset
                        )
                        records.append(
                            record
                        )

                if local_indices:
                    document_ids = (
                        batch.column(0)
                        .to_pylist()
                    )

                    texts = (
                        batch.column(1)
                        .to_pylist()
                    )

                    prepared_texts = [
                        tokenizer_text(
                            texts[index]
                        )
                        for index
                        in local_indices
                    ]

                    encoded_batch = (
                        processor.encode(
                            prepared_texts,
                            out_type=int,
                            num_threads=num_threads,
                        )
                    )

                    for (
                        local_index,
                        record,
                        token_ids,
                    ) in zip(
                        local_indices,
                        records,
                        encoded_batch,
                    ):
                        actual_document_id = (
                            document_ids[
                                local_index
                            ]
                        )

                        if (
                            actual_document_id
                            != record.document_id
                        ):
                            raise ValueError(
                                "Processed document ID "
                                "does not match manifest: "
                                f"expected="
                                f"{record.document_id}, "
                                f"actual="
                                f"{actual_document_id}"
                            )

                        # M1 token counts include
                        # exactly one EOD per document.
                        full_ids = list(
                            token_ids
                        )

                        full_ids.append(
                            eod_id
                        )

                        if (
                            len(full_ids)
                            != record.token_count
                        ):
                            raise ValueError(
                                "M1 token-count mismatch "
                                f"for {record.document_id}: "
                                f"manifest="
                                f"{record.token_count}, "
                                f"recomputed="
                                f"{len(full_ids)}"
                            )

                        consumed_ids = (
                            full_ids[
                                : record.tokens_to_write
                            ]
                        )

                        start = (
                            record.output_start
                        )

                        end = (
                            start
                            + record.tokens_to_write
                        )

                        cache[start:end] = np.asarray(
                            consumed_ids,
                            dtype=np.uint16,
                        )

                        found_documents += 1
                        written_tokens += (
                            record.tokens_to_write
                        )

                row_offset = batch_end

        if found_documents != len(plan):
            raise RuntimeError(
                "Not all M1 documents were found: "
                f"found={found_documents}, "
                f"expected={len(plan)}"
            )

        if written_tokens != target_tokens:
            raise RuntimeError(
                "Incorrect number of tokens written: "
                f"written={written_tokens:,}, "
                f"target={target_tokens:,}"
            )

        cache.flush()

    except Exception:
        del cache

        if temp_path.exists():
            temp_path.unlink()

        raise

    del cache

    os.replace(
        temp_path,
        output_path,
    )

    output_sha256 = sha256_file(
        output_path
    )

    tail_record = plan[-1]

    metadata = {
        "format": "raw_uint16_token_ids",
        "dtype": "uint16",
        "target_tokens": target_tokens,
        "bytes": output_path.stat().st_size,
        "cache_path": str(
            output_path.relative_to(
                repo_root
            )
            if output_path.is_relative_to(
                repo_root
            )
            else output_path
        ),
        "cache_sha256": output_sha256,
        "manifest_path": str(
            manifest_path
        ),
        "manifest_sha256": (
            manifest_sha256
        ),
        "tokenizer_path": str(
            tokenizer_path
        ),
        "tokenizer_sha256": (
            tokenizer_sha256
        ),
        "vocab_size": vocab_size,
        "eod_id": eod_id,
        "consumed_documents": len(plan),
        "final_document": {
            "document_id": (
                tail_record.document_id
            ),
            "sampling_order": (
                tail_record.sampling_order
            ),
            "full_document_tokens": (
                tail_record.token_count
            ),
            "tokens_consumed": (
                tail_record.tokens_to_write
            ),
            "eod_consumed": (
                tail_record.tokens_to_write
                == tail_record.token_count
            ),
        },
    }

    metadata_temp = Path(
        str(metadata_path) + ".tmp"
    )

    metadata_temp.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        metadata_temp,
        metadata_path,
    )

    return metadata


class TokenBlockDataset(Dataset):
    """Read full fixed-length blocks from a uint16 token cache.

    The final short tail is deliberately kept separate because
    DataLoader cannot stack a 128-token tail with 512-token blocks.
    """

    def __init__(
        self,
        cache_path: str | Path,
        *,
        total_tokens: int,
        seq_len: int = 512,
    ) -> None:
        if total_tokens <= 0:
            raise ValueError(
                "total_tokens must be positive"
            )

        if seq_len <= 1:
            raise ValueError(
                "seq_len must be greater than 1"
            )

        self.cache_path = Path(
            cache_path
        ).resolve()

        if not self.cache_path.is_file():
            raise FileNotFoundError(
                f"Token cache not found: "
                f"{self.cache_path}"
            )

        expected_bytes = (
            total_tokens
            * np.dtype(
                np.uint16
            ).itemsize
        )

        actual_bytes = (
            self.cache_path.stat().st_size
        )

        if actual_bytes != expected_bytes:
            raise ValueError(
                "Token-cache size mismatch: "
                f"expected={expected_bytes}, "
                f"actual={actual_bytes}"
            )

        self.total_tokens = total_tokens
        self.seq_len = seq_len

        self.full_blocks = (
            total_tokens // seq_len
        )

        self.tail_tokens = (
            total_tokens % seq_len
        )

        self._tokens = np.memmap(
            self.cache_path,
            dtype=np.uint16,
            mode="r",
            shape=(total_tokens,),
        )

    def __len__(self) -> int:
        """Number of full seq_len blocks."""

        return self.full_blocks

    def __getitem__(
        self,
        index: int,
    ) -> torch.Tensor:
        if index < 0:
            index += self.full_blocks

        if (
            index < 0
            or index >= self.full_blocks
        ):
            raise IndexError(index)

        start = (
            index * self.seq_len
        )

        end = (
            start + self.seq_len
        )

        # Copy avoids returning a tensor backed by
        # read-only memmap memory.
        array = np.array(
            self._tokens[start:end],
            dtype=np.int64,
            copy=True,
        )

        return torch.from_numpy(
            array
        )

    def tail(
        self,
    ) -> torch.Tensor | None:
        """Return final short token block, if one exists."""

        if self.tail_tokens == 0:
            return None

        start = (
            self.full_blocks
            * self.seq_len
        )

        array = np.array(
            self._tokens[
                start:self.total_tokens
            ],
            dtype=np.int64,
            copy=True,
        )

        return torch.from_numpy(
            array
        )


def expected_full_blocks(
    total_tokens: int,
    seq_len: int,
) -> tuple[int, int]:
    """Return (full_blocks, tail_tokens)."""

    if total_tokens <= 0:
        raise ValueError(
            "total_tokens must be positive"
        )

    if seq_len <= 0:
        raise ValueError(
            "seq_len must be positive"
        )

    return (
        total_tokens // seq_len,
        total_tokens % seq_len,
    )