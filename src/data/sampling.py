"""Construct the deterministic source-capped 50M-token train subset."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from .hashing import sha256_file


SUBSET_SCHEMA = pa.schema(
    [
        ("sampling_order", pa.int64()),
        ("document_id", pa.string()),
        ("source", pa.string()),
        ("source_group", pa.string()),
        ("token_count", pa.int64()),
        ("selection_phase", pa.string()),
        ("sampling_rank", pa.string()),
        ("processed_file", pa.string()),
        ("processed_row", pa.int64()),
    ]
)


def enrich_processed_references(output_path: Path, train_manifest_path: Path) -> str:
    """Attach stable processed-row references without loading corpus text."""

    selected = pq.read_table(output_path).to_pylist()
    selected_by_id = {row["document_id"]: row for row in selected}
    found = 0
    parquet = pq.ParquetFile(train_manifest_path)
    for batch in parquet.iter_batches(batch_size=20_000, columns=["document_id", "processed_file", "processed_row"]):
        document_ids, files, row_numbers = [column.to_pylist() for column in batch.columns]
        for document_id, processed_file, processed_row in zip(document_ids, files, row_numbers):
            row = selected_by_id.get(document_id)
            if row is not None:
                row["processed_file"] = processed_file
                row["processed_row"] = processed_row
                found += 1
    if found != len(selected):
        raise RuntimeError(f"Resolved {found:,} of {len(selected):,} selected processed references")
    selected.sort(key=lambda row: row["sampling_order"])
    temporary = output_path.with_suffix(output_path.suffix + ".refs.tmp")
    writer = pq.ParquetWriter(temporary, SUBSET_SCHEMA, compression="zstd")
    try:
        for offset in range(0, len(selected), 10_000):
            writer.write_table(pa.Table.from_pylist(selected[offset : offset + 10_000], schema=SUBSET_SCHEMA))
    finally:
        writer.close()
    os.replace(temporary, output_path)
    return sha256_file(output_path)


def _rank(seed: int, document_id: str, label: str = "quota") -> str:
    return hashlib.sha256(f"{label}\0{seed}\0{document_id}".encode("utf-8")).hexdigest()


def exact_budget_boundary(rows: list[dict], target_tokens: int) -> dict:
    """Locate an exact consumed-token boundary in an ordered document stream."""

    cumulative = 0
    for index, row in enumerate(rows):
        next_total = cumulative + row["token_count"]
        if next_total >= target_tokens:
            consumed = target_tokens - cumulative
            return {
                "sampling_order_zero_based": index,
                "sequence_position_one_based": index + 1,
                "document_id": row["document_id"],
                "cumulative_tokens_before_document": cumulative,
                "full_document_tokens_including_eod": row["token_count"],
                "tokens_consumed_from_document": consumed,
                "unconsumed_document_tokens": row["token_count"] - consumed,
                "eod_consumed": consumed == row["token_count"],
                "stop_semantics": (
                    "Read documents in frozen sampling order, append one <eod> after each "
                    "document, and stop after exactly the target number of token IDs."
                ),
            }
        cumulative = next_total
    raise ValueError(f"Ordered documents contain only {cumulative} tokens, below {target_tokens}")


def load_token_counts(connection: sqlite3.Connection, token_counts_path: Path, seed: int) -> None:
    connection.execute("DROP TABLE IF EXISTS token_counts")
    connection.execute(
        """CREATE TABLE token_counts (
               document_id TEXT PRIMARY KEY, source TEXT NOT NULL, source_group TEXT NOT NULL,
               split TEXT NOT NULL, token_count INTEGER NOT NULL, sampling_rank TEXT NOT NULL
           ) WITHOUT ROWID"""
    )
    parquet = pq.ParquetFile(token_counts_path)
    rows = []
    for batch in parquet.iter_batches(batch_size=10_000):
        for document_id, source, group, split, token_count, _ in zip(
            *[column.to_pylist() for column in batch.columns]
        ):
            rows.append((document_id, source, group, split, token_count, _rank(seed, document_id)))
        connection.executemany("INSERT INTO token_counts VALUES (?, ?, ?, ?, ?, ?)", rows)
        connection.commit()
        rows.clear()
    connection.execute("CREATE INDEX token_counts_group_rank ON token_counts(split, source_group, sampling_rank)")
    connection.commit()


def _group_iterator(
    connection: sqlite3.Connection,
    group: str,
    source_order: list[str],
) -> Iterator[tuple[str, str, str, int, str]]:
    clauses = " ".join(f"WHEN ? THEN {index}" for index, _ in enumerate(source_order))
    query = f"""
        SELECT document_id, source, source_group, token_count, sampling_rank
        FROM token_counts
        WHERE split = 'train' AND source_group = ?
        ORDER BY CASE source {clauses} ELSE {len(source_order)} END, sampling_rank
    """
    parameters = [group, *source_order]
    yield from connection.execute(query, parameters)


def build_training_subset(
    connection: sqlite3.Connection,
    token_counts_path: Path,
    output_path: Path,
    settings: dict[str, Any],
    data_seed: int,
) -> dict[str, Any]:
    """Select unique train documents by frozen group quotas and redistribution rules."""

    load_token_counts(connection, token_counts_path, data_seed)
    target = int(settings["target_tokens"])
    weights = settings["requested_group_weights"]
    requested = {group: int(target * weight) for group, weight in weights.items()}
    requested[next(iter(requested))] += target - sum(requested.values())
    available = {
        group: connection.execute(
            "SELECT COALESCE(SUM(token_count), 0) FROM token_counts WHERE split = 'train' AND source_group = ?",
            (group,),
        ).fetchone()[0]
        for group in weights
    }
    iterators = {
        group: iter(_group_iterator(connection, group, settings["group_source_order"][group]))
        for group in weights
    }
    exhausted = {group: False for group in weights}
    group_totals = Counter()
    selected: list[dict[str, Any]] = []
    selected_ids = set()

    def take(group: str, target_total: int, phase: str) -> None:
        while group_totals[group] < target_total and not exhausted[group]:
            try:
                document_id, source, source_group, token_count, sampling_rank = next(iterators[group])
            except StopIteration:
                exhausted[group] = True
                break
            if document_id in selected_ids:
                raise RuntimeError(f"Quota sampler selected {document_id} twice")
            selected_ids.add(document_id)
            group_totals[group] += token_count
            selected.append(
                {
                    "document_id": document_id,
                    "source": source,
                    "source_group": source_group,
                    "token_count": token_count,
                    "selection_phase": phase,
                    "sampling_rank": sampling_rank,
                }
            )

    for group, quota in requested.items():
        take(group, quota, "requested_quota")
    quota_phase_totals = dict(group_totals)

    redistribution_round = 0
    while sum(group_totals.values()) < target:
        active = [
            group
            for group in settings["shortage_redistribution_weights"]
            if not exhausted[group]
        ]
        if not active:
            break
        remaining = target - sum(group_totals.values())
        active_weight = sum(settings["shortage_redistribution_weights"][group] for group in active)
        allocations = {
            group: int(remaining * settings["shortage_redistribution_weights"][group] / active_weight)
            for group in active
        }
        allocations[active[0]] += remaining - sum(allocations.values())
        before = sum(group_totals.values())
        redistribution_round += 1
        for group in active:
            take(group, group_totals[group] + allocations[group], f"redistribution_{redistribution_round}")
        if sum(group_totals.values()) == before:
            break

    selected_tokens = sum(group_totals.values())
    if selected_tokens < target:
        raise RuntimeError(
            f"Unique train data provides only {selected_tokens:,} selected tokens, below the {target:,} target"
        )

    selected.sort(key=lambda row: (_rank(data_seed, row["document_id"], "order"), row["document_id"]))
    boundary = exact_budget_boundary(selected, target)
    rows = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    writer = pq.ParquetWriter(temporary, SUBSET_SCHEMA, compression="zstd")
    source_totals = Counter()
    try:
        for order, row in enumerate(selected):
            row["sampling_order"] = order
            source_totals[row["source"]] += row["token_count"]
            rows.append(row)
            if len(rows) >= 10_000:
                writer.write_table(pa.Table.from_pylist(rows, schema=SUBSET_SCHEMA))
                rows.clear()
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=SUBSET_SCHEMA))
    finally:
        writer.close()
    os.replace(temporary, output_path)

    return {
        "target_tokens": target,
        "boundary_policy": "Select full documents; downstream training stops at the exact 50M model-token boundary without changing canonical documents.",
        "requested_group_tokens": requested,
        "available_group_tokens": available,
        "quota_phase_group_tokens": quota_phase_totals,
        "quota_shortages": {
            group: max(0, requested[group] - quota_phase_totals.get(group, 0))
            for group in requested
        },
        "actual_group_tokens": dict(group_totals),
        "actual_group_proportions": {
            group: tokens / selected_tokens for group, tokens in group_totals.items()
        },
        "actual_source_tokens": dict(source_totals),
        "selected_documents": len(selected),
        "selected_unique_tokens": selected_tokens,
        "overshoot_tokens": selected_tokens - target,
        "exact_consumption_boundary": boundary,
        "exhausted_groups": [group for group, value in exhausted.items() if value],
        "manifest_sha256": sha256_file(output_path),
        "data_seed": data_seed,
        "model_seed_affects_order": False,
    }
