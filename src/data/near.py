"""Disk-backed near-duplicate candidate generation and clustering."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import struct
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .dedup import bottom_k_fingerprint, character_shingles, fingerprint_bands
from .hashing import sha256_file, sha256_text


LOGGER = logging.getLogger("data_pipeline.near")
_BAND_DTYPE = np.dtype([("key", "S16"), ("rowid", "<u8")])


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, item: int) -> int:
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            smaller, larger = sorted((left_root, right_root))
            self.parent[larger] = smaller


def _band_key(values: tuple[int, ...]) -> bytes:
    packed = b"".join(struct.pack(">Q", value & ((1 << 64) - 1)) for value in values)
    return hashlib.sha256(packed).digest()[:16]


def _ensure_stage_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS m1_stage_state (
            stage TEXT PRIMARY KEY,
            version TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            status TEXT NOT NULL,
            details_json TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL
        ) WITHOUT ROWID
        """
    )


def _update_stage(
    connection: sqlite3.Connection,
    stage: str,
    version: str,
    config_hash: str,
    status: str,
    details: dict[str, Any],
) -> None:
    _ensure_stage_table(connection)
    connection.execute(
        "INSERT OR REPLACE INTO m1_stage_state VALUES (?, ?, ?, ?, ?, ?)",
        (
            stage,
            version,
            config_hash,
            status,
            json.dumps(details, sort_keys=True, separators=(",", ":")),
            datetime.now(UTC).isoformat(),
        ),
    )
    connection.commit()


def stage_is_complete(
    connection: sqlite3.Connection, stage: str, config_hash: str
) -> bool:
    """Return true only for a matching, explicitly completed stage."""

    _ensure_stage_table(connection)
    row = connection.execute(
        "SELECT status, config_hash FROM m1_stage_state WHERE stage = ?", (stage,)
    ).fetchone()
    return bool(row and row[0] == "complete" and row[1] == config_hash)


def build_band_files(
    connection: sqlite3.Connection,
    output_dir: Path,
    *,
    shingle_size: int,
    fingerprint_size: int,
    bands: int,
    batch_size: int = 2048,
) -> tuple[list[Path], int]:
    """Write compact LSH band records for exact-unique documents."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"band-{index:02d}.bin" for index in range(bands)]
    handles = [path.open("wb") for path in paths]
    buffers: list[list[tuple[bytes, int]]] = [[] for _ in range(bands)]
    count = 0
    try:
        cursor = connection.execute("SELECT rowid, text FROM documents ORDER BY rowid")
        while rows := cursor.fetchmany(batch_size):
            for rowid, text in rows:
                fingerprint = bottom_k_fingerprint(text, shingle_size, fingerprint_size)
                for band_index, band in enumerate(fingerprint_bands(fingerprint, bands)):
                    buffers[band_index].append((_band_key(band), rowid))
                count += 1
            for band_index, buffer in enumerate(buffers):
                if not buffer:
                    continue
                array = np.empty(len(buffer), dtype=_BAND_DTYPE)
                array["key"] = [item[0] for item in buffer]
                array["rowid"] = [item[1] for item in buffer]
                array.tofile(handles[band_index])
                buffer.clear()
            if count % 100_000 == 0:
                LOGGER.info("stage=near_fingerprint documents=%d", count)
    finally:
        for handle in handles:
            handle.close()
    return paths, count


def build_candidate_table(
    connection: sqlite3.Connection,
    band_paths: list[Path],
    *,
    max_bucket_size: int,
    batch_size: int = 50_000,
) -> dict[str, Any]:
    """Create complete bucket pairs with bounded memory and band checkpoints."""

    strategy_hash = sha256_text(f"complete-pairs-v2\0{max_bucket_size}\0{_BAND_DTYPE.descr}")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS near_candidates (
            left_rowid INTEGER NOT NULL,
            right_rowid INTEGER NOT NULL,
            PRIMARY KEY(left_rowid, right_rowid)
        ) WITHOUT ROWID
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS near_candidate_progress (
            band_index INTEGER PRIMARY KEY,
            band_sha256 TEXT NOT NULL,
            strategy_hash TEXT NOT NULL,
            status TEXT NOT NULL,
            colliding_buckets INTEGER NOT NULL,
            complete_buckets INTEGER NOT NULL,
            skipped_buckets INTEGER NOT NULL,
            maximum_bucket_size INTEGER NOT NULL,
            pair_attempts INTEGER NOT NULL,
            completed_at_utc TEXT NOT NULL
        )
        """
    )
    _update_stage(connection, "near_candidates", "2", strategy_hash, "running", {})

    band_hashes: dict[str, str] = {}
    for band_index, path in enumerate(band_paths):
        band_hash = sha256_file(path)
        band_hashes[path.name] = band_hash
        progress = connection.execute(
            """SELECT band_sha256, strategy_hash, status
               FROM near_candidate_progress WHERE band_index = ?""",
            (band_index,),
        ).fetchone()
        if progress:
            if progress[0] != band_hash or progress[1] != strategy_hash:
                raise RuntimeError(f"Near-candidate checkpoint mismatch for band {band_index}")
            if progress[2] == "complete":
                LOGGER.info("stage=near_candidates band=%d status=resume_skip", band_index)
                continue

        records = np.memmap(path, dtype=_BAND_DTYPE, mode="r")
        order = np.argsort(records, order=("key", "rowid"), kind="stable")
        batch: list[tuple[int, int]] = []
        colliding_buckets = 0
        complete_buckets = 0
        skipped_buckets = 0
        maximum_bucket_size = 0
        pair_attempts = 0
        start = 0
        while start < len(order):
            end = start + 1
            key = records[order[start]]["key"]
            while end < len(order) and records[order[end]]["key"] == key:
                end += 1
            size = end - start
            if size > 1:
                colliding_buckets += 1
                maximum_bucket_size = max(maximum_bucket_size, size)
                rowids = sorted(int(records[order[index]]["rowid"]) for index in range(start, end))
                if size > max_bucket_size:
                    skipped_buckets += 1
                else:
                    complete_buckets += 1
                    pair_attempts += size * (size - 1) // 2
                    for left in range(size - 1):
                        left_rowid = rowids[left]
                        for right in range(left + 1, size):
                            batch.append((left_rowid, rowids[right]))
                            if len(batch) >= batch_size:
                                connection.executemany(
                                    "INSERT OR IGNORE INTO near_candidates VALUES (?, ?)", batch
                                )
                                connection.commit()
                                batch.clear()
            start = end
        if batch:
            connection.executemany("INSERT OR IGNORE INTO near_candidates VALUES (?, ?)", batch)
            connection.commit()
        del order
        del records
        connection.execute(
            """
            INSERT OR REPLACE INTO near_candidate_progress
            VALUES (?, ?, ?, 'complete', ?, ?, ?, ?, ?, ?)
            """,
            (
                band_index,
                band_hash,
                strategy_hash,
                colliding_buckets,
                complete_buckets,
                skipped_buckets,
                maximum_bucket_size,
                pair_attempts,
                datetime.now(UTC).isoformat(),
            ),
        )
        connection.commit()
        LOGGER.info(
            "stage=near_candidates band=%d buckets=%d pair_attempts=%d",
            band_index,
            colliding_buckets,
            pair_attempts,
        )

    totals = connection.execute(
        """
        SELECT COALESCE(SUM(colliding_buckets), 0),
               COALESCE(SUM(complete_buckets), 0),
               COALESCE(SUM(skipped_buckets), 0),
               COALESCE(MAX(maximum_bucket_size), 0),
               COALESCE(SUM(pair_attempts), 0)
        FROM near_candidate_progress
        WHERE status = 'complete' AND strategy_hash = ?
        """,
        (strategy_hash,),
    ).fetchone()
    unique_pairs = connection.execute("SELECT COUNT(*) FROM near_candidates").fetchone()[0]
    result = {
        "strategy": "complete_bucket_pairs",
        "strategy_version": 2,
        "strategy_hash": strategy_hash,
        "maximum_supported_bucket_size": max_bucket_size,
        "total_colliding": totals[0],
        "complete": totals[1],
        "star": 0,
        "skipped": totals[2],
        "maximum_observed_bucket_size": totals[3],
        "pair_insert_attempts": totals[4],
        "unique_candidate_pairs": unique_pairs,
        "band_hashes": band_hashes,
    }
    status = "complete" if not result["skipped"] else "failed"
    _update_stage(connection, "near_candidates", "2", strategy_hash, status, result)
    return result


def verify_candidate_edges(
    connection: sqlite3.Connection,
    *,
    threshold: float,
    shingle_size: int,
    cache_size: int = 20_000,
    checkpoint_size: int = 25_000,
    maximum_cached_shingles: int = 500_000,
) -> dict[str, Any]:
    """Verify candidates exactly and resume from the last committed pair."""

    verification_hash = sha256_text(f"exact-jaccard-v2\0{threshold:.17g}\0{shingle_size}")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS near_edges (
            left_rowid INTEGER NOT NULL,
            right_rowid INTEGER NOT NULL,
            similarity REAL NOT NULL,
            PRIMARY KEY(left_rowid,right_rowid)
        ) WITHOUT ROWID
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS near_verification_progress (
            singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
            verification_hash TEXT NOT NULL,
            status TEXT NOT NULL,
            last_left_rowid INTEGER NOT NULL,
            last_right_rowid INTEGER NOT NULL,
            checked_pairs INTEGER NOT NULL,
            accepted_edges INTEGER NOT NULL,
            completed_at_utc TEXT
        )
        """
    )
    progress = connection.execute(
        """SELECT verification_hash, status, last_left_rowid, last_right_rowid,
                  checked_pairs, accepted_edges
           FROM near_verification_progress WHERE singleton = 1"""
    ).fetchone()
    if progress and progress[0] != verification_hash:
        raise RuntimeError("Near-verification checkpoint does not match the frozen threshold")
    if progress and progress[1] == "complete":
        return {
            "candidate_pairs_checked": progress[4],
            "accepted_edges": progress[5],
            "verification_hash": verification_hash,
            "resumed": True,
        }

    last_left = progress[2] if progress else -1
    last_right = progress[3] if progress else -1
    checked = progress[4] if progress else 0
    accepted = progress[5] if progress else 0
    _update_stage(
        connection,
        "near_verification",
        "2",
        verification_hash,
        "running",
        {"checked_pairs": checked, "accepted_edges": accepted},
    )

    documents: OrderedDict[int, tuple[str, int]] = OrderedDict()
    shingles: OrderedDict[int, set[str]] = OrderedDict()
    cached_shingle_count = 0

    def get_document(rowid: int) -> tuple[str, int]:
        if rowid in documents:
            documents.move_to_end(rowid)
            return documents[rowid]
        row = connection.execute(
            "SELECT text, character_count FROM documents WHERE rowid = ?", (rowid,)
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Missing near-duplicate document rowid {rowid}")
        documents[rowid] = row
        if len(documents) > cache_size:
            documents.popitem(last=False)
        return row

    def get_shingles(rowid: int, text: str) -> set[str]:
        nonlocal cached_shingle_count
        if rowid in shingles:
            shingles.move_to_end(rowid)
            return shingles[rowid]
        value = character_shingles(text, shingle_size)
        if len(value) <= maximum_cached_shingles:
            while shingles and cached_shingle_count + len(value) > maximum_cached_shingles:
                _, evicted = shingles.popitem(last=False)
                cached_shingle_count -= len(evicted)
            shingles[rowid] = value
            cached_shingle_count += len(value)
        return value

    query = """
        SELECT left_rowid, right_rowid
        FROM near_candidates
        WHERE left_rowid > ? OR (left_rowid = ? AND right_rowid > ?)
        ORDER BY left_rowid, right_rowid
    """
    edge_batch: list[tuple[int, int, float]] = []
    since_checkpoint = 0
    for left_rowid, right_rowid in connection.execute(query, (last_left, last_left, last_right)):
        left_text, left_length = get_document(left_rowid)
        right_text, right_length = get_document(right_rowid)
        checked += 1
        since_checkpoint += 1
        if min(left_length, right_length) / max(left_length, right_length) >= threshold:
            left_set = get_shingles(left_rowid, left_text)
            right_set = get_shingles(right_rowid, right_text)
            union_size = len(left_set | right_set)
            similarity = len(left_set & right_set) / union_size if union_size else 1.0
            if similarity >= threshold:
                edge_batch.append((left_rowid, right_rowid, similarity))
                accepted += 1
        last_left, last_right = left_rowid, right_rowid
        if since_checkpoint >= checkpoint_size:
            if edge_batch:
                connection.executemany(
                    "INSERT OR REPLACE INTO near_edges VALUES (?, ?, ?)", edge_batch
                )
                edge_batch.clear()
            connection.execute(
                """INSERT OR REPLACE INTO near_verification_progress
                   VALUES (1, ?, 'running', ?, ?, ?, ?, NULL)""",
                (verification_hash, last_left, last_right, checked, accepted),
            )
            connection.commit()
            since_checkpoint = 0
            if checked % 100_000 < checkpoint_size:
                LOGGER.info("stage=near_verify pairs=%d accepted=%d", checked, accepted)

    if edge_batch:
        connection.executemany("INSERT OR REPLACE INTO near_edges VALUES (?, ?, ?)", edge_batch)
    connection.execute(
        """INSERT OR REPLACE INTO near_verification_progress
           VALUES (1, ?, 'complete', ?, ?, ?, ?, ?)""",
        (
            verification_hash,
            last_left,
            last_right,
            checked,
            accepted,
            datetime.now(UTC).isoformat(),
        ),
    )
    connection.commit()
    result = {
        "candidate_pairs_checked": checked,
        "accepted_edges": accepted,
        "verification_hash": verification_hash,
        "resumed": progress is not None,
    }
    _update_stage(connection, "near_verification", "2", verification_hash, "complete", result)
    return result


def build_clusters(connection: sqlite3.Connection) -> dict[str, int]:
    """Build deterministic connected components from accepted exact edges."""

    union = UnionFind()
    for left_rowid, right_rowid in connection.execute(
        "SELECT left_rowid, right_rowid FROM near_edges ORDER BY left_rowid, right_rowid"
    ):
        union.union(left_rowid, right_rowid)

    groups: dict[int, list[int]] = {}
    for rowid in union.parent:
        groups.setdefault(union.find(rowid), []).append(rowid)
    connection.execute("DROP TABLE IF EXISTS near_members")
    connection.execute(
        """
        CREATE TABLE near_members (
            rowid INTEGER PRIMARY KEY,
            cluster_id TEXT NOT NULL,
            representative_rowid INTEGER NOT NULL,
            removed INTEGER NOT NULL
        )
        """
    )
    member_rows: list[tuple[int, str, int, int]] = []
    removed = 0
    for members in groups.values():
        marks = ",".join("?" for _ in members)
        identities = list(
            connection.execute(
                f"SELECT rowid, document_id FROM documents WHERE rowid IN ({marks})", members
            )
        )
        representative_rowid, representative_id = min(identities, key=lambda item: item[1])
        cluster_id = sha256_text(f"near\0{representative_id}")
        for rowid, _ in sorted(identities):
            is_removed = int(rowid != representative_rowid)
            removed += is_removed
            member_rows.append((rowid, cluster_id, representative_rowid, is_removed))
        if len(member_rows) >= 25_000:
            connection.executemany("INSERT INTO near_members VALUES (?, ?, ?, ?)", member_rows)
            connection.commit()
            member_rows.clear()
    if member_rows:
        connection.executemany("INSERT INTO near_members VALUES (?, ?, ?, ?)", member_rows)
    connection.commit()
    result = {
        "clusters": len(groups),
        "clustered_documents": len(union.parent),
        "removed_documents": removed,
    }
    _update_stage(connection, "near_clusters", "2", "connected-components-v1", "complete", result)
    return result


def verify_and_cluster(
    connection: sqlite3.Connection,
    *,
    threshold: float,
    shingle_size: int,
    cache_size: int = 20_000,
) -> dict[str, Any]:
    """Verify LSH candidates exactly and freeze connected components."""

    verified = verify_candidate_edges(
        connection,
        threshold=threshold,
        shingle_size=shingle_size,
        cache_size=cache_size,
    )
    return {**verified, **build_clusters(connection)}
