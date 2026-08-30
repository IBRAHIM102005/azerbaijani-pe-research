"""Audit large LSH buckets independently of the accepted cluster graph."""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.config import load_config
from src.data.dedup import character_shingles
from src.data.hashing import atomic_write_json
from src.data.split import assign_split


LOGGER = logging.getLogger("data_pipeline.audit_near_buckets")
BAND_DTYPE = np.dtype([("key", "S16"), ("rowid", "<u8")])
SAMPLE_SIZES = (201, 247, 291, 352, 407, 623, 799, 940)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit exact near-pair recall in frozen large LSH buckets.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "frozen" / "data_pipeline.yaml")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("sample", "all-large"), default="sample")
    parser.add_argument("--include-pairs", action="store_true")
    return parser.parse_args()


def large_bucket_events(band_paths: list[Path], scope: str) -> tuple[list[dict], dict]:
    """Read deterministic large bucket events from the frozen band files."""

    events = []
    selected_sizes: set[int] = set()
    distribution: dict[str, int] = {}
    scanned = 0
    maximum = 0
    started = time.perf_counter()
    for band_index, path in enumerate(band_paths):
        records = np.memmap(path, dtype=BAND_DTYPE, mode="r")
        order = np.argsort(records, order=("key", "rowid"), kind="stable")
        start = 0
        while start < len(order):
            end = start + 1
            key = records[order[start]]["key"]
            while end < len(order) and records[order[end]]["key"] == key:
                end += 1
            size = end - start
            if size > 1:
                scanned += 1
                maximum = max(maximum, size)
                label = (
                    "2-10" if size <= 10 else
                    "11-50" if size <= 50 else
                    "51-200" if size <= 200 else
                    "201-400" if size <= 400 else
                    "401-700" if size <= 700 else
                    "701-1000" if size <= 1000 else ">1000"
                )
                distribution[label] = distribution.get(label, 0) + 1
                keep = size > 200 and (
                    scope == "all-large" or (size in SAMPLE_SIZES and size not in selected_sizes)
                )
                if keep:
                    events.append(
                        {
                            "band": band_index,
                            "key": bytes(key).hex(),
                            "rowids": sorted(
                                int(records[order[index]]["rowid"])
                                for index in range(start, end)
                            ),
                        }
                    )
                    selected_sizes.add(size)
            start = end
        del order, records
        LOGGER.info(
            "stage=large_bucket_scan band=%d events=%d elapsed_seconds=%.1f",
            band_index,
            len(events),
            time.perf_counter() - started,
        )
    if scope == "sample" and selected_sizes != set(SAMPLE_SIZES):
        raise RuntimeError(f"Could not reproduce frozen sample sizes: {sorted(selected_sizes)}")
    return events, {
        "colliding_bucket_events": scanned,
        "large_bucket_events": sum(distribution.get(key, 0) for key in ("201-400", "401-700", "701-1000", ">1000")),
        "maximum_bucket_size": maximum,
        "bucket_distribution": distribution,
    }


def _load_documents(connection: sqlite3.Connection, rowids: list[int]) -> dict[int, tuple[str, int, str]]:
    documents = {}
    for offset in range(0, len(rowids), 500):
        chunk = rowids[offset : offset + 500]
        marks = ",".join("?" for _ in chunk)
        for rowid, text, length, document_id in connection.execute(
            f"SELECT rowid, text, character_count, document_id FROM documents WHERE rowid IN ({marks})",
            chunk,
        ):
            documents[rowid] = (text, length, document_id)
    if len(documents) != len(set(rowids)):
        raise RuntimeError("A large-bucket rowid has no document record")
    return documents


def audit_events(
    connection: sqlite3.Connection,
    events: list[dict],
    *,
    threshold: float,
    shingle_size: int,
    split_seed: int,
    split_settings: dict,
    include_pairs: bool,
) -> dict:
    """Evaluate every pair in the selected bucket events with direct Jaccard."""

    table_names = {
        row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    member_info = (
        {
            rowid: (cluster_id, representative_rowid, removed)
            for rowid, cluster_id, representative_rowid, removed in connection.execute(
                "SELECT rowid, cluster_id, representative_rowid, removed FROM near_members"
            )
        }
        if "near_members" in table_names
        else {}
    )
    has_candidates = "near_candidates" in table_names
    has_edges = "near_edges" in table_names
    document_cache: dict[int, tuple[str, int, str]] = {}
    representative_shingles: dict[int, set[str]] = {}

    def documents_for(rowids: list[int]) -> dict[int, tuple[str, int, str]]:
        missing = [rowid for rowid in rowids if rowid not in document_cache]
        if missing:
            document_cache.update(_load_documents(connection, missing))
        return {rowid: document_cache[rowid] for rowid in rowids}

    totals = {
        "pair_opportunities": 0,
        "true_jaccard_pairs": 0,
        "eligible_true_pairs": 0,
        "candidate_true_pairs": 0,
        "accepted_edge_pairs": 0,
    }
    bucket_results = []
    confirmed_representatives: dict[tuple[int, int], dict] = {}
    started = time.perf_counter()
    for event_index, event in enumerate(events, start=1):
        rowids = event["rowids"]
        documents = documents_for(rowids)
        shingles = {
            rowid: character_shingles(documents[rowid][0], shingle_size) for rowid in rowids
        }
        bucket = {"band": event["band"], "key": event["key"], "size": len(rowids), "pairs": 0, "true": 0, "eligible": 0, "candidates": 0, "edges": 0}
        for left, right in combinations(rowids, 2):
            bucket["pairs"] += 1
            left_set = shingles[left]
            right_set = shingles[right]
            union_size = len(left_set | right_set)
            similarity = len(left_set & right_set) / union_size if union_size else 1.0
            if similarity < threshold:
                continue
            bucket["true"] += 1
            left_length = documents[left][1]
            right_length = documents[right][1]
            if min(left_length, right_length) / max(left_length, right_length) < threshold:
                continue
            bucket["eligible"] += 1
            if has_candidates and connection.execute(
                "SELECT 1 FROM near_candidates WHERE left_rowid = ? AND right_rowid = ?",
                (left, right),
            ).fetchone():
                bucket["candidates"] += 1
            if has_edges and connection.execute(
                "SELECT 1 FROM near_edges WHERE left_rowid = ? AND right_rowid = ?",
                (left, right),
            ).fetchone():
                bucket["edges"] += 1

            left_rep = member_info.get(left, (None, left, 0))[1]
            right_rep = member_info.get(right, (None, right, 0))[1]
            if left_rep == right_rep:
                continue
            pair = tuple(sorted((left_rep, right_rep)))
            if pair in confirmed_representatives:
                continue
            representative_docs = documents_for(list(pair))
            left_cluster = member_info.get(pair[0], (None, pair[0], 0))[0]
            right_cluster = member_info.get(pair[1], (None, pair[1], 0))[0]
            left_identifier = left_cluster or representative_docs[pair[0]][2]
            right_identifier = right_cluster or representative_docs[pair[1]][2]
            left_split = assign_split(left_identifier, split_seed, **split_settings)
            right_split = assign_split(right_identifier, split_seed, **split_settings)
            if left_split == right_split:
                continue
            if pair[0] not in representative_shingles:
                representative_shingles[pair[0]] = character_shingles(
                    representative_docs[pair[0]][0], shingle_size
                )
            if pair[1] not in representative_shingles:
                representative_shingles[pair[1]] = character_shingles(
                    representative_docs[pair[1]][0], shingle_size
                )
            left_rep_set = representative_shingles[pair[0]]
            right_rep_set = representative_shingles[pair[1]]
            rep_union = len(left_rep_set | right_rep_set)
            rep_similarity = len(left_rep_set & right_rep_set) / rep_union if rep_union else 1.0
            rep_length_ratio = min(representative_docs[pair[0]][1], representative_docs[pair[1]][1]) / max(
                representative_docs[pair[0]][1], representative_docs[pair[1]][1]
            )
            if rep_similarity >= threshold and rep_length_ratio >= threshold:
                confirmed_representatives[pair] = {
                    "left_rowid": pair[0],
                    "right_rowid": pair[1],
                    "left_document_id": representative_docs[pair[0]][2],
                    "right_document_id": representative_docs[pair[1]][2],
                    "left_split": left_split,
                    "right_split": right_split,
                    "similarity": rep_similarity,
                    "origin_band": event["band"],
                    "origin_bucket_key": event["key"],
                }
        totals["pair_opportunities"] += bucket["pairs"]
        totals["true_jaccard_pairs"] += bucket["true"]
        totals["eligible_true_pairs"] += bucket["eligible"]
        totals["candidate_true_pairs"] += bucket["candidates"]
        totals["accepted_edge_pairs"] += bucket["edges"]
        bucket_results.append(bucket)
        LOGGER.info(
            "stage=large_bucket_exact bucket=%d/%d size=%d eligible=%d candidates=%d elapsed_seconds=%.1f",
            event_index,
            len(events),
            len(rowids),
            bucket["eligible"],
            bucket["candidates"],
            time.perf_counter() - started,
        )

    eligible = totals["eligible_true_pairs"]
    result = {
        **totals,
        "measured_candidate_recall": totals["candidate_true_pairs"] / eligible if eligible else 1.0,
        "cross_split_retained_representative_pairs": len(confirmed_representatives),
        "bucket_results": bucket_results,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    pairs = list(confirmed_representatives.values())
    result["confirmed_pair_preview"] = pairs[:10]
    if include_pairs:
        result["confirmed_pairs"] = pairs
    return result


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config, ROOT)
    settings = config.values["near_duplicate"]
    band_paths = [
        config.path("interim") / "near_bands" / f"band-{index:02d}.bin"
        for index in range(settings["bands"])
    ]
    events, scan = large_bucket_events(band_paths, args.scope)
    database = args.database.resolve()
    connection = sqlite3.connect("file:" + database.as_posix() + "?mode=ro", uri=True)
    try:
        audit = audit_events(
            connection,
            events,
            threshold=settings["selected_threshold"],
            shingle_size=settings["shingle_size"],
            split_seed=config.values["seeds"]["split"],
            split_settings=config.values["split"],
            include_pairs=args.include_pairs,
        )
    finally:
        connection.close()
    result = {
        "scope": args.scope,
        "database": args.database.name,
        "threshold": settings["selected_threshold"],
        "shingle_size": settings["shingle_size"],
        "bucket_scan": scan,
        "audit": audit,
    }
    atomic_write_json(args.output, result)
    LOGGER.info("stage=large_bucket_audit_complete output=%s", args.output)


if __name__ == "__main__":
    main()
