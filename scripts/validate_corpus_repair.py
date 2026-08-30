"""Independently validate repaired manifests, leakage, and cluster behavior."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from itertools import combinations, islice
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.config import load_config
from src.data.dedup import character_shingles
from src.data.hashing import atomic_write_json
from src.data.paths import resolve_repository_path
from src.data.split import assign_split


SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the staged near-duplicate repair.")
    parser.add_argument(
        "--database", type=Path, default=ROOT / "data" / "interim" / "corpus" / "corpus_index_repair.sqlite"
    )
    parser.add_argument(
        "--manifests", type=Path, default=ROOT / "data" / "manifests_repair"
    )
    parser.add_argument(
        "--prerepair-evidence",
        type=Path,
        default=ROOT / "data" / "metadata" / "near_repair_prerepair_evidence.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "metadata" / "near_repair_validation.json",
    )
    return parser.parse_args()


def iter_column(path: Path, field: str):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=32_768, columns=[field]):
        yield from batch.column(0).to_pylist()


def manifest_audit(manifest_dir: Path, repo_root: Path) -> tuple[dict, dict[str, Path]]:
    paths = {split: manifest_dir / f"{split}.parquet" for split in SPLITS}
    counts = {}
    references = {}
    for split, path in paths.items():
        parquet = pq.ParquetFile(path)
        counts[split] = parquet.metadata.num_rows
        previous = None
        seen_reference = None
        for batch in parquet.iter_batches(
            batch_size=32_768,
            columns=["document_id", "processed_file", "processed_row"],
        ):
            document_ids, processed_files, processed_rows = [
                column.to_pylist() for column in batch.columns
            ]
            for document_id, processed_file, processed_row in zip(
                document_ids, processed_files, processed_rows
            ):
                if not document_id or processed_row is None or processed_row < 0:
                    raise RuntimeError(f"Invalid required manifest value in {split}")
                if previous is not None and document_id <= previous:
                    raise RuntimeError(f"{split} manifest is not strictly document-ID sorted")
                previous = document_id
                seen_reference = seen_reference or processed_file
                if processed_file != seen_reference:
                    raise RuntimeError(f"{split} has inconsistent processed-file references")
        resolved = resolve_repository_path(seen_reference, repo_root)
        references[split] = {
            "stored": seen_reference,
            "resolved": str(resolved),
            "is_relative": not Path(seen_reference).is_absolute(),
        }
        if not references[split]["is_relative"]:
            raise RuntimeError(f"{split} contains an absolute operational path")
    return {"counts": counts, "processed_references": references}, paths


def cross_split_checks(paths: dict[str, Path]) -> dict:
    results = {}
    for field in ("document_id", "canonical_text_hash", "duplicate_cluster_id"):
        validation = set(iter_column(paths["validation"], field))
        test = set(iter_column(paths["test"], field))
        if len(validation) != pq.ParquetFile(paths["validation"]).metadata.num_rows:
            raise RuntimeError(f"Duplicate {field} inside validation")
        if len(test) != pq.ParquetFile(paths["test"]).metadata.num_rows:
            raise RuntimeError(f"Duplicate {field} inside test")
        train_validation = 0
        train_test = 0
        previous = None
        train_rows = 0
        for value in iter_column(paths["train"], field):
            if previous is not None and value <= previous and field == "document_id":
                raise RuntimeError("Train document IDs are not strictly sorted")
            previous = value
            train_validation += value in validation
            train_test += value in test
            train_rows += 1
        intersections = {
            "train_validation": train_validation,
            "train_test": train_test,
            "validation_test": len(validation & test),
        }
        if any(intersections.values()):
            raise RuntimeError(f"Cross-split {field} leakage: {intersections}")
        results[field] = {"intersections": intersections, "train_rows_checked": train_rows}
    return results


def graph_audit(connection: sqlite3.Connection, threshold: float) -> dict:
    counts = {
        "candidate_pairs": connection.execute("SELECT COUNT(*) FROM near_candidates").fetchone()[0],
        "accepted_edges": connection.execute("SELECT COUNT(*) FROM near_edges").fetchone()[0],
        "clustered_documents": connection.execute("SELECT COUNT(*) FROM near_members").fetchone()[0],
        "clusters": connection.execute("SELECT COUNT(DISTINCT cluster_id) FROM near_members").fetchone()[0],
        "removed_documents": connection.execute(
            "SELECT COUNT(*) FROM near_members WHERE removed = 1"
        ).fetchone()[0],
    }
    bad_edges = connection.execute(
        "SELECT COUNT(*) FROM near_edges WHERE similarity < ?", (threshold,)
    ).fetchone()[0]
    cross_component_edges = connection.execute(
        """
        SELECT COUNT(*) FROM near_edges e
        JOIN near_members l ON l.rowid = e.left_rowid
        JOIN near_members r ON r.rowid = e.right_rowid
        WHERE l.cluster_id != r.cluster_id
        """
    ).fetchone()[0]
    representative_errors = connection.execute(
        """
        SELECT COUNT(*) FROM (
          SELECT cluster_id,
                 SUM(CASE WHEN removed = 0 THEN 1 ELSE 0 END) retained,
                 COUNT(DISTINCT representative_rowid) representatives
          FROM near_members GROUP BY cluster_id
          HAVING retained != 1 OR representatives != 1
        )
        """
    ).fetchone()[0]
    if bad_edges or cross_component_edges or representative_errors:
        raise RuntimeError("Repaired near graph failed an internal consistency assertion")
    return {
        **counts,
        "accepted_edges_below_threshold": bad_edges,
        "edges_crossing_components": cross_component_edges,
        "clusters_without_exactly_one_representative": representative_errors,
    }


def prerepair_pair_regression(
    connection: sqlite3.Connection, evidence_path: Path, split_seed: int, split_settings: dict
) -> dict:
    pairs = json.loads(evidence_path.read_text(encoding="utf-8"))["audit"]["confirmed_pairs"]
    outcomes = Counter()
    unresolved = []
    for pair in pairs:
        rowids = (pair["left_rowid"], pair["right_rowid"])
        state = {}
        for rowid in rowids:
            document_id = connection.execute(
                "SELECT document_id FROM documents WHERE rowid = ?", (rowid,)
            ).fetchone()[0]
            member = connection.execute(
                "SELECT cluster_id, representative_rowid, removed FROM near_members WHERE rowid = ?",
                (rowid,),
            ).fetchone()
            if member:
                identifier, representative, removed = member
            else:
                identifier, representative, removed = document_id, rowid, 0
            state[rowid] = {
                "identifier": identifier,
                "representative": representative,
                "removed": bool(removed),
                "split": assign_split(identifier, split_seed, **split_settings),
            }
        left, right = (state[rowids[0]], state[rowids[1]])
        if left["removed"] or right["removed"]:
            outcome = "resolved_by_removal"
        elif left["representative"] == right["representative"]:
            outcome = "resolved_same_representative"
        elif left["split"] == right["split"]:
            outcome = "resolved_same_split"
        else:
            outcome = "unresolved_cross_split"
            unresolved.append({**pair, "postrepair_left": left, "postrepair_right": right})
        outcomes[outcome] += 1
    if unresolved:
        raise RuntimeError(f"{len(unresolved)} prerepair confirmed pairs remain cross-split")
    return {
        "prerepair_pairs_checked": len(pairs),
        "outcomes": dict(sorted(outcomes.items())),
        "unresolved_cross_split": len(unresolved),
        "unresolved_pairs": unresolved,
    }


def jaccard(left: set[str], right: set[str]) -> float:
    union = len(left | right)
    return len(left & right) / union if union else 1.0


def transitivity_audit(connection: sqlite3.Connection, shingle_size: int) -> dict:
    sizes = [
        row[0]
        for row in connection.execute(
            "SELECT COUNT(*) FROM near_members GROUP BY cluster_id ORDER BY COUNT(*) DESC"
        )
    ]
    distribution = {
        "size_2": sum(size == 2 for size in sizes),
        "size_gt_2": sum(size > 2 for size in sizes),
        "maximum_size": max(sizes, default=0),
    }
    cluster_ids = [
        row[0]
        for row in connection.execute(
            """SELECT cluster_id FROM near_members GROUP BY cluster_id
               HAVING COUNT(*) > 2 ORDER BY COUNT(*) DESC, cluster_id LIMIT 25"""
        )
    ]
    sampled = []
    for cluster_id in cluster_ids:
        rows = connection.execute(
            """SELECT m.rowid, m.representative_rowid, d.text
               FROM near_members m JOIN documents d ON d.rowid = m.rowid
               WHERE m.cluster_id = ? ORDER BY m.rowid""",
            (cluster_id,),
        ).fetchall()
        shingles = {rowid: character_shingles(text, shingle_size) for rowid, _, text in rows}
        representative = rows[0][1]
        rep_values = [
            jaccard(shingles[representative], shingles[rowid])
            for rowid, _, _ in rows
            if rowid != representative
        ]
        pair_iter = combinations((rowid for rowid, _, _ in rows), 2)
        evaluated_pairs = list(islice(pair_iter, 5_000))
        endpoint_values = [jaccard(shingles[left], shingles[right]) for left, right in evaluated_pairs]
        sampled.append(
            {
                "cluster_id": cluster_id,
                "size": len(rows),
                "representative_rowid": representative,
                "representative_member_pairs": len(rep_values),
                "minimum_representative_member_similarity": min(rep_values, default=1.0),
                "arbitrary_pairs_evaluated": len(endpoint_values),
                "minimum_arbitrary_pair_similarity": min(endpoint_values, default=1.0),
                "contains_sampled_pair_below_threshold": any(value < 0.95 for value in endpoint_values),
            }
        )
    return {
        "cluster_distribution": distribution,
        "sample_policy": "The 25 largest clusters above size two; all representative-member pairs and up to 5,000 rowid-ordered endpoint pairs per cluster.",
        "sampled_clusters": len(sampled),
        "sampled_clusters_with_endpoint_below_threshold": sum(
            row["contains_sampled_pair_below_threshold"] for row in sampled
        ),
        "minimum_representative_member_similarity": min(
            (row["minimum_representative_member_similarity"] for row in sampled), default=1.0
        ),
        "minimum_arbitrary_endpoint_similarity": min(
            (row["minimum_arbitrary_pair_similarity"] for row in sampled), default=1.0
        ),
        "clusters": sampled,
        "semantics": (
            "Every accepted graph edge is direct Jaccard >= 0.95. Clusters are connected "
            "components, so transitive members are not required to be pairwise >= 0.95."
        ),
    }


def main() -> None:
    args = parse_args()
    config = load_config(repo_root=ROOT)
    manifest, paths = manifest_audit(args.manifests.resolve(), config.repo_root)
    intersections = cross_split_checks(paths)
    database = args.database.resolve()
    connection = sqlite3.connect("file:" + database.as_posix() + "?mode=ro", uri=True)
    try:
        graph = graph_audit(connection, config.values["near_duplicate"]["selected_threshold"])
        regression = prerepair_pair_regression(
            connection,
            args.prerepair_evidence.resolve(),
            config.values["seeds"]["split"],
            config.values["split"],
        )
        transitivity = transitivity_audit(
            connection, config.values["near_duplicate"]["shingle_size"]
        )
    finally:
        connection.close()
    total = sum(manifest["counts"].values())
    if total != 6_209_184 - graph["removed_documents"]:
        raise RuntimeError("Repaired retained-document accounting does not reconcile")
    result = {
        "status": "pass",
        "database": args.database.name,
        "manifest_directory": args.manifests.name,
        "manifest": manifest,
        "cross_split": intersections,
        "near_graph": graph,
        "prerepair_237_pair_regression": regression,
        "transitivity": transitivity,
        "retained_documents": total,
        "hard_gate_passed": True,
    }
    atomic_write_json(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "retained_documents": total,
        "near_graph": graph,
        "prerepair_regression": regression["outcomes"],
    }, indent=2))


if __name__ == "__main__":
    main()
