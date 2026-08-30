"""Replay and validate repaired tokenizer, counts, and 50M sequence."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import sentencepiece as spm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.config import load_config
from src.data.hashing import atomic_write_json, sha256_file
from src.data.paths import resolve_repository_path


def rank(seed: int, document_id: str, label: str) -> str:
    return hashlib.sha256(f"{label}\0{seed}\0{document_id}".encode("utf-8")).hexdigest()


def replay_selection(connection: sqlite3.Connection, settings: dict, seed: int) -> list[dict]:
    target = settings["target_tokens"]
    weights = settings["requested_group_weights"]
    requested = {group: int(target * weight) for group, weight in weights.items()}
    requested[next(iter(requested))] += target - sum(requested.values())
    iterators = {}
    for group in weights:
        source_order = settings["group_source_order"][group]
        clauses = " ".join(f"WHEN ? THEN {index}" for index, _ in enumerate(source_order))
        query = f"""
            SELECT document_id, source, source_group, token_count
            FROM token_counts WHERE split = 'train' AND source_group = ?
            ORDER BY CASE source {clauses} ELSE {len(source_order)} END, sampling_rank
        """
        iterators[group] = iter(connection.execute(query, [group, *source_order]))
    exhausted = {group: False for group in weights}
    totals = Counter()
    selected = []

    def take(group: str, limit: int) -> None:
        while totals[group] < limit and not exhausted[group]:
            try:
                document_id, source, source_group, token_count = next(iterators[group])
            except StopIteration:
                exhausted[group] = True
                break
            totals[group] += token_count
            selected.append(
                {
                    "document_id": document_id,
                    "source": source,
                    "source_group": source_group,
                    "token_count": token_count,
                }
            )

    for group, quota in requested.items():
        take(group, quota)
    while sum(totals.values()) < target:
        active = [
            group
            for group in settings["shortage_redistribution_weights"]
            if not exhausted[group]
        ]
        if not active:
            break
        remaining = target - sum(totals.values())
        weight_total = sum(settings["shortage_redistribution_weights"][group] for group in active)
        allocations = {
            group: int(
                remaining * settings["shortage_redistribution_weights"][group] / weight_total
            )
            for group in active
        }
        allocations[active[0]] += remaining - sum(allocations.values())
        before = sum(totals.values())
        for group in active:
            take(group, totals[group] + allocations[group])
        if sum(totals.values()) == before:
            break
    selected.sort(key=lambda row: (rank(seed, row["document_id"], "order"), row["document_id"]))
    return selected


def main() -> None:
    config = load_config(repo_root=ROOT)
    metadata = config.path("metadata")
    tokenizer_dir = config.path("tokenizer")
    audit = json.loads((metadata / "tokenizer_audit.json").read_text(encoding="utf-8"))
    token_report = json.loads(
        (metadata / "token_counts_by_source_split.json").read_text(encoding="utf-8")
    )
    subset_summary = json.loads(
        (metadata / "training_subset_summary.json").read_text(encoding="utf-8")
    )

    corpus_hash = sha256_file(tokenizer_dir / "training_corpus.txt")
    sample_hash = sha256_file(tokenizer_dir / "training_sample_manifest.parquet")
    provenance = audit["training_provenance"]
    if corpus_hash != provenance["training_corpus_sha256"] or sample_hash != provenance[
        "training_sample_manifest_sha256"
    ]:
        raise RuntimeError("Tokenizer training input hash mismatch")
    sample = pq.ParquetFile(tokenizer_dir / "training_sample_manifest.parquet")
    train = pq.ParquetFile(config.path("processed") / "train.parquet")
    sample_ids = []
    for batch in sample.iter_batches(batch_size=50_000, columns=["document_id", "split", "selection_order"]):
        ids, splits, orders = [column.to_pylist() for column in batch.columns]
        for document_id, split, order in zip(ids, splits, orders):
            if split != "train" or order != len(sample_ids):
                raise RuntimeError("Tokenizer sample is not a stable train-only sequence")
            sample_ids.append(document_id)
    first_train_ids = []
    for batch in train.iter_batches(batch_size=50_000, columns=["document_id"]):
        remaining = len(sample_ids) - len(first_train_ids)
        first_train_ids.extend(batch.column(0).to_pylist()[:remaining])
        if len(first_train_ids) == len(sample_ids):
            break
    if sample_ids != first_train_ids or len(sample_ids) != 1_000_000:
        raise RuntimeError("Tokenizer sample does not match the deterministic repaired-train prefix")

    candidate_checks = {}
    for size in (8_000, 16_000, 32_000):
        values = audit["candidate_comparison"][str(size)]
        model_path = resolve_repository_path(values["training"]["model_path"], ROOT)
        processor = spm.SentencePieceProcessor(model_file=str(model_path))
        if (
            processor.vocab_size() != size
            or values["training"]["training_corpus_sha256"] != corpus_hash
            or values["audit"]["audit_split"] != "train"
        ):
            raise RuntimeError(f"Candidate {size} did not use the common repaired-train input")
        candidate_checks[str(size)] = {
            "vocab_size": processor.vocab_size(),
            "model_sha256": sha256_file(model_path),
            "training_corpus_sha256": corpus_hash,
            "unknown_token_count": values["audit"]["unknown_token_count"],
            "unknown_rate_denominator_token_count": values["audit"][
                "unknown_rate_denominator_token_count"
            ],
        }

    count_path = metadata / "document_token_counts.parquet"
    measured: dict[str, dict[str, Counter]] = {}
    token_parquet = pq.ParquetFile(count_path)
    for batch in token_parquet.iter_batches(
        batch_size=50_000, columns=["document_id", "source", "split", "token_count", "includes_eod"]
    ):
        ids, sources, splits, counts, eod = [column.to_pylist() for column in batch.columns]
        for document_id, source, split, token_count, includes_eod in zip(
            ids, sources, splits, counts, eod
        ):
            if not document_id or not includes_eod:
                raise RuntimeError("Malformed repaired token-count record")
            measured.setdefault(split, {}).setdefault(source, Counter()).update(
                {"documents": 1, "tokens": token_count}
            )
    for split, sources in token_report.items():
        for source, expected in sources.items():
            if measured[split][source]["documents"] != expected["documents"] or measured[split][
                source
            ]["tokens"] != expected["tokens"]:
                raise RuntimeError(f"Token report mismatch for {split}/{source}")

    index_path = config.path("interim") / "token_counts_index.sqlite"
    connection = sqlite3.connect("file:" + index_path.as_posix() + "?mode=ro", uri=True)
    try:
        replayed = replay_selection(
            connection, config.values["training_subset"], config.values["seeds"]["data"]
        )
    finally:
        connection.close()
    subset_path = config.path("manifests") / "train_50m.parquet"
    frozen = pq.read_table(subset_path).to_pylist()
    if len(frozen) != len(replayed):
        raise RuntimeError("50M replay selected a different document count")
    for order, (stored, replay) in enumerate(zip(frozen, replayed)):
        fields = ("document_id", "source", "source_group", "token_count")
        if stored["sampling_order"] != order or any(stored[field] != replay[field] for field in fields):
            raise RuntimeError(f"50M replay first differs at sampling order {order}")
        if Path(stored["processed_file"]).is_absolute() or stored["processed_row"] < 0:
            raise RuntimeError("50M manifest has a non-portable processed reference")
    if len({row["document_id"] for row in frozen}) != len(frozen):
        raise RuntimeError("50M manifest contains duplicate documents")

    cumulative = 0
    boundary = None
    target = subset_summary["target_tokens"]
    for order, row in enumerate(frozen):
        if cumulative + row["token_count"] >= target:
            consumed = target - cumulative
            boundary = {
                "sampling_order_zero_based": order,
                "document_id": row["document_id"],
                "cumulative_tokens_before_document": cumulative,
                "full_document_tokens_including_eod": row["token_count"],
                "tokens_consumed_from_document": consumed,
                "eod_consumed": consumed == row["token_count"],
            }
            break
        cumulative += row["token_count"]
    expected_boundary = subset_summary["exact_consumption_boundary"]
    if any(expected_boundary[key] != value for key, value in boundary.items()):
        raise RuntimeError("Exact 50M consumption boundary did not replay")
    if sha256_file(subset_path) != subset_summary["manifest_sha256"]:
        raise RuntimeError("50M manifest hash mismatch")

    simulated_root = Path("C:/Research/azerbaijani-positional-encoding")
    for row in frozen[:100]:
        resolved = resolve_repository_path(row["processed_file"], simulated_root)
        if not str(resolved).lower().startswith(str(simulated_root.resolve()).lower()):
            raise RuntimeError("Portable processed reference escaped the simulated moved root")

    result = {
        "status": "pass",
        "tokenizer_training": {
            "documents": len(sample_ids),
            "split": "train",
            "matches_first_repaired_train_documents": True,
            "training_corpus_sha256": corpus_hash,
            "training_sample_manifest_sha256": sample_hash,
            "candidate_checks": candidate_checks,
        },
        "token_counts": {
            "documents": token_parquet.metadata.num_rows,
            "sha256": sha256_file(count_path),
        },
        "training_subset": {
            "documents": len(frozen),
            "tokens": sum(row["token_count"] for row in frozen),
            "manifest_sha256": subset_summary["manifest_sha256"],
            "train_only": True,
            "duplicates": 0,
            "ids_and_order_match_independent_replay": True,
            "exact_boundary_replayed": boundary,
        },
        "portability": {
            "status": "pass",
            "simulated_repository_root": simulated_root.as_posix(),
            "sampled_subset_references": 100,
        },
    }
    atomic_write_json(metadata / "downstream_repair_validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
