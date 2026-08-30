"""Count final tokens and build the frozen 50M train manifest."""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.config import load_config
from src.data.hashing import atomic_write_json
from src.data.sampling import build_training_subset, enrich_processed_references
from src.tokenizer.counts import count_processed_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count final tokens and construct the deterministic 50M subset.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "frozen" / "data_pipeline.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config, ROOT)
    token_counts_path = config.path("metadata") / "document_token_counts.parquet"
    logging.info("stage=token_count_start")
    totals = count_processed_tokens(
        config.path("tokenizer") / "tokenizer.model",
        config.path("processed"),
        token_counts_path,
    )
    atomic_write_json(config.path("metadata") / "token_counts_by_source_split.json", totals)
    index_path = config.path("interim") / "token_counts_index.sqlite"
    temporary_index = index_path.with_suffix(index_path.suffix + ".tmp")
    if temporary_index.exists():
        temporary_index.unlink()
    connection = sqlite3.connect(temporary_index)
    try:
        subset_path = config.path("manifests") / "train_50m.parquet"
        summary = build_training_subset(
            connection,
            token_counts_path,
            subset_path,
            config.values["training_subset"],
            config.values["seeds"]["data"],
        )
    finally:
        connection.close()
    os.replace(temporary_index, index_path)
    summary["manifest_sha256"] = enrich_processed_references(
        subset_path,
        config.path("manifests") / "train.parquet",
    )
    summary["processed_reference_policy"] = (
        "Each selected document records its frozen processed parquet path and zero-based row number."
    )
    atomic_write_json(config.path("metadata") / "training_subset_summary.json", summary)
    logging.info("stage=training_subset_complete output=%s", subset_path)


if __name__ == "__main__":
    main()
