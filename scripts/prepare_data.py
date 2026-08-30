"""Normalize, deduplicate, split, and freeze the data pipeline corpus."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.config import load_config
from src.data.hashing import atomic_write_json, canonical_json_hash
from src.data.prepare import run_prepare


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the canonical data pipeline corpus and frozen document splits.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "frozen" / "data_pipeline.yaml")
    parser.add_argument("--rebuild", action="store_true", help="Replace an incomplete generated data pipeline index and outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config, ROOT)
    summary_path = config.path("metadata") / "preparation_summary.json"
    index_path = config.path("interim") / "corpus_index.sqlite"
    if summary_path.exists() and index_path.exists() and not args.rebuild:
        logging.info("stage=prepare_skip reason=completed output=%s", summary_path)
        return
    if index_path.exists() and not args.rebuild:
        logging.info("Incomplete data pipeline index exists at %s; attempting to resume...", index_path)
    if args.rebuild:
        for path in (index_path, index_path.with_suffix(".sqlite-wal"), index_path.with_suffix(".sqlite-shm")):
            if path.exists():
                path.unlink()
        for directory in (config.path("interim") / "near_bands", config.path("processed")):
            if directory.exists():
                shutil.rmtree(directory)
        for path in config.path("manifests").glob("*.parquet"):
            path.unlink()
    config.path("interim").mkdir(parents=True, exist_ok=True)
    result = run_prepare(config, index_path)
    result["config_sha256"] = canonical_json_hash(config.values)
    atomic_write_json(summary_path, result)
    atomic_write_json(config.path("metadata") / "manifest_hashes.json", result["manifest_hashes"])
    atomic_write_json(config.path("metadata") / "exact_duplicate_report.json", result["exact_duplicates"])
    atomic_write_json(config.path("metadata") / "near_duplicate_report.json", result["near_duplicates"])
    logging.info("stage=prepare_complete output=%s", summary_path)


if __name__ == "__main__":
    main()
