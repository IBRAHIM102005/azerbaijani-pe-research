"""Run the near-duplicate threshold pilot on a stable core sample."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.config import load_config
from src.data.dedup import PilotDocument, StableSample, run_near_duplicate_pilot
from src.data.hashing import atomic_write_json, raw_record_id
from src.data.io import stream_source
from src.data.normalize import normalize_text, unicode_letter_count


LOGGER = logging.getLogger("data_pipeline.near_pilot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit near-duplicate thresholds on a train-independent raw sample.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "frozen" / "data_pipeline.yaml")
    parser.add_argument("--batch-size", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config, ROOT)
    settings = config.values["near_duplicate"]
    minimum_letters = config.values["normalization"]["minimum_unicode_letters"]
    samples = []
    started = time.perf_counter()

    for source in config.included_sources:
        sampler = StableSample(settings["pilot_documents_per_source"], config.values["seeds"]["data"])
        source_settings = config.source(source)
        count = 0
        for record in stream_source(config.path("raw_core"), source, source_settings["text_column"], args.batch_size):
            sampler.add(PilotDocument(source, raw_record_id(source, record.shard, record.row_index), record.text or ""))
            count += 1
        retained = 0
        for document in sampler.documents():
            normalized = normalize_text(document.text).text
            if unicode_letter_count(normalized) >= minimum_letters:
                samples.append(PilotDocument(source, document.record_id, normalized))
                retained += 1
        LOGGER.info("stage=sample source=%s records=%d pilot_retained=%d", source, count, retained)

    result = run_near_duplicate_pilot(
        samples,
        shingle_size=settings["shingle_size"],
        fingerprint_size=settings["fingerprint_size"],
        bands=settings["bands"],
        thresholds=settings["candidate_thresholds"],
    )
    result["selection_method"] = "Stable provenance-hash sample, normalized and filtered at 50 Unicode letters, then exact-deduplicated before near-pair scoring."
    result["runtime_seconds"] = round(time.perf_counter() - started, 3)
    output = config.path("metadata") / "near_duplicate_pilot.json"
    atomic_write_json(output, result)
    LOGGER.info("stage=near_pilot_complete output=%s", output)


if __name__ == "__main__":
    main()
