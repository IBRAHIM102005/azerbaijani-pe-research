import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.config import DataPipelineConfig
from src.data.prepare import run_prepare
from src.data.sampling import build_training_subset
from src.tokenizer.corpus import build_training_corpus
from src.tokenizer.counts import count_processed_tokens
from src.tokenizer.train import train_candidate


def _config(root: Path) -> DataPipelineConfig:
    values = {
        "paths": {
            "raw_core": "raw",
            "original_dollma": "original",
            "interim": "interim",
            "processed": "processed",
            "manifests": "manifests",
            "metadata": "metadata",
            "tokenizer": "tokenizer",
        },
        "sources": {
            "source-a": {
                "group": "News",
                "text_column": "text",
                "included_in_core": True,
                "evidence": "synthetic test",
            }
        },
        "seeds": {"data": 2026, "split": 2026},
        "normalization": {"minimum_unicode_letters": 10},
        "near_duplicate": {
            "shingle_size": 5,
            "fingerprint_size": 16,
            "bands": 4,
            "selected_threshold": 0.95,
            "max_complete_band_bucket": 200,
            "max_star_band_bucket": 10000,
        },
        "split": {"train_upper": 900, "validation_upper": 950, "modulus": 1000},
    }
    return DataPipelineConfig(root, root / "config.yaml", values)


def test_tiny_end_to_end_data_tokenizer_and_quota(tmp_path):
    config = _config(tmp_path)
    raw_dir = tmp_path / "raw" / "source-a"
    raw_dir.mkdir(parents=True)
    base = "Azərbaycan dilində sınaq üçün kifayət qədər uzun sənəd nömrəsi"
    texts = [f"{base} {index}. Fərqli məzmun {index * 17}." for index in range(240)]
    texts.extend([texts[0], texts[1] + " Kiçik əlavə.", "qısa"])
    pq.write_table(pa.table({"text": texts}), raw_dir / "part.parquet")
    (tmp_path / "metadata").mkdir()

    result = run_prepare(config, tmp_path / "interim" / "index.sqlite")
    assert result["accounting"]["sources"]["source-a"]["raw"] == len(texts)
    assert result["accounting"]["sources"]["source-a"]["removed_short"] == 1
    assert result["exact_duplicates"]["removed_documents"] >= 1

    corpus = tmp_path / "tokenizer" / "training.txt"
    sample_manifest = tmp_path / "tokenizer" / "sample.parquet"
    provenance = build_training_corpus(
        tmp_path / "processed" / "train.parquet", corpus, sample_manifest, 1000
    )
    assert provenance["documents"] > 100
    settings = {
        "character_coverage": 1.0,
        "normalization_rule_name": "identity",
        "unk_id": 0,
        "unk_piece": "<unk>",
        "eos_id": 1,
        "eos_piece": "<eod>",
        "bos_id": -1,
        "pad_id": -1,
        "num_threads": 1,
    }
    trained = train_candidate(corpus, tmp_path / "tokenizer" / "candidate", settings, 128)
    counts_path = tmp_path / "metadata" / "counts.parquet"
    totals = count_processed_tokens(Path(trained["model_path"]), tmp_path / "processed", counts_path)
    assert totals["train"]["source-a"]["tokens"] > 100

    connection = sqlite3.connect(tmp_path / "interim" / "index.sqlite")
    subset = build_training_subset(
        connection,
        counts_path,
        tmp_path / "manifests" / "train_100.parquet",
        {
            "target_tokens": 100,
            "requested_group_weights": {"News": 1.0},
            "shortage_redistribution_weights": {"News": 1.0},
            "group_source_order": {"News": ["source-a"]},
        },
        2026,
    )
    connection.close()
    assert subset["selected_unique_tokens"] >= 100
