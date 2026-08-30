"""Train and audit the data pipeline SentencePiece candidates."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.config import load_config
from src.data.hashing import atomic_write_json
from src.data.paths import repository_relative
from src.tokenizer.audit import audit_candidate
from src.tokenizer.corpus import build_training_corpus
from src.tokenizer.train import freeze_final_tokenizer, train_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 8K, 16K, and 32K train-only SentencePiece BPE candidates.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "frozen" / "data_pipeline.yaml")
    parser.add_argument(
        "--recover-existing-candidates",
        action="store_true",
        help="Validate and reuse complete candidate models left by an interrupted audit.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        help="Processed split directory; defaults to the frozen config path.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        help="Tokenizer output directory; defaults to the frozen config path.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        help="Audit metadata path; defaults to data/metadata/tokenizer_audit.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config, ROOT)
    settings = config.values["tokenizer"]
    processed_dir = (args.processed_dir or config.path("processed")).resolve()
    tokenizer_dir = (args.tokenizer_dir or config.path("tokenizer")).resolve()
    metadata_output = (
        args.metadata_output or config.path("metadata") / "tokenizer_audit.json"
    ).resolve()
    corpus_path = tokenizer_dir / "training_corpus.txt"
    sample_manifest = tokenizer_dir / "training_sample_manifest.parquet"
    provenance = build_training_corpus(
        processed_dir / "train.parquet",
        corpus_path,
        sample_manifest,
        settings["max_training_documents"],
    )
    provenance["sample_manifest_path"] = repository_relative(sample_manifest, config.repo_root)
    provenance["training_corpus_path"] = repository_relative(corpus_path, config.repo_root)
    provenance["source_processed_train"] = repository_relative(
        processed_dir / "train.parquet", config.repo_root
    )
    candidates = {}
    for vocab_size in settings["candidate_vocab_sizes"]:
        logging.info("stage=tokenizer_train vocab_size=%d", vocab_size)
        output_dir = tokenizer_dir / "candidates" / f"vocab_{vocab_size}"
        training = train_candidate(
            corpus_path,
            output_dir,
            settings,
            vocab_size,
            recover_existing=args.recover_existing_candidates,
            input_sha256=provenance["training_corpus_sha256"],
            repo_root=config.repo_root,
        )
        audit = audit_candidate(
            output_dir / "tokenizer.model",
            processed_dir / "train.parquet",
            settings["audit_documents"],
        )
        candidates[str(vocab_size)] = {"training": training, "audit": audit}
    final_size = settings["final_vocab_size"]
    final_audit = candidates[str(final_size)]["audit"]
    if final_audit["actual_vocab_size"] != final_size:
        raise RuntimeError("The 16K tokenizer did not produce the requested vocabulary")
    if final_audit["unknown_token_rate"] > 0.001:
        raise RuntimeError("The 16K tokenizer exceeded the pre-declared 0.1% train-audit unknown-token gate")
    frozen = freeze_final_tokenizer(
        tokenizer_dir / "candidates" / f"vocab_{final_size}",
        tokenizer_dir,
        settings,
        provenance,
    )
    result = {
        "training_provenance": provenance,
        "candidate_comparison": candidates,
        "selection": {
            "selected_vocab_size": final_size,
            "reason": "Protocol default retained; no technical or coverage failure was observed on the train-only audit.",
        },
        "final": frozen,
    }
    atomic_write_json(metadata_output, result)
    logging.info("stage=tokenizer_complete output=%s", tokenizer_dir)


if __name__ == "__main__":
    main()
