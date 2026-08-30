"""Validate and promote repaired tokenizer artifacts from their staging directory."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq
import sentencepiece as spm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.hashing import atomic_write_json, sha256_file


def replace_staging_paths(value):
    if isinstance(value, dict):
        return {key: replace_staging_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_staging_paths(item) for item in value]
    if isinstance(value, str):
        return value.replace("tokenizer_repair/", "tokenizer/")
    return value


def main() -> None:
    staging = ROOT / "tokenizer_repair"
    final = ROOT / "tokenizer"
    metadata = ROOT / "data" / "metadata"
    audit_path = metadata / "tokenizer_audit_repair.json"
    if final.exists() or not staging.is_dir() or not audit_path.is_file():
        raise RuntimeError("Tokenizer promotion paths are not in the expected staged state")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    provenance = audit["training_provenance"]
    if provenance["documents"] != 1_000_000:
        raise RuntimeError("Tokenizer training corpus does not contain one million documents")
    corpus = staging / "training_corpus.txt"
    sample = staging / "training_sample_manifest.parquet"
    if sha256_file(corpus) != provenance["training_corpus_sha256"]:
        raise RuntimeError("Tokenizer training corpus hash changed")
    if sha256_file(sample) != provenance["training_sample_manifest_sha256"]:
        raise RuntimeError("Tokenizer training sample manifest hash changed")
    if pq.ParquetFile(sample).metadata.num_rows != provenance["documents"]:
        raise RuntimeError("Tokenizer training sample manifest is incomplete")
    for batch in pq.ParquetFile(sample).iter_batches(batch_size=100_000, columns=["split"]):
        if set(batch.column(0).to_pylist()) != {"train"}:
            raise RuntimeError("Tokenizer training sample contains a non-train document")

    for size in (8_000, 16_000, 32_000):
        candidate = audit["candidate_comparison"][str(size)]
        training = candidate["training"]
        candidate_dir = staging / "candidates" / f"vocab_{size}"
        model = candidate_dir / "tokenizer.model"
        vocab = candidate_dir / "tokenizer.vocab"
        processor = spm.SentencePieceProcessor(model_file=str(model))
        if (
            processor.vocab_size() != size
            or training["training_corpus_sha256"] != provenance["training_corpus_sha256"]
            or sha256_file(model) != training["model_sha256"]
            or sha256_file(vocab) != training["vocab_sha256"]
        ):
            raise RuntimeError(f"Tokenizer candidate {size} failed promotion validation")
        training_metadata = replace_staging_paths(
            json.loads((candidate_dir / "training_metadata.json").read_text(encoding="utf-8"))
        )
        atomic_write_json(candidate_dir / "training_metadata.json", training_metadata)

    final_model = staging / "tokenizer.model"
    processor = spm.SentencePieceProcessor(model_file=str(final_model))
    if (
        processor.vocab_size() != 16_000
        or (processor.unk_id(), processor.eos_id(), processor.bos_id(), processor.pad_id())
        != (0, 1, -1, -1)
        or processor.id_to_piece(1) != "<eod>"
    ):
        raise RuntimeError("The repaired final 16K tokenizer has invalid special-token settings")

    audit = replace_staging_paths(audit)
    config_path = staging / "tokenizer_config.json"
    tokenizer_config = replace_staging_paths(json.loads(config_path.read_text(encoding="utf-8")))
    tokenizer_config["embedded_trainer_paths"] = (
        "Historical SentencePiece TrainerSpec paths are not used to load or run the tokenizer."
    )
    atomic_write_json(config_path, tokenizer_config)
    hashes = {
        name: sha256_file(staging / name)
        for name in (
            "tokenizer.model",
            "tokenizer.vocab",
            "special_tokens.json",
            "tokenizer_config.json",
        )
    }
    atomic_write_json(staging / "tokenizer_hashes.json", hashes)
    audit["final"]["hashes"] = hashes

    os.replace(staging, final)
    atomic_write_json(metadata / "tokenizer_audit.json", audit)
    audit_path.unlink()
    for name, expected in hashes.items():
        if sha256_file(final / name) != expected:
            raise RuntimeError(f"Promoted tokenizer hash mismatch: {name}")
    promotion = {
        "status": "complete",
        "training_documents": provenance["documents"],
        "training_corpus_sha256": provenance["training_corpus_sha256"],
        "training_sample_manifest_sha256": provenance["training_sample_manifest_sha256"],
        "final_vocab_size": processor.vocab_size(),
        "artifact_hashes": hashes,
    }
    atomic_write_json(metadata / "tokenizer_repair_promotion.json", promotion)
    print(json.dumps(promotion, indent=2))


if __name__ == "__main__":
    main()
