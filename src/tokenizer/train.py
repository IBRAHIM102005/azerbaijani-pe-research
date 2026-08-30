"""Train and freeze SentencePiece BPE candidates."""

from __future__ import annotations

import os
import json
import shutil
from pathlib import Path
from typing import Any

import sentencepiece as spm
import sentencepiece.sentencepiece_model_pb2 as model_pb2

from src.data.hashing import atomic_write_json, sha256_file
from src.data.paths import repository_relative


def trainer_arguments(input_path: Path, model_prefix: Path, settings: dict[str, Any], vocab_size: int) -> dict[str, Any]:
    return {
        "input": str(input_path.resolve()),
        "model_prefix": str(model_prefix.resolve()),
        "model_type": "bpe",
        "vocab_size": vocab_size,
        "character_coverage": settings["character_coverage"],
        "normalization_rule_name": settings["normalization_rule_name"],
        "remove_extra_whitespaces": False,
        "unk_id": settings["unk_id"],
        "unk_piece": settings["unk_piece"],
        "eos_id": settings["eos_id"],
        "eos_piece": settings["eos_piece"],
        "bos_id": settings["bos_id"],
        "pad_id": settings["pad_id"],
        "num_threads": settings["num_threads"],
        "shuffle_input_sentence": False,
        "input_sentence_size": 0,
        "hard_vocab_limit": True,
        "max_sentence_length": 4_194_304,
        "train_extremely_large_corpus": True,
        "byte_fallback": bool(settings.get("byte_fallback", False)),
    }


def _validate_existing_model(model_path: Path, settings: dict[str, Any], vocab_size: int) -> None:
    """Check that a recovered candidate matches the frozen trainer settings."""

    proto = model_pb2.ModelProto()
    proto.ParseFromString(model_path.read_bytes())
    trainer = proto.trainer_spec
    normalizer = proto.normalizer_spec
    expected = {
        "vocab_size": vocab_size,
        "model_type": model_pb2.TrainerSpec.BPE,
        "character_coverage": settings["character_coverage"],
        "unk_id": settings["unk_id"],
        "eos_id": settings["eos_id"],
        "bos_id": settings["bos_id"],
        "pad_id": settings["pad_id"],
        "byte_fallback": bool(settings.get("byte_fallback", False)),
    }
    actual = {name: getattr(trainer, name) for name in expected}
    if actual != expected or normalizer.name != settings["normalization_rule_name"]:
        raise RuntimeError(f"Existing tokenizer candidate does not match frozen settings: {model_path}")


def train_candidate(
    input_path: Path,
    output_dir: Path,
    settings: dict[str, Any],
    vocab_size: int,
    recover_existing: bool = False,
    input_sha256: str | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Train one SentencePiece candidate from the frozen train-only corpus."""

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "tokenizer"
    arguments = trainer_arguments(input_path, prefix, settings, vocab_size)
    recorded_arguments = dict(arguments)
    if repo_root is not None:
        recorded_arguments["input"] = repository_relative(input_path, repo_root)
        recorded_arguments["model_prefix"] = repository_relative(prefix, repo_root)
    model_path = prefix.with_suffix(".model")
    vocab_path = prefix.with_suffix(".vocab")
    metadata_path = output_dir / "training_metadata.json"
    corpus_sha256 = input_sha256 or sha256_file(input_path)
    reused = model_path.exists() or vocab_path.exists()
    if reused:
        metadata_matches = False
        if metadata_path.exists():
            saved = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata_matches = (
                saved.get("training_corpus_sha256") == corpus_sha256
                and saved.get("trainer_arguments") == recorded_arguments
                and saved.get("model_sha256") == sha256_file(model_path)
                and saved.get("vocab_sha256") == sha256_file(vocab_path)
            )
        if not (model_path.exists() and vocab_path.exists() and (recover_existing or metadata_matches)):
            raise RuntimeError(f"Incomplete or unapproved existing candidate in {output_dir}")
        _validate_existing_model(model_path, settings, vocab_size)
    else:
        spm.SentencePieceTrainer.Train(**arguments)
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    if processor.vocab_size() != vocab_size:
        raise RuntimeError(f"Requested vocab {vocab_size}, produced {processor.vocab_size()}")
    result = {
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": processor.vocab_size(),
        "model_path": repository_relative(model_path, repo_root) if repo_root else str(model_path.resolve()),
        "vocab_path": repository_relative(vocab_path, repo_root) if repo_root else str(vocab_path.resolve()),
        "model_sha256": sha256_file(model_path),
        "vocab_sha256": sha256_file(vocab_path),
        "model_bytes": model_path.stat().st_size,
        "vocab_bytes": vocab_path.stat().st_size,
        "trainer_arguments": recorded_arguments,
        "training_corpus_sha256": corpus_sha256,
        "reused_after_interrupted_audit": reused,
    }
    atomic_write_json(metadata_path, result)
    return result


def freeze_final_tokenizer(
    candidate_dir: Path,
    tokenizer_dir: Path,
    settings: dict[str, Any],
    training_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Copy the accepted 16K candidate to stable data pipeline artifact names."""

    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    source_model = candidate_dir / "tokenizer.model"
    source_vocab = candidate_dir / "tokenizer.vocab"
    final_model = tokenizer_dir / "tokenizer.model"
    final_vocab = tokenizer_dir / "tokenizer.vocab"
    for source, destination in ((source_model, final_model), (source_vocab, final_vocab)):
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)

    processor = spm.SentencePieceProcessor(model_file=str(final_model))
    special_tokens = {
        "unk": {"piece": settings["unk_piece"], "id": processor.unk_id()},
        "eod": {"piece": settings["eos_piece"], "id": processor.eos_id()},
        "bos": {"piece": None, "id": processor.bos_id(), "enabled": False},
        "pad": {"piece": None, "id": processor.pad_id(), "enabled": False},
        "document_boundary_policy": "Append one <eod> token after each document in downstream token counts and exports.",
    }
    atomic_write_json(tokenizer_dir / "special_tokens.json", special_tokens)
    config = {
        "type": "SentencePiece BPE",
        "vocab_size": processor.vocab_size(),
        "model_file": "tokenizer.model",
        "vocab_file": "tokenizer.vocab",
        "normalization_rule_name": settings["normalization_rule_name"],
        "character_coverage": settings["character_coverage"],
        "byte_fallback": bool(settings.get("byte_fallback", False)),
        "text_projection": "Replace document-internal line breaks with spaces before SentencePiece encoding; canonical processed text is unchanged.",
        "sentencepiece_version": spm.__version__,
        "training_provenance": training_provenance,
    }
    atomic_write_json(tokenizer_dir / "tokenizer_config.json", config)
    hashes = {
        path.name: sha256_file(path)
        for path in (
            final_model,
            final_vocab,
            tokenizer_dir / "special_tokens.json",
            tokenizer_dir / "tokenizer_config.json",
        )
    }
    atomic_write_json(tokenizer_dir / "tokenizer_hashes.json", hashes)
    return {"vocab_size": processor.vocab_size(), "hashes": hashes, "special_tokens": special_tokens}
