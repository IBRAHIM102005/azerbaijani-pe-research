import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_SIZES = (8_000, 16_000, 32_000)


def _metadata(size):
    path = (
        ROOT
        / "tokenizer"
        / "candidates"
        / f"vocab_{size}"
        / "training_metadata.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _shared_trainer_arguments(metadata):
    return {
        key: value
        for key, value in metadata["trainer_arguments"].items()
        if key not in {"vocab_size", "model_prefix"}
    }


def test_tokenizer_candidates_use_identical_training_corpus():
    candidates = [_metadata(size) for size in CANDIDATE_SIZES]

    corpus_hashes = {
        candidate["training_corpus_sha256"]
        for candidate in candidates
    }

    assert corpus_hashes == {
        "da0ff4b8209ab40e98afc96c71584a15defbd962d2f50e9b4f5ebc4e0a65a1d1"
    }


def test_tokenizer_candidates_differ_only_in_vocab_specific_arguments():
    candidates = [_metadata(size) for size in CANDIDATE_SIZES]

    shared = [_shared_trainer_arguments(candidate) for candidate in candidates]
    assert shared[0] == shared[1] == shared[2]

    for size, candidate in zip(CANDIDATE_SIZES, candidates):
        assert candidate["requested_vocab_size"] == size
        assert candidate["actual_vocab_size"] == size

    common = shared[0]
    assert common["model_type"] == "bpe"
    assert common["normalization_rule_name"] == "identity"
    assert common["character_coverage"] == 1.0
    assert common["shuffle_input_sentence"] is False
    assert common["unk_id"] == 0
    assert common["eos_id"] == 1
