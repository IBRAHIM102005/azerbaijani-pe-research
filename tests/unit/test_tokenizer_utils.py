import sentencepiece as spm

from src.tokenizer.corpus import tokenizer_text
from src.tokenizer.train import train_candidate


def test_tokenizer_text_projects_line_breaks_without_changing_letters():
    assert tokenizer_text("Azərbaycan\r\nşəhərləri\n\nGəncə") == "Azərbaycan şəhərləri  Gəncə"


def test_sentencepiece_protobuf_binding_is_available():
    import google.protobuf
    import sentencepiece.sentencepiece_model_pb2 as model_pb2

    assert google.protobuf.__version__
    assert model_pb2.ModelProto.DESCRIPTOR.full_name == "sentencepiece.ModelProto"


def test_sentencepiece_candidate_has_stable_special_tokens(tmp_path):
    corpus = tmp_path / "train.txt"
    corpus.write_text(
        "Əlifba öyrənmək şəhərlərimizdə.\n" + "\n".join(
            f"Azərbaycan dilində müxtəlif şəkilçili sözlər və cümlələr {index}"
            for index in range(300)
        ),
        encoding="utf-8",
    )
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
    result = train_candidate(corpus, tmp_path / "candidate", settings, 128)
    processor = spm.SentencePieceProcessor(model_file=result["model_path"])
    text = "Əlifba, öyrənmək, şəhərlərimizdə"
    assert processor.vocab_size() == 128
    assert processor.unk_id() == 0
    assert processor.eos_id() == 1
    assert processor.encode(text) == processor.encode(text)
    decoded = processor.decode(processor.encode(text))
    assert "Ə" in decoded and "ö" in decoded and "ş" in decoded
