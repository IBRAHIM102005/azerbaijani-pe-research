"""Measure tokenizer candidates on a shared train-only sample."""

from __future__ import annotations

import heapq
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import sentencepiece as spm

from .corpus import tokenizer_text


_WORD = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*", re.UNICODE)
_AZERBAIJANI_EXAMPLES = [
    "Azərbaycan dili zəngin şəkilçi sisteminə malikdir.",
    "Öyrəncilər gördüklərini müəllimləri ilə bölüşdülər.",
    "Şəhərlərimizdəki dəyişiklikləri qiymətləndirəcəyik.",
]


def _quantiles(counter: Counter[int]) -> dict[str, float]:
    total = sum(counter.values())
    if not total:
        return {"p50": 0, "p95": 0, "p99": 0}
    output = {}
    cumulative = 0
    targets = [("p50", 0.50), ("p95", 0.95), ("p99", 0.99)]
    index = 0
    for value, count in sorted(counter.items()):
        cumulative += count
        while index < len(targets) and cumulative >= total * targets[index][1]:
            output[targets[index][0]] = value
            index += 1
    return output


def audit_candidate(model_path: Path, train_parquet: Path, max_documents: int) -> dict[str, Any]:
    """Audit one candidate without reading validation or test text."""

    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    totals = Counter()
    sources: dict[str, Counter[str]] = defaultdict(Counter)
    token_lengths: Counter[int] = Counter()
    longest_words: list[tuple[int, str]] = []
    seen_words = set()
    parquet = pq.ParquetFile(train_parquet)

    for batch in parquet.iter_batches(
        batch_size=2048,
        columns=["source", "text", "split"],
    ):
        for source, text, split in zip(*[column.to_pylist() for column in batch.columns]):
            if totals["documents"] >= max_documents:
                break
            if split != "train":
                raise RuntimeError("Tokenizer audit received a non-train document")
            prepared_text = tokenizer_text(text)
            ids = processor.encode(prepared_text, out_type=int)
            words = prepared_text.split()
            totals["documents"] += 1
            totals["characters"] += len(prepared_text)
            totals["words"] += len(words)
            totals["tokens"] += len(ids)
            unknowns = sum(token_id == processor.unk_id() for token_id in ids)
            totals["unknown_tokens"] += unknowns
            totals["documents_with_unknown"] += bool(unknowns)
            sources[source]["documents"] += 1
            sources[source]["words"] += len(words)
            sources[source]["tokens"] += len(ids)
            for token_id in ids:
                token_lengths[len(processor.id_to_piece(token_id).replace("▁", ""))] += 1
            for word in _WORD.findall(prepared_text):
                lowered = word.casefold()
                if len(word) >= 12 and lowered not in seen_words:
                    seen_words.add(lowered)
                    item = (len(word), word)
                    if len(longest_words) < 20:
                        heapq.heappush(longest_words, item)
                    elif item > longest_words[0]:
                        heapq.heapreplace(longest_words, item)
        if totals["documents"] >= max_documents:
            break

    morphology_examples = []
    for _, word in sorted(longest_words, reverse=True)[:12]:
        pieces = processor.encode(word, out_type=str)
        morphology_examples.append({"word": word, "pieces": pieces, "piece_count": len(pieces)})
    azerbaijani_checks = []
    for text in _AZERBAIJANI_EXAMPLES:
        ids = processor.encode(text, out_type=int)
        decoded = processor.decode(ids)
        azerbaijani_checks.append(
            {
                "text": text,
                "ids": ids,
                "pieces": processor.encode(text, out_type=str),
                "decoded": decoded,
                "deterministic_ids": ids == processor.encode(text, out_type=int),
                "special_letters_preserved": all(
                    character in decoded for character in set(text) & set("əƏıİöÖüÜşŞçÇğĞ")
                ),
            }
        )
    return {
        "actual_vocab_size": processor.vocab_size(),
        "audit_split": "train",
        "documents": totals["documents"],
        "characters": totals["characters"],
        "approximate_words": totals["words"],
        "tokens": totals["tokens"],
        "token_per_word_fertility": totals["tokens"] / max(1, totals["words"]),
        "unknown_token_count": totals["unknown_tokens"],
        "average_characters_per_token": totals["characters"] / max(1, totals["tokens"]),
        "unknown_token_rate": totals["unknown_tokens"] / max(1, totals["tokens"]),
        "unknown_rate_denominator_token_count": totals["tokens"],
        "unknown_rate_denominator_policy": (
            "SentencePiece tokens on the shared train-only audit sample; EOD is not appended."
        ),
        "documents_with_unknown_tokens": totals["documents_with_unknown"],
        "token_piece_length_quantiles": _quantiles(token_lengths),
        "source_fertility": {
            source: {
                "documents": values["documents"],
                "tokens": values["tokens"],
                "approximate_words": values["words"],
                "token_per_word_fertility": values["tokens"] / max(1, values["words"]),
            }
            for source, values in sorted(sources.items())
        },
        "suffix_rich_long_word_examples": morphology_examples,
        "azerbaijani_round_trip_checks": azerbaijani_checks,
    }
