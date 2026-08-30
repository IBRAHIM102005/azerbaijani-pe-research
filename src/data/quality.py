"""Streaming raw-corpus quality measurements."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .dedup import PilotDocument, StableSample
from .hashing import raw_record_id


_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_UNUSUAL_SPACE = re.compile(r"[^\S\r\n\t ]")
_MARKUP = re.compile(r"<[A-Za-z/][^>]{0,200}>")
_URL = re.compile(r"https?://|www\.", re.IGNORECASE)
_REPEATED = re.compile(r"([^\W\d_])\1{5,}", re.IGNORECASE)
_MOJIBAKE = re.compile(r"(?:[РС][А-Яа-я]|[ДЖГЙ][™±јњ])")
_AZERBAIJANI = set("əƏıİöÖüÜşŞçÇğĞ")


def quality_flags(text: str) -> set[str]:
    """Return non-destructive quality flags for one text."""

    flags = set()
    if "\ufffd" in text:
        flags.add("unicode_replacement")
    if "\x00" in text:
        flags.add("null_byte")
    if _CONTROL.search(text):
        flags.add("control_character")
    if _UNUSUAL_SPACE.search(text):
        flags.add("unusual_whitespace")
    if _MARKUP.search(text):
        flags.add("html_xml_like_markup")
    url_matches = len(_URL.findall(text))
    if url_matches and url_matches * 30 > max(1, len(text)):
        flags.add("url_heavy")
    if _REPEATED.search(text):
        flags.add("repeated_character")
    if text and text.count("\n") / len(text) > 0.05:
        flags.add("high_line_break_density")
    if len(text) >= 100_000:
        flags.add("extremely_long")
    if _MOJIBAKE.search(text):
        flags.add("mojibake_signature")
    return flags


def _quantiles(histogram: Counter[int], probabilities: tuple[float, ...]) -> dict[str, int]:
    total = sum(histogram.values())
    if not total:
        return {f"p{int(probability * 100):02d}": 0 for probability in probabilities}
    targets = [max(1, int(total * probability + 0.999999)) for probability in probabilities]
    output: dict[str, int] = {}
    cumulative = 0
    target_index = 0
    for value, count in sorted(histogram.items()):
        cumulative += count
        while target_index < len(targets) and cumulative >= targets[target_index]:
            output[f"p{int(probabilities[target_index] * 100):02d}"] = value
            target_index += 1
    return output


@dataclass
class SourceProfile:
    source: str
    sample_limit: int
    sample_seed: int
    documents: int = 0
    null_documents: int = 0
    empty_documents: int = 0
    whitespace_only_documents: int = 0
    total_characters: int = 0
    approximate_words: int = 0
    flags: Counter[str] = field(default_factory=Counter)
    character_lengths: Counter[int] = field(default_factory=Counter)
    word_lengths: Counter[int] = field(default_factory=Counter)
    samples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    sample: StableSample = field(init=False)

    def __post_init__(self) -> None:
        self.sample = StableSample(self.sample_limit, self.sample_seed)

    def _flag(self, name: str, record: dict[str, Any], text: str) -> None:
        self.flags[name] += 1
        examples = self.samples.setdefault(name, [])
        if len(examples) < 5:
            examples.append({**record, "excerpt": text[:240].replace("\n", " ")})

    def update(self, shard: str, row_index: int, text: str | None) -> None:
        self.documents += 1
        record_id = raw_record_id(self.source, shard, row_index)
        record = {"source": self.source, "shard": shard, "row_index": row_index, "raw_record_id": record_id}
        if text is None:
            self.null_documents += 1
            self._flag("null_text", record, "")
            return
        if not text:
            self.empty_documents += 1
            self._flag("empty_text", record, text)
        elif not text.strip():
            self.whitespace_only_documents += 1
            self._flag("whitespace_only", record, text)

        character_count = len(text)
        word_count = len(text.split())
        self.total_characters += character_count
        self.approximate_words += word_count
        self.character_lengths[character_count] += 1
        self.word_lengths[word_count] += 1

        for flag in quality_flags(text):
            self._flag(flag, record, text)

        self.sample.add(PilotDocument(self.source, record_id, text))

    def finalize(self) -> dict[str, Any]:
        sampled = self.sample.documents()
        sample_letter_lengths: Counter[int] = Counter()
        sample_alpha_ratio: list[float] = []
        sample_latin_ratio: list[float] = []
        az_occurrences = Counter()
        below_50_letters = 0
        for document in sampled:
            letters = [character for character in document.text if character.isalpha()]
            letter_count = len(letters)
            sample_letter_lengths[letter_count] += 1
            below_50_letters += letter_count < 50
            sample_alpha_ratio.append(letter_count / max(1, len(document.text)))
            latin = sum(
                ("A" <= character <= "Z")
                or ("a" <= character <= "z")
                or ("À" <= character <= "ɏ")
                for character in letters
            )
            sample_latin_ratio.append(latin / max(1, letter_count))
            az_occurrences.update(character for character in document.text if character in _AZERBAIJANI)

        probabilities = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
        return {
            "source": self.source,
            "documents": self.documents,
            "null_documents": self.null_documents,
            "empty_documents": self.empty_documents,
            "whitespace_only_documents": self.whitespace_only_documents,
            "total_characters": self.total_characters,
            "approximate_whitespace_words": self.approximate_words,
            "character_quantiles": _quantiles(self.character_lengths, probabilities),
            "word_quantiles": _quantiles(self.word_lengths, probabilities),
            "flags": dict(sorted(self.flags.items())),
            "flag_samples": self.samples,
            "deterministic_sample": {
                "documents": len(sampled),
                "letter_quantiles": _quantiles(sample_letter_lengths, probabilities),
                "estimated_fraction_below_50_letters": below_50_letters / max(1, len(sampled)),
                "mean_alphabetic_character_ratio": sum(sample_alpha_ratio) / max(1, len(sample_alpha_ratio)),
                "mean_latin_script_ratio_among_letters": sum(sample_latin_ratio) / max(1, len(sample_latin_ratio)),
                "azerbaijani_specific_letter_occurrences": dict(sorted(az_occurrences.items())),
            },
        }
