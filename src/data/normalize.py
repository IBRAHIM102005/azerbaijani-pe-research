"""Conservative normalization for Azerbaijani source text."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


_HORIZONTAL_SPACE = re.compile(r"[^\S\r\n]+")
_MANY_BLANK_LINES = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")
_DISALLOWED_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


@dataclass(frozen=True)
class NormalizationResult:
    text: str
    changed_nfc: bool
    changed_newlines: bool
    changed_horizontal_space: bool
    removed_control_characters: int
    trimmed_outer_space: bool


def normalize_text(value: str | None) -> NormalizationResult:
    """Normalize formatting while preserving Azerbaijani orthography."""

    if value is None:
        value = ""
    original = value
    nfc = unicodedata.normalize("NFC", value)
    changed_nfc = nfc != value
    newline_normalized = nfc.replace("\r\n", "\n").replace("\r", "\n")
    changed_newlines = newline_normalized != nfc
    control_matches = _DISALLOWED_CONTROL.findall(newline_normalized)
    without_controls = _DISALLOWED_CONTROL.sub("", newline_normalized)
    lines = [_HORIZONTAL_SPACE.sub(" ", line).strip() for line in without_controls.split("\n")]
    spaced = "\n".join(lines)
    changed_horizontal_space = spaced != without_controls
    compact = _MANY_BLANK_LINES.sub("\n\n", spaced)
    normalized = compact.strip()
    return NormalizationResult(
        text=normalized,
        changed_nfc=changed_nfc,
        changed_newlines=changed_newlines,
        changed_horizontal_space=changed_horizontal_space,
        removed_control_characters=len(control_matches),
        trimmed_outer_space=normalized != compact or (not original and bool(normalized)),
    )


def unicode_letter_count(text: str) -> int:
    return sum(character.isalpha() for character in text)
