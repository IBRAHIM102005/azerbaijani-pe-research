"""Independent exact-similarity checks for frozen split audits."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .dedup import character_shingles


@dataclass(frozen=True)
class LeakageDocument:
    document_id: str
    split: str
    text: str


def cross_split_near_pairs(
    documents: list[LeakageDocument], *, threshold: float, shingle_size: int
) -> list[dict]:
    """Find exact high-Jaccard pairs across splits without using the candidate graph."""

    prepared = {
        document.document_id: (character_shingles(document.text, shingle_size), len(document.text))
        for document in documents
    }
    findings = []
    for left, right in combinations(documents, 2):
        if left.split == right.split:
            continue
        left_set, left_length = prepared[left.document_id]
        right_set, right_length = prepared[right.document_id]
        if min(left_length, right_length) / max(left_length, right_length) < threshold:
            continue
        union_size = len(left_set | right_set)
        similarity = len(left_set & right_set) / union_size if union_size else 1.0
        if similarity >= threshold:
            findings.append(
                {
                    "left_document_id": left.document_id,
                    "right_document_id": right.document_id,
                    "left_split": left.split,
                    "right_split": right.split,
                    "similarity": similarity,
                }
            )
    return findings
