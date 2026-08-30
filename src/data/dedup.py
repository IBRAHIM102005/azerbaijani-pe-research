"""Exact and near-duplicate helpers."""

from __future__ import annotations

import hashlib
import heapq
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .hashing import stable_int


def character_shingles(text: str, width: int = 5) -> set[str]:
    compact = " ".join(text.split())
    if len(compact) <= width:
        return {compact} if compact else set()
    return {compact[index : index + width] for index in range(len(compact) - width + 1)}


def jaccard_similarity(left: str, right: str, width: int = 5) -> float:
    left_shingles = character_shingles(left, width)
    right_shingles = character_shingles(right, width)
    union = left_shingles | right_shingles
    return len(left_shingles & right_shingles) / len(union) if union else 1.0


def bottom_k_fingerprint(text: str, width: int = 5, size: int = 32) -> tuple[int, ...]:
    compact = " ".join(text.split())
    if not compact:
        return tuple(range(size))
    if len(compact) <= width:
        hashes = [stable_int([compact])]
    else:
        mask = (1 << 64) - 1
        base = 1_000_003
        power = pow(base, width - 1, 1 << 64)
        rolling = 0
        for character in compact[:width]:
            rolling = ((rolling * base) + ord(character)) & mask
        smallest: set[int] = {rolling}
        heap = [-rolling]
        for index in range(width, len(compact)):
            rolling = (
                ((rolling - ord(compact[index - width]) * power) * base)
                + ord(compact[index])
            ) & mask
            if rolling in smallest:
                continue
            if len(smallest) < size:
                smallest.add(rolling)
                heapq.heappush(heap, -rolling)
            elif rolling < -heap[0]:
                removed = -heapq.heapreplace(heap, -rolling)
                smallest.remove(removed)
                smallest.add(rolling)
        hashes = sorted(smallest)
    if len(hashes) < size:
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        while len(hashes) < size:
            offset = len(hashes) % len(seed)
            hashes.append(0x1_0000_0000_0000_0000 + seed[offset] * size + len(hashes))
    return tuple(hashes)


def fingerprint_bands(fingerprint: tuple[int, ...], bands: int) -> tuple[tuple[int, ...], ...]:
    if len(fingerprint) % bands:
        raise ValueError("Fingerprint size must be divisible by the number of bands")
    width = len(fingerprint) // bands
    return tuple(
        fingerprint[index : index + width]
        for index in range(0, len(fingerprint), width)
    )


@dataclass(frozen=True)
class PilotDocument:
    source: str
    record_id: str
    text: str


class StableSample:
    """Keep records with the smallest stable provenance hashes."""

    def __init__(self, limit: int, seed: int):
        self.limit = limit
        self.seed = seed
        self._items: list[tuple[int, str, PilotDocument]] = []

    def add(self, document: PilotDocument) -> None:
        rank = stable_int([str(self.seed), document.record_id])
        item = (-rank, document.record_id, document)
        if len(self._items) < self.limit:
            heapq.heappush(self._items, item)
        elif item > self._items[0]:
            heapq.heapreplace(self._items, item)

    def documents(self) -> list[PilotDocument]:
        return [item[2] for item in sorted(self._items, key=lambda value: (-value[0], value[1]))]


def run_near_duplicate_pilot(
    documents: Iterable[PilotDocument],
    *,
    shingle_size: int,
    fingerprint_size: int,
    bands: int,
    thresholds: list[float],
    example_limit: int = 12,
) -> dict:
    """Evaluate conservative character-shingle thresholds on a fixed sample."""

    original_items = list(documents)
    exact_groups: dict[str, list[PilotDocument]] = defaultdict(list)
    for document in original_items:
        text_hash = hashlib.sha256(document.text.encode("utf-8")).hexdigest()
        exact_groups[text_hash].append(document)
    items = [
        min(group, key=lambda document: (document.source, document.record_id))
        for group in exact_groups.values()
    ]
    items.sort(key=lambda document: (document.source, document.record_id))
    exact_duplicate_documents = sum(len(group) - 1 for group in exact_groups.values())
    cross_source_exact_groups = sum(
        len({document.source for document in group}) > 1
        for group in exact_groups.values()
    )
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    candidates: set[tuple[int, int]] = set()
    for index, document in enumerate(items):
        fingerprint = bottom_k_fingerprint(document.text, shingle_size, fingerprint_size)
        for band_index, band in enumerate(fingerprint_bands(fingerprint, bands)):
            key = (band_index, band)
            for other in buckets[key]:
                candidates.add((other, index))
            buckets[key].append(index)

    similarities: list[tuple[float, int, int]] = []
    for left_index, right_index in sorted(candidates):
        similarity = jaccard_similarity(
            items[left_index].text, items[right_index].text, shingle_size
        )
        similarities.append((similarity, left_index, right_index))
    similarities.sort(reverse=True)

    results = []
    for threshold in thresholds:
        accepted = [item for item in similarities if item[0] >= threshold]
        source_pairs: dict[str, int] = defaultdict(int)
        examples = []
        for similarity, left_index, right_index in accepted:
            left = items[left_index]
            right = items[right_index]
            source_pairs[" / ".join(sorted((left.source, right.source)))] += 1
            if len(examples) < example_limit:
                examples.append(
                    {
                        "similarity": round(similarity, 6),
                        "left_source": left.source,
                        "right_source": right.source,
                        "left_record_id": left.record_id,
                        "right_record_id": right.record_id,
                        "left_excerpt": left.text[:240].replace("\n", " "),
                        "right_excerpt": right.text[:240].replace("\n", " "),
                    }
                )
        results.append(
            {
                "threshold": threshold,
                "accepted_pair_count": len(accepted),
                "source_pairs": dict(sorted(source_pairs.items())),
                "examples": examples,
            }
        )
    return {
        "sample_documents_before_exact_dedup": len(original_items),
        "sample_documents_after_exact_dedup": len(items),
        "sample_exact_duplicate_documents": exact_duplicate_documents,
        "sample_cross_source_exact_groups": cross_source_exact_groups,
        "lsh_candidate_pairs": len(candidates),
        "shingle_size": shingle_size,
        "fingerprint_size": fingerprint_size,
        "bands": bands,
        "threshold_results": results,
    }
