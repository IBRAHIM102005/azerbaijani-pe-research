"""Deterministic document-level split assignment."""

from __future__ import annotations

from .hashing import stable_int


def split_bucket(identifier: str, seed: int, modulus: int = 1000) -> int:
    return stable_int([identifier, str(seed)]) % modulus


def assign_split(
    identifier: str,
    seed: int,
    train_upper: int = 900,
    validation_upper: int = 950,
    modulus: int = 1000,
) -> str:
    """Assign a stable 90/5/5 split from a document or cluster identifier."""

    bucket = split_bucket(identifier, seed, modulus)
    if bucket < train_upper:
        return "train"
    if bucket < validation_upper:
        return "validation"
    return "test"
