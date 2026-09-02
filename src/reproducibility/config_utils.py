"""Config canonicalization + pairwise diffing.

No dependency on internals: operates purely on already-resolved
config dicts (however they were produced), so it works whether M1/M2/M3
resolve configs via OmegaConf, plain YAML merge, or something else.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level, got {type(data)}")
    return data


def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict into dotted-path -> scalar/leaf-list mapping."""
    out: dict[str, Any] = {}
    if isinstance(d, dict):
        for k in sorted(d.keys()):
            path = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(d[k], path))
    elif isinstance(d, list):
        # Lists are treated as leaves (compared as a whole) to avoid
        # index-based false positives on reordered-but-equivalent lists
        # being silently accepted; exact equality is required.
        out[prefix] = d
    else:
        out[prefix] = d
    return out


def canonicalize(config: dict) -> dict[str, Any]:
    """Return a flat, dotted-path -> value view of a resolved config,
    suitable for deterministic pairwise comparison and hashing."""
    return _flatten(config)


def config_hash(config: dict) -> str:
    canon = canonicalize(config)
    blob = json.dumps(canon, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def combined_hash(mapping: dict[str, str]) -> str:
    """Deterministic single hash summarizing a {name: hash} mapping, e.g.
    tokenizer.artifact_hashes -> one 'tokenizer_hash' for run metadata."""
    blob = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def diff_configs(
    config_a: dict,
    config_b: dict,
    allowlist_prefixes: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Pairwise diff of two resolved configs.

    Returns a dict with:
      - "allowed_differences": {path: (value_a, value_b)} for keys matching
        allowlist_prefixes
      - "forbidden_differences": {path: (value_a, value_b)} for every other
        differing key (present-in-one-only counts as a forbidden difference)
      - "compared_paths": sorted list of every path that was present in
        either config
    """
    flat_a = canonicalize(config_a)
    flat_b = canonicalize(config_b)
    all_paths = sorted(set(flat_a) | set(flat_b))

    allowed: dict[str, tuple[Any, Any]] = {}
    forbidden: dict[str, tuple[Any, Any]] = {}

    _MISSING = object()
    for path in all_paths:
        va = flat_a.get(path, _MISSING)
        vb = flat_b.get(path, _MISSING)
        if va == vb and va is not _MISSING and vb is not _MISSING:
            continue
        is_allowed = any(
            path == prefix or path.startswith(prefix + ".")
            for prefix in allowlist_prefixes
        )
        target = allowed if is_allowed else forbidden
        target[path] = (
            None if va is _MISSING else va,
            None if vb is _MISSING else vb,
        )

    return {
        "compared_paths": all_paths,
        "allowed_differences": allowed,
        "forbidden_differences": forbidden,
    }
