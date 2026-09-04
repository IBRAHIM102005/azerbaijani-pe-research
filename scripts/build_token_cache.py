#!/usr/bin/env python3
"""Build the frozen exact-50M M3 uint16 training token cache.

Source of truth
---------------
All scientific identity comes from M1's frozen:

    data/metadata/training_data_contract.json

The script verifies:

    tokenizer.model identity
    train_50m.parquet identity
    processed train.parquet identity
    target token budget
    vocabulary size
    EOD token ID
    exact final consumption boundary

Then it calls the optimized M3 cache builder, producing:

    data/cache/train_50m.uint16.bin
    data/cache/train_50m.uint16.json

The binary contains exactly 50,000,000 uint16 token IDs.

This script must be run only when the frozen M1 large artifacts are
available, normally on the training server.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow
import sentencepiece


# ============================================================
# Repository
# ============================================================

REPO_ROOT = Path(
    __file__
).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )


from src.models.data_contract import (
    load_contract,
)

from src.training.cache_builder import (
    build_fast_token_cache,
    sha256_file,
)


# ============================================================
# Helpers
# ============================================================


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def resolve_path(
    path: Path,
) -> Path:
    if path.is_absolute():
        return path.resolve()

    return (
        REPO_ROOT
        / path
    ).resolve()


def human_bytes(
    value: int,
) -> str:
    size = float(
        value
    )

    for unit in (
        "B",
        "KiB",
        "MiB",
        "GiB",
        "TiB",
    ):
        if (
            size < 1024.0
            or unit == "TiB"
        ):
            return (
                f"{size:.2f} {unit}"
            )

        size /= 1024.0

    return f"{size:.2f} TiB"


def atomic_write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = Path(
        str(path) + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def require_file(
    *,
    name: str,
    path: Path,
    expected_bytes: int | None = None,
) -> None:
    """Require one frozen input artifact."""

    if not path.is_file():
        raise FileNotFoundError(
            f"{name} is missing:\n"
            f"  {path}"
        )

    if expected_bytes is not None:

        actual_bytes = (
            path.stat().st_size
        )

        if actual_bytes != expected_bytes:
            raise RuntimeError(
                f"{name} size mismatch:\n"
                f"  expected = "
                f"{expected_bytes:,} bytes\n"
                f"  actual   = "
                f"{actual_bytes:,} bytes\n"
                f"  path     = {path}"
            )


def verify_sha256(
    *,
    name: str,
    path: Path,
    expected: str,
) -> str:
    """Hash one frozen artifact and refuse drift."""

    print(
        f"Hashing {name}..."
    )

    started = (
        time.perf_counter()
    )

    actual = sha256_file(
        path
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    if actual != expected:
        raise RuntimeError(
            f"{name} SHA-256 mismatch:\n"
            f"  expected = {expected}\n"
            f"  actual   = {actual}\n"
            f"  path     = {path}"
        )

    print(
        f"  PASS "
        f"({elapsed:.2f}s)"
    )

    return actual


def contract_artifact(
    contract,
    *keys: str,
) -> dict[str, Any]:
    value: Any = (
        contract.raw[
            "artifacts"
        ]
    )

    for key in keys:
        value = value[
            key
        ]

    if not isinstance(
        value,
        dict,
    ):
        raise TypeError(
            "Expected artifact metadata "
            f"dictionary at {keys}."
        )

    return value


# ============================================================
# CLI
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-contract",
        type=Path,
        default=Path(
            "data/metadata/"
            "training_data_contract.json"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/cache/"
            "train_50m.uint16.bin"
        ),
    )

    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path(
            "data/cache/"
            "train_50m.uint16.json"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16_384,
        help=(
            "Parquet scan batch size."
        ),
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=16,
        help=(
            "SentencePiece encoding threads."
        ),
    )

    parser.add_argument(
        "--skip-processed-hash",
        action="store_true",
        help=(
            "Skip SHA-256 verification of the "
            "~2GB processed train parquet. "
            "Not recommended for the final "
            "scientific cache build."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite an existing cache and "
            "metadata file."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================


def main():
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be positive."
        )

    if args.num_threads <= 0:
        raise ValueError(
            "--num-threads must be positive."
        )

    contract_path = resolve_path(
        args.data_contract
    )

    output_path = resolve_path(
        args.output
    )

    metadata_path = resolve_path(
        args.metadata_output
    )

    # ========================================================
    # Refuse accidental overwrite
    # ========================================================

    existing = [
        path
        for path in (
            output_path,
            metadata_path,
        )
        if path.exists()
    ]

    if (
        existing
        and not args.force
    ):
        formatted = "\n".join(
            f"  {path}"
            for path in existing
        )

        raise FileExistsError(
            "Cache artifact already exists.\n"
            "Refusing to overwrite:\n"
            f"{formatted}\n\n"
            "If intentional, rerun with --force."
        )

    # ========================================================
    # Frozen M1 contract
    # ========================================================

    if not contract_path.is_file():
        raise FileNotFoundError(
            "M1 frozen training data contract "
            f"is missing:\n  {contract_path}"
        )

    contract = load_contract(
        contract_path
    )

    target_tokens = int(
        contract.target_tokens
    )

    selected_tokens = int(
        contract.selected_tokens
    )

    vocab_size = int(
        contract.vocab_size
    )

    eod_id = int(
        contract.eod_id
    )

    data_seed = int(
        contract.data_seed
    )

    if target_tokens <= 0:
        raise RuntimeError(
            "Contract target token count "
            "must be positive."
        )

    if (
        selected_tokens
        < target_tokens
    ):
        raise RuntimeError(
            "Frozen selected subset is "
            "smaller than target token budget."
        )

    if contract.model_seed_affects_order:
        raise RuntimeError(
            "M1 contract unexpectedly states "
            "that model seed affects data order."
        )

    # ========================================================
    # Resolve artifact metadata
    # ========================================================

    tokenizer_entry = (
        contract_artifact(
            contract,
            "tokenizer",
            "tokenizer.model",
        )
    )

    subset_entry = (
        contract_artifact(
            contract,
            "training_subset_manifest",
        )
    )

    processed_entry = (
        contract_artifact(
            contract,
            "processed_corpus",
            "train",
        )
    )

    tokenizer_path = (
        REPO_ROOT
        / tokenizer_entry[
            "path"
        ]
    ).resolve()

    manifest_path = (
        REPO_ROOT
        / subset_entry[
            "path"
        ]
    ).resolve()

    processed_train_path = (
        REPO_ROOT
        / processed_entry[
            "path"
        ]
    ).resolve()

    # ========================================================
    # Console header
    # ========================================================

    print()
    print("=" * 78)
    print("M3 FROZEN 50M TOKEN CACHE BUILD")
    print("=" * 78)

    print(
        f"repository:       "
        f"{REPO_ROOT}"
    )

    print(
        f"contract:         "
        f"{contract_path}"
    )

    print(
        f"manifest:         "
        f"{manifest_path}"
    )

    print(
        f"processed train:  "
        f"{processed_train_path}"
    )

    print(
        f"tokenizer:        "
        f"{tokenizer_path}"
    )

    print(
        f"target tokens:    "
        f"{target_tokens:,}"
    )

    print(
        f"selected tokens:  "
        f"{selected_tokens:,}"
    )

    print(
        f"overshoot:        "
        f"{selected_tokens - target_tokens:,}"
    )

    print(
        f"data seed:        "
        f"{data_seed}"
    )

    print(
        f"vocab size:       "
        f"{vocab_size:,}"
    )

    print(
        f"EOD id:           "
        f"{eod_id}"
    )

    print(
        f"output:           "
        f"{output_path}"
    )

    print(
        f"expected size:    "
        f"{human_bytes(target_tokens * 2)}"
    )

    print(
        f"batch size:       "
        f"{args.batch_size:,}"
    )

    print(
        f"SP threads:       "
        f"{args.num_threads}"
    )

    print()

    # ========================================================
    # Frozen file existence / size
    # ========================================================

    require_file(
        name="tokenizer.model",
        path=tokenizer_path,
        expected_bytes=int(
            tokenizer_entry[
                "bytes"
            ]
        ),
    )

    require_file(
        name="train_50m manifest",
        path=manifest_path,
        expected_bytes=int(
            subset_entry[
                "bytes"
            ]
        ),
    )

    require_file(
        name="processed train corpus",
        path=processed_train_path,
        expected_bytes=int(
            processed_entry[
                "bytes"
            ]
        ),
    )

    print(
        "[PASS] Frozen source files "
        "exist with expected sizes."
    )

    # ========================================================
    # Hash audit
    # ========================================================

    tokenizer_hash = (
        verify_sha256(
            name="tokenizer.model",
            path=tokenizer_path,
            expected=str(
                tokenizer_entry[
                    "sha256"
                ]
            ),
        )
    )

    manifest_hash = (
        verify_sha256(
            name="train_50m manifest",
            path=manifest_path,
            expected=str(
                subset_entry[
                    "sha256"
                ]
            ),
        )
    )

    if args.skip_processed_hash:

        processed_hash = None

        print(
            "WARNING: processed train "
            "SHA-256 verification skipped."
        )

    else:

        processed_hash = (
            verify_sha256(
                name="processed train corpus",
                path=processed_train_path,
                expected=str(
                    processed_entry[
                        "sha256"
                    ]
                ),
            )
        )

    # ========================================================
    # Contract consistency
    # ========================================================

    if (
        tokenizer_hash
        != contract.tokenizer_sha256
    ):
        raise RuntimeError(
            "Tokenizer artifact hash and "
            "DataContract tokenizer hash disagree."
        )

    if (
        manifest_hash
        != contract.training_subset_sha256
    ):
        raise RuntimeError(
            "Training subset artifact hash and "
            "DataContract subset hash disagree."
        )

    # The current frozen M1 design explicitly
    # includes one EOD after each full document.
    includes_eod = bool(
        contract.raw[
            "token_counts"
        ][
            "includes_one_eod_per_document"
        ]
    )

    if not includes_eod:
        raise RuntimeError(
            "M1 contract no longer states "
            "one EOD per document. "
            "Cache semantics must be audited "
            "before proceeding."
        )

    # ========================================================
    # Frozen exact boundary
    # ========================================================

    frozen_boundary = (
        contract.raw[
            "training_subset"
        ][
            "exact_consumption_boundary"
        ]
    )

    expected_final_document_id = str(
        frozen_boundary[
            "document_id"
        ]
    )

    expected_final_sampling_order = int(
        frozen_boundary[
            "sampling_order_zero_based"
        ]
    )

    expected_tokens_before = int(
        frozen_boundary[
            "cumulative_tokens_before_document"
        ]
    )

    expected_final_take = int(
        frozen_boundary[
            "tokens_consumed_from_document"
        ]
    )

    expected_full_final_tokens = int(
        frozen_boundary[
            "full_document_tokens_including_eod"
        ]
    )

    expected_eod_consumed = bool(
        frozen_boundary[
            "eod_consumed"
        ]
    )

    if (
        expected_tokens_before
        + expected_final_take
        != target_tokens
    ):
        raise RuntimeError(
            "Frozen boundary does not sum "
            "to target_tokens."
        )

    print()
    print("FROZEN EXACT BOUNDARY")
    print(
        f"  sampling order:       "
        f"{expected_final_sampling_order}"
    )

    print(
        f"  final document:       "
        f"{expected_final_document_id}"
    )

    print(
        f"  tokens before doc:    "
        f"{expected_tokens_before:,}"
    )

    print(
        f"  full document tokens: "
        f"{expected_full_final_tokens:,}"
    )

    print(
        f"  consumed from doc:    "
        f"{expected_final_take:,}"
    )

    print(
        f"  EOD consumed:         "
        f"{expected_eod_consumed}"
    )

    # ========================================================
    # Build
    # ========================================================

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    build_started_utc = (
        utc_now()
    )

    build_started = (
        time.perf_counter()
    )

    print()
    print("=" * 78)
    print("BUILDING CACHE")
    print("=" * 78)
    print()

    cache_metadata = (
        build_fast_token_cache(
            repo_root=REPO_ROOT,
            manifest_path=(
                manifest_path
            ),
            tokenizer_path=(
                tokenizer_path
            ),
            output_path=(
                output_path
            ),
            target_tokens=(
                target_tokens
            ),
            eod_id=(
                eod_id
            ),
            vocab_size=(
                vocab_size
            ),
            batch_size=(
                args.batch_size
            ),
            num_threads=(
                args.num_threads
            ),
        )
    )

    build_elapsed = (
        time.perf_counter()
        - build_started
    )

    # ========================================================
    # Validate generated cache
    # ========================================================

    if not output_path.is_file():
        raise RuntimeError(
            "Cache builder returned but "
            "output file does not exist."
        )

    expected_cache_bytes = (
        target_tokens
        * 2
    )

    actual_cache_bytes = (
        output_path.stat().st_size
    )

    if (
        actual_cache_bytes
        != expected_cache_bytes
    ):
        raise RuntimeError(
            "Generated cache has wrong "
            "byte size."
        )

    if (
        int(
            cache_metadata[
                "target_tokens"
            ]
        )
        != target_tokens
    ):
        raise RuntimeError(
            "Generated metadata target "
            "token mismatch."
        )

    final_document = (
        cache_metadata[
            "final_document"
        ]
    )

    if (
        str(
            final_document[
                "document_id"
            ]
        )
        != expected_final_document_id
    ):
        raise RuntimeError(
            "Generated cache final document "
            "does not match frozen M1 boundary."
        )

    if (
        int(
            final_document[
                "sampling_order"
            ]
        )
        != expected_final_sampling_order
    ):
        raise RuntimeError(
            "Generated cache final sampling "
            "order does not match M1."
        )

    if (
        int(
            final_document[
                "tokens_consumed"
            ]
        )
        != expected_final_take
    ):
        raise RuntimeError(
            "Generated cache final-document "
            "consumption differs from M1."
        )

    if (
        int(
            final_document[
                "full_token_count"
            ]
        )
        != expected_full_final_tokens
    ):
        raise RuntimeError(
            "Generated cache final document "
            "token count differs from M1."
        )

    generated_eod_consumed = (
        int(
            final_document[
                "tokens_consumed"
            ]
        )
        == int(
            final_document[
                "full_token_count"
            ]
        )
    )

    if (
        generated_eod_consumed
        != expected_eod_consumed
    ):
        raise RuntimeError(
            "Generated cache EOD-boundary "
            "semantics differ from frozen M1."
        )

    cache_sha256 = str(
        cache_metadata[
            "cache_sha256"
        ]
    )

    # ========================================================
    # Final reproducibility metadata
    # ========================================================

    final_metadata = {
        **cache_metadata,

        "artifact_role": (
            "M3 frozen sequential "
            "training token cache"
        ),

        "created_at_utc": (
            utc_now()
        ),

        "build_started_at_utc": (
            build_started_utc
        ),

        "build_elapsed_seconds": (
            build_elapsed
        ),

        "data_seed": (
            data_seed
        ),

        "model_seed_affects_order": (
            contract
            .model_seed_affects_order
        ),

        "source_identity": {
            "contract_path": str(
                contract_path
            ),

            "tokenizer": {
                "path": str(
                    tokenizer_path
                ),
                "sha256": (
                    tokenizer_hash
                ),
            },

            "training_subset_manifest": {
                "path": str(
                    manifest_path
                ),
                "sha256": (
                    manifest_hash
                ),
            },

            "processed_train": {
                "path": str(
                    processed_train_path
                ),
                "expected_sha256": str(
                    processed_entry[
                        "sha256"
                    ]
                ),
                "verified_sha256": (
                    processed_hash
                ),
                "hash_verified": (
                    processed_hash
                    is not None
                ),
            },
        },

        "frozen_boundary": (
            frozen_boundary
        ),

        "build_environment": {
            "python_version": (
                platform.python_version()
            ),

            "platform": (
                platform.platform()
            ),

            "pyarrow_version": (
                pyarrow.__version__
            ),

            "sentencepiece_version": (
                sentencepiece.__version__
            ),

            "batch_size": (
                args.batch_size
            ),

            "num_threads": (
                args.num_threads
            ),
        },
    }

    atomic_write_json(
        metadata_path,
        final_metadata,
    )

    # ========================================================
    # Final console report
    # ========================================================

    print()
    print("=" * 78)
    print("CACHE BUILD COMPLETE")
    print("=" * 78)

    print(
        f"tokens:             "
        f"{target_tokens:,}"
    )

    print(
        f"bytes:              "
        f"{actual_cache_bytes:,}"
    )

    print(
        f"size:               "
        f"{human_bytes(actual_cache_bytes)}"
    )

    print(
        f"documents consumed: "
        f"{cache_metadata['documents_consumed']:,}"
    )

    print(
        f"final document:     "
        f"{final_document['document_id']}"
    )

    print(
        f"final doc consumed: "
        f"{final_document['tokens_consumed']:,}"
        f"/"
        f"{final_document['full_token_count']:,}"
    )

    print(
        f"cache SHA-256:      "
        f"{cache_sha256}"
    )

    print(
        f"elapsed:            "
        f"{build_elapsed / 60:.2f} min"
    )

    print(
        f"cache:              "
        f"{output_path}"
    )

    print(
        f"metadata:           "
        f"{metadata_path}"
    )

    print()
    print(
        "PASS: frozen exact-50M "
        "token cache generated."
    )


if __name__ == "__main__":
    main()