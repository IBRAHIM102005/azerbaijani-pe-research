#!/usr/bin/env python3
"""
Collect run metadata.

Thin CLI around src.reproducibility.metadata.collect_metadata. Intended to be called at the
end of a training/eval run by Fidan's launcher, with the run-specific fields
(hashes, tokens_seen, exit_code, ...) passed in as arguments or a JSON
side-file Fidan already writes.

Usage:
    python scripts/collect_metadata.py \
        --run-id p31az-rope-s17-t50m-c512 \
        --pe rope --model-seed 17 --data-seed 2026 \
        --resolved-config-hash <sha256> \
        --out experiments/manifests/p31az-rope-s17-t50m-c512.json

Exit codes:
    0  metadata written and validated as JSON
    1  metadata collection or validation failed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.reproducibility.adapters import (  # noqa: E402
    MissingInterfaceError,
    data_seed_from_contract,
    load_training_data_contract,
    manifest_hashes_from_contract,
    tokenizer_hashes_from_contract,
    training_subset_hash_from_contract,
)
from src.reproducibility.metadata import collect_metadata, write_metadata  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pe", required=True, choices=["learned", "sinusoidal", "rope", "alibi", "nope"])
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, default=None, help="Required unless --from-contract is given")
    parser.add_argument("--resolved-config-hash", required=True)
    parser.add_argument("--tokenizer-hash", default=None)
    parser.add_argument("--train-manifest-hash", default=None)
    parser.add_argument("--validation-manifest-hash", default=None)
    parser.add_argument("--test-manifest-hash", default=None)
    parser.add_argument(
        "--training-subset-manifest-hash",
        default=None,
        help="sha256 of data/manifests/train_50m.parquet",
    )
    parser.add_argument("--dataset-source-revision", default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--tokens-seen", type=int, default=None)
    parser.add_argument(
        "--peak-allocated-vram-bytes",
        type=int,
        default=None,
        help="Peak allocated CUDA memory, measured in the training process itself",
    )
    parser.add_argument(
        "--peak-reserved-vram-bytes",
        type=int,
        default=None,
        help="Peak reserved CUDA memory, measured in the training process itself",
    )
    parser.add_argument(
        "--checkpoint-hashes-json",
        default=None,
        help="JSON string or path to a JSON file mapping checkpoint tag -> hash",
    )
    parser.add_argument("--exit-code", type=int, default=None)
    parser.add_argument("--metrics-path", default=None)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--from-contract",
        action="store_true",
        help="Auto-fill tokenizer/manifest hashes and data seed from Ibrahim's "
        "data/metadata/training_data_contract.json when the corresponding "
        "--*-hash / --data-seed flags aren't explicitly given.",
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    checkpoint_hashes = None
    if args.checkpoint_hashes_json:
        raw = args.checkpoint_hashes_json
        if Path(raw).is_file():
            raw = Path(raw).read_text()
        checkpoint_hashes = json.loads(raw)

    tokenizer_hash = args.tokenizer_hash
    train_hash = args.train_manifest_hash
    val_hash = args.validation_manifest_hash
    test_hash = args.test_manifest_hash
    subset_hash = args.training_subset_manifest_hash
    data_seed = args.data_seed

    if args.from_contract:
        try:
            contract = load_training_data_contract(args.repo_root)
            manifest_hashes = manifest_hashes_from_contract(contract)
            tokenizer_hashes = tokenizer_hashes_from_contract(contract)
            # tokenizer_hash matches Yasin's tokenizer_sha256 (tokenizer.model hash).
            tokenizer_hash = tokenizer_hash or tokenizer_hashes.get("tokenizer.model")
            train_hash = train_hash or manifest_hashes.get("train")
            val_hash = val_hash or manifest_hashes.get("validation")
            test_hash = test_hash or manifest_hashes.get("test")
            if subset_hash is None:
                subset_hash = training_subset_hash_from_contract(contract)
            if data_seed is None:
                data_seed = data_seed_from_contract(contract)
        except MissingInterfaceError as exc:
            print(f"[collect_metadata] ERROR: --from-contract failed: {exc}", file=sys.stderr)
            return 1

    if data_seed is None:
        print("[collect_metadata] ERROR: --data-seed is required (or pass --from-contract)", file=sys.stderr)
        return 1

    try:
        meta = collect_metadata(
            run_id=args.run_id,
            pe_method=args.pe,
            model_seed=args.model_seed,
            data_seed=data_seed,
            resolved_config_hash=args.resolved_config_hash,
            tokenizer_hash=tokenizer_hash,
            train_manifest_hash=train_hash,
            validation_manifest_hash=val_hash,
            test_manifest_hash=test_hash,
            training_subset_manifest_hash=subset_hash,
            dataset_source_revision=args.dataset_source_revision,
            precision=args.precision,
            tokens_seen=args.tokens_seen,
            checkpoint_hashes=checkpoint_hashes,
            exit_code=args.exit_code,
            metrics_path=args.metrics_path,
            repo_dir=args.repo_root,
            peak_allocated_vram_bytes=args.peak_allocated_vram_bytes,
            peak_reserved_vram_bytes=args.peak_reserved_vram_bytes,
        )
    except ValueError as exc:
        print(f"[collect_metadata] ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        write_metadata(meta, args.out)
    except Exception as exc:  # noqa: BLE001
        print(f"[collect_metadata] ERROR: failed to write/validate metadata: {exc}", file=sys.stderr)
        return 1

    print(f"[collect_metadata] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
