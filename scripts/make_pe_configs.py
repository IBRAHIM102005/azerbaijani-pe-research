#!/usr/bin/env python
"""Single source of truth for every generated model config.

``configs/model_base.json``, ``configs/pe/*.json`` and ``configs/run_matrix.json``
are produced *only* by this script.  They are never hand-edited: patching a
generated file is how one repair silently reverts another.  ``--check``
regenerates everything in memory and fails if anything on disk has drifted.

The production architecture is frozen here, explicitly, field by field.  The
tokenizer path and hash are not restated as literals -- they are read from
``data/metadata/training_data_contract.json``, the artifact M1 froze.

Usage
-----
    python scripts/make_pe_configs.py
    python scripts/make_pe_configs.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.config import PE_TYPES, TEMPLATE_SEED, ModelConfig  # noqa: E402
from src.models.data_contract import load_contract  # noqa: E402

BASE_PATH = ROOT / "configs" / "model_base.json"
ARM_DIR = ROOT / "configs" / "pe"
RUN_MATRIX_PATH = ROOT / "configs" / "run_matrix.json"

#: Preregistered model initialisation seeds.  Adding or removing a seed is a
#: protocol amendment, not an implementation detail: it changes how many runs
#: the study reports and therefore what variance claims it can make.
RUN_SEEDS = (17, 42, 1234)

#: Fixed by M1 and independent of RUN_SEEDS.  Verified against the contract.
DATA_SEED = 2026

#: The frozen production architecture, stated field by field so that a change
#: to any of it is a visible diff in this file.
FROZEN_ARCHITECTURE = {
    "vocab_size": 16000,
    "n_layer": 6,
    "n_head": 8,
    "d_model": 256,
    "d_ff": 1024,
    "max_seq_len": 512,
    "dropout": 0.0,
    "layer_norm_eps": 1e-5,
    "tie_embeddings": False,
    "init_scheme": "pythia",
    "rotary_pct": 0.25,
    "rope_theta": 10000.0,
    "sinusoidal_theta": 10000.0,
    "alibi_max_slope_exponent": 3.0,
}

#: Packed-document position policy.  Documents are concatenated with one
#: ``<eod>`` between them and positions are NOT reset at the boundary, and the
#: causal mask is NOT reset either -- the GPT-NeoX/Pythia default.  The policy
#: is a property of the data pipeline, not of any positional encoding, so it is
#: identical in all five arms; a per-arm choice here would confound the study.
PACKING_POLICY = {
    "reset_position_ids": False,
    "reset_attention_mask": False,
}


def base_config() -> ModelConfig:
    """The frozen M2 base model, with the template seed."""
    contract = load_contract()
    if contract.data_seed != DATA_SEED:
        raise SystemExit(
            f"data_seed mismatch: generator says {DATA_SEED}, contract says "
            f"{contract.data_seed}"
        )
    if contract.vocab_size != FROZEN_ARCHITECTURE["vocab_size"]:
        raise SystemExit(
            f"vocab_size mismatch: generator says "
            f"{FROZEN_ARCHITECTURE['vocab_size']}, contract says "
            f"{contract.vocab_size}"
        )

    return ModelConfig(
        pe_type="nope",
        init_seed=TEMPLATE_SEED,      # resolved per run from run_matrix.json
        data_seed=DATA_SEED,
        **FROZEN_ARCHITECTURE,
        **PACKING_POLICY,
        meta={
            "milestone": "M2",
            "token_budget": contract.target_tokens,
            "selected_tokens": contract.selected_tokens,
            "tokenizer": contract.tokenizer_path,
            "tokenizer_sha256": contract.tokenizer_sha256,
            "eod_token_id": contract.eod_id,
            "training_subset_manifest": contract.training_subset_path,
            "training_subset_sha256": contract.training_subset_sha256,
            "manifest_sha256": contract.manifest_hashes,
            "data_contract": "data/metadata/training_data_contract.json",
            "seed_policy": (
                "init_seed is a template here; production seeds come from "
                "configs/run_matrix.json. data_seed is fixed by M1 and is "
                "independent of the model seed."
            ),
        },
    )


def run_matrix(base: ModelConfig) -> dict:
    return {
        "run_seeds": list(RUN_SEEDS),
        "data_seed": DATA_SEED,
        "pe_types": list(PE_TYPES),
        "run_id_template": "p31az-<pe>-s<seed>-t50m-c512-v16k-<conf8>",
        "runs": [
            {"pe_type": pe, "init_seed": seed}
            for seed in RUN_SEEDS
            for pe in PE_TYPES
        ],
        "contract": (
            "Within one run seed, all five arms share init_seed; only pe_type "
            "varies. Optimizer, schedule, token budget and data order are held "
            "fixed across arms. data_seed=2026 is fixed by M1 and is never "
            "derived from the model seed. Adding seeds requires an explicit "
            "protocol amendment."
        ),
    }


def payloads() -> dict[Path, dict]:
    base = base_config()
    out: dict[Path, dict] = {BASE_PATH: base.to_dict()}
    for pe_type in PE_TYPES:
        out[ARM_DIR / f"{pe_type}.json"] = base.with_pe(pe_type).to_dict()
    out[RUN_MATRIX_PATH] = run_matrix(base)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify, do not write")
    args = parser.parse_args()

    drifted = []
    for path, payload in payloads().items():
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                drifted.append(path.relative_to(ROOT))
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")

    if args.check:
        if drifted:
            print("config drift detected in:")
            for path in drifted:
                print(f"  - {path}")
            print("run: python scripts/make_pe_configs.py")
            return 1
        print("all configs match the generator.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
