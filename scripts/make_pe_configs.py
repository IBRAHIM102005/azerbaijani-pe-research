#!/usr/bin/env python
"""Emit ``configs/model_base.json`` and the five per-arm configs.

The arm configs are generated, never hand-edited: they are byte-identical to
the base config except for ``pe_type``.  ``--check`` re-generates them in
memory and fails if the files on disk have drifted, which is what CI runs.

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

from src.models.config import PE_TYPES, ModelConfig  # noqa: E402

BASE_PATH = ROOT / "configs" / "model_base.json"
ARM_DIR = ROOT / "configs" / "pe"


#: The multiple-seed contract.  For run seed ``s``, all five arms are built
#: with ``init_seed = s``; arms are never given different seeds within a run,
#: and a seed is never reused for a different arm ordering.  M3 iterates over
#: this list, producing 5 arms x len(RUN_SEEDS) runs.
RUN_SEEDS = (2026, 2027, 2028)


def base_config() -> ModelConfig:
    """The frozen M2 base model.

    Sized so that five arms x three seeds fit inside a free Kaggle weekly GPU
    quota at the 50,000,000-token budget fixed in M1.
    """
    return ModelConfig(
        pe_type="nope",
        vocab_size=16000,          # frozen shared SentencePiece tokenizer
        n_layer=6,
        n_head=8,
        d_model=256,
        d_ff=1024,
        max_seq_len=512,
        dropout=0.0,
        tie_embeddings=False,   # frozen spec: separate output head
        init_seed=2026,
        meta={
            "milestone": "M2",
            "token_budget": 50_000_000,
            "tokenizer": "tokenizer/az_bpe_16000",
            "data_contract": "data/metadata/training_data_contract.json",
        },
    )


def payloads() -> dict[Path, dict]:
    base = base_config()
    out = {BASE_PATH: base.to_dict()}
    for pe_type in PE_TYPES:
        out[ARM_DIR / f"{pe_type}.json"] = base.with_pe(pe_type).to_dict()
    out[ROOT / "configs" / "run_matrix.json"] = {
        "run_seeds": list(RUN_SEEDS),
        "pe_types": list(PE_TYPES),
        "runs": [
            {"pe_type": pe, "init_seed": seed, "run_id": f"{pe}-s{seed}"}
            for seed in RUN_SEEDS
            for pe in PE_TYPES
        ],
        "contract": (
            "Within one run seed, all five arms share init_seed; only pe_type "
            "varies. Data order, optimizer, schedule and token budget are also "
            "held fixed across arms."
        ),
    }
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
