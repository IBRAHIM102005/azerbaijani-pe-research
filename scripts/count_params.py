#!/usr/bin/env python
"""Print the M2 parameter-fairness table and store it as a results artifact.

Usage
-----
    python scripts/count_params.py
    python scripts/count_params.py --config configs/model_base.json
    python scripts/count_params.py --out results/m2_parameter_fairness.json

The table is what goes into the paper's experimental-setup section and what
answers the "were the arms actually comparable?" question at the defence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.config import PE_TYPES, ModelConfig  # noqa: E402
from src.models.params import fairness_report, format_fairness_table  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "model_base.json",
        help="base model config (its pe_type field is ignored)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "m2_parameter_fairness.json",
        help="where to write the machine-readable report",
    )
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    config = (
        ModelConfig.from_json(args.config)
        if args.config.exists()
        else ModelConfig()
    )
    if not args.config.exists():
        print(f"[warn] {args.config} not found; using built-in defaults\n")

    report = fairness_report(config, PE_TYPES)
    print(f"base config: {args.config.name}")
    print(
        f"  n_layer={config.n_layer} n_head={config.n_head} d_model={config.d_model} "
        f"d_ff={config.d_ff} max_seq_len={config.max_seq_len} "
        f"vocab_size={config.vocab_size} tied={config.tie_embeddings}\n"
    )
    print(format_fairness_table(report))

    if not args.no_write:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"base_config": config.to_dict(), "report": report}
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out.relative_to(ROOT)}")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
