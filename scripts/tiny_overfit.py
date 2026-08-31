#!/usr/bin/env python
"""Run the preregistered tiny-overfit gate for all five arms.

The gate contract lives in ``src/models/gates.py`` so that this script and
``tests/test_tiny_overfit.py`` cannot drift apart: 32 fixed sequences,
dropout 0, five arms, and a frozen pass threshold.

This is the M2 -> M3 hand-off gate, not an experimental result.  The JSON it
writes is evidence that the arms are wired correctly and says nothing about
which positional encoding is better.

Usage
-----
    python scripts/tiny_overfit.py
    python scripts/tiny_overfit.py --device cuda
    python scripts/tiny_overfit.py --pilot      # rewrite the pilot evidence
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.gates import (  # noqa: E402
    GATE_INIT_SEED,
    GATE_LOSS_THRESHOLD,
    GATE_LR,
    GATE_SEQ_LEN,
    GATE_SEQUENCES,
    GATE_STEPS,
    run_gate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=GATE_STEPS)
    parser.add_argument(
        "--pilot", action="store_true",
        help="write results/m2_tiny_overfit_pilot.json instead of the gate artifact",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    results = run_gate(steps=args.steps, device=device)

    print(
        f"tiny-overfit gate: {GATE_SEQUENCES} fixed sequences x {GATE_SEQ_LEN} tokens, "
        f"dropout=0, init_seed={GATE_INIT_SEED}, {args.steps} steps, lr={GATE_LR}"
    )
    print(f"frozen pass threshold: final loss < {GATE_LOSS_THRESHOLD}\n")
    print(f"{'PE':<12}{'initial':>10}{'final':>12}{'sec':>8}  gate")
    print("-" * 50)
    for row in results:
        print(
            f"{row.pe_type:<12}{row.initial_loss:>10.3f}{row.final_loss:>12.5f}"
            f"{row.seconds:>8.1f}  {'pass' if row.passed else 'FAIL'}"
        )

    failed = [r.pe_type for r in results if not r.passed]
    name = "m2_tiny_overfit_pilot.json" if args.pilot else "m2_tiny_overfit.json"
    out = args.out or ROOT / "results" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    # Wall-clock timestamps and per-arm durations are deliberately NOT written:
    # they would make this artifact differ on every run, so `git status` could
    # never be clean after executing the validation gates.  The losses are the
    # evidence; they are deterministic for a given environment.
    out.write_text(
        json.dumps(
            {
                "artifact": "pilot evidence" if args.pilot else "gate result",
                "note": "wiring check only; not an experimental result",
                "contract": {
                    "sequences": GATE_SEQUENCES,
                    "seq_len": GATE_SEQ_LEN,
                    "dropout": 0.0,
                    "steps": args.steps,
                    "lr": GATE_LR,
                    "init_seed": GATE_INIT_SEED,
                    "threshold": GATE_LOSS_THRESHOLD,
                },
                "environment": {
                    "device": str(device),
                    "torch": torch.__version__,
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                },
                "passed": not failed,
                "worst_final_loss": max(r.final_loss for r in results),
                "arms": [r.to_dict() for r in results],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {out.relative_to(ROOT)}")

    if failed:
        print(f"gate FAILED for: {', '.join(failed)}")
        return 1
    print("gate passed: all five arms memorise the 32 fixed sequences.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
