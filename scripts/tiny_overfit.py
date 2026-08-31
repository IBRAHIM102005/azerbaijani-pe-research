#!/usr/bin/env python
"""Run the tiny-overfit gate for all five arms and record the curves.

This is the M2 -> M3 hand-off gate: no arm is allowed onto the real corpus
until it can memorise a small fixed batch.  The JSON artifact it writes is
evidence for the report, not an experimental result -- it says nothing about
which positional encoding is better.

Usage
-----
    python scripts/tiny_overfit.py
    python scripts/tiny_overfit.py --steps 400 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.config import PE_TYPES, ModelConfig 
from src.models.transformer import PELanguageModel

VOCAB = 32
SEQ_LEN = 24
BATCH = 4


def build_batch(device: torch.device) -> torch.Tensor:
    gen = torch.Generator().manual_seed(1234)
    return torch.randint(0, VOCAB, (BATCH, SEQ_LEN), generator=gen).to(device)


def run_arm(pe_type: str, steps: int, lr: float, device: torch.device) -> dict:
    torch.manual_seed(0)
    config = ModelConfig(
        pe_type=pe_type,
        vocab_size=VOCAB,
        n_layer=2,
        n_head=4,
        d_model=64,
        d_ff=128,
        max_seq_len=SEQ_LEN,
        dropout=0.0,
    )
    model = PELanguageModel(config).to(device)
    batch = build_batch(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    curve = []
    started = time.time()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        _, loss = model(batch, labels=batch)
        loss.backward()
        opt.step()
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            curve.append({"step": step, "loss": round(loss.item(), 6)})

    return {
        "pe_type": pe_type,
        "steps": steps,
        "initial_loss": curve[0]["loss"],
        "final_loss": curve[-1]["loss"],
        "seconds": round(time.time() - started, 2),
        "curve": curve,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results" / "m2_tiny_overfit.json"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    results = [run_arm(pe, args.steps, args.lr, device) for pe in PE_TYPES]

    print(f"{'PE':<12}{'initial':>10}{'final':>10}{'sec':>8}  gate")
    print("-" * 48)
    failed = []
    for row in results:
        ok = row["final_loss"] < args.threshold
        failed += [] if ok else [row["pe_type"]]
        print(
            f"{row['pe_type']:<12}{row['initial_loss']:>10.3f}"
            f"{row['final_loss']:>10.4f}{row['seconds']:>8.1f}  "
            f"{'pass' if ok else 'FAIL'}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "threshold": args.threshold,
                "lr": args.lr,
                "device": str(device),
                "passed": not failed,
                "arms": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {args.out.relative_to(ROOT)}")

    if failed:
        print(f"gate FAILED for: {', '.join(failed)}")
        return 1
    print("gate passed: all five arms memorise the batch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
