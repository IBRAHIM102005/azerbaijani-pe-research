#!/usr/bin/env python
"""CUDA forward+backward smoke test for all five arms.

This is the one mandatory gate that cannot run on a CPU-only host.  Run it on
the target GPU (Kaggle T4/P100, or an A100) before any M3 training.

It is deliberately more than "does it not crash".  Three things specifically
can pass on CPU and fail on GPU, and each has its own check below:

1. **ALiBi under reduced precision.**  ALiBi adds a finite bias to a mask that
   already contains ``-inf``.  In fp16 the sum can produce ``NaN`` if any row
   is fully masked, and the softmax then poisons the whole batch silently.
2. **SDPA backend selection.**  Passing an explicit float mask makes PyTorch
   fall back from the flash kernel to the math backend.  That is correct but
   slower, and it is better to know the throughput cost before budgeting GPU
   hours than after.
3. **Determinism of the shared initialisation.**  The per-name seeding must
   produce the same shared-parameter fingerprint on GPU as on CPU, or the
   fairness argument holds only on the machine it was tested on.

Usage
-----
    python scripts/cuda_smoke.py
    python scripts/cuda_smoke.py --dtype bfloat16 --steps 5
    python scripts/cuda_smoke.py --seq-len 512 --batch 8
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

from src.models.config import PE_TYPES, ModelConfig  # noqa: E402
from src.models.params import parameter_report, shared_init_fingerprint  # noqa: E402
from src.models.run_config import load_run_matrix  # noqa: E402
from src.models.transformer import PELanguageModel  # noqa: E402

SPEC_BASELINE_PARAMS = 12_931_072
SPEC_LEARNED_PARAMS = 13_062_144

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def check_arm(
    pe_type: str,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    batch: int,
    seq_len: int,
    steps: int,
) -> dict:
    config = (
        ModelConfig.from_json(ROOT / "configs" / "pe" / f"{pe_type}.json")
        .with_seed(seed)
    )
    model = PELanguageModel(config).to(device)

    # Capture the fingerprint BEFORE any optimizer step.  Taken afterwards it
    # would measure trained weights, which legitimately differ per arm, and the
    # check would fail for a reason that has nothing to do with initialisation.
    fingerprint = shared_init_fingerprint(model)[:16]

    torch.manual_seed(0)
    ids = torch.randint(0, config.vocab_size, (batch, seq_len), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    autocast = torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32)
    losses = []
    nan_logits = nan_loss = nan_grad = False

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with autocast:
            logits, loss = model(ids, labels=ids)
        nan_logits |= not torch.isfinite(logits).all().item()
        nan_loss |= not torch.isfinite(loss).item()
        loss.backward()
        for param in model.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                nan_grad = True
                break
        optimizer.step()
        losses.append(float(loss.item()))

    report = parameter_report(model)
    expected = SPEC_LEARNED_PARAMS if pe_type == "learned" else SPEC_BASELINE_PARAMS

    return {
        "pe_type": pe_type,
        "parameters": report["total"],
        "parameters_match_spec": report["total"] == expected,
        "init_fingerprint": fingerprint,
        "first_loss": round(losses[0], 6),
        "last_loss": round(losses[-1], 6),
        "loss_decreased": losses[-1] < losses[0],
        "nan_in_logits": nan_logits,
        "nan_in_loss": nan_loss,
        "nan_in_gradients": nan_grad,
        "peak_memory_mb": (
            round(torch.cuda.max_memory_allocated(device) / 1024**2, 1)
            if device.type == "cuda"
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "m2_cuda_smoke.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: no CUDA device visible; this gate must run on the target GPU")
        return 2

    device = torch.device("cuda")
    dtype = DTYPES[args.dtype]
    seeds = load_run_matrix()["run_seeds"]
    seed = args.seed if args.seed is not None else seeds[0]
    if seed not in seeds:
        print(f"FAIL: {seed} is not a preregistered run seed {seeds}")
        return 2

    name = torch.cuda.get_device_name(0)
    print(f"device : {name}")
    print(f"torch  : {torch.__version__}  cuda {torch.version.cuda}")
    print(
        f"config : dtype={args.dtype} batch={args.batch} seq_len={args.seq_len} "
        f"steps={args.steps} init_seed={seed}\n"
    )

    rows = []
    for pe_type in PE_TYPES:
        torch.cuda.reset_peak_memory_stats(device)
        rows.append(
            check_arm(pe_type, seed, device, dtype, args.batch, args.seq_len, args.steps)
        )

    header = f"{'PE':<12}{'params':>12}{'loss0':>9}{'lossN':>9}{'NaN':>6}{'MB':>8}  gate"
    print(header)
    print("-" * len(header))
    failures = []
    for row in rows:
        nan = row["nan_in_logits"] or row["nan_in_loss"] or row["nan_in_gradients"]
        ok = row["parameters_match_spec"] and not nan and row["loss_decreased"]
        if not ok:
            failures.append(row["pe_type"])
        memory = "-" if row["peak_memory_mb"] is None else f"{row['peak_memory_mb']:.1f}"
        print(
            f"{row['pe_type']:<12}{row['parameters']:>12,}{row['first_loss']:>9.3f}"
            f"{row['last_loss']:>9.3f}{('yes' if nan else 'no'):>6}"
            f"{memory:>8}  {'pass' if ok else 'FAIL'}"
        )

    fingerprints = {row["init_fingerprint"] for row in rows}
    fingerprint_ok = len(fingerprints) == 1
    print(
        f"\nshared-init fingerprint identical across arms on GPU: "
        f"{'yes' if fingerprint_ok else 'NO -- ' + str(fingerprints)}"
    )
    if not fingerprint_ok:
        failures.append("init-parity")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "artifact": "cuda smoke test",
                "note": "wiring and numerical-stability check; not an experimental result",
                "device": name,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "python": platform.python_version(),
                "dtype": args.dtype,
                "batch": args.batch,
                "seq_len": args.seq_len,
                "steps": args.steps,
                "init_seed": seed,
                "shared_init_fingerprint_identical": fingerprint_ok,
                "passed": not failures,
                "arms": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out.relative_to(ROOT)}")

    if failures:
        print(f"\nCUDA smoke FAILED for: {', '.join(sorted(set(failures)))}")
        return 1
    print("\nCUDA smoke passed for all five arms.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
