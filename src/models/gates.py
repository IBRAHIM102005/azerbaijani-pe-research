"""The tiny-overfit gate, defined once and used by both the script and the test.

This is a *wiring* check, not an experiment.  It answers one question: can each
arm drive the loss to near zero on a batch small enough to memorise?  An arm
that cannot has a broken gradient path, and no GPU time should be spent on it.

Nothing produced here is a result.  It says nothing about which positional
encoding is better, and it must never appear in the results table.

The contract below is preregistered: 32 fixed sequences, dropout 0, all five
arms, and a pass threshold frozen from pilot evidence before the real runs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List

import torch

from .config import PE_TYPES, ModelConfig
from .transformer import PELanguageModel

# --- preregistered gate contract -------------------------------------------
#: number of fixed sequences the arm must memorise
GATE_SEQUENCES = 32
#: length of each sequence
GATE_SEQ_LEN = 64
#: toy vocabulary for the gate
GATE_VOCAB = 128
#: seed for the fixed batch; the batch is identical for every arm and every run
GATE_BATCH_SEED = 1234
#: optimisation budget
GATE_STEPS = 300
GATE_LR = 3e-3
#: model seed used for the gate (a preregistered run seed)
GATE_INIT_SEED = 17

#: Frozen pass threshold.  Pilot evidence (results/m2_tiny_overfit_pilot.json)
#: put every arm's final loss between 0.0035 and 0.0058, from a starting loss
#: near ln(128) = 4.85.  The threshold is frozen at roughly nine times the
#: worst observed value: high enough to be insensitive to hardware, dtype and
#: BLAS ordering, low enough that an arm which fails to memorise 32 sequences
#: cannot pass.  Changing it is a protocol amendment, not a tuning knob.
GATE_LOSS_THRESHOLD = 0.05

__all__ = [
    "GATE_SEQUENCES",
    "GATE_SEQ_LEN",
    "GATE_VOCAB",
    "GATE_STEPS",
    "GATE_LR",
    "GATE_INIT_SEED",
    "GATE_LOSS_THRESHOLD",
    "ArmResult",
    "gate_config",
    "gate_batch",
    "run_arm",
    "run_gate",
]


def gate_config(pe_type: str) -> ModelConfig:
    """The gate model.  Dropout is 0, as the contract requires."""
    return ModelConfig(
        pe_type=pe_type,
        init_seed=GATE_INIT_SEED,
        vocab_size=GATE_VOCAB,
        n_layer=2,
        n_head=4,
        d_model=64,
        d_ff=128,
        max_seq_len=GATE_SEQ_LEN,
        dropout=0.0,
    )


def gate_batch(device: torch.device | None = None) -> torch.Tensor:
    """The 32 fixed sequences.  Identical for every arm, every run, every host."""
    generator = torch.Generator().manual_seed(GATE_BATCH_SEED)
    batch = torch.randint(
        0, GATE_VOCAB, (GATE_SEQUENCES, GATE_SEQ_LEN), generator=generator
    )
    return batch if device is None else batch.to(device)


@dataclass
class ArmResult:
    pe_type: str
    initial_loss: float
    final_loss: float
    seconds: float
    curve: List[Dict[str, float]]

    @property
    def passed(self) -> bool:
        return self.final_loss < GATE_LOSS_THRESHOLD

    def to_dict(self) -> Dict[str, object]:
        """Serialisable form.  ``seconds`` is excluded on purpose: it varies
        run to run and would make the stored artifact dirty the working tree
        every time the gate is executed."""
        return {
            "pe_type": self.pe_type,
            "initial_loss": round(self.initial_loss, 6),
            "final_loss": round(self.final_loss, 6),
            "passed": self.passed,
            "curve": self.curve,
        }


def run_arm(
    pe_type: str,
    steps: int = GATE_STEPS,
    lr: float = GATE_LR,
    device: torch.device | None = None,
) -> ArmResult:
    device = device or torch.device("cpu")
    torch.manual_seed(0)

    config = gate_config(pe_type)
    assert config.dropout == 0.0, "the gate contract requires dropout=0"
    model = PELanguageModel(config).to(device)
    batch = gate_batch(device)
    assert batch.shape == (GATE_SEQUENCES, GATE_SEQ_LEN)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    curve: List[Dict[str, float]] = []
    started = time.time()

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(batch, labels=batch)
        loss.backward()
        optimizer.step()
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            curve.append({"step": step, "loss": round(loss.item(), 6)})

    return ArmResult(
        pe_type=pe_type,
        initial_loss=curve[0]["loss"],
        final_loss=curve[-1]["loss"],
        seconds=time.time() - started,
        curve=curve,
    )


def run_gate(
    steps: int = GATE_STEPS, lr: float = GATE_LR, device: torch.device | None = None
) -> List[ArmResult]:
    return [run_arm(pe, steps, lr, device) for pe in PE_TYPES]
