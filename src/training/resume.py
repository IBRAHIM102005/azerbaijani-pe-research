"""High-level save/resume integration for M3 training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .checkpoint import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)

from .trainer import Trainer


TRAINER_STATE_KEY = "trainer_state"
USER_EXTRA_KEY = "user_extra"


def save_training_checkpoint(
    path: str | Path,
    trainer: Trainer,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save everything required to resume training safely."""

    if not trainer.at_accumulation_boundary:
        raise RuntimeError(
            "Training checkpoints may only be saved "
            "at gradient-accumulation boundaries."
        )

    payload_extra = {
        TRAINER_STATE_KEY: trainer.state_dict(),
        USER_EXTRA_KEY: extra or {},
    }

    save_checkpoint(
        path=path,
        model=trainer.model,
        optimizer=trainer.optimizer,
        rng_state=capture_rng_state(),
        tokens_seen=trainer.state.tokens_seen,
        extra=payload_extra,
    )


def load_training_checkpoint(
    path: str | Path,
    trainer: Trainer,
) -> dict[str, Any]:
    """Restore model, optimizer, trainer state and RNG."""

    metadata = load_checkpoint(
        path=path,
        model=trainer.model,
        optimizer=trainer.optimizer,
        map_location=trainer.device,
    )

    extra = metadata.get(
        "extra",
        {},
    )

    if TRAINER_STATE_KEY not in extra:
        raise ValueError(
            "Checkpoint does not contain "
            "trainer resume state."
        )

    trainer.load_state_dict(
        extra[TRAINER_STATE_KEY]
    )

    if (
        trainer.state.tokens_seen
        != metadata["tokens_seen"]
    ):
        raise ValueError(
            "tokens_seen mismatch between "
            "checkpoint and trainer state."
        )

    restore_rng_state(
        metadata["rng_state"]
    )

    return {
        "tokens_seen": (
            trainer.state.tokens_seen
        ),
        "micro_step": (
            trainer.state.micro_step
        ),
        "optimizer_step": (
            trainer.state.optimizer_step
        ),
        "extra": extra.get(
            USER_EXTRA_KEY,
            {},
        ),
    }