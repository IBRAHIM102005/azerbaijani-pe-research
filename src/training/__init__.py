"""Training and compute utilities for M3."""

from .batching import (
    BatchCursor,
    SequentialTokenBatcher,
    TokenBatch,
)

from .checkpoint import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)

from .data import (
    ConsumptionRecord,
    TokenBlockDataset,
    build_consumption_plan,
    build_token_cache,
    expected_full_blocks,
)

from .optimizer import (
    build_optimizer,
    learning_rate_at_step,
    set_optimizer_lr,
)

from .resume import (
    load_training_checkpoint,
    save_training_checkpoint,
)

from .runner import (
    RunSummary,
    TrainingRunner,
)

from .trainer import (
    StepResult,
    Trainer,
    TrainingState,
)


__all__ = [
    "BatchCursor",
    "SequentialTokenBatcher",
    "TokenBatch",
    "capture_rng_state",
    "load_checkpoint",
    "restore_rng_state",
    "save_checkpoint",
    "ConsumptionRecord",
    "TokenBlockDataset",
    "build_consumption_plan",
    "build_token_cache",
    "expected_full_blocks",
    "build_optimizer",
    "learning_rate_at_step",
    "set_optimizer_lr",
    "load_training_checkpoint",
    "save_training_checkpoint",
    "RunSummary",
    "TrainingRunner",
    "StepResult",
    "Trainer",
    "TrainingState",
]