"""End-to-end training runner for M3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .batching import (
    SequentialTokenBatcher,
)

from .resume import (
    load_training_checkpoint,
    save_training_checkpoint,
)

from .trainer import (
    StepResult,
    Trainer,
)


@dataclass(frozen=True)
class RunSummary:
    """Summary of one runner invocation."""

    start_tokens: int
    end_tokens: int

    microbatches_processed: int
    optimizer_steps_processed: int

    last_loss: float | None

    exhausted: bool
    final_accumulation_flushed: bool
    checkpointable: bool


class TrainingRunner:
    """Connect the frozen token stream to the Trainer.

    Responsibilities:

    - read deterministic batches
    - feed them to Trainer
    - keep data cursor and tokens_seen synchronized
    - save both Trainer and Batcher state
    - resume without token skip/repeat
    - flush final incomplete accumulation cycle
    """

    def __init__(
        self,
        trainer: Trainer,
        batcher: SequentialTokenBatcher,
    ) -> None:
        self.trainer = trainer
        self.batcher = batcher

        self._assert_synchronized()

    # --------------------------------------------------
    # State consistency
    # --------------------------------------------------

    def _assert_synchronized(
        self,
    ) -> None:
        """Ensure trainer and data stream describe the same position."""

        trainer_tokens = (
            self.trainer.state.tokens_seen
        )

        batcher_tokens = (
            self.batcher.token_offset
        )

        if trainer_tokens != batcher_tokens:
            raise RuntimeError(
                "Trainer/batcher token position mismatch: "
                f"trainer.tokens_seen={trainer_tokens}, "
                f"batcher.token_offset={batcher_tokens}"
            )

    @property
    def can_checkpoint(self) -> bool:
        """True when checkpointing is currently safe."""

        return (
            self.trainer.at_accumulation_boundary
            and self.trainer.state.tokens_seen
            == self.batcher.token_offset
        )

    # --------------------------------------------------
    # Checkpoint
    # --------------------------------------------------

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Save model + optimizer + RNG + trainer + data cursor."""

        self._assert_synchronized()

        if not self.can_checkpoint:
            raise RuntimeError(
                "Cannot checkpoint in the middle "
                "of a gradient accumulation cycle."
            )

        runner_state = {
            "batcher_state": (
                self.batcher.state_dict()
            ),
            "runner_extra": (
                extra or {}
            ),
        }

        save_training_checkpoint(
            path,
            self.trainer,
            extra=runner_state,
        )

    def load_checkpoint(
        self,
        path: str | Path,
    ) -> dict[str, Any]:
        """Restore complete training state."""

        restored = load_training_checkpoint(
            path,
            self.trainer,
        )

        extra = restored.get(
            "extra",
            {},
        )

        if "batcher_state" not in extra:
            raise ValueError(
                "Checkpoint does not contain "
                "batcher state."
            )

        self.batcher.load_state_dict(
            extra["batcher_state"]
        )

        self._assert_synchronized()

        return extra.get(
            "runner_extra",
            {},
        )

    # --------------------------------------------------
    # Run
    # --------------------------------------------------

    def run(
        self,
        *,
        max_microbatches: int | None = None,
        on_step: Callable[
            [StepResult],
            None,
        ]
        | None = None,
    ) -> RunSummary:
        """Run training from the current cursor.

        If ``max_microbatches`` is None, consume the
        entire remaining frozen token stream.

        If a limit is supplied, training stops after
        that many microbatches. A partial accumulation
        cycle is NOT flushed unless the dataset itself
        is exhausted.

        This behavior is important because stopping a
        smoke/debug invocation must not silently alter
        the optimization trajectory.
        """

        if (
            max_microbatches is not None
            and max_microbatches <= 0
        ):
            raise ValueError(
                "max_microbatches must be positive"
            )

        self._assert_synchronized()

        start_tokens = (
            self.trainer.state.tokens_seen
        )

        start_optimizer_steps = (
            self.trainer.state.optimizer_step
        )

        processed = 0

        last_result: StepResult | None = None

        while not self.batcher.exhausted:

            if (
                max_microbatches is not None
                and processed
                >= max_microbatches
            ):
                break

            batch = (
                self.batcher.next_batch()
            )

            if batch is None:
                break

            result = (
                self.trainer.train_microbatch(
                    batch.input_ids,
                    batch.labels,
                    real_tokens=(
                        batch.real_tokens
                    ),
                )
            )

            processed += 1

            last_result = result

            # Critical invariant:
            #
            # trainer's token budget and the frozen
            # dataset cursor must never diverge.
            self._assert_synchronized()

            if on_step is not None:
                on_step(
                    result
                )

        final_flushed = False

        # Only flush incomplete gradient accumulation
        # when the actual frozen stream has ended.
        #
        # Do NOT flush simply because a smoke-test
        # max_microbatches limit was reached.
        if self.batcher.exhausted:

            flush_result = (
                self.trainer
                .finalize_accumulation()
            )

            if flush_result is not None:
                final_flushed = True

        self._assert_synchronized()

        end_tokens = (
            self.trainer.state.tokens_seen
        )

        optimizer_steps_processed = (
            self.trainer.state.optimizer_step
            - start_optimizer_steps
        )

        return RunSummary(
            start_tokens=start_tokens,
            end_tokens=end_tokens,
            microbatches_processed=(
                processed
            ),
            optimizer_steps_processed=(
                optimizer_steps_processed
            ),
            last_loss=(
                None
                if last_result is None
                else last_result.loss
            ),
            exhausted=(
                self.batcher.exhausted
            ),
            final_accumulation_flushed=(
                final_flushed
            ),
            checkpointable=(
                self.can_checkpoint
            ),
        )