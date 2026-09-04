"""Deterministic sequential batching for the frozen M1 token stream."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class BatchCursor:
    """Current position in the frozen token stream."""

    token_offset: int = 0

    def state_dict(self) -> dict[str, int]:
        return {
            "token_offset": int(self.token_offset),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: dict,
    ) -> "BatchCursor":
        return cls(
            token_offset=int(
                state["token_offset"]
            )
        )


@dataclass
class TokenBatch:
    """One deterministic training microbatch."""

    input_ids: torch.Tensor
    labels: torch.Tensor

    # Number of genuine M1 tokens consumed.
    real_tokens: int

    # Offset before this batch.
    start_offset: int

    # Offset immediately after this batch.
    end_offset: int

    @property
    def is_partial(self) -> bool:
        return bool(
            (self.labels == -100).any().item()
        )


class SequentialTokenBatcher:
    """Read the frozen uint16 token stream without shuffling.

    Full microbatches have shape:

        [micro_batch_sequences, seq_len]

    At the very end of the 50M-token stream, missing positions are
    padded in ``input_ids`` with ``eod_id`` and ignored in ``labels``
    with ``-100``.

    The cursor always counts only genuine M1 tokens, never padding.
    """

    def __init__(
        self,
        cache_path: str | Path,
        *,
        total_tokens: int,
        seq_len: int,
        micro_batch_sequences: int,
        eod_id: int,
        cursor: BatchCursor | None = None,
    ) -> None:
        if total_tokens <= 0:
            raise ValueError(
                "total_tokens must be positive"
            )

        if seq_len <= 1:
            raise ValueError(
                "seq_len must be greater than 1"
            )

        if micro_batch_sequences <= 0:
            raise ValueError(
                "micro_batch_sequences must be positive"
            )

        if eod_id < 0:
            raise ValueError(
                "eod_id must be non-negative"
            )

        self.cache_path = Path(
            cache_path
        ).resolve()

        if not self.cache_path.is_file():
            raise FileNotFoundError(
                f"Token cache not found: "
                f"{self.cache_path}"
            )

        expected_bytes = (
            total_tokens
            * np.dtype(np.uint16).itemsize
        )

        actual_bytes = (
            self.cache_path.stat().st_size
        )

        if actual_bytes != expected_bytes:
            raise ValueError(
                "Token-cache size mismatch: "
                f"expected={expected_bytes}, "
                f"actual={actual_bytes}"
            )

        self.total_tokens = int(
            total_tokens
        )

        self.seq_len = int(
            seq_len
        )

        self.micro_batch_sequences = int(
            micro_batch_sequences
        )

        self.micro_batch_tokens = (
            self.seq_len
            * self.micro_batch_sequences
        )

        self.eod_id = int(
            eod_id
        )

        self.cursor = (
            cursor
            if cursor is not None
            else BatchCursor()
        )

        if not (
            0
            <= self.cursor.token_offset
            <= self.total_tokens
        ):
            raise ValueError(
                "cursor token_offset is outside "
                "the token stream"
            )

        self._tokens = np.memmap(
            self.cache_path,
            dtype=np.uint16,
            mode="r",
            shape=(self.total_tokens,),
        )

    @property
    def token_offset(self) -> int:
        return self.cursor.token_offset

    @property
    def remaining_tokens(self) -> int:
        return (
            self.total_tokens
            - self.token_offset
        )

    @property
    def exhausted(self) -> bool:
        return self.token_offset >= self.total_tokens

    def state_dict(self) -> dict:
        return {
            "cursor": (
                self.cursor.state_dict()
            ),
            "total_tokens": (
                self.total_tokens
            ),
            "seq_len": (
                self.seq_len
            ),
            "micro_batch_sequences": (
                self.micro_batch_sequences
            ),
            "eod_id": (
                self.eod_id
            ),
        }

    def load_state_dict(
        self,
        state: dict,
    ) -> None:
        """Restore the exact frozen-stream position."""

        expected = {
            "total_tokens": (
                self.total_tokens
            ),
            "seq_len": (
                self.seq_len
            ),
            "micro_batch_sequences": (
                self.micro_batch_sequences
            ),
            "eod_id": (
                self.eod_id
            ),
        }

        for key, expected_value in expected.items():
            actual = state.get(key)

            if actual != expected_value:
                raise ValueError(
                    "Batcher config mismatch "
                    f"while resuming: {key}: "
                    f"checkpoint={actual!r}, "
                    f"current={expected_value!r}"
                )

        cursor = BatchCursor.from_state_dict(
            state["cursor"]
        )

        if not (
            0
            <= cursor.token_offset
            <= self.total_tokens
        ):
            raise ValueError(
                "Restored cursor lies outside "
                "the token stream."
            )

        self.cursor = cursor

    def next_batch(
        self,
    ) -> TokenBatch | None:
        """Return the next microbatch and advance the cursor."""

        if self.exhausted:
            return None

        start = self.token_offset

        real_tokens = min(
            self.micro_batch_tokens,
            self.remaining_tokens,
        )

        end = (
            start + real_tokens
        )

        flat = np.array(
            self._tokens[start:end],
            dtype=np.int64,
            copy=True,
        )

        # Full rectangular microbatch capacity.
        capacity = self.micro_batch_tokens

        input_flat = np.full(
            capacity,
            fill_value=self.eod_id,
            dtype=np.int64,
        )

        label_flat = np.full(
            capacity,
            fill_value=-100,
            dtype=np.int64,
        )

        input_flat[:real_tokens] = flat
        label_flat[:real_tokens] = flat

        input_ids = torch.from_numpy(
            input_flat.reshape(
                self.micro_batch_sequences,
                self.seq_len,
            )
        )

        labels = torch.from_numpy(
            label_flat.reshape(
                self.micro_batch_sequences,
                self.seq_len,
            )
        )

        self.cursor = BatchCursor(
            token_offset=end
        )

        return TokenBatch(
            input_ids=input_ids,
            labels=labels,
            real_tokens=real_tokens,
            start_offset=start,
            end_offset=end,
        )