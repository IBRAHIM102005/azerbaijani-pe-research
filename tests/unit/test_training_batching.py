import numpy as np
import torch

from src.training.batching import (
    BatchCursor,
    SequentialTokenBatcher,
)


def make_cache(
    path,
    total_tokens,
):
    np.arange(
        total_tokens,
        dtype=np.uint16,
    ).tofile(path)


def test_full_microbatch():
    path = "unused"


def test_full_microbatch_reads_exact_stream(
    tmp_path,
):
    path = tmp_path / "tokens.bin"

    make_cache(
        path,
        total_tokens=24,
    )

    batcher = SequentialTokenBatcher(
        path,
        total_tokens=24,
        seq_len=4,
        micro_batch_sequences=2,
        eod_id=1,
    )

    batch = batcher.next_batch()

    assert batch is not None

    assert batch.real_tokens == 8
    assert batch.start_offset == 0
    assert batch.end_offset == 8

    assert batch.input_ids.shape == (
        2,
        4,
    )

    assert torch.equal(
        batch.input_ids,
        torch.tensor(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
            ]
        ),
    )

    assert torch.equal(
        batch.labels,
        batch.input_ids,
    )

    assert not batch.is_partial

    assert batcher.token_offset == 8


def test_batches_preserve_sequential_order(
    tmp_path,
):
    path = tmp_path / "tokens.bin"

    make_cache(
        path,
        total_tokens=20,
    )

    batcher = SequentialTokenBatcher(
        path,
        total_tokens=20,
        seq_len=5,
        micro_batch_sequences=2,
        eod_id=1,
    )

    first = batcher.next_batch()
    second = batcher.next_batch()

    assert first is not None
    assert second is not None

    assert first.start_offset == 0
    assert first.end_offset == 10

    assert second.start_offset == 10
    assert second.end_offset == 20

    assert batcher.exhausted
    assert batcher.next_batch() is None


def test_final_partial_batch_is_padded_and_masked(
    tmp_path,
):
    path = tmp_path / "tokens.bin"

    make_cache(
        path,
        total_tokens=11,
    )

    batcher = SequentialTokenBatcher(
        path,
        total_tokens=11,
        seq_len=4,
        micro_batch_sequences=2,
        eod_id=1,
    )

    first = batcher.next_batch()
    final = batcher.next_batch()

    assert first is not None
    assert final is not None

    assert final.real_tokens == 3
    assert final.start_offset == 8
    assert final.end_offset == 11

    assert final.input_ids.shape == (
        2,
        4,
    )

    # Genuine final tokens.
    assert torch.equal(
        final.input_ids.reshape(-1)[:3],
        torch.tensor(
            [8, 9, 10]
        ),
    )

    # Padding input uses EOD.
    assert torch.equal(
        final.input_ids.reshape(-1)[3:],
        torch.tensor(
            [1, 1, 1, 1, 1]
        ),
    )

    # Padding labels are ignored by CE.
    assert torch.equal(
        final.labels.reshape(-1)[:3],
        torch.tensor(
            [8, 9, 10]
        ),
    )

    assert torch.equal(
        final.labels.reshape(-1)[3:],
        torch.tensor(
            [-100, -100, -100, -100, -100]
        ),
    )

    assert final.is_partial
    assert batcher.exhausted


def test_cursor_resume_reads_next_tokens_only(
    tmp_path,
):
    path = tmp_path / "tokens.bin"

    make_cache(
        path,
        total_tokens=24,
    )

    first_batcher = SequentialTokenBatcher(
        path,
        total_tokens=24,
        seq_len=4,
        micro_batch_sequences=2,
        eod_id=1,
    )

    first = first_batcher.next_batch()

    assert first is not None
    assert first.end_offset == 8

    saved_state = (
        first_batcher.state_dict()
    )

    resumed = SequentialTokenBatcher(
        path,
        total_tokens=24,
        seq_len=4,
        micro_batch_sequences=2,
        eod_id=1,
    )

    resumed.load_state_dict(
        saved_state
    )

    second = resumed.next_batch()

    assert second is not None

    assert second.start_offset == 8
    assert second.end_offset == 16

    assert torch.equal(
        second.input_ids.reshape(-1),
        torch.tensor(
            [
                8, 9, 10, 11,
                12, 13, 14, 15,
            ]
        ),
    )


def test_explicit_cursor_start(
    tmp_path,
):
    path = tmp_path / "tokens.bin"

    make_cache(
        path,
        total_tokens=16,
    )

    batcher = SequentialTokenBatcher(
        path,
        total_tokens=16,
        seq_len=4,
        micro_batch_sequences=2,
        eod_id=1,
        cursor=BatchCursor(
            token_offset=8
        ),
    )

    batch = batcher.next_batch()

    assert batch is not None

    assert torch.equal(
        batch.input_ids.reshape(-1),
        torch.tensor(
            [
                8, 9, 10, 11,
                12, 13, 14, 15,
            ]
        ),
    )


def test_cursor_never_counts_padding(
    tmp_path,
):
    path = tmp_path / "tokens.bin"

    make_cache(
        path,
        total_tokens=10,
    )

    batcher = SequentialTokenBatcher(
        path,
        total_tokens=10,
        seq_len=4,
        micro_batch_sequences=2,
        eod_id=1,
    )

    batcher.next_batch()
    final = batcher.next_batch()

    assert final is not None
    assert final.real_tokens == 2

    # Even though returned tensor has 8 positions,
    # frozen-data cursor stops at exactly 10.
    assert batcher.token_offset == 10
    assert batcher.remaining_tokens == 0