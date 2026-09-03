import sys
import warnings
from pathlib import Path

import pytest


from src.reproducibility.adapters import MissingInterfaceError, checkpoint_adapter  # noqa: E402


@pytest.fixture(scope="session")
def checkpoint_fns():
    """(save_checkpoint, load_checkpoint) -- Fidan's real module if importable,
    otherwise the local reference implementation (SYNTHETIC_FIXTURE) so this
    integration-test *pattern* can run before Fidan lands. Once Fidan's real
    training.checkpoint module exists, this fixture picks it up with zero
    changes to the test bodies below.
    """
    try:
        return checkpoint_adapter()
    except MissingInterfaceError:
        warnings.warn(
            "Fidan's training.checkpoint module not found; running the "
            "checkpoint-integration TEST PATTERN against the local reference "
            "checkpoint implementation instead. This validates the test "
            "harness, not Fidan's real checkpoint system.",
            stacklevel=2,
        )
        from src.reproducibility.reference_checkpoint import save_checkpoint, load_checkpoint

        return save_checkpoint, load_checkpoint
