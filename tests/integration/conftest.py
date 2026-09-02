import sys
import warnings
from pathlib import Path

import pytest


from src.reproducibility.adapters import MissingInterfaceError, checkpoint_adapter 


@pytest.fixture(scope="session")
def checkpoint_fns():
    """(save_checkpoint, load_checkpoint) -- real other's module if importable,
    otherwise the local reference implementation (SYNTHETIC_FIXTURE) so this
    integration-test *pattern* can run before other's lands. Once other's real
    training.checkpoint module exists, this fixture picks it up with zero
    changes to the test bodies below.
    """
    try:
        return checkpoint_adapter()
    except MissingInterfaceError:
        warnings.warn(
            "other's training.checkpoint module not found; running the "
            "checkpoint-integration TEST PATTERN against the local reference "
            "checkpoint implementation instead. This validates the test "
            "harness, not other's's real checkpoint system.",
            stacklevel=2,
        )
        from src.reproducibility.reference_checkpoint import save_checkpoint, load_checkpoint

        return save_checkpoint, load_checkpoint
