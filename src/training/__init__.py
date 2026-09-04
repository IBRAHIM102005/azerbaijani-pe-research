"""Training and compute utilities for M3."""

from .batching import (
    BatchCursor,
    SequentialTokenBatcher,
    TokenBatch,
)

from .benchmark import (
    BenchmarkCandidate,
    make_benchmark_candidates,
)

from .cache_builder import (
    CacheDocument,
    build_fast_token_cache,
    group_documents_by_processed_file,
    load_cache_documents,
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

from .launch import (
    CheckpointMilestone,
    RunPlan,
    build_plan_manifest,
    make_run_plans,
    resolve_checkpoint_milestones,
    write_plan_manifest,
)

from .optimizer import (
    build_optimizer,
    describe_optimizer_groups,
    learning_rate_at_step,
    set_optimizer_lr,
)

from .queue import (
    MatrixJob,
    build_matrix_jobs,
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
    # batching
    "BatchCursor",
    "SequentialTokenBatcher",
    "TokenBatch",

    # benchmark
    "BenchmarkCandidate",
    "make_benchmark_candidates",

    # cache builder
    "CacheDocument",
    "build_fast_token_cache",
    "group_documents_by_processed_file",
    "load_cache_documents",

    # checkpoint
    "capture_rng_state",
    "load_checkpoint",
    "restore_rng_state",
    "save_checkpoint",

    # data
    "ConsumptionRecord",
    "TokenBlockDataset",
    "build_consumption_plan",
    "build_token_cache",
    "expected_full_blocks",

    # launch
    "CheckpointMilestone",
    "RunPlan",
    "build_plan_manifest",
    "make_run_plans",
    "resolve_checkpoint_milestones",
    "write_plan_manifest",

    # optimizer
    "build_optimizer",
    "describe_optimizer_groups",
    "learning_rate_at_step",
    "set_optimizer_lr",

    # queue
    "MatrixJob",
    "build_matrix_jobs",

    # resume
    "load_training_checkpoint",
    "save_training_checkpoint",

    # runner
    "RunSummary",
    "TrainingRunner",

    # trainer
    "StepResult",
    "Trainer",
    "TrainingState",
]