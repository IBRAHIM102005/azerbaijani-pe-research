"""Matrix queue planning utilities for M3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class MatrixJob:
    """One training run waiting in the GPU queue."""

    index: int
    run_id: str
    pe_type: str
    init_seed: int
    run_dir: Path

    @property
    def latest_checkpoint(self) -> Path:
        return (
            self.run_dir
            / "checkpoints"
            / "latest.pt"
        )

    @property
    def completion_manifest(self) -> Path:
        return (
            self.run_dir
            / "completed.json"
        )

    @property
    def is_completed(self) -> bool:
        return (
            self.completion_manifest
            .is_file()
        )

    @property
    def can_resume(self) -> bool:
        return (
            self.latest_checkpoint
            .is_file()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "run_id": self.run_id,
            "pe_type": self.pe_type,
            "init_seed": self.init_seed,
            "run_dir": str(
                self.run_dir
            ),
            "is_completed": (
                self.is_completed
            ),
            "can_resume": (
                self.can_resume
            ),
        }


def build_matrix_jobs(
    plan_payload: dict[str, Any],
    *,
    repo_root: str | Path,
    pe_types: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
    include_completed: bool = False,
) -> list[MatrixJob]:
    """Build queue jobs from a frozen M3 run plan.

    Ordering is preserved exactly as it appears in the plan.
    """

    if "runs" not in plan_payload:
        raise ValueError(
            "Run plan is missing 'runs'."
        )

    runs = plan_payload["runs"]

    if not isinstance(
        runs,
        list,
    ):
        raise ValueError(
            "'runs' must be a list."
        )

    repo_root = Path(
        repo_root
    ).resolve()

    pe_filter = (
        None
        if pe_types is None
        else set(pe_types)
    )

    seed_filter = (
        None
        if seeds is None
        else {
            int(seed)
            for seed in seeds
        }
    )

    seen_run_ids: set[str] = set()

    jobs: list[
        MatrixJob
    ] = []

    for index, run in enumerate(
        runs,
        start=1,
    ):
        run_id = str(
            run["run_id"]
        )

        if run_id in seen_run_ids:
            raise ValueError(
                f"Duplicate run_id in plan: "
                f"{run_id}"
            )

        seen_run_ids.add(
            run_id
        )

        pe_type = str(
            run["pe_type"]
        )

        init_seed = int(
            run["init_seed"]
        )

        if (
            pe_filter is not None
            and pe_type not in pe_filter
        ):
            continue

        if (
            seed_filter is not None
            and init_seed not in seed_filter
        ):
            continue

        run_dir = Path(
            run["run_dir"]
        )

        if not run_dir.is_absolute():
            run_dir = (
                repo_root
                / run_dir
            )

        run_dir = (
            run_dir.resolve()
        )

        job = MatrixJob(
            index=index,
            run_id=run_id,
            pe_type=pe_type,
            init_seed=init_seed,
            run_dir=run_dir,
        )

        if (
            job.is_completed
            and not include_completed
        ):
            continue

        jobs.append(
            job
        )

    return jobs