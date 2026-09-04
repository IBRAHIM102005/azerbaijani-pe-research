#!/usr/bin/env python3
"""Launch the frozen M3 run matrix across one or more GPUs.

Design:

    GPU 0 -> one independent experiment
    GPU 1 -> another independent experiment
    ...

When a run finishes, the next queued run is assigned to the freed GPU.

This is experiment-level parallelism, not DDP.

Examples
--------

Dry run locally:

    python scripts/m3_launch_matrix.py --dry-run

On a machine where CUDA_VISIBLE_DEVICES is already configured:

    python scripts/m3_launch_matrix.py

Explicit GPUs:

    python scripts/m3_launch_matrix.py --gpus 0 1

Only RoPE:

    python scripts/m3_launch_matrix.py --pe rope

Only one seed:

    python scripts/m3_launch_matrix.py --seeds 17
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

import torch


# ============================================================
# Repository
# ============================================================

REPO_ROOT = Path(
    __file__
).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )


from src.training.queue import (
    MatrixJob,
    build_matrix_jobs,
)


TRAIN_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "m3_train.py"
)


# ============================================================
# Helpers
# ============================================================


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def atomic_write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = Path(
        str(path) + ".tmp"
    )

    temp.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        temp,
        path,
    )


def load_json(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def discover_gpu_slots(
    explicit: list[str] | None,
) -> list[str]:
    """Determine physical/allocated GPU identifiers.

    If CUDA_VISIBLE_DEVICES is already supplied by Slurm or another
    scheduler, preserve those identifiers instead of assuming physical
    GPU 0/1.
    """

    if explicit:
        return [
            str(value)
            for value in explicit
        ]

    visible = os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    )

    if (
        visible is not None
        and visible.strip()
        and visible.strip() != "-1"
    ):
        slots = [
            item.strip()
            for item
            in visible.split(",")
            if item.strip()
        ]

        if slots:
            return slots

    if not torch.cuda.is_available():
        return []

    return [
        str(index)
        for index
        in range(
            torch.cuda.device_count()
        )
    ]


def build_command(
    job: MatrixJob,
    *,
    plan_path: Path,
    cache_path: Path,
    log_every: int,
    fresh: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(
            TRAIN_SCRIPT
        ),
        "--plan",
        str(
            plan_path
        ),
        "--run-id",
        job.run_id,
        "--cache",
        str(
            cache_path
        ),
        "--device",
        "cuda",
        "--log-every",
        str(
            log_every
        ),
    ]

    # Default queue behavior is resume-safe.
    if not fresh:
        command.append(
            "--resume"
        )

    return command


# ============================================================
# Running-process metadata
# ============================================================


@dataclass
class RunningProcess:
    job: MatrixJob

    gpu_slot: str

    process: subprocess.Popen

    log_handle: IO[str]

    started_at: str

    log_path: Path


# ============================================================
# CLI
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--plan",
        type=Path,
        default=Path(
            "results/manifests/"
            "m3_run_plan.json"
        ),
    )

    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "data/cache/"
            "train_50m.uint16.bin"
        ),
    )

    parser.add_argument(
        "--gpus",
        nargs="+",
        default=None,
        help=(
            "GPU identifiers. "
            "If omitted, CUDA_VISIBLE_DEVICES "
            "or all visible CUDA devices are used."
        ),
    )

    parser.add_argument(
        "--pe",
        nargs="+",
        choices=(
            "learned",
            "sinusoidal",
            "rope",
            "alibi",
            "nope",
        ),
        default=None,
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Launch at most N pending runs. "
            "Useful for server smoke tests."
        ),
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path(
            "results/manifests/"
            "m3_queue_state.json"
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print queue/commands without "
            "starting training."
        ),
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Do not pass --resume to m3_train.py."
        ),
    )

    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help=(
            "Include runs that already have "
            "completed.json."
        ),
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Continue launching new jobs after "
            "a run exits non-zero."
        ),
    )

    return parser.parse_args()


# ============================================================
# Queue-state report
# ============================================================


def write_queue_state(
    *,
    path: Path,
    status: str,
    pending: list[MatrixJob],
    running: dict[
        str,
        RunningProcess,
    ],
    completed: list[dict[str, Any]],
    failed: list[dict[str, Any]],
) -> None:

    atomic_write_json(
        path,
        {
            "status": status,
            "updated_at_utc": (
                utc_now()
            ),
            "pending": [
                job.to_dict()
                for job in pending
            ],
            "running": [
                {
                    **item.job.to_dict(),
                    "gpu_slot": (
                        item.gpu_slot
                    ),
                    "pid": (
                        item.process.pid
                    ),
                    "started_at_utc": (
                        item.started_at
                    ),
                    "log_path": str(
                        item.log_path
                    ),
                }
                for item
                in running.values()
            ],
            "completed": (
                completed
            ),
            "failed": (
                failed
            ),
            "counts": {
                "pending": len(
                    pending
                ),
                "running": len(
                    running
                ),
                "completed": len(
                    completed
                ),
                "failed": len(
                    failed
                ),
            },
        },
    )


# ============================================================
# Launch one process
# ============================================================


def start_job(
    job: MatrixJob,
    *,
    gpu_slot: str,
    plan_path: Path,
    cache_path: Path,
    log_every: int,
    fresh: bool,
) -> RunningProcess:

    logs_dir = (
        job.run_dir
        / "logs"
    )

    logs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        logs_dir
        / "matrix_process.log"
    )

    log_handle = log_path.open(
        "a",
        encoding="utf-8",
        buffering=1,
    )

    command = build_command(
        job,
        plan_path=plan_path,
        cache_path=cache_path,
        log_every=log_every,
        fresh=fresh,
    )

    env = os.environ.copy()

    # The child sees exactly one GPU, so inside
    # m3_train.py it is simply cuda:0.
    env[
        "CUDA_VISIBLE_DEVICES"
    ] = str(
        gpu_slot
    )

    started_at = (
        utc_now()
    )

    log_handle.write(
        "\n"
        + "=" * 72
        + "\n"
    )

    log_handle.write(
        f"launcher start: "
        f"{started_at}\n"
    )

    log_handle.write(
        f"GPU slot: "
        f"{gpu_slot}\n"
    )

    log_handle.write(
        "command: "
        + " ".join(command)
        + "\n"
    )

    log_handle.write(
        "=" * 72
        + "\n"
    )

    process = subprocess.Popen(
        command,
        cwd=str(
            REPO_ROOT
        ),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )

    return RunningProcess(
        job=job,
        gpu_slot=gpu_slot,
        process=process,
        log_handle=log_handle,
        started_at=started_at,
        log_path=log_path,
    )


# ============================================================
# Main
# ============================================================


def main():
    args = parse_args()

    plan_path = (
        args.plan
        if args.plan.is_absolute()
        else REPO_ROOT / args.plan
    ).resolve()

    cache_path = (
        args.cache
        if args.cache.is_absolute()
        else REPO_ROOT / args.cache
    ).resolve()

    state_path = (
        args.state_file
        if args.state_file.is_absolute()
        else REPO_ROOT
        / args.state_file
    ).resolve()

    payload = load_json(
        plan_path
    )

    jobs = build_matrix_jobs(
        payload,
        repo_root=REPO_ROOT,
        pe_types=args.pe,
        seeds=args.seeds,
        include_completed=(
            args.rerun_completed
        ),
    )

    if (
        args.limit is not None
    ):
        if args.limit <= 0:
            raise ValueError(
                "--limit must be positive"
            )

        jobs = jobs[
            :args.limit
        ]

    gpu_slots = (
        discover_gpu_slots(
            args.gpus
        )
    )

    print()
    print("=" * 78)
    print("M3 MATRIX QUEUE")
    print("=" * 78)

    print(
        f"plan:        "
        f"{plan_path}"
    )

    print(
        f"cache:       "
        f"{cache_path}"
    )

    print(
        f"pending runs:"
        f" {len(jobs)}"
    )

    print(
        f"GPU slots:   "
        f"{gpu_slots if gpu_slots else 'none'}"
    )

    print(
        f"resume mode: "
        f"{not args.fresh}"
    )

    print()

    for number, job in enumerate(
        jobs,
        start=1,
    ):
        resume_text = (
            "resume"
            if job.can_resume
            and not args.fresh
            else "start"
        )

        print(
            f"{number:02d}. "
            f"{job.pe_type:<10} "
            f"seed={job.init_seed:<4} "
            f"{resume_text:<6} "
            f"{job.run_id}"
        )

    # ========================================================
    # Dry run
    # ========================================================

    if args.dry_run:

        print()
        print("=" * 78)
        print("DRY-RUN COMMANDS")
        print("=" * 78)

        dry_slots = (
            gpu_slots
            if gpu_slots
            else [
                "GPU_SLOT"
            ]
        )

        for index, job in enumerate(
            jobs
        ):
            slot = dry_slots[
                index
                % len(
                    dry_slots
                )
            ]

            command = build_command(
                job,
                plan_path=plan_path,
                cache_path=cache_path,
                log_every=args.log_every,
                fresh=args.fresh,
            )

            print()
            print(
                f"CUDA_VISIBLE_DEVICES="
                f"{slot}"
            )

            print(
                " ".join(
                    command
                )
            )

        print()
        print(
            "Dry run complete. "
            "No training processes started."
        )

        return

    # ========================================================
    # Real launch validation
    # ========================================================

    if not TRAIN_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Training script missing: "
            f"{TRAIN_SCRIPT}"
        )

    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Frozen token cache missing: "
            f"{cache_path}"
        )

    if not gpu_slots:
        raise RuntimeError(
            "No CUDA GPU slots found. "
            "Run with --dry-run locally, "
            "or launch on the CUDA server."
        )

    if args.poll_seconds <= 0:
        raise ValueError(
            "--poll-seconds must be positive"
        )

    pending = list(
        jobs
    )

    running: dict[
        str,
        RunningProcess
    ] = {}

    completed: list[
        dict[str, Any]
    ] = []

    failed: list[
        dict[str, Any]
    ] = []

    free_slots = list(
        gpu_slots
    )

    stop_launching = False

    write_queue_state(
        path=state_path,
        status="running",
        pending=pending,
        running=running,
        completed=completed,
        failed=failed,
    )

    try:

        while (
            pending
            or running
        ):

            # -----------------------------------------------
            # Fill free GPUs.
            # -----------------------------------------------

            while (
                pending
                and free_slots
                and not stop_launching
            ):

                job = pending.pop(
                    0
                )

                gpu_slot = (
                    free_slots.pop(
                        0
                    )
                )

                item = start_job(
                    job,
                    gpu_slot=gpu_slot,
                    plan_path=plan_path,
                    cache_path=cache_path,
                    log_every=(
                        args.log_every
                    ),
                    fresh=args.fresh,
                )

                running[
                    gpu_slot
                ] = item

                print(
                    f"[START] "
                    f"GPU={gpu_slot} "
                    f"{job.run_id} "
                    f"pid={item.process.pid}"
                )

            write_queue_state(
                path=state_path,
                status="running",
                pending=pending,
                running=running,
                completed=completed,
                failed=failed,
            )

            if not running:
                break

            time.sleep(
                args.poll_seconds
            )

            # -----------------------------------------------
            # Check running jobs.
            # -----------------------------------------------

            finished_slots: list[
                str
            ] = []

            for gpu_slot, item in list(
                running.items()
            ):

                return_code = (
                    item.process.poll()
                )

                if return_code is None:
                    continue

                item.log_handle.flush()
                item.log_handle.close()

                result = {
                    "run_id": (
                        item.job.run_id
                    ),
                    "pe_type": (
                        item.job.pe_type
                    ),
                    "init_seed": (
                        item.job.init_seed
                    ),
                    "gpu_slot": (
                        gpu_slot
                    ),
                    "return_code": (
                        return_code
                    ),
                    "started_at_utc": (
                        item.started_at
                    ),
                    "finished_at_utc": (
                        utc_now()
                    ),
                    "log_path": str(
                        item.log_path
                    ),
                }

                if return_code == 0:

                    completed.append(
                        result
                    )

                    print(
                        f"[DONE ] "
                        f"GPU={gpu_slot} "
                        f"{item.job.run_id}"
                    )

                else:

                    failed.append(
                        result
                    )

                    print(
                        f"[FAIL ] "
                        f"GPU={gpu_slot} "
                        f"{item.job.run_id} "
                        f"exit={return_code}"
                    )

                    if not (
                        args.continue_on_error
                    ):
                        stop_launching = True

                finished_slots.append(
                    gpu_slot
                )

            # -----------------------------------------------
            # Release GPUs.
            # -----------------------------------------------

            for gpu_slot in (
                finished_slots
            ):

                del running[
                    gpu_slot
                ]

                free_slots.append(
                    gpu_slot
                )

            write_queue_state(
                path=state_path,
                status=(
                    "stopping_after_error"
                    if stop_launching
                    else "running"
                ),
                pending=pending,
                running=running,
                completed=completed,
                failed=failed,
            )

            # If one run failed and default
            # stop-on-error policy is active,
            # finish already-running processes
            # but launch nothing new.
            if (
                stop_launching
                and not running
            ):
                break

    except KeyboardInterrupt:

        print()
        print(
            "Keyboard interrupt received. "
            "Terminating running jobs..."
        )

        for item in (
            running.values()
        ):
            item.process.terminate()

        for item in (
            running.values()
        ):
            try:
                item.process.wait(
                    timeout=30
                )
            except subprocess.TimeoutExpired:
                item.process.kill()
                item.process.wait()

            if not (
                item.log_handle.closed
            ):
                item.log_handle.close()

        write_queue_state(
            path=state_path,
            status="interrupted",
            pending=pending,
            running={},
            completed=completed,
            failed=failed,
        )

        raise SystemExit(
            130
        )

    # ========================================================
    # Final status
    # ========================================================

    final_status = (
        "completed"
        if (
            not failed
            and not pending
        )
        else "failed"
    )

    write_queue_state(
        path=state_path,
        status=final_status,
        pending=pending,
        running={},
        completed=completed,
        failed=failed,
    )

    print()
    print("=" * 78)
    print("M3 MATRIX QUEUE SUMMARY")
    print("=" * 78)

    print(
        f"completed: "
        f"{len(completed)}"
    )

    print(
        f"failed:    "
        f"{len(failed)}"
    )

    print(
        f"pending:   "
        f"{len(pending)}"
    )

    print(
        f"state:     "
        f"{state_path}"
    )

    if failed:

        print()
        print(
            "Failed runs:"
        )

        for item in failed:
            print(
                f"  {item['run_id']} "
                f"exit="
                f"{item['return_code']} "
                f"log="
                f"{item['log_path']}"
            )

        raise SystemExit(
            1
        )

    if pending:
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()