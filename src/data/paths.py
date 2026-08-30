"""Resolve portable repository artifacts and explicit external data roots."""

from __future__ import annotations

from pathlib import Path


def repository_relative(path: Path, repo_root: Path) -> str:
    """Return a stable POSIX path that stays inside the repository."""

    resolved = path.resolve()
    root = repo_root.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Artifact path is outside the repository: {resolved}") from exc


def resolve_repository_path(value: str, repo_root: Path) -> Path:
    """Resolve a repository-relative artifact reference under a chosen root."""

    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError(f"Operational artifact paths must be repository-relative: {value}")
    resolved = (repo_root.resolve() / candidate).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Artifact reference escapes the repository: {value}") from exc
    return resolved
