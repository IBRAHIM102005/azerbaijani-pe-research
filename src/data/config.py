"""Load and validate the frozen data-pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataPipelineConfig:
    repo_root: Path
    config_path: Path
    values: dict[str, Any]

    def path(self, name: str) -> Path:
        configured = (self.repo_root / self.values["paths"][name]).resolve()
        if name != "original_dollma":
            return configured

        override = os.environ.get("AZ_PE_DOLLMA_ROOT")
        resolved = Path(override).expanduser().resolve() if override else configured
        if not resolved.is_dir():
            source = "AZ_PE_DOLLMA_ROOT" if override else "the configured relative path"
            raise FileNotFoundError(
                f"DOLLMA root from {source} does not exist: {resolved}. "
                "Set AZ_PE_DOLLMA_ROOT to the external DOLLMA directory."
            )
        return resolved

    @property
    def included_sources(self) -> list[str]:
        return [
            name
            for name, settings in self.values["sources"].items()
            if settings["included_in_core"]
        ]

    def source(self, name: str) -> dict[str, Any]:
        try:
            return self.values["sources"][name]
        except KeyError as exc:
            raise ValueError(f"Source {name!r} is not declared in the data-pipeline config") from exc


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository root from the current script or working directory."""

    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "configs" / "frozen" / "data_pipeline.yaml").is_file():
            return candidate
    raise FileNotFoundError("Could not find configs/frozen/data_pipeline.yaml")


def load_config(path: Path | None = None, repo_root: Path | None = None) -> DataPipelineConfig:
    """Load the data-pipeline YAML file and resolve its repository root."""

    root = (repo_root or find_repo_root(path.parent if path else None)).resolve()
    config_path = (path or root / "configs" / "frozen" / "data_pipeline.yaml").resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError(f"Invalid data-pipeline configuration: {config_path}")
    return DataPipelineConfig(root, config_path, values)
