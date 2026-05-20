"""Base classes for dataset adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from urban_imagenet_lib.manifest import BenchmarkRecord


class BaseAdapter(ABC):
    """Convert a dataset split into a stream of BenchmarkRecord objects."""

    dataset_name: str
    supported_tasks: tuple[str, ...]

    def __init__(self, root: str | Path, split: str = "test") -> None:
        self.root = Path(root)
        self.split = split

    @abstractmethod
    def iter_records(self, task: str) -> Iterator[BenchmarkRecord]:
        """Yield records for a supported task."""

    def require_task(self, task: str) -> None:
        if task not in self.supported_tasks:
            supported = ", ".join(self.supported_tasks)
            raise ValueError(f"{self.dataset_name} supports tasks [{supported}], got {task!r}.")
