"""Places365 adapter for scene-classification comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from urban_imagenet_lib.adapters.base import BaseAdapter
from urban_imagenet_lib.manifest import BenchmarkRecord

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class Places365Adapter(BaseAdapter):
    """Read Places365 from either ImageFolder layout or a standard file list."""

    dataset_name = "places365"
    supported_tasks = ("t1",)

    def __init__(self, root: str | Path, split: str = "val", file_list: str | Path | None = None) -> None:
        super().__init__(root, split)
        self.file_list = Path(file_list) if file_list else None

    def iter_records(self, task: str) -> Iterator[BenchmarkRecord]:
        self.require_task(task)
        if self.file_list:
            yield from self._iter_file_list()
        else:
            yield from self._iter_image_folder()

    def _iter_file_list(self) -> Iterator[BenchmarkRecord]:
        if not self.file_list or not self.file_list.exists():
            raise FileNotFoundError(f"Missing Places365 file list: {self.file_list}")
        for line in self.file_list.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            rel_path = parts[0]
            label_id = int(parts[1]) if len(parts) > 1 and parts[1].lstrip("-").isdigit() else None
            label = Path(rel_path).parts[0] if Path(rel_path).parts else None
            yield BenchmarkRecord(
                dataset=self.dataset_name,
                task="t1",
                split=self.split,
                image_path=str(self.root / rel_path),
                label=label,
                label_id=label_id,
            )

    def _iter_image_folder(self) -> Iterator[BenchmarkRecord]:
        split_root = self.root / self.split
        if not split_root.exists():
            split_root = self.root
        class_dirs = sorted(p for p in split_root.iterdir() if p.is_dir())
        for label_id, class_dir in enumerate(class_dirs):
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                yield BenchmarkRecord(
                    dataset=self.dataset_name,
                    task="t1",
                    split=self.split,
                    image_path=str(image_path),
                    label=class_dir.name,
                    label_id=label_id,
                )
