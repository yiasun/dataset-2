"""Cityscapes adapter for semantic/instance segmentation comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from urban_imagenet_lib.adapters.base import BaseAdapter
from urban_imagenet_lib.manifest import BenchmarkRecord


class CityscapesAdapter(BaseAdapter):
    """Scan Cityscapes leftImg8bit and gtFine files into a manifest."""

    dataset_name = "cityscapes"
    supported_tasks = ("t3",)

    def __init__(self, root: str | Path, split: str = "val", gt_kind: str = "gtFine") -> None:
        super().__init__(root, split)
        self.gt_kind = gt_kind

    def iter_records(self, task: str) -> Iterator[BenchmarkRecord]:
        self.require_task(task)
        image_root = self.root / "leftImg8bit" / self.split
        gt_root = self.root / self.gt_kind / self.split
        if not image_root.exists():
            raise FileNotFoundError(f"Missing Cityscapes image folder: {image_root}")
        for image_path in sorted(image_root.rglob("*_leftImg8bit.png")):
            city = image_path.parent.name
            stem = image_path.name.replace("_leftImg8bit.png", "")
            instance_path = gt_root / city / f"{stem}_{self.gt_kind}_instanceIds.png"
            label_path = gt_root / city / f"{stem}_{self.gt_kind}_labelIds.png"
            annotation_path = instance_path if instance_path.exists() else label_path
            yield BenchmarkRecord(
                dataset=self.dataset_name,
                task="t3",
                split=self.split,
                image_path=str(image_path),
                annotation_path=str(annotation_path) if annotation_path.exists() else None,
                metadata={"city": city, "stem": stem, "gt_kind": self.gt_kind},
            )
