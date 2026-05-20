"""MS-COCO adapter for detection/segmentation comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from urban_imagenet_lib.adapters.base import BaseAdapter
from urban_imagenet_lib.manifest import BenchmarkRecord


class COCOAdapter(BaseAdapter):
    """Read a COCO annotation file into the unified manifest format."""

    dataset_name = "ms-coco"
    supported_tasks = ("t3",)

    def __init__(
        self,
        root: str | Path,
        split: str = "val2017",
        ann_file: str | Path | None = None,
        image_root: str | Path | None = None,
    ) -> None:
        super().__init__(root, split)
        self.ann_file = Path(ann_file) if ann_file else self.root / "annotations" / f"instances_{split}.json"
        self.image_root = Path(image_root) if image_root else self.root / split

    def iter_records(self, task: str) -> Iterator[BenchmarkRecord]:
        self.require_task(task)
        if not self.ann_file.exists():
            raise FileNotFoundError(f"Missing COCO annotation file: {self.ann_file}")
        data = json.loads(self.ann_file.read_text(encoding="utf-8"))
        ann_count_by_image: dict[int, int] = {}
        cat_ids_by_image: dict[int, set[int]] = {}
        for ann in data.get("annotations", []):
            image_id = int(ann["image_id"])
            ann_count_by_image[image_id] = ann_count_by_image.get(image_id, 0) + 1
            cat_ids_by_image.setdefault(image_id, set()).add(int(ann.get("category_id", -1)))

        for image in data.get("images", []):
            image_id = int(image["id"])
            yield BenchmarkRecord(
                dataset=self.dataset_name,
                task="t3",
                split=self.split,
                image_path=str(self.image_root / image["file_name"]),
                annotation_path=str(self.ann_file),
                metadata={
                    "image_id": image_id,
                    "instance_count": ann_count_by_image.get(image_id, 0),
                    "category_ids": sorted(cat_ids_by_image.get(image_id, set())),
                    "height": image.get("height"),
                    "width": image.get("width"),
                },
            )
