"""Urban-ImageNet adapter."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterator

from urban_imagenet_lib.adapters.base import BaseAdapter
from urban_imagenet_lib.manifest import BenchmarkRecord
from urban_imagenet_lib.taxonomy import HUSIC_CLASS_TO_ID, prompt_for_class

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class UrbanImageNetAdapter(BaseAdapter):
    """Adapter for released Urban-ImageNet balanced tiers."""

    dataset_name = "urban-imagenet"
    supported_tasks = ("t1", "t2", "t3")

    def __init__(self, root: str | Path, split: str = "test", dataset_size: str | None = None) -> None:
        super().__init__(root, split)
        self.dataset_dir = self.root / dataset_size if dataset_size else self.root

    def iter_records(self, task: str) -> Iterator[BenchmarkRecord]:
        self.require_task(task)
        if task == "t1":
            yield from self.iter_t1()
        elif task == "t2":
            yield from self.iter_t2()
        elif task == "t3":
            yield from self.iter_t3()

    def iter_t1(self) -> Iterator[BenchmarkRecord]:
        image_root = self.dataset_dir / "01 Images with labels" / self.split
        if not image_root.exists():
            raise FileNotFoundError(f"Missing Task 1 image folder: {image_root}")

        for class_dir in sorted(p for p in image_root.iterdir() if p.is_dir()):
            label = class_dir.name
            label_id = HUSIC_CLASS_TO_ID.get(label)
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                yield BenchmarkRecord(
                    dataset=self.dataset_name,
                    task="t1",
                    split=self.split,
                    image_path=str(image_path),
                    label=label,
                    label_id=label_id,
                )

    def iter_t2(self) -> Iterator[BenchmarkRecord]:
        table_dir = self.dataset_dir / "02 Text-Image Pairs"
        rows = list(_read_split_table(table_dir, self.split))
        image_root = self.dataset_dir / "01 Images with labels" / self.split

        for idx, row in enumerate(rows):
            label = _clean(row.get("Image Label", ""))
            image_name = _clean(row.get("Image Filename", ""))
            if not image_name:
                continue
            if not Path(image_name).suffix:
                image_name = f"{image_name}.jpg"
            image_path = _resolve_image(image_root, label, image_name)
            post_text = _clean(row.get("Post Text", row.get("Text", "")))
            post_id = _clean(row.get("Post ID", row.get("post_id", "")))
            group_id = post_id or post_text or f"row-{idx}"
            text = post_text or prompt_for_class(label)
            yield BenchmarkRecord(
                dataset=self.dataset_name,
                task="t2",
                split=self.split,
                image_path=str(image_path),
                label=label,
                label_id=HUSIC_CLASS_TO_ID.get(label),
                text=text,
                group_id=group_id,
                metadata={"category_prompt": prompt_for_class(label)},
            )

    def iter_t3(self) -> Iterator[BenchmarkRecord]:
        ann_path = _find_t3_annotation(self.dataset_dir, self.split)
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        image_root = self.dataset_dir / "01 Images with labels" / self.split
        cats = {cat["id"]: cat.get("name", str(cat["id"])) for cat in data.get("categories", [])}
        ann_count_by_image: dict[int, int] = {}
        for ann in data.get("annotations", []):
            ann_count_by_image[int(ann["image_id"])] = ann_count_by_image.get(int(ann["image_id"]), 0) + 1

        for image in data.get("images", []):
            label_id = image.get("classification_label")
            label = cats.get(label_id)
            image_name = image.get("file_name", "")
            image_path = _resolve_image(image_root, label or "", Path(image_name).name)
            yield BenchmarkRecord(
                dataset=self.dataset_name,
                task="t3",
                split=self.split,
                image_path=str(image_path),
                label=label,
                label_id=label_id,
                annotation_path=str(ann_path),
                metadata={
                    "image_id": image.get("id"),
                    "instance_count": ann_count_by_image.get(int(image.get("id", -1)), 0),
                    "height": image.get("height"),
                    "width": image.get("width"),
                },
            )


def _clean(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _read_split_table(table_dir: Path, split: str) -> Iterator[dict[str, str]]:
    csv_path = table_dir / f"{split}.csv"
    xlsx_path = table_dir / f"{split}.xlsx"
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            yield from csv.DictReader(f)
        return
    if xlsx_path.exists():
        try:
            import openpyxl
        except ImportError as exc:
            raise RuntimeError("Install openpyxl to read Urban-ImageNet split xlsx files.") from exc
        wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
        ws = wb.active
        rows = ws.iter_rows(values_only=True)
        header = [str(cell or "").strip() for cell in next(rows)]
        for values in rows:
            yield {key: "" if val is None else str(val) for key, val in zip(header, values)}
        return
    raise FileNotFoundError(f"Missing split table: {csv_path} or {xlsx_path}")


def _resolve_image(image_root: Path, label: str, image_name: str) -> Path:
    direct = image_root / label / image_name
    if direct.exists():
        return direct
    matches = list(image_root.rglob(Path(image_name).name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Cannot locate image {image_name!r} under {image_root}")


def _find_t3_annotation(dataset_dir: Path, split: str) -> Path:
    candidates = [
        dataset_dir / "03 Instance Segmentation labels.json",
        dataset_dir / "03 Instance Segmentation labels" / f"{split}.json",
        dataset_dir / "annotations" / f"{split}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Cannot find Task 3 COCO annotation JSON.")
