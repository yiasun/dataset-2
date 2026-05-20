"""Common manifest format for Urban-ImageNet and external benchmark adapters."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass(frozen=True)
class BenchmarkRecord:
    """A single image/text/annotation record in the unified benchmark view."""

    dataset: str
    task: str
    split: str
    image_path: str
    label: str | None = None
    label_id: int | None = None
    text: str | None = None
    group_id: str | None = None
    annotation_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["metadata"] = dict(self.metadata)
        return row


def write_jsonl(records: Iterable[BenchmarkRecord], path: str | Path) -> int:
    """Write records to JSONL and return the number of rows."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[BenchmarkRecord]:
    """Read records from a JSONL manifest."""

    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            yield BenchmarkRecord(**row)


def write_csv(records: Iterable[BenchmarkRecord], path: str | Path) -> int:
    """Write records to a flat CSV manifest."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "task",
        "split",
        "image_path",
        "label",
        "label_id",
        "text",
        "group_id",
        "annotation_path",
        "metadata",
    ]
    count = 0
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = record.to_dict()
            row["metadata"] = json.dumps(row["metadata"], ensure_ascii=False)
            writer.writerow(row)
            count += 1
    return count
