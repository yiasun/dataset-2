"""Adapter registry for command-line manifest generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .cityscapes import CityscapesAdapter
from .coco import COCOAdapter
from .places365 import Places365Adapter
from .urban_imagenet import UrbanImageNetAdapter

ADAPTERS = {
    "urban-imagenet": UrbanImageNetAdapter,
    "places365": Places365Adapter,
    "coco": COCOAdapter,
    "ms-coco": COCOAdapter,
    "cityscapes": CityscapesAdapter,
}


def list_adapters() -> list[str]:
    return sorted(ADAPTERS)


def create_adapter(name: str, root: str | Path, split: str, **kwargs: Any):
    key = name.lower()
    if key not in ADAPTERS:
        choices = ", ".join(list_adapters())
        raise ValueError(f"Unknown adapter {name!r}. Available adapters: {choices}")
    return ADAPTERS[key](root=root, split=split, **kwargs)
