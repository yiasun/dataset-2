"""Dataset adapters for unified benchmark manifests."""

from .base import BaseAdapter
from .cityscapes import CityscapesAdapter
from .coco import COCOAdapter
from .places365 import Places365Adapter
from .registry import create_adapter, list_adapters
from .urban_imagenet import UrbanImageNetAdapter

__all__ = [
    "BaseAdapter",
    "CityscapesAdapter",
    "COCOAdapter",
    "Places365Adapter",
    "UrbanImageNetAdapter",
    "create_adapter",
    "list_adapters",
]
