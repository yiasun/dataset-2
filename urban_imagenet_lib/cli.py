"""Command-line helpers for Urban-ImageNet-lib."""

from __future__ import annotations

import argparse
from pathlib import Path

from urban_imagenet_lib.adapters import create_adapter, list_adapters
from urban_imagenet_lib.manifest import write_csv, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Urban-ImageNet-lib manifest utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-adapters", help="List supported dataset adapters.")
    list_parser.set_defaults(func=cmd_list_adapters)

    manifest = subparsers.add_parser("build-manifest", help="Build a unified JSONL/CSV manifest.")
    manifest.add_argument("--dataset", required=True, choices=list_adapters())
    manifest.add_argument("--root", required=True, type=Path)
    manifest.add_argument("--task", required=True, choices=["t1", "t2", "t3"])
    manifest.add_argument("--split", default="test")
    manifest.add_argument("--out", required=True, type=Path)
    manifest.add_argument("--format", default="jsonl", choices=["jsonl", "csv"])
    manifest.add_argument("--dataset-size", default=None, help="Urban-ImageNet tier folder, e.g. '10K Dataset'.")
    manifest.add_argument("--file-list", default=None, help="Optional Places365 file list.")
    manifest.add_argument("--ann-file", default=None, help="Optional COCO annotation file.")
    manifest.add_argument("--image-root", default=None, help="Optional COCO image root.")
    manifest.add_argument("--gt-kind", default="gtFine", help="Cityscapes ground-truth type.")
    manifest.set_defaults(func=cmd_build_manifest)
    return parser.parse_args()


def cmd_list_adapters(_: argparse.Namespace) -> None:
    for name in list_adapters():
        print(name)


def cmd_build_manifest(args: argparse.Namespace) -> None:
    kwargs = {}
    if args.dataset == "urban-imagenet" and args.dataset_size:
        kwargs["dataset_size"] = args.dataset_size
    if args.dataset == "places365" and args.file_list:
        kwargs["file_list"] = args.file_list
    if args.dataset in {"coco", "ms-coco"}:
        if args.ann_file:
            kwargs["ann_file"] = args.ann_file
        if args.image_root:
            kwargs["image_root"] = args.image_root
    if args.dataset == "cityscapes":
        kwargs["gt_kind"] = args.gt_kind

    adapter = create_adapter(args.dataset, root=args.root, split=args.split, **kwargs)
    records = adapter.iter_records(args.task)
    if args.format == "jsonl":
        count = write_jsonl(records, args.out)
    else:
        count = write_csv(records, args.out)
    print(f"[DONE] wrote {count} records -> {args.out}")


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
