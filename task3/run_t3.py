import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from tools.mmdet_config_builder import build_mmdet_config


def read_categories(data_root):
    ann_path = Path(data_root) / "train.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Cannot find training annotation: {ann_path}")

    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    categories = sorted(data.get("categories", []), key=lambda x: int(x["id"]))
    if not categories:
        raise ValueError(f"No categories found in {ann_path}")

    expected_ids = list(range(len(categories)))
    observed_ids = [int(cat["id"]) for cat in categories]
    if observed_ids != expected_ids:
        raise ValueError(
            "MMDetection expects contiguous zero-based category ids. "
            f"Observed ids are {observed_ids[:20]}..."
        )

    return [str(cat["name"]) for cat in categories]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["cascade_mask_rcnn", "mask_rcnn", "solov2", "mask2former"])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--img-root", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--auto-classes", action="store_true",
                        help="Read class names and class count from data-root/train.json.")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-iters", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--load-from", default=None,
                        help="Optional pretrained checkpoint path or URL.")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.auto_classes or args.num_classes is None:
        class_names = read_categories(args.data_root)
        num_classes = len(class_names)
    else:
        num_classes = int(args.num_classes)
        class_names = [str(i) for i in range(num_classes)]

    cfg_path = work_dir / f"{args.model}_auto.py"

    build_mmdet_config(
        model=args.model,
        cfg_path=cfg_path,
        data_root=args.data_root,
        img_root=args.img_root,
        work_dir=str(work_dir),
        num_classes=num_classes,
        class_names=class_names,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        warmup_iters=args.warmup_iters,
        num_workers=args.num_workers,
        load_from=args.load_from,
    )

    print(f"[INFO] config saved -> {cfg_path}")
    print(f"[INFO] num_classes={num_classes}")
    print(f"[INFO] class_names={class_names}")

    cmd = [
        sys.executable,
        "-m",
        "mim",
        "train",
        "mmdet",
        str(cfg_path),
    ]

    print("[CMD]", " ".join(cmd))
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
