import argparse
import subprocess
import sys
from pathlib import Path

from tools.mmdet_config_builder import build_mmdet_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["cascade_mask_rcnn", "solov2", "mask2former"])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--img-root", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-classes", type=int, required=True)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = work_dir / f"{args.model}_auto.py"

    build_mmdet_config(
        model=args.model,
        cfg_path=cfg_path,
        data_root=args.data_root,
        img_root=args.img_root,
        work_dir=str(work_dir),
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    print(f"[INFO] config saved -> {cfg_path}")

    cmd = [
        sys.executable,
        "-m",
        "mim",
        "train",
        "mmdet",
        str(cfg_path),
    ]

    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()