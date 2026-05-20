import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def scan_one(ann_path, img_root):
    ann_path = Path(ann_path)
    img_root = Path(img_root)

    with ann_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    rows = []
    for img in coco.get("images", []):
        rel = img["file_name"]
        path = img_root / rel
        status = "ok"
        size = -1
        height = None
        width = None

        if not path.exists():
            status = "missing"
        else:
            size = path.stat().st_size
            if size == 0:
                status = "empty"
            else:
                image = read_image(path)
                if image is None:
                    status = "unreadable"
                else:
                    height, width = image.shape[:2]

        rows.append(
            {
                "ann": str(ann_path),
                "image_id": img.get("id"),
                "file_name": rel,
                "path": str(path),
                "status": status,
                "size": size,
                "decoded_height": height,
                "decoded_width": width,
                "json_height": img.get("height"),
                "json_width": img.get("width"),
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", action="append", required=True, help="COCO annotation JSON. Can be passed multiple times.")
    parser.add_argument("--img-root", required=True)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--fail-on-bad", action="store_true")
    args = parser.parse_args()

    rows = []
    for ann in args.ann:
        rows.extend(scan_one(ann, args.img_root))

    df = pd.DataFrame(rows)
    bad = df[df["status"] != "ok"].copy()

    print(f"[INFO] scanned images: {len(df)}")
    if bad.empty:
        print("[DONE] all referenced images are readable by OpenCV.")
    else:
        print(f"[WARN] bad images: {len(bad)}")
        print(bad[["ann", "image_id", "file_name", "status", "size", "path"]].to_string(index=False))

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[DONE] scan csv -> {out_csv}")

    if args.fail_on_bad and not bad.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
