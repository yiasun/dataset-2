import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def is_readable(path):
    if not path.exists() or path.stat().st_size == 0:
        return False
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return False
    return cv2.imdecode(data, cv2.IMREAD_COLOR) is not None


def repair_split(split, ann_dir, img_root, out_ann_dir):
    ann_path = ann_dir / f"{split}.json"
    with ann_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    good_images = []
    bad_images = []
    for img in coco.get("images", []):
        path = img_root / img["file_name"]
        if is_readable(path):
            good_images.append(img)
        else:
            bad_images.append(img)

    good_ids = {img["id"] for img in good_images}
    good_annotations = [
        ann for ann in coco.get("annotations", [])
        if ann.get("image_id") in good_ids
    ]

    repaired = dict(coco)
    repaired["images"] = good_images
    repaired["annotations"] = good_annotations
    info = dict(repaired.get("info", {}))
    info["repair_note"] = "Images unreadable by OpenCV were removed with their annotations."
    info["removed_images"] = len(bad_images)
    repaired["info"] = info

    out_path = out_ann_dir / f"{split}.json"
    out_path.write_text(json.dumps(repaired, ensure_ascii=False), encoding="utf-8")

    print(
        f"[DONE] {split}: kept_images={len(good_images)} "
        f"removed_images={len(bad_images)} kept_annotations={len(good_annotations)} "
        f"removed_annotations={len(coco.get('annotations', [])) - len(good_annotations)}"
    )
    for img in bad_images[:20]:
        print(f"  [BAD] {split}/{img.get('id')}: {img.get('file_name')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-dir", required=True)
    parser.add_argument("--img-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--copy-images", action="store_true")
    args = parser.parse_args()

    ann_dir = Path(args.ann_dir)
    img_root = Path(args.img_root)
    out_root = Path(args.out_root)
    out_ann_dir = out_root / "annotations"
    out_ann_dir.mkdir(parents=True, exist_ok=True)

    if args.copy_images:
        dst_img_root = out_root / "images"
        if dst_img_root.exists():
            shutil.rmtree(dst_img_root)
        shutil.copytree(img_root, dst_img_root)
        print(f"[DONE] copied images -> {dst_img_root}")

    for split in ("train", "val", "test"):
        repair_split(split, ann_dir, img_root, out_ann_dir)

    if not (out_root / "images").exists():
        print("[INFO] images were not copied. Use the original --img-root, or rerun with --copy-images.")
    else:
        print(f"[DONE] repaired dataset root -> {out_root}")


if __name__ == "__main__":
    main()
