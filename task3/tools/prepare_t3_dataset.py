import argparse
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

from pycocotools import mask as mask_utils


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}


def clean_label(text):
    text = str(text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" .", ".").replace(". ", " ")
    return text or "object"


def ann_to_rle(ann, height, width):
    seg = ann["segmentation"]
    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, height, width)
        return mask_utils.merge(rles)
    if isinstance(seg, dict) and isinstance(seg.get("counts"), list):
        return mask_utils.frPyObjects(seg, height, width)
    return seg


def recompute_bbox_area(ann, image_info):
    height = int(image_info["height"])
    width = int(image_info["width"])
    rle = ann_to_rle(ann, height, width)
    x, y, w, h = mask_utils.toBbox(rle).tolist()
    area = float(mask_utils.area(rle))
    ann["bbox"] = [float(x), float(y), float(w), float(h)]
    ann["area"] = area
    return ann


def find_image(image_root, split, file_name, scene_name=None):
    split_root = image_root / split
    direct = split_root / file_name
    if direct.exists():
        return direct
    if scene_name:
        by_scene = split_root / scene_name / file_name
        if by_scene.exists():
            return by_scene
    stem = Path(file_name).stem
    for path in split_root.rglob(stem + ".*"):
        if path.suffix in IMAGE_SUFFIXES:
            return path
    raise FileNotFoundError(f"Cannot find image for {split}/{file_name} under {split_root}")


def collect_detected_label_counts(ann_root):
    counts = Counter()
    for split in ["train", "val", "test"]:
        data = json.loads((ann_root / f"{split}.json").read_text(encoding="utf-8"))
        for ann in data.get("annotations", []):
            counts[clean_label(ann.get("detected_label", "object"))] += 1
    return counts


def build_category_map(ann_root, mode, min_count):
    if mode == "class_agnostic":
        return {"object": 0}, [{"id": 0, "name": "object"}]

    counts = collect_detected_label_counts(ann_root)
    names = sorted([name for name, count in counts.items() if count >= min_count])
    if not names:
        names = ["object"]
    cat_map = {name: idx for idx, name in enumerate(names)}
    cats = [{"id": idx, "name": name} for name, idx in cat_map.items()]
    return cat_map, cats


def convert_split(split, image_root, ann_root, out_root, mode, cat_map, categories, recompute_bbox, overwrite_images):
    src_ann = ann_root / f"{split}.json"
    data = json.loads(src_ann.read_text(encoding="utf-8"))
    old_cats = {cat["id"]: cat["name"] for cat in data.get("categories", [])}

    out_images = out_root / "images" / split
    out_images.mkdir(parents=True, exist_ok=True)

    image_by_id = {img["id"]: img for img in data["images"]}
    new_images = []
    for img in data["images"]:
        scene_name = old_cats.get(img.get("classification_label"), None)
        src_img = find_image(image_root, split, img["file_name"], scene_name)
        dst_rel = f"{split}/{Path(img['file_name']).name}"
        dst_img = out_root / "images" / dst_rel
        if overwrite_images or not dst_img.exists() or dst_img.stat().st_size == 0:
            shutil.copy2(src_img, dst_img)
        new_img = dict(img)
        new_img["file_name"] = dst_rel.replace("\\", "/")
        new_images.append(new_img)

    new_annotations = []
    kept = 0
    dropped = 0
    for ann in data["annotations"]:
        new_ann = dict(ann)
        if mode == "class_agnostic":
            new_ann["category_id"] = 0
        else:
            label = clean_label(new_ann.get("detected_label", "object"))
            if label not in cat_map:
                dropped += 1
                continue
            new_ann["category_id"] = cat_map[label]
        if recompute_bbox:
            new_ann = recompute_bbox_area(new_ann, image_by_id[new_ann["image_id"]])
        new_annotations.append(new_ann)
        kept += 1

    out_data = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": categories,
        "info": {
            "conversion_mode": mode,
            "source_annotation": str(src_ann),
            "bbox": "tight mask bbox" if recompute_bbox else "source bbox",
        },
    }
    out_ann_dir = out_root / "annotations"
    out_ann_dir.mkdir(parents=True, exist_ok=True)
    (out_ann_dir / f"{split}.json").write_text(json.dumps(out_data, ensure_ascii=False), encoding="utf-8")
    return kept, dropped, len(new_images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", required=True, help="Root with train/val/test class folders.")
    parser.add_argument("--ann-root", required=True, help="Root with train.json, val.json, test.json.")
    parser.add_argument("--out-root", required=True, help="Prepared output root.")
    parser.add_argument("--mode", choices=["class_agnostic", "detected_label"], default="class_agnostic")
    parser.add_argument("--min-count", type=int, default=5, help="For detected_label mode, drop labels with fewer total instances.")
    parser.add_argument("--keep-source-bbox", action="store_true", help="Do not recompute tight bboxes from masks.")
    parser.add_argument("--overwrite-images", action="store_true", help="Overwrite copied images in out-root/images.")
    args = parser.parse_args()

    image_root = Path(args.image_root)
    ann_root = Path(args.ann_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cat_map, categories = build_category_map(ann_root, args.mode, args.min_count)
    print(f"[INFO] mode={args.mode}")
    print(f"[INFO] categories={len(categories)}")
    print(f"[INFO] out_root={out_root}")

    summary = []
    for split in ["train", "val", "test"]:
        kept, dropped, n_images = convert_split(
            split=split,
            image_root=image_root,
            ann_root=ann_root,
            out_root=out_root,
            mode=args.mode,
            cat_map=cat_map,
            categories=categories,
            recompute_bbox=not args.keep_source_bbox,
            overwrite_images=args.overwrite_images,
        )
        summary.append({"split": split, "images": n_images, "kept_annotations": kept, "dropped_annotations": dropped})
        print(f"[DONE] {split}: images={n_images}, kept={kept}, dropped={dropped}")

    (out_root / "conversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[DONE] prepared dataset is ready for MMDetection and SAM scripts.")
    print(f"       data-root: {out_root / 'annotations'}")
    print(f"       img-root : {out_root / 'images'}")


if __name__ == "__main__":
    main()
