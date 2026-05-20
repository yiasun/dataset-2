from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils


PALETTE = [
    (64, 128, 255),
    (72, 180, 120),
    (230, 120, 64),
    (170, 96, 220),
    (220, 170, 64),
    (80, 190, 190),
]


def parse_pred_arg(value: str) -> tuple[str, Path]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("--pred must look like Name:/path/to/predictions.json")
    name, path = value.split(":", 1)
    if not name:
        raise argparse.ArgumentTypeError("Prediction name is empty")
    return name, Path(path)


def decode_segmentation(segmentation, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
        return mask_utils.decode(rle).astype(bool)
    if isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
        return mask_utils.decode(rle).astype(bool)
    return np.zeros((height, width), dtype=bool)


def colorize(base: np.ndarray, anns: list[dict], height: int, width: int, score_thr: float, max_per_img: int) -> np.ndarray:
    out = base.copy()
    selected = [a for a in anns if float(a.get("score", 1.0)) >= score_thr]
    selected = sorted(selected, key=lambda x: float(x.get("score", 1.0)), reverse=True)[:max_per_img]
    for idx, ann in enumerate(selected):
        mask = decode_segmentation(ann.get("segmentation"), height, width)
        if mask.sum() == 0:
            continue
        color = np.array(PALETTE[idx % len(PALETTE)], dtype=np.float32)
        alpha = 0.46
        out[mask] = (out[mask].astype(np.float32) * (1 - alpha) + color * alpha).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, tuple(int(c) for c in color), 2)
    return out


def put_title(img: np.ndarray, title: str) -> np.ndarray:
    pad = 34
    out = np.full((img.shape[0] + pad, img.shape[1], 3), 255, dtype=np.uint8)
    out[pad:] = img
    cv2.putText(out, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2, cv2.LINE_AA)
    return out


def fit_height(img: np.ndarray, target_h: int) -> np.ndarray:
    if img.shape[0] == target_h:
        return img
    scale = target_h / img.shape[0]
    return cv2.resize(img, (int(round(img.shape[1] * scale)), target_h), interpolation=cv2.INTER_AREA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", required=True)
    parser.add_argument("--img-root", required=True)
    parser.add_argument("--pred", type=parse_pred_arg, action="append", default=[])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--score-thr", type=float, default=0.25)
    parser.add_argument("--max-per-img", type=int, default=20)
    args = parser.parse_args()

    ann_path = Path(args.ann)
    img_root = Path(args.img_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    images = {int(img["id"]): img for img in coco["images"]}
    gt_by_img: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        gt_by_img[int(ann["image_id"])].append(ann)

    preds_by_name: list[tuple[str, dict[int, list[dict]]]] = []
    for name, pred_path in args.pred:
        preds = json.loads(pred_path.read_text(encoding="utf-8"))
        grouped: dict[int, list[dict]] = defaultdict(list)
        for pred in preds:
            grouped[int(pred["image_id"])].append(pred)
        preds_by_name.append((name, grouped))

    ranked = sorted(images, key=lambda img_id: len(gt_by_img[img_id]), reverse=True)
    saved = 0
    for img_id in ranked:
        info = images[img_id]
        file_name = info["file_name"]
        img_path = Path(file_name)
        if not img_path.is_absolute():
            img_path = img_root / file_name
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        height, width = image.shape[:2]
        panels = [
            put_title(colorize(image, gt_by_img[img_id], height, width, 0.0, args.max_per_img), f"GT image_id={img_id}")
        ]
        for name, grouped in preds_by_name:
            panel = colorize(image, grouped.get(img_id, []), height, width, args.score_thr, args.max_per_img)
            panels.append(put_title(panel, name))

        target_h = min(520, max(p.shape[0] for p in panels))
        panels = [fit_height(p, target_h) for p in panels]
        canvas = np.concatenate(panels, axis=1)
        out_path = out_dir / f"qual_{saved + 1:02d}_image_{img_id}.jpg"
        cv2.imwrite(str(out_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        saved += 1
        if saved >= args.num_images:
            break

    print(f"[DONE] saved {saved} qualitative panels -> {out_dir}")


if __name__ == "__main__":
    main()
