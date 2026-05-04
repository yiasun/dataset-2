import argparse
import json
import os
import time
from collections import defaultdict

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm


def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def xywh_to_xyxy(box):
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def encode_binary_mask(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", required=True)
    parser.add_argument("--img-root", required=True)
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument(
        "--model-type",
        default="vit_b",
        choices=["vit_h", "vit_l", "vit_b"],
    )
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-thr", type=float, default=0.0)
    args = parser.parse_args()

    coco = COCO(args.ann)

    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)

    anns_by_img = defaultdict(list)
    for ann in coco.dataset["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    results = []
    total_time = 0.0
    total_instances = 0
    failed_images = 0

    for img_id in tqdm(coco.getImgIds(), desc="SAM zero-shot"):
        img_info = coco.loadImgs([img_id])[0]
        img_path = os.path.join(args.img_root, img_info["file_name"])

        image = imread_unicode(img_path)
        if image is None:
            print(f"[WARN] failed to read {img_path}")
            failed_images += 1
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        start_img = time.perf_counter()
        predictor.set_image(image)
        set_image_time = time.perf_counter() - start_img

        img_anns = anns_by_img.get(img_id, [])
        if not img_anns:
            continue

        total_time += set_image_time

        for ann in img_anns:
            box = xywh_to_xyxy(ann["bbox"])

            start = time.perf_counter()
            masks, scores, _ = predictor.predict(
                box=box,
                point_coords=None,
                point_labels=None,
                multimask_output=True,
            )
            total_time += time.perf_counter() - start
            total_instances += 1

            best_idx = int(np.argmax(scores))
            best_mask = masks[best_idx]
            best_score = float(scores[best_idx])

            if best_score < args.score_thr:
                continue

            rle = encode_binary_mask(best_mask)
            area = float(mask_utils.area(rle))
            x, y, w, h = mask_utils.toBbox(rle).tolist()

            results.append(
                {
                    "image_id": int(img_id),
                    "category_id": int(ann["category_id"]),
                    "segmentation": rle,
                    "score": best_score,
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "area": area,
                }
            )

    fps = total_instances / total_time if total_time > 0 else 0.0

    out_dir = os.path.dirname(args.out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f)

    print(f"[DONE] saved predictions -> {args.out_json}")
    print(f"[INFO] total predictions: {len(results)}")
    print(f"[INFO] failed images: {failed_images}")
    print(f"[INFO] Approx instance FPS: {fps:.4f}")


if __name__ == "__main__":
    main()