import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from mmdet.apis import init_detector, inference_detector
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm


def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def encode_binary_mask(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def extract_pred_instances(result):
    pred = result.pred_instances
    if len(pred) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    bboxes = pred.bboxes.detach().cpu().numpy()
    scores = pred.scores.detach().cpu().numpy()
    labels = pred.labels.detach().cpu().numpy()
    return bboxes, scores, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-config", required=True)
    parser.add_argument("--det-checkpoint", required=True)
    parser.add_argument("--ann", required=True)
    parser.add_argument("--img-root", required=True)
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--model-type", default="vit_b", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--det-score-thr", type=float, default=0.05)
    parser.add_argument("--max-per-img", type=int, default=100)
    parser.add_argument(
        "--score-mode",
        default="det",
        choices=["det", "sam", "product"],
        help="Score stored in the COCO prediction: detector score, SAM mask score, or product.",
    )
    args = parser.parse_args()

    coco = COCO(args.ann)
    cats = coco.loadCats(coco.getCatIds())
    cat_ids = [cat["id"] for cat in cats]
    label_to_cat = {idx: cat_id for idx, cat_id in enumerate(cat_ids)}
    fallback_cat_id = cat_ids[0]

    detector = init_detector(args.det_config, args.det_checkpoint, device=args.device)

    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)

    results = []
    failed_images = 0
    total_instances = 0
    total_time = 0.0

    for idx, img_id in enumerate(tqdm(coco.getImgIds(), desc="SAM refine detector boxes")):
        info = coco.loadImgs([img_id])[0]
        img_path = str(Path(args.img_root) / info["file_name"])

        image_bgr = imread_unicode(img_path)
        if image_bgr is None:
            print(f"[WARN] failed to read {img_path}")
            failed_images += 1
            continue

        det_result = inference_detector(detector, img_path)
        bboxes, det_scores, labels = extract_pred_instances(det_result)
        keep = det_scores >= args.det_score_thr
        bboxes = bboxes[keep]
        det_scores = det_scores[keep]
        labels = labels[keep]

        if len(det_scores) > args.max_per_img:
            order = np.argsort(-det_scores)[: args.max_per_img]
            bboxes = bboxes[order]
            det_scores = det_scores[order]
            labels = labels[order]

        if len(det_scores) == 0:
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        start_img = time.perf_counter()
        predictor.set_image(image_rgb)
        total_time += time.perf_counter() - start_img

        h, w = image_rgb.shape[:2]
        for box, det_score, label in zip(bboxes, det_scores, labels):
            x1, y1, x2, y2 = box.astype(np.float32).tolist()
            x1 = max(0.0, min(float(x1), w - 1.0))
            y1 = max(0.0, min(float(y1), h - 1.0))
            x2 = max(0.0, min(float(x2), w - 1.0))
            y2 = max(0.0, min(float(y2), h - 1.0))
            if x2 <= x1 or y2 <= y1:
                continue

            start = time.perf_counter()
            with torch.no_grad():
                masks, sam_scores, _ = predictor.predict(
                    box=np.array([x1, y1, x2, y2], dtype=np.float32),
                    point_coords=None,
                    point_labels=None,
                    multimask_output=True,
                )
            total_time += time.perf_counter() - start
            total_instances += 1

            best_idx = int(np.argmax(sam_scores))
            mask = masks[best_idx]
            sam_score = float(sam_scores[best_idx])
            det_score = float(det_score)

            if args.score_mode == "sam":
                score = sam_score
            elif args.score_mode == "product":
                score = det_score * sam_score
            else:
                score = det_score

            rle = encode_binary_mask(mask)
            area = float(mask_utils.area(rle))
            mx, my, mw, mh = mask_utils.toBbox(rle).tolist()

            results.append(
                {
                    "image_id": int(img_id),
                    "category_id": int(label_to_cat.get(int(label), fallback_cat_id)),
                    "segmentation": rle,
                    "bbox": [float(mx), float(my), float(mw), float(mh)],
                    "score": float(score),
                    "area": area,
                }
            )

        if (idx + 1) % 100 == 0:
            print(f"[INFO] refined {idx + 1}/{len(coco.getImgIds())}")

    out_dir = os.path.dirname(args.out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f)

    fps = total_instances / total_time if total_time > 0 else 0.0
    print(f"[DONE] saved predictions -> {args.out_json}")
    print(f"[INFO] total predictions: {len(results)}")
    print(f"[INFO] failed images: {failed_images}")
    print(f"[INFO] Approx instance FPS: {fps:.4f}")


if __name__ == "__main__":
    main()
