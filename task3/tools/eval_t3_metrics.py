import argparse
import json
import os
import time
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
import torch
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from pycocotools.cocoeval import COCOeval

from mmdet.apis import init_detector, inference_detector


def mask_to_rle(mask):
    mask = np.asfortranarray(mask.astype(np.uint8))
    rle = maskUtils.encode(mask)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def decode_gt_mask(coco, ann):
    h = coco.imgs[ann["image_id"]]["height"]
    w = coco.imgs[ann["image_id"]]["width"]
    rle = coco.annToRLE(ann)
    m = maskUtils.decode(rle)
    if m.ndim == 3:
        m = np.any(m, axis=2)
    return m.astype(bool)


def compute_iou(m1, m2):
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter / union) if union > 0 else 0.0


def extract_instances(result):
    pred = result.pred_instances
    bboxes = pred.bboxes.detach().cpu().numpy() if len(pred) else np.zeros((0, 4))
    scores = pred.scores.detach().cpu().numpy() if len(pred) else np.zeros((0,))
    labels = pred.labels.detach().cpu().numpy() if len(pred) else np.zeros((0,), dtype=int)

    masks = []
    if hasattr(pred, "masks") and pred.masks is not None:
        raw = pred.masks
        if torch.is_tensor(raw):
            raw = raw.detach().cpu().numpy()
        else:
            raw = raw.to_ndarray()
        masks = [(m > 0.5) for m in raw]

    return bboxes, scores, labels, masks


def run_inference(args):
    coco = COCO(args.ann)
    img_ids = coco.getImgIds()
    cats = coco.loadCats(coco.getCatIds())
    cat_ids = [c["id"] for c in cats]
    label_to_cat = {i: cid for i, cid in enumerate(cat_ids)}

    model = init_detector(args.config, args.checkpoint, device=args.device)

    bbox_results = []
    segm_results = []
    pred_masks_by_img = {}

    t0 = time.time()
    infer_count = 0

    for idx, img_id in enumerate(img_ids):
        info = coco.loadImgs([img_id])[0]
        img_path = str(Path(args.img_root) / info["file_name"])

        result = inference_detector(model, img_path)
        bboxes, scores, labels, masks = extract_instances(result)

        pred_masks_by_img[img_id] = []

        for box, score, label, mask in zip(bboxes, scores, labels, masks):
            if score < args.score_thr:
                continue

            x1, y1, x2, y2 = box.tolist()
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            cat_id = label_to_cat[int(label)]

            bbox_results.append({
                "image_id": int(img_id),
                "category_id": int(cat_id),
                "bbox": [x1, y1, w, h],
                "score": float(score),
            })

            segm_results.append({
                "image_id": int(img_id),
                "category_id": int(cat_id),
                "segmentation": mask_to_rle(mask),
                "score": float(score),
            })

            pred_masks_by_img[img_id].append({
                "category_id": int(cat_id),
                "score": float(score),
                "mask": mask,
            })

        infer_count += 1
        if (idx + 1) % 100 == 0:
            print(f"[INFO] inferred {idx + 1}/{len(img_ids)}")

    total_time = time.time() - t0
    fps = infer_count / total_time

    return coco, cats, bbox_results, segm_results, pred_masks_by_img, fps


def coco_eval(coco, results, iou_type, out_json):
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f)

    if len(results) == 0:
        print(f"[WARN] no {iou_type} predictions above score threshold.")
        return {
            "AP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "AP_small": 0.0,
            "AP_medium": 0.0,
            "AP_large": 0.0,
        }

    coco_dt = coco.loadRes(str(out_json))
    ev = COCOeval(coco, coco_dt, iou_type)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    return {
        "AP": float(ev.stats[0]),
        "AP50": float(ev.stats[1]),
        "AP75": float(ev.stats[2]),
        "AP_small": float(ev.stats[3]),
        "AP_medium": float(ev.stats[4]),
        "AP_large": float(ev.stats[5]),
    }


def per_class_ap(coco, cats, results, iou_type, out_json):
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f)

    if len(results) == 0:
        return [
            {
                "category_id": cat["id"],
                "class_name": cat["name"],
                f"{iou_type}_AP": 0.0,
                f"{iou_type}_AP50": 0.0,
                f"{iou_type}_AP75": 0.0,
            }
            for cat in cats
        ]

    coco_dt = coco.loadRes(str(out_json))
    rows = []

    for cat in cats:
        ev = COCOeval(coco, coco_dt, iou_type)
        ev.params.catIds = [cat["id"]]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

        rows.append({
            "category_id": cat["id"],
            "class_name": cat["name"],
            f"{iou_type}_AP": float(ev.stats[0]),
            f"{iou_type}_AP50": float(ev.stats[1]),
            f"{iou_type}_AP75": float(ev.stats[2]),
        })

    return rows


def compute_miou(coco, pred_masks_by_img, score_thr):
    ious = []
    per_class = {cid: [] for cid in coco.getCatIds()}

    for img_id in coco.getImgIds():
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        preds = [p for p in pred_masks_by_img.get(img_id, []) if p["score"] >= score_thr]

        for ann in anns:
            gt_mask = decode_gt_mask(coco, ann)
            gt_cat = ann["category_id"]

            best = 0.0
            for p in preds:
                if p["category_id"] != gt_cat:
                    continue
                best = max(best, compute_iou(gt_mask, p["mask"]))

            ious.append(best)
            per_class[gt_cat].append(best)

    miou = float(np.mean(ious)) if ious else 0.0
    per_class_miou = {
        cid: float(np.mean(vals)) if vals else 0.0
        for cid, vals in per_class.items()
    }
    return miou, per_class_miou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ann", required=True)
    parser.add_argument("--img-root", required=True)
    parser.add_argument("--model-name", default="Cascade Mask R-CNN")
    parser.add_argument("--out-dir", default="outputs/t3_metrics")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-thr", type=float, default=0.001)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coco, cats, bbox_results, segm_results, pred_masks_by_img, fps = run_inference(args)

    bbox_json = out_dir / "bbox_predictions.json"
    segm_json = out_dir / "segm_predictions.json"

    bbox_metrics = coco_eval(coco, bbox_results, "bbox", bbox_json)
    segm_metrics = coco_eval(coco, segm_results, "segm", segm_json)

    bbox_pc = per_class_ap(coco, cats, bbox_results, "bbox", out_dir / "bbox_predictions_for_pc.json")
    segm_pc = per_class_ap(coco, cats, segm_results, "segm", out_dir / "segm_predictions_for_pc.json")

    miou, pc_miou = compute_miou(coco, pred_masks_by_img, args.score_thr)

    main_row = {
        "Model": args.model_name,
        "Checkpoint": args.checkpoint,
        "AP": segm_metrics["AP"],
        "AP50": segm_metrics["AP50"],
        "AP75": segm_metrics["AP75"],
        "bbox_AP": bbox_metrics["AP"],
        "bbox_AP50": bbox_metrics["AP50"],
        "bbox_AP75": bbox_metrics["AP75"],
        "mIoU": miou,
        "FPS": fps,
        "score_thr": args.score_thr,
    }

    main_df = pd.DataFrame([main_row])

    pc_rows = []
    segm_map = {r["category_id"]: r for r in segm_pc}
    bbox_map = {r["category_id"]: r for r in bbox_pc}

    for cat in cats:
        cid = cat["id"]
        pc_rows.append({
            "Model": args.model_name,
            "category_id": cid,
            "class_name": cat["name"],
            "segm_AP": segm_map[cid]["segm_AP"],
            "segm_AP50": segm_map[cid]["segm_AP50"],
            "segm_AP75": segm_map[cid]["segm_AP75"],
            "bbox_AP": bbox_map[cid]["bbox_AP"],
            "bbox_AP50": bbox_map[cid]["bbox_AP50"],
            "bbox_AP75": bbox_map[cid]["bbox_AP75"],
            "mIoU": pc_miou.get(cid, 0.0),
        })

    pc_df = pd.DataFrame(pc_rows)

    main_csv = out_dir / "t3_main_metrics.csv"
    pc_csv = out_dir / "t3_per_class_metrics.csv"
    xlsx_path = out_dir / "t3_metrics.xlsx"

    main_df.to_csv(main_csv, index=False, encoding="utf-8-sig")
    pc_df.to_csv(pc_csv, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        main_df.to_excel(writer, sheet_name="Main Metrics", index=False)
        pc_df.to_excel(writer, sheet_name="Per-class Metrics", index=False)

    print(f"[DONE] main csv -> {main_csv}")
    print(f"[DONE] per-class csv -> {pc_csv}")
    print(f"[DONE] xlsx -> {xlsx_path}")


if __name__ == "__main__":
    main()
