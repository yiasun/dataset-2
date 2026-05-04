import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils
from PIL import Image


def ann_to_rle(ann, height, width):
    segm = ann["segmentation"]
    if isinstance(segm, list):
        rles = mask_utils.frPyObjects(segm, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segm, dict) and isinstance(segm.get("counts"), list):
        rle = mask_utils.frPyObjects(segm, height, width)
    else:
        rle = segm
    return rle


def compute_simple_matched_miou(coco_gt, coco_dt):
    gt_img_to_anns = defaultdict(list)
    dt_img_to_anns = defaultdict(list)

    for ann in coco_gt.dataset["annotations"]:
        img = coco_gt.imgs[ann["image_id"]]
        gt_img_to_anns[ann["image_id"]].append((ann, img["height"], img["width"]))
    for ann in coco_dt.dataset["annotations"]:
        img = coco_gt.imgs[ann["image_id"]]
        dt_img_to_anns[ann["image_id"]].append((ann, img["height"], img["width"]))

    ious = []
    for img_id, gt_items in gt_img_to_anns.items():
        pred_items = dt_img_to_anns.get(img_id, [])
        used = set()
        for gt_ann, h, w in gt_items:
            gt_rle = ann_to_rle(gt_ann, h, w)
            best_iou = 0.0
            best_j = -1
            for j, (dt_ann, _, _) in enumerate(pred_items):
                if j in used:
                    continue
                if dt_ann["category_id"] != gt_ann["category_id"]:
                    continue
                dt_rle = ann_to_rle(dt_ann, h, w)
                iou = mask_utils.iou([dt_rle], [gt_rle], [0])[0][0]
                if iou > best_iou:
                    best_iou = float(iou)
                    best_j = j
            if best_j >= 0:
                used.add(best_j)
                ious.append(best_iou)
    return float(np.mean(ious)) if ious else 0.0


def save_metrics_csv(out_csv, overall, per_class):
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "name", "value"])
        for k, v in overall.items():
            writer.writerow(["overall", k, v])
        for cname, ap in per_class.items():
            writer.writerow(["per_class_ap", cname, ap])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--img-root", required=False, default="")
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    coco_gt = COCO(args.gt)
    coco_dt = coco_gt.loadRes(args.pred)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    overall = {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "AP_small": float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large": float(stats[5]),
    }

    precisions = coco_eval.eval["precision"]
    cat_ids = coco_gt.getCatIds()
    cats = coco_gt.loadCats(cat_ids)
    per_class = {}
    for idx, cat in enumerate(cats):
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        per_class[cat["name"]] = float(np.mean(precision)) if precision.size else 0.0

    miou = compute_simple_matched_miou(coco_gt, coco_dt)
    overall["mIoU"] = miou

    with open(args.pred, "r", encoding="utf-8") as f:
        pred_data = json.load(f)
    if isinstance(pred_data, dict) and "_fps" in pred_data:
        overall["FPS"] = float(pred_data["_fps"])

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    save_metrics_csv(args.out_csv, overall, per_class)
    print(f"[DONE] metrics saved to {args.out_csv}")


if __name__ == "__main__":
    main()
