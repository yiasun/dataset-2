import argparse
import json
import os
import pandas as pd

os.environ["MPLBACKEND"] = "Agg"

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def per_class_ap(coco_gt, coco_dt, cats, iou_type):
    rows = []

    for cat in cats:
        evaluator = COCOeval(coco_gt, coco_dt, iou_type)
        evaluator.params.catIds = [cat["id"]]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

        rows.append({
            "category_id": cat["id"],
            "class_name": cat["name"],
            f"{iou_type}_AP": float(evaluator.stats[0]),
            f"{iou_type}_AP50": float(evaluator.stats[1]),
            f"{iou_type}_AP75": float(evaluator.stats[2]),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=float, default=0.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    coco_gt = COCO(args.ann)
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    with open(args.pred, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    if len(predictions) == 0:
        print("[WARN] prediction file is empty. Writing zero metrics.")
        main_df = pd.DataFrame([{
            "Model": args.model_name,
            "AP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "AP_small": 0.0,
            "AP_medium": 0.0,
            "AP_large": 0.0,
            "AR": 0.0,
            "FPS": float(args.fps),
        }])
        per_class_df = pd.DataFrame([
            {
                "category_id": cat["id"],
                "class_name": cat["name"],
                "segm_AP": 0.0,
                "segm_AP50": 0.0,
                "segm_AP75": 0.0,
            }
            for cat in cats
        ])
        csv_main = os.path.join(args.out_dir, "t3_main_metrics.csv")
        csv_pc = os.path.join(args.out_dir, "t3_per_class_metrics.csv")
        xlsx_path = os.path.join(args.out_dir, "t3_metrics.xlsx")
        main_df.to_csv(csv_main, index=False, encoding="utf-8-sig")
        per_class_df.to_csv(csv_pc, index=False, encoding="utf-8-sig")
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            main_df.to_excel(writer, sheet_name="main_metrics", index=False)
            per_class_df.to_excel(writer, sheet_name="per_class_ap", index=False)
        return

    coco_dt = coco_gt.loadRes(predictions)

    print("=" * 80)
    print("SEGMENTATION METRICS")
    print("=" * 80)

    seg_eval = COCOeval(coco_gt, coco_dt, "segm")
    seg_eval.evaluate()
    seg_eval.accumulate()
    seg_eval.summarize()

    main_df = pd.DataFrame([{
        "Model": args.model_name,
        "AP": float(seg_eval.stats[0]),
        "AP50": float(seg_eval.stats[1]),
        "AP75": float(seg_eval.stats[2]),
        "AP_small": float(seg_eval.stats[3]),
        "AP_medium": float(seg_eval.stats[4]),
        "AP_large": float(seg_eval.stats[5]),
        "AR": float(seg_eval.stats[8]),
        "FPS": float(args.fps)
    }])

    print("=" * 80)
    print("PER-CLASS AP")
    print("=" * 80)

    per_class_df = per_class_ap(coco_gt, coco_dt, cats, "segm")

    csv_main = os.path.join(args.out_dir, "t3_main_metrics.csv")
    csv_pc = os.path.join(args.out_dir, "t3_per_class_metrics.csv")
    xlsx_path = os.path.join(args.out_dir, "t3_metrics.xlsx")

    main_df.to_csv(csv_main, index=False, encoding="utf-8-sig")
    per_class_df.to_csv(csv_pc, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        main_df.to_excel(writer, sheet_name="main_metrics", index=False)
        per_class_df.to_excel(writer, sheet_name="per_class_ap", index=False)

    print(f"[DONE] saved -> {csv_main}")
    print(f"[DONE] saved -> {csv_pc}")
    print(f"[DONE] saved -> {xlsx_path}")


if __name__ == "__main__":
    main()
