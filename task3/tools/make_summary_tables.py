import argparse
import glob
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-root", required=True, help="Folder that contains per-model metric subfolders.")
    parser.add_argument("--out-xlsx", required=True)
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    metric_files = glob.glob(str(Path(args.metrics_root) / "*" / "t3_main_metrics.csv"))
    rows = []
    for path in sorted(metric_files):
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        row["metric_dir"] = Path(path).parent.name
        rows.append(row)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        preferred = [
            "metric_dir",
            "Model",
            "AP",
            "AP50",
            "AP75",
            "bbox_AP",
            "bbox_AP50",
            "bbox_AP75",
            "mIoU",
            "FPS",
            "score_thr",
        ]
        cols = [c for c in preferred if c in summary.columns]
        cols += [c for c in summary.columns if c not in cols]
        summary = summary[cols]

    out_xlsx = Path(args.out_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="T3_main_metrics", index=False)

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"[DONE] merged {len(summary)} metric rows")
    print(f"[DONE] summary xlsx -> {out_xlsx}")
    if args.out_csv:
        print(f"[DONE] summary csv  -> {args.out_csv}")


if __name__ == "__main__":
    main()
