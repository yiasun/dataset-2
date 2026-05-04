import argparse
import glob
import os

import pandas as pd


def load_metrics_csv(path):
    df = pd.read_csv(path)
    overall = df[df["section"] == "overall"][["name", "value"]].copy()
    row = {r["name"]: r["value"] for _, r in overall.iterrows()}
    row["model_dir"] = os.path.basename(os.path.dirname(path))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t3-dir", required=True)
    parser.add_argument("--out-xlsx", required=True)
    args = parser.parse_args()

    metric_files = glob.glob(os.path.join(args.t3_dir, "*", "metrics.csv"))
    rows = [load_metrics_csv(p) for p in metric_files]
    t3_df = pd.DataFrame(rows)
    if not t3_df.empty:
        cols = [c for c in ["model_dir", "AP", "AP50", "AP75", "mIoU", "FPS"] if c in t3_df.columns]
        t3_df = t3_df[cols]
    with pd.ExcelWriter(args.out_xlsx, engine="openpyxl") as writer:
        t3_df.to_excel(writer, sheet_name="T3_overall", index=False)
    print(f"[DONE] summary saved -> {args.out_xlsx}")


if __name__ == "__main__":
    main()
