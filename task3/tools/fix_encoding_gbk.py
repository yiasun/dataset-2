import json
from pathlib import Path

files = [
    r"D:\data2\10K Dataset_Segmentation Labelled\train.json",
    r"D:\data2\10K Dataset_Segmentation Labelled\val.json",
    r"D:\data2\10K Dataset_Segmentation Labelled\test.json",
]

for f in files:
    p = Path(f)
    print(f"[FIX-GBK] {p}")

    # 先用 UTF-8 读
    with open(p, "r", encoding="utf-8") as fr:
        data = json.load(fr)

    # 再写成 GBK
    with open(p, "w", encoding="gbk", errors="ignore") as fw:
        json.dump(data, fw, ensure_ascii=False)

print("[DONE] converted to GBK")