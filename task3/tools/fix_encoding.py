import json
from pathlib import Path

files = [
    r"D:\data2\10K Dataset_Segmentation Labelled\train.json",
    r"D:\data2\10K Dataset_Segmentation Labelled\val.json",
    r"D:\data2\10K Dataset_Segmentation Labelled\test.json",
]

for f in files:
    p = Path(f)
    print(f"[FIX] {p}")

    # 读（自动识别）
    with open(p, "r", encoding="utf-8", errors="ignore") as fr:
        data = json.load(fr)

    # 写成标准 UTF-8（无 BOM）
    with open(p, "w", encoding="utf-8") as fw:
        json.dump(data, fw, ensure_ascii=False)

print("[DONE] all json converted to utf-8")