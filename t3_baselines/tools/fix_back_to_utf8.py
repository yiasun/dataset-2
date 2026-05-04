import json

files = [
    r"D:\data2\10K Dataset_Segmentation Labelled\train.json",
    r"D:\data2\10K Dataset_Segmentation Labelled\val.json",
    r"D:\data2\10K Dataset_Segmentation Labelled\test.json",
]

for f in files:
    print("[FIX UTF-8]", f)

    # 用 GBK 读（因为你刚刚转过）
    with open(f, "r", encoding="gbk", errors="ignore") as fr:
        data = json.load(fr)

    # 写回 UTF-8
    with open(f, "w", encoding="utf-8") as fw:
        json.dump(data, fw, ensure_ascii=False)

print("[DONE]")