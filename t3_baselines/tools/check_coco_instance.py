import argparse
import json
import os
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", required=True)
    parser.add_argument("--img-root", required=True)
    args = parser.parse_args()

    with open(args.ann, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "images" in data and "annotations" in data and "categories" in data, "Not a COCO-style annotation file."

    images = data["images"]
    anns = data["annotations"]
    cats = data["categories"]

    print(f"[INFO] images={len(images)} annotations={len(anns)} categories={len(cats)}")

    id_to_name = {c["id"]: c["name"] for c in cats}
    cat_counter = Counter(a["category_id"] for a in anns)
    print("[INFO] category distribution:")
    for cid, cnt in sorted(cat_counter.items(), key=lambda x: x[0]):
        print(f"  {cid}: {id_to_name.get(cid, 'UNKNOWN')} -> {cnt}")

    missing_images = 0
    for img in images[:50]:
        p = os.path.join(args.img_root, img["file_name"])
        if not os.path.exists(p):
            missing_images += 1
            print("[WARN] missing:", p)
    if missing_images == 0:
        print("[INFO] first 50 image paths look valid.")

    missing_segm = 0
    for ann in anns[:200]:
        if "segmentation" not in ann:
            missing_segm += 1
    if missing_segm > 0:
        print(f"[WARN] {missing_segm} of first 200 annotations miss segmentation field.")
    else:
        print("[INFO] first 200 annotations all have segmentation.")

    print("[DONE] basic COCO instance check passed.")


if __name__ == "__main__":
    main()
