import os
import shutil
from pathlib import Path

src_root = Path(r"D:\data2\10K Dataset_Labelled")
dst_root = Path(r"D:\data2\t3_flat_images")
dst_root.mkdir(parents=True, exist_ok=True)
cd
exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
count = 0
skip = 0

for p in src_root.rglob("*"):
    if p.is_file() and p.suffix.lower() in exts:
        dst = dst_root / p.name
        if dst.exists():
            skip += 1
            continue
        try:
            os.link(p, dst)   # 硬链接，不额外占空间
        except Exception:
            shutil.copy2(p, dst)
        count += 1

print(f"[DONE] linked/copied {count} images to {dst_root}")
print(f"[INFO] skipped duplicated names: {skip}")