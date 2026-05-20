"""
Task 2 multi-positive image-text retrieval baseline for Urban-ImageNet.

This script fixes the main pitfall in row-level social-media retrieval:
one Weibo post can contain multiple images. For text-to-image retrieval,
all images attached to the same post group are positives. For image-to-text
retrieval, the corresponding post group text is the positive target.

Typical quick 1K runs:

  python scripts/run_task2_multipositive.py ^
    --dataset-root "E:\\NIPS 2026 Dataset\\00 Dataset\\最终上传版" ^
    --dataset-size "1K Dataset" ^
    --split test ^
    --text-source post ^
    --output-dir runs_t2_post_zs

  python scripts/run_task2_multipositive.py ^
    --dataset-root "E:\\NIPS 2026 Dataset\\00 Dataset\\最终上传版" ^
    --dataset-size "1K Dataset" ^
    --text-source label ^
    --do-finetune --epochs 3 --batch-size 16 ^
    --output-dir runs_t2_label_ft

When moving to another computer, normally change only:
  --dataset-root, --dataset-size, --output-dir, --batch-size, --device.
Use a smaller --batch-size if GPU memory is limited. Use --device cpu only
for debugging; fine-tuning CLIP on CPU will be slow.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


CLASS_NAMES = [
    "Exterior urban spaces with people",
    "Exterior urban spaces without people",
    "Food or drink items",
    "Hotel or commercial lodging spaces",
    "Human-centered portrait",
    "Interior urban spaces with people",
    "Interior urban spaces without people",
    "Other non-spatial content",
    "Private home interiors",
    "Retail products and merchandise",
]

CLASS_PROMPTS = {
    "Exterior urban spaces with people": (
        "a social media photo of an activated exterior urban commercial space "
        "with visible pedestrians or people"
    ),
    "Exterior urban spaces without people": (
        "a social media photo of an exterior urban commercial space without "
        "visible people, emphasizing street, plaza, facade, or landscape design"
    ),
    "Food or drink items": "a social media photo centered on food, drinks, dessert, or dining items",
    "Hotel or commercial lodging spaces": "a social media photo of a hotel room or commercial lodging space",
    "Human-centered portrait": "a social media portrait or group photo where people are the main subject",
    "Interior urban spaces with people": (
        "a social media photo of an activated interior commercial space with "
        "visible shoppers, visitors, or workers"
    ),
    "Interior urban spaces without people": (
        "a social media photo of an interior commercial space without visible people"
    ),
    "Other non-spatial content": (
        "a social media image that is a screenshot, poster, advertisement, graphic, "
        "meme, or other non-spatial content"
    ),
    "Private home interiors": "a social media photo of a private home interior",
    "Retail products and merchandise": (
        "a social media photo centered on retail products, merchandise, fashion, cosmetics, or displays"
    ),
}

IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"]


@dataclass(frozen=True)
class PairItem:
    row_id: int
    image_path: str
    image_filename: str
    label: str
    post_id: str
    post_text: str
    user_id: str
    post_time: str
    group_key: str
    text: str


@dataclass(frozen=True)
class TextCandidate:
    group_key: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Urban-ImageNet Task 2 multi-positive CLIP baseline")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"E:\NIPS 2026 Dataset\00 Dataset\最终上传版"),
        help="Root folder containing 1K Dataset, 10K Dataset, 100K Dataset, and Full Dataset-2M.",
    )
    parser.add_argument("--dataset-size", default="1K Dataset", help="Dataset folder, e.g. '1K Dataset'.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Evaluation split.")
    parser.add_argument("--train-split", default="train", choices=["train", "val", "test"])
    parser.add_argument(
        "--text-source",
        default="post",
        choices=["label", "post", "label_plus_post"],
        help="Use HUSIC label prompts, original post text, or both.",
    )
    parser.add_argument(
        "--group-mode",
        default="auto",
        choices=["auto", "post_id", "post_text", "label"],
        help="Positive grouping. 'auto' uses labels for label mode and post IDs for post modes.",
    )
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32", help="Hugging Face CLIP model name.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs_task2_multipositive"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="'auto', 'cuda', 'cpu', or any torch device string.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap for quick debugging.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--do-finetune", action="store_true", help="Fine-tune CLIP with row-level contrastive loss.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision during fine-tuning on CUDA.")
    parser.add_argument("--save-model", action="store_true", help="Save the fine-tuned CLIP checkpoint.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def clean_cell(value) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return re.sub(r"\s+", " ", text).strip()


def normalize_key(text: str, max_len: int = 240) -> str:
    text = clean_cell(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text[:max_len]


def read_split_table(dataset_dir: Path, split: str):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Please install pandas and openpyxl to read the split Excel files.") from exc

    table_path = dataset_dir / "02 Text-Image Pairs" / f"{split}.xlsx"
    if not table_path.exists():
        raise FileNotFoundError(f"Missing split table: {table_path}")
    return pd.read_excel(table_path)


def resolve_image_path(dataset_dir: Path, split: str, label: str, image_filename: str) -> str:
    image_root = dataset_dir / "01 Images with labels" / split
    class_dir = image_root / label
    candidates: List[Path] = []

    raw = Path(image_filename)
    if raw.suffix:
        candidates.append(class_dir / raw.name)
        candidates.append(image_root / raw.name)
    else:
        for suffix in IMAGE_SUFFIXES:
            candidates.append(class_dir / f"{image_filename}{suffix}")
            candidates.append(image_root / f"{image_filename}{suffix}")

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    search_roots = [class_dir, image_root]
    target_stems = {raw.stem if raw.suffix else image_filename}
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.stem in target_stems and path.suffix in IMAGE_SUFFIXES:
                return str(path)

    raise FileNotFoundError(f"Could not find image '{image_filename}' under {image_root}")


def choose_group_key(row: Dict[str, str], label: str, text_source: str, group_mode: str) -> str:
    if group_mode == "label" or (group_mode == "auto" and text_source == "label"):
        return f"label::{label}"

    if group_mode in {"auto", "post_id"}:
        post_id = clean_cell(row.get("Post ID", ""))
        if post_id:
            return f"post::{post_id}"

    if group_mode in {"auto", "post_text"}:
        user_id = clean_cell(row.get("User ID", ""))
        post_time = clean_cell(row.get("Post Time", ""))
        post_text = normalize_key(row.get("Post Text", ""))
        return f"post_text::{user_id}|{post_time}|{post_text}"

    return f"label::{label}"


def choose_text(label: str, post_text: str, text_source: str) -> str:
    if text_source == "label":
        return CLASS_PROMPTS.get(label, f"a social media photo of {label}")
    if text_source == "post":
        return post_text if post_text else CLASS_PROMPTS.get(label, label)
    return f"{CLASS_PROMPTS.get(label, label)}. {post_text}".strip()


def build_items(dataset_dir: Path, split: str, text_source: str, group_mode: str, max_samples: int = 0) -> List[PairItem]:
    df = read_split_table(dataset_dir, split)
    required = {"Image Label", "Image Filename"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {split}.xlsx: {sorted(missing)}")

    items: List[PairItem] = []
    for row_id, row_obj in df.iterrows():
        row = {str(k): clean_cell(v) for k, v in row_obj.to_dict().items()}
        label = clean_cell(row.get("Image Label", ""))
        image_filename = clean_cell(row.get("Image Filename", ""))
        post_text = clean_cell(row.get("Post Text", ""))
        if not label or not image_filename:
            continue

        image_path = resolve_image_path(dataset_dir, split, label, image_filename)
        group_key = choose_group_key(row, label, text_source, group_mode)
        text = choose_text(label, post_text, text_source)
        if not text:
            continue

        items.append(
            PairItem(
                row_id=int(row_id),
                image_path=image_path,
                image_filename=image_filename,
                label=label,
                post_id=clean_cell(row.get("Post ID", "")),
                post_text=post_text,
                user_id=clean_cell(row.get("User ID", "")),
                post_time=clean_cell(row.get("Post Time", "")),
                group_key=group_key,
                text=text,
            )
        )
        if max_samples and len(items) >= max_samples:
            break

    if not items:
        raise RuntimeError(f"No valid image-text items found for split={split}")
    return items


def build_text_candidates(items: Sequence[PairItem]) -> List[TextCandidate]:
    seen: Set[str] = set()
    candidates: List[TextCandidate] = []
    for item in items:
        if item.group_key in seen:
            continue
        seen.add(item.group_key)
        candidates.append(TextCandidate(group_key=item.group_key, text=item.text))
    return candidates


def positives_for_text_to_image(items: Sequence[PairItem], texts: Sequence[TextCandidate]) -> List[Set[int]]:
    group_to_images: Dict[str, Set[int]] = {}
    for idx, item in enumerate(items):
        group_to_images.setdefault(item.group_key, set()).add(idx)
    return [group_to_images.get(text.group_key, set()) for text in texts]


def positives_for_image_to_text(items: Sequence[PairItem], texts: Sequence[TextCandidate]) -> List[Set[int]]:
    group_to_texts: Dict[str, Set[int]] = {}
    for idx, text in enumerate(texts):
        group_to_texts.setdefault(text.group_key, set()).add(idx)
    return [group_to_texts.get(item.group_key, set()) for item in items]


def load_clip(model_name: str, device_arg: str):
    import torch
    from transformers import CLIPModel, CLIPProcessor

    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    return model, processor, device


def batched(seq: Sequence, batch_size: int) -> Iterable[Sequence]:
    for start in range(0, len(seq), batch_size):
        yield seq[start : start + batch_size]


def encode_images(model, processor, device: str, image_paths: Sequence[str], batch_size: int):
    import torch
    import torch.nn.functional as F
    from PIL import Image

    features = []
    model.eval()
    with torch.no_grad():
        for batch_paths in batched(image_paths, batch_size):
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            inputs = processor(images=images, return_tensors="pt").to(device)
            feats = model.get_image_features(**inputs)
            feats = F.normalize(feats, dim=-1)
            features.append(feats.cpu())
    return torch.cat(features, dim=0)


def encode_texts(model, processor, device: str, texts: Sequence[str], batch_size: int):
    import torch
    import torch.nn.functional as F

    features = []
    model.eval()
    with torch.no_grad():
        for batch_texts in batched(list(texts), batch_size):
            inputs = processor(text=list(batch_texts), padding=True, truncation=True, return_tensors="pt").to(device)
            feats = model.get_text_features(**inputs)
            feats = F.normalize(feats, dim=-1)
            features.append(feats.cpu())
    return torch.cat(features, dim=0)


class ClipPairDataset:
    def __init__(self, items: Sequence[PairItem]):
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> PairItem:
        return self.items[idx]


def finetune_clip(
    model,
    processor,
    device: str,
    train_items: Sequence[PairItem],
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    amp: bool,
) -> None:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader
    from transformers.optimization import get_cosine_schedule_with_warmup

    def collate(batch: Sequence[PairItem]):
        images = [Image.open(item.image_path).convert("RGB") for item in batch]
        texts = [item.text for item in batch]
        return processor(text=texts, images=images, padding=True, truncation=True, return_tensors="pt")

    loader = DataLoader(
        ClipPairDataset(train_items),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, epochs * len(loader))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 20),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.startswith("cuda"))
    model.train()

    for epoch in range(epochs):
        running = 0.0
        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp and device.startswith("cuda")):
                out = model(**batch, return_loss=True)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += float(loss.detach().cpu())
            if step % 20 == 0 or step == len(loader):
                print(f"epoch={epoch + 1}/{epochs} step={step}/{len(loader)} loss={running / step:.4f}")


def average_precision(ranked: Sequence[int], positives: Set[int]) -> float:
    if not positives:
        return math.nan
    hits = 0
    total = 0.0
    for rank, idx in enumerate(ranked, start=1):
        if idx in positives:
            hits += 1
            total += hits / rank
            if hits == len(positives):
                break
    return total / len(positives)


def evaluate_similarity(similarity, positives: Sequence[Set[int]], direction: str) -> Dict[str, float]:
    import numpy as np

    recalls = {1: [], 5: [], 10: []}
    aps = []
    first_ranks = []
    valid_queries = 0

    sim_np = similarity.numpy() if hasattr(similarity, "numpy") else similarity
    for q_idx in range(sim_np.shape[0]):
        pos = positives[q_idx]
        if not pos:
            continue
        valid_queries += 1
        ranked = np.argsort(-sim_np[q_idx]).tolist()
        ranked_set_prefix = {}
        for k in recalls:
            top_k = ranked[:k]
            recalls[k].append(float(any(idx in pos for idx in top_k)))
        aps.append(average_precision(ranked, pos))
        first = min((rank + 1 for rank, idx in enumerate(ranked) if idx in pos), default=math.nan)
        first_ranks.append(first)

    if valid_queries == 0:
        raise RuntimeError(f"No valid positives for {direction}")

    return {
        "direction": direction,
        "R@1": 100.0 * float(np.mean(recalls[1])),
        "R@5": 100.0 * float(np.mean(recalls[5])),
        "R@10": 100.0 * float(np.mean(recalls[10])),
        "mAP": 100.0 * float(np.nanmean(aps)),
        "MedianRank": float(np.nanmedian(first_ranks)),
        "num_queries": int(valid_queries),
        "num_candidates": int(sim_np.shape[1]),
        "avg_positives": float(np.mean([len(p) for p in positives if p])),
    }


def write_results(output_dir: Path, rows: List[Dict[str, float]], metadata: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "task2_multipositive_results.csv"
    fieldnames = [
        "setting",
        "model",
        "split",
        "text_source",
        "direction",
        "R@1",
        "R@5",
        "R@10",
        "mAP",
        "MedianRank",
        "num_queries",
        "num_candidates",
        "avg_positives",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Wrote results to {csv_path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset_dir = args.dataset_root / args.dataset_size
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    group_mode = args.group_mode
    if group_mode == "auto" and args.text_source == "label":
        group_mode = "label"

    print(f"Loading CLIP model: {args.model_name}")
    model, processor, device = load_clip(args.model_name, args.device)
    print(f"Using device: {device}")

    if args.do_finetune:
        train_items = build_items(dataset_dir, args.train_split, args.text_source, group_mode, args.max_samples)
        print(f"Fine-tuning on {len(train_items)} row-level image-text pairs")
        finetune_clip(
            model,
            processor,
            device,
            train_items,
            args.batch_size,
            args.epochs,
            args.lr,
            args.weight_decay,
            args.amp,
        )
        if args.save_model:
            ckpt_dir = args.output_dir / "clip_checkpoint"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            processor.save_pretrained(ckpt_dir)
            print(f"Saved checkpoint to {ckpt_dir}")

    eval_items = build_items(dataset_dir, args.split, args.text_source, group_mode, args.max_samples)
    text_candidates = build_text_candidates(eval_items)
    print(f"Evaluation images: {len(eval_items)}")
    print(f"Evaluation text groups: {len(text_candidates)}")

    image_features = encode_images(model, processor, device, [item.image_path for item in eval_items], args.batch_size)
    text_features = encode_texts(model, processor, device, [text.text for text in text_candidates], args.batch_size)

    # Text-to-image: rows are text groups, columns are images.
    similarity_t2i = text_features @ image_features.T
    t2i_pos = positives_for_text_to_image(eval_items, text_candidates)
    t2i = evaluate_similarity(similarity_t2i, t2i_pos, "text-to-image")

    # Image-to-text: rows are images, columns are text groups.
    similarity_i2t = image_features @ text_features.T
    i2t_pos = positives_for_image_to_text(eval_items, text_candidates)
    i2t = evaluate_similarity(similarity_i2t, i2t_pos, "image-to-text")

    mode = "fine-tuned" if args.do_finetune else "zero-shot"
    rows = []
    for row in [t2i, i2t]:
        row = dict(row)
        row.update(
            {
                "setting": "post-group" if args.text_source != "label" else "category-label",
                "model": f"CLIP {mode}",
                "split": args.split,
                "text_source": args.text_source,
            }
        )
        rows.append(row)

    avg = {
        "setting": rows[0]["setting"],
        "model": f"CLIP {mode}",
        "split": args.split,
        "text_source": args.text_source,
        "direction": "average",
    }
    for metric in ["R@1", "R@5", "R@10", "mAP", "MedianRank", "num_queries", "num_candidates", "avg_positives"]:
        avg[metric] = sum(float(row[metric]) for row in rows) / len(rows)
    rows.append(avg)

    metadata = {
        "dataset_root": str(args.dataset_root),
        "dataset_size": args.dataset_size,
        "split": args.split,
        "text_source": args.text_source,
        "group_mode": group_mode,
        "model_name": args.model_name,
        "do_finetune": args.do_finetune,
        "epochs": args.epochs if args.do_finetune else 0,
        "batch_size": args.batch_size,
        "max_samples": args.max_samples,
    }
    write_results(args.output_dir, rows, metadata)


if __name__ == "__main__":
    main()
