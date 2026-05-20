
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set


CLASS_PROMPTS = {
    "Exterior urban spaces with people": "a social media photo of an activated exterior urban commercial space with visible pedestrians or people",
    "Exterior urban spaces without people": "a social media photo of an exterior urban commercial space without visible people, emphasizing street, plaza, facade, or landscape design",
    "Food or drink items": "a social media photo centered on food, drinks, dessert, or dining items",
    "Hotel or commercial lodging spaces": "a social media photo of a hotel room or commercial lodging space",
    "Human-centered portrait": "a social media portrait or group photo where people are the main subject",
    "Interior urban spaces with people": "a social media photo of an activated interior commercial space with visible shoppers, visitors, or workers",
    "Interior urban spaces without people": "a social media photo of an interior commercial space without visible people",
    "Other non-spatial content": "a social media image that is a screenshot, poster, advertisement, graphic, meme, or other non-spatial content",
    "Private home interiors": "a social media photo of a private home interior",
    "Retail products and merchandise": "a social media photo centered on retail products, merchandise, fashion, cosmetics, or displays",
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
    p = argparse.ArgumentParser(description="Task2 multi-positive retrieval for CLIP/BLIP/BLIP2")
    p.add_argument("--dataset-root", type=Path, default=Path(r"D:\data2"))
    p.add_argument("--dataset-size", default="10K Dataset")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--train-split", default="train", choices=["train", "val", "test"])
    p.add_argument("--text-source", default="label", choices=["label", "post", "label_plus_post"])
    p.add_argument("--group-mode", default="auto", choices=["auto", "post_id", "post_text", "label"])

    p.add_argument("--model-family", default="clip", choices=["clip", "blip", "blip2"])
    p.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    p.add_argument("--output-dir", type=Path, default=Path("runs_task2_multipositive_vlm"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--do-finetune", action="store_true")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--save-model", action="store_true")
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--temperature", type=float, default=0.07)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
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
    return re.sub(r"\s+", " ", clean_cell(text).lower())[:max_len]


def read_split_table(dataset_dir: Path, split: str):
    import pandas as pd
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
        candidates += [class_dir / raw.name, image_root / raw.name]
    else:
        for suf in IMAGE_SUFFIXES:
            candidates += [class_dir / f"{image_filename}{suf}", image_root / f"{image_filename}{suf}"]

    for c in candidates:
        if c.exists():
            return str(c)

    target_stems = {raw.stem if raw.suffix else image_filename}
    for root in [class_dir, image_root]:
        if root.exists():
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
    out: List[TextCandidate] = []
    for item in items:
        if item.group_key not in seen:
            seen.add(item.group_key)
            out.append(TextCandidate(item.group_key, item.text))
    return out


def positives_for_text_to_image(items: Sequence[PairItem], texts: Sequence[TextCandidate]) -> List[Set[int]]:
    group_to_images: Dict[str, Set[int]] = {}
    for idx, item in enumerate(items):
        group_to_images.setdefault(item.group_key, set()).add(idx)
    return [group_to_images.get(t.group_key, set()) for t in texts]


def positives_for_image_to_text(items: Sequence[PairItem], texts: Sequence[TextCandidate]) -> List[Set[int]]:
    group_to_texts: Dict[str, Set[int]] = {}
    for idx, text in enumerate(texts):
        group_to_texts.setdefault(text.group_key, set()).add(idx)
    return [group_to_texts.get(item.group_key, set()) for item in items]


def batched(seq: Sequence, batch_size: int):
    for s in range(0, len(seq), batch_size):
        yield seq[s:s + batch_size]


class VLMWrapper:
    def __init__(self, model_family: str, model_name: str, device_arg: str, freeze_backbone: bool = False):
        import torch
        self.model_family = model_family.lower()
        self.model_name = model_name
        self.device = "cuda" if device_arg == "auto" and torch.cuda.is_available() else ("cpu" if device_arg == "auto" else device_arg)

        if self.model_family == "clip":
            from transformers import CLIPModel, CLIPProcessor
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        elif self.model_family == "blip":
            from transformers import BlipModel, BlipProcessor
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipModel.from_pretrained(model_name).to(self.device)
        elif self.model_family == "blip2":
            from transformers import Blip2Model, Blip2Processor
            self.processor = Blip2Processor.from_pretrained(model_name)
            self.model = Blip2Model.from_pretrained(model_name).to(self.device)
        else:
            raise ValueError(f"Unsupported model_family: {model_family}")

        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

    def _masked_mean(self, hidden, attention_mask):
        if attention_mask is None:
            return hidden.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def encode_images_tensor(self, images, grad: bool = False):
        import torch
        import torch.nn.functional as F
        self.model.train(grad)
        with torch.set_grad_enabled(grad):
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)

            if self.model_family == "clip":
                feats = self.model.get_image_features(**inputs)

            elif self.model_family == "blip":
                if hasattr(self.model, "get_image_features"):
                    out = self.model.get_image_features(**inputs)
                    feats = out if isinstance(out, torch.Tensor) else getattr(out, "image_embeds", None)
                    if feats is None:
                        feats = out.last_hidden_state[:, 0, :]
                else:
                    out = self.model.vision_model(pixel_values=inputs["pixel_values"], return_dict=True)
                    feats = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0, :]
                    if hasattr(self.model, "visual_projection"):
                        feats = self.model.visual_projection(feats)

            elif self.model_family == "blip2":
                if hasattr(self.model, "get_image_features"):
                    out = self.model.get_image_features(pixel_values=inputs["pixel_values"])
                    feats = out if isinstance(out, torch.Tensor) else getattr(out, "image_embeds", None)
                    if feats is None and hasattr(out, "last_hidden_state"):
                        feats = out.last_hidden_state
                else:
                    out = self.model.vision_model(pixel_values=inputs["pixel_values"], return_dict=True)
                    feats = out.last_hidden_state.mean(dim=1)
                if feats.dim() == 3:
                    feats = feats.mean(dim=1)

            return F.normalize(feats.float(), dim=-1)

    def encode_texts_tensor(self, texts: Sequence[str], grad: bool = False):
        import torch
        import torch.nn.functional as F
        self.model.train(grad)
        with torch.set_grad_enabled(grad):
            inputs = self.processor(text=list(texts), padding=True, truncation=True, return_tensors="pt").to(self.device)

            if self.model_family == "clip":
                feats = self.model.get_text_features(**inputs)

            elif self.model_family == "blip":
                if hasattr(self.model, "get_text_features"):
                    out = self.model.get_text_features(**inputs)
                    feats = out if isinstance(out, torch.Tensor) else getattr(out, "text_embeds", None)
                    if feats is None:
                        feats = out.last_hidden_state[:, 0, :]
                else:
                    out = self.model.text_encoder(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"), return_dict=True)
                    feats = out.last_hidden_state[:, 0, :]
                    if hasattr(self.model, "text_projection"):
                        feats = self.model.text_projection(feats)

            elif self.model_family == "blip2":
                if hasattr(self.model, "get_text_features"):
                    out = self.model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
                    feats = out if isinstance(out, torch.Tensor) else getattr(out, "text_embeds", None)
                    if feats is None and hasattr(out, "last_hidden_state"):
                        feats = out.last_hidden_state
                elif hasattr(self.model, "qformer"):
                    out = self.model.qformer(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"), return_dict=True)
                    feats = out.last_hidden_state
                elif hasattr(self.model, "language_model"):
                    out = self.model.language_model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    feats = out.hidden_states[-1]
                else:
                    raise RuntimeError("Cannot find usable BLIP-2 text encoder.")
                if feats.dim() == 3:
                    feats = self._masked_mean(feats, inputs.get("attention_mask"))

            return F.normalize(feats.float(), dim=-1)


def encode_images(wrapper: VLMWrapper, image_paths: Sequence[str], batch_size: int):
    import torch
    from PIL import Image
    feats = []
    wrapper.model.eval()
    with torch.no_grad():
        for paths in batched(image_paths, batch_size):
            images = [Image.open(p).convert("RGB") for p in paths]
            feats.append(wrapper.encode_images_tensor(images, grad=False).cpu())
    return torch.cat(feats, dim=0)


def encode_texts(wrapper: VLMWrapper, texts: Sequence[str], batch_size: int):
    import torch
    feats = []
    wrapper.model.eval()
    with torch.no_grad():
        for batch_text in batched(list(texts), batch_size):
            feats.append(wrapper.encode_texts_tensor(list(batch_text), grad=False).cpu())
    return torch.cat(feats, dim=0)


class PairDataset:
    def __init__(self, items: Sequence[PairItem]):
        self.items = list(items)
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return self.items[idx]


def contrastive_loss(image_feats, text_feats, temperature: float = 0.07):
    import torch
    import torch.nn.functional as F
    logits = image_feats @ text_feats.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def finetune_vlm(wrapper: VLMWrapper, train_items: Sequence[PairItem], batch_size: int, epochs: int, lr: float, weight_decay: float, amp: bool, temperature: float):
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader
    from transformers.optimization import get_cosine_schedule_with_warmup

    def collate(batch: Sequence[PairItem]):
        images = [Image.open(item.image_path).convert("RGB") for item in batch]
        texts = [item.text for item in batch]
        return images, texts

    loader = DataLoader(PairDataset(train_items), batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate, drop_last=True)
    params = [p for p in wrapper.model.parameters() if p.requires_grad]
    if not params:
        print("[WARN] No trainable parameters. Unfreezing all parameters.")
        for p in wrapper.model.parameters():
            p.requires_grad = True
        params = list(wrapper.model.parameters())

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    total_steps = max(1, epochs * len(loader))
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=max(1, total_steps // 20), num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and str(wrapper.device).startswith("cuda"))

    print(f"[INFO] trainable params: {sum(p.numel() for p in params):,}")
    print(f"[INFO] batches/epoch: {len(loader)}")

    for ep in range(epochs):
        wrapper.model.train()
        running = 0.0
        for step, (images, texts) in enumerate(loader, 1):
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp and str(wrapper.device).startswith("cuda")):
                img = wrapper.encode_images_tensor(images, grad=True)
                txt = wrapper.encode_texts_tensor(texts, grad=True)
                if img.shape[-1] != txt.shape[-1]:
                    raise RuntimeError(f"Image/text dim mismatch: {img.shape[-1]} vs {txt.shape[-1]}. Use zero-shot or add projection head.")
                loss = contrastive_loss(img, txt, temperature)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += float(loss.detach().cpu())
            if step % 20 == 0 or step == len(loader):
                print(f"epoch={ep+1}/{epochs} step={step}/{len(loader)} loss={running/step:.4f}")


def average_precision(ranked: Sequence[int], positives: Set[int]) -> float:
    if not positives:
        return math.nan
    hits, total = 0, 0.0
    for rank, idx in enumerate(ranked, 1):
        if idx in positives:
            hits += 1
            total += hits / rank
            if hits == len(positives):
                break
    return total / len(positives)


def evaluate_similarity(similarity, positives: Sequence[Set[int]], direction: str) -> Dict[str, float]:
    import numpy as np
    sim_np = similarity.numpy() if hasattr(similarity, "numpy") else similarity
    recalls = {1: [], 5: [], 10: []}
    aps, first_ranks = [], []
    valid = 0
    for q in range(sim_np.shape[0]):
        pos = positives[q]
        if not pos:
            continue
        valid += 1
        ranked = np.argsort(-sim_np[q]).tolist()
        for k in recalls:
            recalls[k].append(float(any(idx in pos for idx in ranked[:k])))
        aps.append(average_precision(ranked, pos))
        first_ranks.append(min((r + 1 for r, idx in enumerate(ranked) if idx in pos), default=math.nan))

    if valid == 0:
        raise RuntimeError(f"No valid positives for {direction}")

    return {
        "direction": direction,
        "R@1": 100.0 * float(np.mean(recalls[1])),
        "R@5": 100.0 * float(np.mean(recalls[5])),
        "R@10": 100.0 * float(np.mean(recalls[10])),
        "mAP": 100.0 * float(np.nanmean(aps)),
        "MedianRank": float(np.nanmedian(first_ranks)),
        "num_queries": int(valid),
        "num_candidates": int(sim_np.shape[1]),
        "avg_positives": float(np.mean([len(p) for p in positives if p])),
    }


def write_results(output_dir: Path, rows: List[Dict[str, float]], metadata: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "task2_multipositive_results.csv"
    fields = ["setting", "model", "split", "text_source", "direction", "R@1", "R@5", "R@10", "mAP", "MedianRank", "num_queries", "num_candidates", "avg_positives"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Wrote results to {csv_path}")


def main() -> None:
    import torch
    args = parse_args()
    set_seed(args.seed)

    dataset_dir = args.dataset_root / args.dataset_size
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    group_mode = args.group_mode
    if group_mode == "auto" and args.text_source == "label":
        group_mode = "label"

    print(f"Loading model family: {args.model_family}")
    print(f"Loading model name  : {args.model_name}")
    wrapper = VLMWrapper(args.model_family, args.model_name, args.device, args.freeze_backbone)
    print(f"Using device: {wrapper.device}")

    if args.do_finetune:
        train_items = build_items(dataset_dir, args.train_split, args.text_source, group_mode, args.max_samples)
        print(f"Fine-tuning on {len(train_items)} row-level image-text pairs")
        finetune_vlm(wrapper, train_items, args.batch_size, args.epochs, args.lr, args.weight_decay, args.amp, args.temperature)
        if args.save_model:
            ckpt_dir = args.output_dir / f"{args.model_family}_checkpoint"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            wrapper.model.save_pretrained(ckpt_dir)
            wrapper.processor.save_pretrained(ckpt_dir)
            print(f"Saved checkpoint to {ckpt_dir}")

    eval_items = build_items(dataset_dir, args.split, args.text_source, group_mode, args.max_samples)
    text_candidates = build_text_candidates(eval_items)
    print(f"Evaluation images: {len(eval_items)}")
    print(f"Evaluation text groups: {len(text_candidates)}")

    image_features = encode_images(wrapper, [item.image_path for item in eval_items], args.batch_size)
    text_features = encode_texts(wrapper, [tc.text for tc in text_candidates], args.batch_size)

    if image_features.shape[-1] != text_features.shape[-1]:
        raise RuntimeError(f"Image/text feature dimension mismatch during evaluation: {image_features.shape[-1]} vs {text_features.shape[-1]}")

    sim_t2i = text_features @ image_features.T
    t2i = evaluate_similarity(sim_t2i, positives_for_text_to_image(eval_items, text_candidates), "text-to-image")

    sim_i2t = image_features @ text_features.T
    i2t = evaluate_similarity(sim_i2t, positives_for_image_to_text(eval_items, text_candidates), "image-to-text")

    mode = "fine-tuned" if args.do_finetune else "zero-shot"
    model_label = f"{args.model_family.upper()} {mode}"
    rows = []
    for row in [t2i, i2t]:
        row = dict(row)
        row.update({
            "setting": "post-group" if args.text_source != "label" else "category-label",
            "model": model_label,
            "split": args.split,
            "text_source": args.text_source,
        })
        rows.append(row)

    avg = {
        "setting": rows[0]["setting"],
        "model": model_label,
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
        "model_family": args.model_family,
        "model_name": args.model_name,
        "do_finetune": args.do_finetune,
        "epochs": args.epochs if args.do_finetune else 0,
        "batch_size": args.batch_size,
        "max_samples": args.max_samples,
        "freeze_backbone": args.freeze_backbone,
    }
    write_results(args.output_dir, rows, metadata)


if __name__ == "__main__":
    main()
