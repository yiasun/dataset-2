import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from transformers import (
        CLIPModel,
        CLIPProcessor,
        BlipForImageTextRetrieval,
        BlipProcessor,
        Blip2ForConditionalGeneration,
        Blip2Processor,
        LlavaForConditionalGeneration,
        LlavaProcessor,
    )
except Exception:
    CLIPModel = None
    CLIPProcessor = None
    BlipForImageTextRetrieval = None
    BlipProcessor = None
    Blip2ForConditionalGeneration = None
    Blip2Processor = None
    LlavaForConditionalGeneration = None
    LlavaProcessor = None


# ============================================================
# Utils
# ============================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=1e-6)


def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Image path resolver
# 1) direct match
# 2) append suffix
# 3) recursive search in subfolders
# 4) stem-based fuzzy match
# ============================================================
def resolve_image_file(image_root: Path, image_file_raw: str) -> str:
    image_file = str(image_file_raw).strip()
    if not image_file:
        raise ValueError("Empty image filename")

    suffixes = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

    # 1) direct
    candidate = image_root / image_file
    if candidate.exists():
        return candidate.as_posix()

    # 2) append suffix if missing
    lower_name = image_file.lower()
    has_suffix = any(lower_name.endswith(s) for s in suffixes)
    if not has_suffix:
        for suf in suffixes:
            candidate = image_root / f"{image_file}{suf}"
            if candidate.exists():
                return candidate.as_posix()

    # 3) recursive exact-name search
    target_names = [image_file]
    if not has_suffix:
        target_names.extend([f"{image_file}{suf}" for suf in suffixes])

    for name in target_names:
        matches = list(image_root.rglob(name))
        if matches:
            return matches[0].as_posix()

    # 4) recursive stem-based search
    stem = Path(image_file).stem
    for p in image_root.rglob("*"):
        if p.is_file() and p.stem == stem:
            return p.as_posix()

    raise FileNotFoundError(f"Image not found under {image_root}: {image_file_raw}")


# ============================================================
# Excel -> retrieval json
# Expected columns:
# Image Filename, Image Label, Post Text
# ============================================================
def build_pairs_from_excel(
    input_xlsx: str,
    image_root: str,
    output_json: str,
    text_mode: str = "label_plus_post",
) -> None:
    df = pd.read_excel(input_xlsx)

    required_cols = ["Image Filename", "Image Label", "Post Text"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    pairs = []
    missing = []
    image_root = Path(image_root)

    for idx, row in df.iterrows():
        image_file_raw = str(row["Image Filename"]).strip()
        label = str(row["Image Label"]).strip()
        post_text = str(row["Post Text"]).strip()

        try:
            image_path = resolve_image_file(image_root, image_file_raw)
        except Exception as e:
            missing.append(
                {
                    "row": int(idx),
                    "image_filename": image_file_raw,
                    "error": str(e),
                }
            )
            continue

        if text_mode == "post_only":
            text = post_text
        elif text_mode == "label_only":
            text = label
        else:
            text = f"A photo of {label}. {post_text}"

        pairs.append(
            {
                "id": int(idx),
                "image": image_path,
                "text": text,
                "label": label,
                "file_name": Path(image_path).name,
            }
        )

    out = {
        "info": {
            "source_xlsx": input_xlsx,
            "image_root": str(image_root),
            "text_mode": text_mode,
            "num_pairs": len(pairs),
            "num_missing": len(missing),
        },
        "data": pairs,
        "missing": missing,
    }
    save_json(out, output_json)
    print(f"[OK] saved {len(pairs)} pairs -> {output_json}")

    if missing:
        miss_path = str(Path(output_json).with_name(Path(output_json).stem + "_missing.json"))
        save_json(missing, miss_path)
        print(f"[WARN] skipped {len(missing)} missing images -> {miss_path}")


# ============================================================
# Retrieval dataset
# ============================================================
@dataclass
class PairSample:
    image_path: str
    text: str
    pair_id: str


class RetrievalDataset(Dataset):
    def __init__(self, json_path: str):
        raw = load_json(json_path)
        data = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
        self.samples: List[PairSample] = []
        for i, item in enumerate(data):
            self.samples.append(
                PairSample(
                    image_path=str(item["image"]),
                    text=str(item["text"]),
                    pair_id=str(item.get("id", i)),
                )
            )

        if len(self.samples) == 0:
            raise ValueError(f"No valid pairs found in {json_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        image = Image.open(s.image_path).convert("RGB")
        return {
            "image": image,
            "text": s.text,
            "pair_id": s.pair_id,
            "image_path": s.image_path,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "images": [b["image"] for b in batch],
        "texts": [b["text"] for b in batch],
        "pair_ids": [b["pair_id"] for b in batch],
        "image_paths": [b["image_path"] for b in batch],
    }


# ============================================================
# Model components
# ============================================================
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaseRetrievalModel(nn.Module):
    def encode_image(self, images: List[Image.Image]) -> torch.Tensor:
        raise NotImplementedError

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        raise NotImplementedError

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


class CLIPRetrievalModel(BaseRetrievalModel):
    def __init__(self, model_name: str, device: torch.device, ft: bool, embed_dim: int, train_backbone: bool):
        super().__init__()
        if CLIPModel is None or CLIPProcessor is None:
            raise ImportError("Please install transformers")
        self.device = device
        self.ft = ft
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.backbone = CLIPModel.from_pretrained(model_name).to(device)

        feat_dim = self.backbone.visual_projection.out_features
        self.image_proj = ProjectionHead(feat_dim, embed_dim)
        self.text_proj = ProjectionHead(feat_dim, embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), dtype=torch.float32))

        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.to(device)

    def encode_image(self, images: List[Image.Image]) -> torch.Tensor:
        with torch.set_grad_enabled(self.ft):
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            feats = self.backbone.get_image_features(**inputs)
        feats = self.image_proj(feats)
        return l2_normalize(feats)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        with torch.set_grad_enabled(self.ft):
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            feats = self.backbone.get_text_features(**inputs)
        feats = self.text_proj(feats)
        return l2_normalize(feats)


class BLIPRetrievalModel(BaseRetrievalModel):
    def __init__(self, model_name: str, device: torch.device, ft: bool, embed_dim: int, train_backbone: bool):
        super().__init__()
        if BlipForImageTextRetrieval is None or BlipProcessor is None:
            raise ImportError("Please install transformers")
        self.device = device
        self.ft = ft
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.backbone = BlipForImageTextRetrieval.from_pretrained(model_name).to(device)

        feat_dim = self.backbone.vision_proj.out_features
        self.image_proj = ProjectionHead(feat_dim, embed_dim)
        self.text_proj = ProjectionHead(feat_dim, embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), dtype=torch.float32))

        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.to(device)

    def encode_image(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.set_grad_enabled(self.ft):
            vision_outputs = self.backbone.vision_model(pixel_values=inputs["pixel_values"], return_dict=True)
            cls_feat = vision_outputs.last_hidden_state[:, 0, :]
            feats = self.backbone.vision_proj(cls_feat)
        feats = self.image_proj(feats)
        return l2_normalize(feats)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.set_grad_enabled(self.ft):
            text_outputs = self.backbone.text_encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict=True,
            )
            cls_feat = text_outputs.last_hidden_state[:, 0, :]
            feats = self.backbone.text_proj(cls_feat)

        feats = self.text_proj(feats)
        return l2_normalize(feats)


class BLIP2RetrievalModel(BaseRetrievalModel):
    def __init__(self, model_name: str, device: torch.device, ft: bool, embed_dim: int, train_backbone: bool):
        super().__init__()
        if Blip2ForConditionalGeneration is None or Blip2Processor is None:
            raise ImportError("Please install transformers")
        self.device = device
        self.ft = ft
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.backbone = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype).to(device)

        q_dim = self.backbone.qformer.config.hidden_size
        lm_dim = self.backbone.language_model.config.hidden_size
        self.image_proj = ProjectionHead(q_dim, embed_dim)
        self.text_proj = ProjectionHead(lm_dim, embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), dtype=torch.float32))

        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.to(device)

    def encode_image(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.set_grad_enabled(self.ft):
            vision_outputs = self.backbone.vision_model(pixel_values=inputs["pixel_values"], return_dict=True)
            image_embeds = vision_outputs.last_hidden_state
            image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=self.device)
            query_tokens = self.backbone.query_tokens.expand(image_embeds.shape[0], -1, -1)
            qformer_outputs = self.backbone.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                return_dict=True,
            )
            feats = qformer_outputs.last_hidden_state.mean(dim=1)
        feats = self.image_proj(feats.float())
        return l2_normalize(feats)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.set_grad_enabled(self.ft):
            outputs = self.backbone.language_model.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict=True,
            )
            feats = outputs.last_hidden_state[:, 0, :]
        feats = self.text_proj(feats.float())
        return l2_normalize(feats)


class LLaVARetrievalModel(BaseRetrievalModel):
    def __init__(self, model_name: str, device: torch.device, ft: bool, embed_dim: int, train_backbone: bool):
        super().__init__()
        if LlavaForConditionalGeneration is None or LlavaProcessor is None:
            raise ImportError("Please install transformers")
        self.device = device
        self.ft = ft
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.processor = LlavaProcessor.from_pretrained(model_name)
        self.backbone = LlavaForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype).to(device)

        hidden_dim = self.backbone.config.text_config.hidden_size
        self.image_proj = ProjectionHead(hidden_dim, embed_dim)
        self.text_proj = ProjectionHead(hidden_dim, embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), dtype=torch.float32))

        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.to(device)

    def encode_image(self, images: List[Image.Image]) -> torch.Tensor:
        prompts = ["<image> Describe this image briefly."] * len(images)
        inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True).to(self.device)
        with torch.set_grad_enabled(self.ft):
            outputs = self.backbone(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                output_hidden_states=True,
                return_dict=True,
            )
            feats = outputs.hidden_states[-1][:, 0, :]
        feats = self.image_proj(feats.float())
        return l2_normalize(feats)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.set_grad_enabled(self.ft):
            outputs = self.backbone(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
            feats = outputs.hidden_states[-1][:, 0, :]

        feats = self.text_proj(feats.float())
        return l2_normalize(feats)


# ============================================================
# Factory
# ============================================================
def build_model(model_name: str, device: torch.device, mode: str, embed_dim: int, train_backbone: bool) -> BaseRetrievalModel:
    ft = mode == "ft"
    key = model_name.lower()
    if key == "clip":
        return CLIPRetrievalModel("openai/clip-vit-base-patch32", device, ft, embed_dim, train_backbone)
    if key == "blip":
        return BLIPRetrievalModel("Salesforce/blip-itm-base-coco", device, ft, embed_dim, train_backbone)
    if key == "blip2":
        return BLIP2RetrievalModel("Salesforce/blip2-opt-2.7b", device, ft, embed_dim, train_backbone)
    if key == "llava":
        return LLaVARetrievalModel("llava-hf/llava-1.5-7b-hf", device, ft, embed_dim, train_backbone)
    raise ValueError(f"Unsupported model_name: {model_name}")


# ============================================================
# Loss / metrics
# ============================================================
def contrastive_loss(image_embeds: torch.Tensor, text_embeds: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    scale = logit_scale.exp().clamp(max=100)
    logits_i2t = scale * (image_embeds @ text_embeds.t())
    logits_t2i = logits_i2t.t()
    target = torch.arange(image_embeds.size(0), device=image_embeds.device)
    loss_i = F.cross_entropy(logits_i2t, target)
    loss_t = F.cross_entropy(logits_t2i, target)
    return 0.5 * (loss_i + loss_t)


@torch.no_grad()
def encode_split(model: BaseRetrievalModel, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
    model.eval()
    img_feats_all = []
    txt_feats_all = []
    pair_ids = []
    image_paths = []

    for batch in tqdm(loader, desc="Encoding", leave=False):
        img_feats = model.encode_image(batch["images"])
        txt_feats = model.encode_text(batch["texts"])
        img_feats_all.append(img_feats.cpu())
        txt_feats_all.append(txt_feats.cpu())
        pair_ids.extend(batch["pair_ids"])
        image_paths.extend(batch["image_paths"])

    return torch.cat(img_feats_all, dim=0), torch.cat(txt_feats_all, dim=0), pair_ids, image_paths


def retrieval_metrics(sim: np.ndarray) -> Dict[str, float]:
    n = sim.shape[0]
    gt = np.arange(n)
    ranks = []
    ap_scores = []

    for i in range(n):
        order = np.argsort(-sim[i])
        rank = int(np.where(order == gt[i])[0][0]) + 1
        ranks.append(rank)
        ap_scores.append(1.0 / rank)

    ranks = np.array(ranks)
    return {
        "R@1": float(np.mean(ranks <= 1)),
        "R@5": float(np.mean(ranks <= 5)),
        "R@10": float(np.mean(ranks <= 10)),
        "mAP": float(np.mean(ap_scores)),
        "MedianRank": float(np.median(ranks)),
    }


# ============================================================
# Benchmark Lib v0.1
# ============================================================
class T2BenchmarkLibV01:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Device: {self.device}")

        self.train_ds = RetrievalDataset(args.train_json)
        self.val_ds = RetrievalDataset(args.val_json)
        self.test_ds = RetrievalDataset(args.test_json)

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
        self.test_loader = DataLoader(
            self.test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )

        self.model = build_model(
            model_name=args.model_name,
            device=self.device,
            mode=args.mode,
            embed_dim=args.embed_dim,
            train_backbone=args.train_backbone,
        )

        self.output_dir = Path(args.output_dir) / f"{args.model_name}_{args.mode}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_pt = self.output_dir / "best.pt"
        self.result_json = self.output_dir / "test_results.json"
        self.history_json = self.output_dir / "history.json"
        self.summary_csv = self.output_dir / "summary.csv"

        if args.mode == "ft":
            self.optimizer = torch.optim.AdamW(
                self.model.trainable_parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        else:
            self.optimizer = None

    def save_ckpt(self):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_name": self.args.model_name,
                "mode": self.args.mode,
                "embed_dim": self.args.embed_dim,
            },
            self.best_pt,
        )
        print(f"[INFO] saved checkpoint -> {self.best_pt}")

    def load_ckpt(self):
        ckpt = torch.load(self.best_pt, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"[INFO] loaded checkpoint <- {self.best_pt}")

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        losses = []

        prog = tqdm(self.train_loader, desc=f"Train {epoch:03d}", leave=False)
        for batch in prog:
            image_embeds = self.model.encode_image(batch["images"])
            text_embeds = self.model.encode_text(batch["texts"])
            loss = contrastive_loss(image_embeds, text_embeds, self.model.logit_scale)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            prog.set_postfix(loss=f"{loss.item():.4f}")

        mean_loss = float(np.mean(losses))
        print(f"[Train] Epoch {epoch:03d} | Loss: {mean_loss:.4f}")
        return mean_loss

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split: str) -> Dict[str, Any]:
        img_feats, txt_feats, pair_ids, image_paths = encode_split(self.model, loader)
        sim_t2i = (txt_feats @ img_feats.t()).numpy()
        sim_i2t = sim_t2i.T

        t2i = retrieval_metrics(sim_t2i)
        i2t = retrieval_metrics(sim_i2t)

        print(
            f"[{split}][T2I] R@1={t2i['R@1']:.4f} "
            f"R@5={t2i['R@5']:.4f} "
            f"R@10={t2i['R@10']:.4f} "
            f"mAP={t2i['mAP']:.4f} "
            f"MedR={t2i['MedianRank']:.1f}"
        )
        print(
            f"[{split}][I2T] R@1={i2t['R@1']:.4f} "
            f"R@5={i2t['R@5']:.4f} "
            f"R@10={i2t['R@10']:.4f} "
            f"mAP={i2t['mAP']:.4f} "
            f"MedR={i2t['MedianRank']:.1f}"
        )

        return {
            "split": split,
            "num_pairs": len(pair_ids),
            "t2i": t2i,
            "i2t": i2t,
            "pair_ids": pair_ids,
            "image_paths": image_paths,
        }

    def save_summary_csv(self, result: Dict[str, Any]):
        rows = [
            {
                "model": self.args.model_name,
                "mode": self.args.mode,
                "direction": "t2i",
                "R@1": result["t2i"]["R@1"],
                "R@5": result["t2i"]["R@5"],
                "R@10": result["t2i"]["R@10"],
                "mAP": result["t2i"]["mAP"],
                "MedianRank": result["t2i"]["MedianRank"],
            },
            {
                "model": self.args.model_name,
                "mode": self.args.mode,
                "direction": "i2t",
                "R@1": result["i2t"]["R@1"],
                "R@5": result["i2t"]["R@5"],
                "R@10": result["i2t"]["R@10"],
                "mAP": result["i2t"]["mAP"],
                "MedianRank": result["i2t"]["MedianRank"],
            },
        ]
        pd.DataFrame(rows).to_csv(self.summary_csv, index=False, encoding="utf-8-sig")
        print(f"[INFO] summary saved -> {self.summary_csv}")

    def run_zero_shot(self):
        _ = self.evaluate(self.val_loader, "Val")
        test_result = self.evaluate(self.test_loader, "Test")
        save_json({"test": test_result}, str(self.result_json))
        self.save_summary_csv(test_result)
        print(f"[INFO] result saved -> {self.result_json}")

    def run_finetune(self):
        history = []
        best_score = -1.0

        for epoch in range(1, self.args.epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_result = self.evaluate(self.val_loader, "Val")
            score = val_result["t2i"]["R@1"]

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_t2i": val_result["t2i"],
                    "val_i2t": val_result["i2t"],
                }
            )

            if score > best_score:
                best_score = score
                self.save_ckpt()

        save_json(history, str(self.history_json))
        self.load_ckpt()
        test_result = self.evaluate(self.test_loader, "Test")
        save_json({"history": history, "test": test_result}, str(self.result_json))
        self.save_summary_csv(test_result)
        print(f"[INFO] result saved -> {self.result_json}")

    def run(self):
        if self.args.mode == "zero_shot":
            self.run_zero_shot()
        else:
            self.run_finetune()


# ============================================================
# Summary merge
# ============================================================
def merge_summaries(base_dir: str, save_name: str = "T2_results_merged.csv"):
    base = Path(base_dir)
    dfs = []
    for csv_path in base.rglob("summary.csv"):
        try:
            dfs.append(pd.read_csv(csv_path))
        except Exception as e:
            print(f"[WARN] skip {csv_path}: {e}")

    if not dfs:
        print("[WARN] no summary.csv found")
        return

    merged = pd.concat(dfs, ignore_index=True)
    out_path = base / save_name
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] merged summary -> {out_path}")


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Lib v0.1 for T2 cross-modal retrieval")

    parser.add_argument("--build_pairs_from_excel", action="store_true")
    parser.add_argument("--input_xlsx", type=str, default="")
    parser.add_argument("--image_root", type=str, default="")
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument(
        "--text_mode",
        type=str,
        default="label_plus_post",
        choices=["label_plus_post", "post_only", "label_only"],
    )

    parser.add_argument("--merge_summaries_only", action="store_true")
    parser.add_argument("--merge_base_dir", type=str, default="")

    parser.add_argument("--train_json", type=str, default="")
    parser.add_argument("--val_json", type=str, default="")
    parser.add_argument("--test_json", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./outputs_t2")

    parser.add_argument("--model_name", type=str, default="clip", choices=["clip", "blip", "blip2", "llava"])
    parser.add_argument("--mode", type=str, default="zero_shot", choices=["zero_shot", "ft"])
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_backbone", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.build_pairs_from_excel:
        if not args.input_xlsx or not args.image_root or not args.output_json:
            raise ValueError(
                "When using --build_pairs_from_excel, you must provide "
                "--input_xlsx --image_root --output_json"
            )
        build_pairs_from_excel(args.input_xlsx, args.image_root, args.output_json, args.text_mode)
        return

    if args.merge_summaries_only:
        if not args.merge_base_dir:
            raise ValueError("When using --merge_summaries_only, you must provide --merge_base_dir")
        merge_summaries(args.merge_base_dir)
        return

    if not args.train_json or not args.val_json or not args.test_json:
        raise ValueError("You must provide --train_json --val_json --test_json")

    runner = T2BenchmarkLibV01(args)
    runner.run()


if __name__ == "__main__":
    main()
