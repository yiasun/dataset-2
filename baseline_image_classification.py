import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import timm
except ImportError:
    timm = None

try:
    import open_clip
except ImportError:
    open_clip = None


# =========================
# Utils
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================
# CLIP Dataset Wrapper
# =========================
class CLIPImageFolder(Dataset):
    def __init__(self, root: str, preprocess):
        self.base = datasets.ImageFolder(root=root)
        self.samples = self.base.samples
        self.classes = self.base.classes
        self.class_to_idx = self.base.class_to_idx
        self.preprocess = preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.preprocess(image)
        return image, label


# =========================
# Models
# =========================
class CLIPFineTuneClassifier(nn.Module):
    def __init__(self, clip_model_name: str, pretrained_tag: str, num_classes: int, train_full_clip: bool = False):
        super().__init__()
        if open_clip is None:
            raise ImportError("需要安装 open_clip_torch：pip install open_clip_torch")

        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name=clip_model_name,
            pretrained=pretrained_tag,
        )
        self.feature_dim = self.clip_model.visual.output_dim
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        if not train_full_clip:
            for p in self.clip_model.parameters():
                p.requires_grad = False

    def forward(self, x):
        features = self.clip_model.encode_image(x)
        features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        logits = self.classifier(features)
        return logits


# =========================
# Runner
# =========================
class BaselineRunner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Device: {self.device}")

        self.data_dir = Path(args.data_dir)
        self.output_dir = Path(args.output_dir)
        ensure_dir(self.output_dir)

        self.train_dir = self.data_dir / "train"
        self.val_dir = self.data_dir / "val"
        self.test_dir = self.data_dir / "test"

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_names = None
        self.num_classes = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.best_model_path = self.output_dir / f"best_{self.args.model_name}.pth"

        self.setup_data()
        self.setup_model_and_optim()

    # -------------------------
    # Data
    # -------------------------
    def build_standard_transforms(self):
        train_tf = transforms.Compose([
            transforms.Resize((self.args.image_size, self.args.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        eval_tf = transforms.Compose([
            transforms.Resize((self.args.image_size, self.args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return train_tf, eval_tf

    def setup_data(self):
        model_name = self.args.model_name.lower()

        if model_name in ["clip_zero_shot", "clip_finetune"]:
            if open_clip is None:
                raise ImportError("需要安装 open_clip_torch：pip install open_clip_torch")
            _, train_clip_tf, eval_clip_tf = open_clip.create_model_and_transforms(
                model_name=self.args.clip_model_name,
                pretrained=self.args.clip_pretrained,
            )
            self.train_dataset = CLIPImageFolder(str(self.train_dir), preprocess=train_clip_tf)
            self.val_dataset = CLIPImageFolder(str(self.val_dir), preprocess=eval_clip_tf)
            self.test_dataset = CLIPImageFolder(str(self.test_dir), preprocess=eval_clip_tf)
        else:
            train_tf, eval_tf = self.build_standard_transforms()
            self.train_dataset = datasets.ImageFolder(str(self.train_dir), transform=train_tf)
            self.val_dataset = datasets.ImageFolder(str(self.val_dir), transform=eval_tf)
            self.test_dataset = datasets.ImageFolder(str(self.test_dir), transform=eval_tf)

        self.class_names = self.train_dataset.classes
        assert self.class_names == self.val_dataset.classes == self.test_dataset.classes, "train/val/test 类别顺序不一致"
        self.num_classes = len(self.class_names)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

        print(f"[INFO] Num classes: {self.num_classes}")
        print(f"[INFO] Classes: {self.class_names}")
        print(f"[INFO] Train samples: {len(self.train_dataset)}")
        print(f"[INFO] Val samples: {len(self.val_dataset)}")
        print(f"[INFO] Test samples: {len(self.test_dataset)}")

        save_json(self.train_dataset.class_to_idx, self.output_dir / "class_to_idx.json")

    # -------------------------
    # Model
    # -------------------------
    def build_model(self):
        name = self.args.model_name.lower()
        pretrained = self.args.pretrained

        if name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            return model

        if name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            return model

        if name == "resnet152":
            weights = models.ResNet152_Weights.DEFAULT if pretrained else None
            model = models.resnet152(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            return model

        if name == "efficientnet_b4":
            weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b4(weights=weights)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
            return model

        if name == "vit_b_16":
            weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
            model = models.vit_b_16(weights=weights)
            model.heads.head = nn.Linear(model.heads.head.in_features, self.num_classes)
            return model

        if name == "deit_b":
            if timm is None:
                raise ImportError("DeiT-B 需要安装 timm：pip install timm")
            model = timm.create_model("deit_base_patch16_224", pretrained=pretrained, num_classes=self.num_classes)
            return model

        if name == "clip_finetune":
            model = CLIPFineTuneClassifier(
                clip_model_name=self.args.clip_model_name,
                pretrained_tag=self.args.clip_pretrained,
                num_classes=self.num_classes,
                train_full_clip=self.args.train_full_clip,
            )
            return model

        raise ValueError(f"Unsupported model_name: {name}")

    def setup_model_and_optim(self):
        if self.args.model_name.lower() == "clip_zero_shot":
            return

        self.model = self.build_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs)

    # -------------------------
    # Eval / Save
    # -------------------------
    @torch.no_grad()
    def evaluate(self, loader, split="Val") -> Dict:
        self.model.eval()
        running_loss = 0.0
        all_labels = []
        all_preds = []

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = self.criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)

            running_loss += loss.item() * images.size(0)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

        avg_loss = running_loss / len(loader.dataset)
        top1_acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        report = classification_report(
            all_labels,
            all_preds,
            target_names=self.class_names,
            digits=4,
            zero_division=0,
            output_dict=True,
        )
        cm = confusion_matrix(all_labels, all_preds)

        print(f"[{split}] Loss: {avg_loss:.4f} | Top-1 Acc: {top1_acc:.4f} | Macro F1: {macro_f1:.4f}")

        return {
            "loss": avg_loss,
            "top1_acc": top1_acc,
            "macro_f1": macro_f1,
            "report": report,
            "confusion_matrix": cm,
            "labels": all_labels,
            "preds": all_preds,
        }

    def save_reports(self, metrics: Dict):
        model_name = self.args.model_name.lower()

        summary_df = pd.DataFrame([
            {
                "model": model_name,
                "dataset": self.data_dir.name,
                "top1_acc": metrics["top1_acc"],
                "macro_f1": metrics["macro_f1"],
                "num_classes": self.num_classes,
                "test_samples": len(self.test_dataset),
            }
        ])
        summary_df.to_csv(self.output_dir / f"summary_{model_name}.csv", index=False, encoding="utf-8-sig")

        rows = []
        for cls_name in self.class_names:
            rows.append({
                "class_name": cls_name,
                "precision": metrics["report"][cls_name]["precision"],
                "recall": metrics["report"][cls_name]["recall"],
                "f1_score": metrics["report"][cls_name]["f1-score"],
                "support": metrics["report"][cls_name]["support"],
            })
        pd.DataFrame(rows).to_csv(self.output_dir / f"per_class_metrics_{model_name}.csv", index=False, encoding="utf-8-sig")

        cm_df = pd.DataFrame(metrics["confusion_matrix"], index=self.class_names, columns=self.class_names)
        cm_df.to_csv(self.output_dir / f"confusion_matrix_{model_name}.csv", encoding="utf-8-sig")

        fig_size = max(10, int(self.num_classes * 0.8))
        plt.figure(figsize=(fig_size, fig_size))
        plt.imshow(metrics["confusion_matrix"], interpolation="nearest")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.colorbar()
        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, self.class_names, rotation=90)
        plt.yticks(tick_marks, self.class_names)
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(self.output_dir / f"confusion_matrix_{model_name}.png", dpi=300, bbox_inches="tight")
        plt.close()

        json_ready = {
            "model": model_name,
            "dataset": self.data_dir.name,
            "top1_acc": metrics["top1_acc"],
            "macro_f1": metrics["macro_f1"],
            "report": metrics["report"],
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
        }
        save_json(json_ready, self.output_dir / f"test_metrics_{model_name}.json")

    # -------------------------
    # Train / Test
    # -------------------------
    def train_one_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for images, labels in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        loss = running_loss / total
        acc = correct / total
        print(f"[Train] Epoch {epoch:03d} | Loss: {loss:.4f} | Acc: {acc:.4f}")
        return loss, acc

    def fit(self):
        history = []
        best_val_acc = -1.0

        for epoch in range(1, self.args.epochs + 1):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader, split="Val")
            self.scheduler.step()

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_metrics["loss"],
                "val_top1_acc": val_metrics["top1_acc"],
                "val_macro_f1": val_metrics["macro_f1"],
            })

            if val_metrics["top1_acc"] > best_val_acc:
                best_val_acc = val_metrics["top1_acc"]
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "class_names": self.class_names,
                        "model_name": self.args.model_name,
                        "num_classes": self.num_classes,
                        "image_size": self.args.image_size,
                    },
                    self.best_model_path,
                )
                print(f"[INFO] Best model saved to: {self.best_model_path}")

        save_json(history, self.output_dir / f"history_{self.args.model_name.lower()}.json")

    def test(self):
        ckpt = torch.load(self.best_model_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        print(f"[INFO] Loaded best model from: {self.best_model_path}")
        test_metrics = self.evaluate(self.test_loader, split="Test")
        self.save_reports(test_metrics)
        return test_metrics

    # -------------------------
    # CLIP Zero-shot
    # -------------------------
    @torch.no_grad()
    def run_clip_zero_shot(self):
        if open_clip is None:
            raise ImportError("需要安装 open_clip_torch：pip install open_clip_torch")

        device = self.device
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=self.args.clip_model_name,
            pretrained=self.args.clip_pretrained,
        )
        model = model.to(device)
        model.eval()

        test_dataset = CLIPImageFolder(str(self.test_dir), preprocess=preprocess)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

        prompts = [self.args.clip_prompt_template.format(label=c) for c in test_dataset.classes]
        tokenizer = open_clip.get_tokenizer(self.args.clip_model_name)
        text_tokens = tokenizer(prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        all_labels = []
        all_preds = []

        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            logits = 100.0 * image_features @ text_features.T
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

        top1_acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        report = classification_report(
            all_labels,
            all_preds,
            target_names=test_dataset.classes,
            digits=4,
            zero_division=0,
            output_dict=True,
        )
        cm = confusion_matrix(all_labels, all_preds)

        print(f"[CLIP Zero-shot][Test] Top-1 Acc: {top1_acc:.4f} | Macro F1: {macro_f1:.4f}")

        metrics = {
            "top1_acc": top1_acc,
            "macro_f1": macro_f1,
            "report": report,
            "confusion_matrix": cm,
        }
        self.save_reports(metrics)
        return metrics

    def run(self):
        if self.args.model_name.lower() == "clip_zero_shot":
            return self.run_clip_zero_shot()
        self.fit()
        return self.test()


# =========================
# Aggregate helper
# =========================
def merge_summaries(base_output_dir: str, save_name: str = "T1_results_merged.csv"):
    base_path = Path(base_output_dir)
    rows = []
    for csv_file in base_path.rglob("summary_*.csv"):
        try:
            df = pd.read_csv(csv_file)
            rows.append(df)
        except Exception as e:
            print(f"[WARN] Skip {csv_file}: {e}")
    if rows:
        merged = pd.concat(rows, ignore_index=True)
        merged.to_csv(base_path / save_name, index=False, encoding="utf-8-sig")
        print(f"[INFO] Merged summary saved to: {base_path / save_name}")
    else:
        print("[WARN] No summary files found.")


# =========================
# Main
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="数据根目录，必须有 train/val/test")
    parser.add_argument("--output_dir", type=str, default="./outputs_baseline")
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet18",
        choices=[
            "resnet18",
            "resnet50",
            "resnet152",
            "vit_b_16",
            "deit_b",
            "efficientnet_b4",
            "clip_zero_shot",
            "clip_finetune",
        ],
    )
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true")

    # CLIP specific
    parser.add_argument("--clip_model_name", type=str, default="ViT-B-16")
    parser.add_argument("--clip_pretrained", type=str, default="openai")
    parser.add_argument("--clip_prompt_template", type=str, default="a photo of {label}")
    parser.add_argument("--train_full_clip", action="store_true", help="CLIP fine-tune 时是否连视觉主干一起训练")

    # helper
    parser.add_argument("--merge_summaries_only", action="store_true")
    parser.add_argument("--merge_base_dir", type=str, default="")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.merge_summaries_only:
        if not args.merge_base_dir:
            raise ValueError("使用 --merge_summaries_only 时，必须提供 --merge_base_dir")
        merge_summaries(args.merge_base_dir)
        return

    runner = BaselineRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
