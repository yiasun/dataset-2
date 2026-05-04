import argparse
import os
import time
import json
import random
import numpy as np
import cv2

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

from segment_anything import sam_model_registry
from tqdm import tqdm


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def xywh_to_xyxy(box):
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def ann_to_mask(ann, h, w):
    if isinstance(ann["segmentation"], list):
        rles = mask_utils.frPyObjects(
            ann["segmentation"], h, w
        )
        rle = mask_utils.merge(rles)
    elif isinstance(ann["segmentation"]["counts"], list):
        rle = mask_utils.frPyObjects(
            ann["segmentation"], h, w
        )
    else:
        rle = ann["segmentation"]

    mask = mask_utils.decode(rle)
    if len(mask.shape) == 3:
        mask = np.any(mask, axis=2)
    return mask.astype(np.uint8)


class SAMDataset(Dataset):
    def __init__(self, ann_path, img_root):
        self.coco = COCO(ann_path)
        self.img_root = img_root

        self.samples = []
        for ann in self.coco.dataset["annotations"]:
            img_info = self.coco.imgs[ann["image_id"]]
            self.samples.append((img_info, ann))

        print(f"[INFO] total train samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_info, ann = self.samples[idx]

        img_path = os.path.join(
            self.img_root,
            img_info["file_name"]
        )

        image = imread_unicode(img_path)
        if image is None:
            raise RuntimeError(f"failed read: {img_path}")

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        h, w = image.shape[:2]

        gt_mask = ann_to_mask(ann, h, w)
        box = xywh_to_xyxy(ann["bbox"])

        return {
            "image": image,
            "mask": gt_mask,
            "box": box,
            "image_id": ann["image_id"]
        }


def collate_fn(batch):
    return batch


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-ann",
        required=True
    )
    parser.add_argument(
        "--img-root",
        required=True
    )
    parser.add_argument(
        "--sam-checkpoint",
        required=True
    )
    parser.add_argument(
        "--model-type",
        default="vit_b",
        choices=["vit_b", "vit_l", "vit_h"]
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )
    parser.add_argument(
        "--save-dir",
        required=True
    )

    args = parser.parse_args()

    os.makedirs(
        args.save_dir,
        exist_ok=True
    )

    seed_everything()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device = {device}")

    dataset = SAMDataset(
        args.train_ann,
        args.img_root
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    sam = sam_model_registry[
        args.model_type
    ](
        checkpoint=args.sam_checkpoint
    )

    sam.to(device)
    sam.train()

    # freeze image encoder
    for p in sam.image_encoder.parameters():
        p.requires_grad = False

    # freeze prompt encoder
    for p in sam.prompt_encoder.parameters():
        p.requires_grad = False

    # only train mask decoder
    optimizer = torch.optim.AdamW(
        sam.mask_decoder.parameters(),
        lr=args.lr
    )

    best_loss = 1e9

    for epoch in range(args.epochs):
        epoch_loss = 0.0

        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch+1}/{args.epochs}"
        )

        for batch in pbar:
            for sample in batch:
                image = sample["image"]
                gt_mask = sample["mask"]
                box = sample["box"]

                image_tensor = torch.as_tensor(
                    image,
                    device=device
                ).permute(2, 0, 1).float()

                image_tensor = sam.preprocess(
                    image_tensor
                ).unsqueeze(0)

                with torch.no_grad():
                    image_embedding = sam.image_encoder(
                        image_tensor
                    )

                    input_box = torch.tensor(
                        box,
                        device=device
                    ).unsqueeze(0)

                    sparse_embeddings, dense_embeddings = \
                        sam.prompt_encoder(
                            points=None,
                            boxes=input_box,
                            masks=None
                        )

                low_res_masks, iou_predictions = \
                    sam.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False
                    )

                pred_mask = F.interpolate(
                    low_res_masks,
                    size=gt_mask.shape,
                    mode="bilinear",
                    align_corners=False
                )

                gt_tensor = torch.tensor(
                    gt_mask,
                    device=device
                ).float().unsqueeze(0).unsqueeze(0)

                loss = F.binary_cross_entropy_with_logits(
                    pred_mask,
                    gt_tensor
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.set_postfix(
                    loss=f"{loss_val:.4f}"
                )

        avg_loss = epoch_loss / len(dataset)

        print(
            f"[Train] Epoch {epoch+1} "
            f"| Avg Loss: {avg_loss:.6f}"
        )

        save_path = os.path.join(
            args.save_dir,
            f"epoch_{epoch+1}.pth"
        )

        torch.save(
            sam.state_dict(),
            save_path
        )

        print(
            f"[INFO] saved -> {save_path}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(
                args.save_dir,
                "best.pth"
            )

            torch.save(
                sam.state_dict(),
                best_path
            )

            print(
                f"[INFO] best saved -> {best_path}"
            )


if __name__ == "__main__":
    main()