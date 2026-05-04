import argparse
import json
import os
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from segment_anything import sam_model_registry
from tqdm import tqdm


def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def xywh_to_xyxy(box):
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def encode_binary_mask(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", required=True)
    parser.add_argument("--img-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-type", default="vit_b")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    coco = COCO(args.ann)

    sam = sam_model_registry[args.model_type](checkpoint=None)
    state_dict = torch.load(args.checkpoint, map_location=args.device)
    sam.load_state_dict(state_dict)
    sam.to(args.device)
    sam.eval()

    anns_by_img = defaultdict(list)
    for ann in coco.dataset["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    results = []
    total_time = 0.0
    total_instances = 0

    for img_id in tqdm(coco.getImgIds(), desc="SAM FT inference fixed"):
        img_info = coco.loadImgs([img_id])[0]
        img_path = os.path.join(args.img_root, img_info["file_name"])

        image = imread_unicode(img_path)
        if image is None:
            print(f"[WARN] failed read: {img_path}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]

        image_tensor = torch.as_tensor(image, device=args.device).permute(2, 0, 1).float()
        input_image = sam.preprocess(image_tensor)
        input_size = tuple(image_tensor.shape[-2:])
        input_image = input_image.unsqueeze(0)

        start_img = time.perf_counter()
        with torch.no_grad():
            image_embedding = sam.image_encoder(input_image)
        total_time += time.perf_counter() - start_img

        for ann in anns_by_img.get(img_id, []):
            box = torch.tensor(xywh_to_xyxy(ann["bbox"]), device=args.device).unsqueeze(0)

            start = time.perf_counter()
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                    points=None,
                    boxes=box,
                    masks=None,
                )

                low_res_masks, _ = sam.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                upscaled_masks = sam.postprocess_masks(
                    low_res_masks,
                    input_size=input_size,
                    original_size=original_size,
                )

                pred_mask = (torch.sigmoid(upscaled_masks)[0, 0].cpu().numpy() > 0.5).astype(np.uint8)

            total_time += time.perf_counter() - start
            total_instances += 1

            rle = encode_binary_mask(pred_mask)
            area = float(mask_utils.area(rle))
            x, y, w, h = mask_utils.toBbox(rle).tolist()

            results.append({
                "image_id": int(img_id),
                "category_id": int(ann["category_id"]),
                "segmentation": rle,
                "score": 1.0,
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": area,
            })

    fps = total_instances / total_time if total_time > 0 else 0.0

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f)

    print(f"[DONE] saved -> {args.out_json}")
    print(f"[INFO] total predictions: {len(results)}")
    print(f"[INFO] Approx instance FPS: {fps:.4f}")


if __name__ == "__main__":
    main()