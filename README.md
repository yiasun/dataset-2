# T2 Benchmark Library — Cross-Modal Retrieval Baselines

## Overview

This module implements **Task 2 (T2): Cross-Modal Retrieval Baselines** for text-image retrieval in fine-grained street-level multimodal visual place recognition.

The benchmark supports both zero-shot and fine-tuning settings for strong vision-language models.

### Supported Models

* CLIP
* BLIP
* BLIP-2
* LLaVA-1.5

Each model supports:

* zero-shot evaluation
* fine-tuning (where applicable)

---

## Evaluation Metrics

### Text-to-Image Retrieval (T2I)

* Recall@1
* Recall@5
* Recall@10
* mAP
* Median Rank

### Image-to-Text Retrieval (I2T)

* Recall@1
* Recall@5
* Recall@10
* mAP
* Median Rank

---

## Dataset Structure

Recommended structure:

```text
10K Dataset_Text-Image Pairs/
├── train_pairs.json
├── val_pairs.json
├── test_pairs.json
├── train.xlsx
├── val.xlsx
└── test.xlsx
```

### JSON Format Example

```json
{
  "image": "path/to/image.jpg",
  "text": "A photo of Exterior urban spaces with people. Example post text here.",
  "label": "Exterior urban spaces with people",
  "file_name": "xxx.jpg"
}
```

Important:

The benchmark actually uses:

* `image`
* `text`

The fields below are auxiliary:

* `label`
* `file_name`

---

## Environment Setup

Recommended environment:

* Python 3.10
* PyTorch 2.x
* CUDA 12.x

Install dependencies:

```bash
pip install torch torchvision transformers timm
pip install sentencepiece protobuf accelerate
pip install scikit-learn pandas openpyxl tqdm pillow
```

For LLaVA:

```bash
pip install transformers==4.40.0
```

For BLIP-2 / large models, server GPU is strongly recommended.

---

## Running Baselines

### Example: CLIP Zero-shot

```bash
python t2_benchmark_v01.py \
  --train_json "./10K Dataset_Text-Image Pairs/train_pairs.json" \
  --val_json "./10K Dataset_Text-Image Pairs/val_pairs.json" \
  --test_json "./10K Dataset_Text-Image Pairs/test_pairs.json" \
  --model_name clip \
  --mode zero_shot \
  --batch_size 32 \
  --output_dir "./outputs_t2"
```

### Example: BLIP-2 Fine-tuning

```bash
python t2_benchmark_v01.py \
  --train_json "./10K Dataset_Text-Image Pairs/train_pairs.json" \
  --val_json "./10K Dataset_Text-Image Pairs/val_pairs.json" \
  --test_json "./10K Dataset_Text-Image Pairs/test_pairs.json" \
  --model_name blip2 \
  --mode ft \
  --batch_size 4 \
  --epochs 3 \
  --lr 1e-4 \
  --output_dir "./outputs_t2"
```

### Example: LLaVA Zero-shot

```bash
python t2_benchmark_v01.py \
  --train_json "./10K Dataset_Text-Image Pairs/train_pairs.json" \
  --val_json "./10K Dataset_Text-Image Pairs/val_pairs.json" \
  --test_json "./10K Dataset_Text-Image Pairs/test_pairs.json" \
  --model_name llava \
  --mode zero_shot \
  --batch_size 1 \
  --output_dir "./outputs_t2"
```

---

## Output Structure

```text
outputs_t2/
├── summary.csv
├── test_results.json
├── history.json
├── best.pt
└── T2_results_merged.csv
```

---

## Important Notes

### LLaVA

LLaVA often requires:

* use_fast=False
* protobuf
* sentencepiece
* manual local model download

Recommended deployment:

* RTX 4090 server
* Linux environment

### BLIP-2

BLIP-2 is extremely slow on local GPUs.

Recommended:

* server-side execution
* batch_size ≤ 4

### Path Issues

If JSON contains Windows paths like:

```text
D:/data2/xxx.jpg
```

they must be converted before running on Linux servers.

---

## Reproducibility

Recommended:

* random seed = 42
* same train/val/test split across all models
* unified evaluation protocol

---

## Citation

If you use this benchmark library in academic work, please cite the corresponding thesis/project repository.
