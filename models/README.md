---
library_name: pytorch
pipeline_tag: image-classification
tags:
  - urban-imagenet
  - husic
  - image-classification
  - resnet
datasets:
  - Urban-ImageNet
license: cc-by-nc-sa-4.0
---

# Urban-ImageNet Task 1 Scene Classification Checkpoint

This folder contains inference utilities for the released **Urban-ImageNet Task 1: Urban Scene Semantic Classification** checkpoint. The model predicts the 10 HUSIC scene categories used by Urban-ImageNet.

## Files

| File | Description |
|---|---|
| `Task 1_Urban Scene Classification.pth` | PyTorch checkpoint for the Task 1 scene classifier, tracked with Git LFS in the GitHub repository. |
| `Task 1_Class_to_Index.xlsx` | Mapping between HUSIC class names and integer label indices. |
| `Task 1_Prediction.py` | Standalone inference script for image files or folders. |
| `requirements.txt` | Minimal Python dependencies for inference. |

## Model

The checkpoint uses a ResNet-style image classifier with a 10-way HUSIC output head. It corresponds to the high-performing Task 1 baseline reported in the Urban-ImageNet paper and is intended for lightweight scene-level filtering and routing of urban social-media imagery.

## HUSIC Classes

| Index | Class |
|---:|---|
| 0 | Exterior urban spaces with people |
| 1 | Exterior urban spaces without people |
| 2 | Food or drink items |
| 3 | Hotel or commercial lodging spaces |
| 4 | Human-centered portrait |
| 5 | Interior urban spaces with people |
| 6 | Interior urban spaces without people |
| 7 | Other non-spatial content |
| 8 | Private home interiors |
| 9 | Retail products and merchandise |

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

When cloning the GitHub repository, install Git LFS first if you want the checkpoint to be downloaded automatically.

Run prediction on a single image:

```bash
python "Task 1_Prediction.py" \
  --input /path/to/image.jpg \
  --output predictions.csv
```

Run prediction on a folder:

```bash
python "Task 1_Prediction.py" \
  --input /path/to/images \
  --output predictions.csv \
  --batch-size 32 \
  --recursive
```

Use explicit checkpoint and class mapping paths:

```bash
python "Task 1_Prediction.py" \
  --input /path/to/images \
  --checkpoint "Task 1_Urban Scene Classification.pth" \
  --class-map "Task 1_Class_to_Index.xlsx" \
  --output predictions.xlsx
```

The output includes the predicted class index, HUSIC class name, confidence score, and top-k predictions.

## Task 2 and Task 3 Checkpoints

This initial model release focuses on Task 1 because it is compact, directly reusable, and provides strong scene-level performance. For Task 2 retrieval and Task 3 instance segmentation, the repository currently releases the training and evaluation code rather than packaged checkpoints. These tasks involve architecture-specific retrieval models, MMDetection/SAM dependencies, and larger experiment artifacts. We plan to add additional checkpoints once the inference APIs and model cards are packaged cleanly.

## Citation

```bibtex
@article{ou2026urbanimagenet,
  title   = {Urban-ImageNet: A Large-Scale Multi-Modal Dataset and Evaluation Framework for Urban Space Perception},
  author  = {Ou, Yiwei and Cheung, Chung Ching and Ang, Jun Yang and Ren, Xiaobin and Sun, Ronggui and Gao, Guansong and Zhao, Kaiqi and Manfredini, Manfredo},
  journal = {arXiv preprint arXiv:2605.09936},
  year    = {2026},
  eprint  = {2605.09936},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url     = {https://arxiv.org/abs/2605.09936}
}
```
