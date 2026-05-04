# T3 Instance Segmentation Baselines

This repository contains baseline code for the T3 instance segmentation task.

Supported models:

* Cascade Mask R-CNN
* SOLOv2
* Mask2Former
* SAM zero-shot
* SAM fine-tuned

Evaluation metrics:

* AP
* AP50
* AP75
* Per-class AP
* mIoU
* FPS

---

## 1. Dataset Structure

Expected dataset structure:

```text
data2/
├── 10K Dataset_Labelled/
├── 10K Dataset_Segmentation Labelled/
│   ├── train.json
│   ├── val.json
│   └── test.json
└── t3_baselines/
```

The annotation files must follow COCO instance segmentation format.

---

## 2. Environment Setup

```bash
conda create -n t3_mmdet python=3.10 -y
conda activate t3_mmdet
```

```bash
pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html
pip install mmdet==3.0.0 mmengine==0.10.7
pip install numpy==1.24.4 pandas openpyxl pycocotools opencv-python tqdm matplotlib
```

---

## 3. Prepare Images

```bash
python tools/flatten_images.py
```

```bash
python tools/check_coco_instance.py --ann "D:/data2/10K Dataset_Segmentation Labelled/train.json" --img-root "D:/data2/t3_flat_images"
```

---

## 4. Training Baselines

### Cascade Mask R-CNN

```bash
python run_t3.py --model cascade_mask_rcnn --data-root "D:/data2/10K Dataset_Segmentation Labelled" --img-root "D:/data2/t3_flat_images" --work-dir "outputs/cascade_mask_rcnn" --epochs 12 --batch-size 1 --num-classes 10
```

### SOLOv2

```bash
python run_t3.py --model solov2 --data-root "D:/data2/10K Dataset_Segmentation Labelled" --img-root "D:/data2/t3_flat_images" --work-dir "outputs/solov2" --epochs 12 --batch-size 1 --num-classes 10
```

### Mask2Former

```bash
python run_t3.py --model mask2former --data-root "D:/data2/10K Dataset_Segmentation Labelled" --img-root "D:/data2/t3_flat_images" --work-dir "outputs/mask2former" --epochs 4 --batch-size 1 --num-classes 10
```

### SAM Zero-shot

```bash
python models/sam_zero_shot.py --ann "D:/data2/10K Dataset_Segmentation Labelled/val.json" --img-root "D:/data2/t3_flat_images" --sam-checkpoint "D:/data2/checkpoints/sam_vit_b_01ec64.pth" --model-type vit_b --out-json "outputs/t3_metrics/sam_zeroshot/sam_predictions.json"
```

### SAM Fine-tuned

```bash
python models/sam_ft.py --train-ann "D:/data2/10K Dataset_Segmentation Labelled/train.json" --img-root "D:/data2/t3_flat_images" --sam-checkpoint "D:/data2/checkpoints/sam_vit_b_01ec64.pth" --model-type vit_b --epochs 5 --batch-size 1 --save-dir "outputs/sam_ft"
```

---

## 5. Evaluation

```bash
python tools/eval_t3_metrics.py ...
python tools/eval_sam_predictions.py ...
```

Outputs:

```text
outputs/t3_metrics/<model_name>/
├── t3_main_metrics.csv
├── t3_per_class_metrics.csv
└── t3_metrics.xlsx
```

Main metrics include:

* AP
* AP50
* AP75
* mIoU
* FPS

---

## 6. Recommended Settings

| Model              | Epochs | Batch Size |
| ------------------ | -----: | ---------: |
| Cascade Mask R-CNN |     12 |          1 |
| SOLOv2             |     12 |          1 |
| Mask2Former        |      4 |        1–2 |
| SAM zero-shot      |      0 |        N/A |
| SAM fine-tuned     |      5 |          1 |
