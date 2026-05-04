# T3 Instance Segmentation Baselines

这套代码面向你的 T3 实例分割任务，默认假设数据是 **COCO instance segmentation** 格式：
- `train.json`
- `val.json`
- `test.json`
- 图片目录下能通过 `images[].file_name` 找到对应图片

包含以下 baseline：
- Cascade Mask R-CNN
- SOLOv2
- DETR-based：Mask2Former
- SAM zero-shot
- SAM fine-tune（轻量版：基于 box prompt 训练 mask decoder）

## 1. 建议目录

```text
project_root/
├─ data/
│  └─ t3/
│     ├─ train.json
│     ├─ val.json
│     ├─ test.json
│     └─ images/
│        ├─ train/
│        ├─ val/
│        └─ test/
└─ t3_baselines/
```

如果你的 `file_name` 已经是相对路径，例如：
- `train/0001.jpg`
- `val/0123.jpg`

那么 `--img-root data/t3/images` 即可。

## 2. 环境建议

### 方案 A：MMDetection（推荐）
用于：
- Cascade Mask R-CNN
- SOLOv2
- Mask2Former

安装示例：

```bash
pip install -U openmim
mim install "mmengine>=0.10.0"
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.3.0"
pip install pycocotools pandas openpyxl tqdm pillow pyyaml
```

### 方案 B：SAM
用于：
- SAM zero-shot
- SAM fine-tune

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install pycocotools opencv-python pandas openpyxl tqdm pillow
```

## 3. 先检查标注是否可用

```bash
python tools/check_coco_instance.py \
  --ann data/t3/train.json \
  --img-root data/t3/images
```

## 4. 运行基线

### 4.1 Cascade Mask R-CNN
```bash
python run_t3.py \
  --model cascade_mask_rcnn \
  --data-root data/t3 \
  --img-root data/t3/images \
  --work-dir outputs/cascade_mask_rcnn \
  --epochs 12 \
  --batch-size 2 \
  --num-classes 80
```

### 4.2 SOLOv2
```bash
python run_t3.py \
  --model solov2 \
  --data-root data/t3 \
  --img-root data/t3/images \
  --work-dir outputs/solov2 \
  --epochs 12 \
  --batch-size 2 \
  --num-classes 80
```

### 4.3 Mask2Former
```bash
python run_t3.py \
  --model mask2former \
  --data-root data/t3 \
  --img-root data/t3/images \
  --work-dir outputs/mask2former \
  --epochs 12 \
  --batch-size 2 \
  --num-classes 80
```

### 4.4 SAM zero-shot
```bash
python models/sam_zero_shot.py \
  --ann data/t3/val.json \
  --img-root data/t3/images \
  --sam-checkpoint sam_vit_h_4b8939.pth \
  --model-type vit_h \
  --out-json outputs/sam_zeroshot/preds_val.json \
  --device cuda
```

### 4.5 SAM fine-tune
```bash
python models/sam_ft.py \
  --train-ann data/t3/train.json \
  --val-ann data/t3/val.json \
  --img-root data/t3/images \
  --sam-checkpoint sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --save-dir outputs/sam_ft \
  --epochs 10 \
  --batch-size 1 \
  --lr 1e-4 \
  --device cuda
```

## 5. 评估

对任意 COCO 预测结果 json：

```bash
python tools/eval_instance_seg.py \
  --gt data/t3/val.json \
  --pred outputs/sam_zeroshot/preds_val.json \
  --img-root data/t3/images \
  --out-csv outputs/sam_zeroshot/metrics.csv
```

输出：
- AP
- AP50
- AP75
- Per-class AP
- mIoU
- FPS（如果 prediction json 里有 `_fps` 字段，或测试日志里可额外记录）

## 6. 三任务总表

把每个模型的 `metrics.csv` 整理好后：

```bash
python tools/make_summary_tables.py \
  --t3-dir outputs \
  --out-xlsx outputs/T3_results_and_summary.xlsx
```

目前这个脚本先聚合 T3；你后面可以再把 T1/T2 csv 一起并进去。

## 7. 重要说明

1. 这套代码默认你的 `train/val/test.json` 是 **实例分割**，不是只有 bbox 的检测标注。
2. MMDetection 三个基线是通过**动态生成配置文件**来跑的，不需要你手改官方 config。
3. SAM zero-shot 这里采用的是 **GT box prompt + SAM 出 mask** 的严格可评估版本。这样你能稳定得到 AP / AP50 / AP75 / mIoU。这个更像 *prompted zero-shot segmentation*。
4. SAM fine-tune 是轻量化版本，训练目标是让 SAM 在给定 GT box prompt 的情况下更好地产生实例 mask；它不是完整的大规模官方训练 recipe，但足够作为项目 baseline。
5. 如果你想要“真正自动类别预测的 SAM zero-shot”，那需要额外接 detector 或 CLIP 分类器，那是下一版工作。
