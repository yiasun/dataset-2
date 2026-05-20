#!/usr/bin/env bash
set -euo pipefail

# Clean Task 3 reproduction script for Linux GPU environments.
#
# Required environment variables:
#   RAW_IMAGE_ROOT  Original Urban-ImageNet image root with train/val/test folders.
#   RAW_ANN_ROOT    Original Task 3 COCO annotation root with train.json/val.json/test.json.
#   PREP_ROOT       Output root for class-agnostic prepared data.
#   OUT_ROOT        Output root for model checkpoints and metrics.
#
# Optional:
#   SAM_CHECKPOINT  Path to sam_vit_b_01ec64.pth for SAM box refinement.
#   DEVICE          cuda or cpu. Defaults to cuda.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK3_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${TASK3_DIR}"

: "${RAW_IMAGE_ROOT:?Set RAW_IMAGE_ROOT to the original image root.}"
: "${RAW_ANN_ROOT:?Set RAW_ANN_ROOT to the original Task 3 annotation root.}"
: "${PREP_ROOT:?Set PREP_ROOT to the prepared dataset output root.}"
: "${OUT_ROOT:?Set OUT_ROOT to the experiment output root.}"

DEVICE="${DEVICE:-cuda}"
PREP_ANN="${PREP_ROOT}/annotations"
PREP_IMG="${PREP_ROOT}/images"

echo "[1/6] Preparing class-agnostic COCO data"
python tools/prepare_t3_dataset.py \
  --image-root "${RAW_IMAGE_ROOT}" \
  --ann-root "${RAW_ANN_ROOT}" \
  --out-root "${PREP_ROOT}" \
  --mode class_agnostic \
  --recompute-bbox \
  --overwrite-images

echo "[2/6] Checking prepared annotations"
python tools/check_coco_instance.py --ann "${PREP_ANN}/train.json" --img-root "${PREP_IMG}"
python tools/check_coco_instance.py --ann "${PREP_ANN}/test.json" --img-root "${PREP_IMG}"

echo "[3/6] Training Mask R-CNN"
MASK_DIR="${OUT_ROOT}/mask_rcnn_class_agnostic_clean"
python run_t3.py \
  --model mask_rcnn \
  --data-root "${PREP_ANN}" \
  --img-root "${PREP_IMG}" \
  --work-dir "${MASK_DIR}" \
  --epochs 8 \
  --batch-size 2 \
  --auto-classes \
  --lr 0.0002 \
  --warmup-iters 50 \
  --num-workers 2

echo "[4/6] Evaluating Mask R-CNN"
python tools/eval_t3_metrics.py \
  --config "${MASK_DIR}/mask_rcnn_auto.py" \
  --checkpoint "${MASK_DIR}/epoch_8.pth" \
  --ann "${PREP_ANN}/test.json" \
  --img-root "${PREP_IMG}" \
  --model-name "Mask R-CNN class-agnostic" \
  --out-dir "${OUT_ROOT}/t3_metrics/mask_rcnn_class_agnostic_clean" \
  --score-thr 0.001 \
  --device "${DEVICE}"

echo "[5/6] Training Cascade Mask R-CNN"
CASCADE_DIR="${OUT_ROOT}/cascade_mask_rcnn_class_agnostic_clean"
python run_t3.py \
  --model cascade_mask_rcnn \
  --data-root "${PREP_ANN}" \
  --img-root "${PREP_IMG}" \
  --work-dir "${CASCADE_DIR}" \
  --epochs 8 \
  --batch-size 2 \
  --auto-classes \
  --lr 0.0002 \
  --warmup-iters 50 \
  --num-workers 2

echo "[6/6] Evaluating Cascade Mask R-CNN"
python tools/eval_t3_metrics.py \
  --config "${CASCADE_DIR}/cascade_mask_rcnn_auto.py" \
  --checkpoint "${CASCADE_DIR}/epoch_8.pth" \
  --ann "${PREP_ANN}/test.json" \
  --img-root "${PREP_IMG}" \
  --model-name "Cascade Mask R-CNN class-agnostic" \
  --out-dir "${OUT_ROOT}/t3_metrics/cascade_mask_rcnn_class_agnostic_clean" \
  --score-thr 0.001 \
  --device "${DEVICE}"

if [[ -n "${SAM_CHECKPOINT:-}" ]]; then
  echo "[optional] Running SAM refinement from Mask R-CNN boxes"
  SAM_OUT="${OUT_ROOT}/sam_refine_mask_rcnn_boxes_class_agnostic"
  mkdir -p "${SAM_OUT}"
  python models/sam_refine_mmdet_boxes.py \
    --det-config "${MASK_DIR}/mask_rcnn_auto.py" \
    --det-checkpoint "${MASK_DIR}/epoch_8.pth" \
    --ann "${PREP_ANN}/test.json" \
    --img-root "${PREP_IMG}" \
    --sam-checkpoint "${SAM_CHECKPOINT}" \
    --model-type vit_b \
    --out-json "${SAM_OUT}/predictions.json" \
    --device "${DEVICE}" \
    --det-score-thr 0.05 \
    --max-per-img 100 \
    --score-mode det
  python tools/eval_sam_predictions.py \
    --ann "${PREP_ANN}/test.json" \
    --pred "${SAM_OUT}/predictions.json" \
    --model-name "SAM refined Mask R-CNN boxes class-agnostic" \
    --out-dir "${OUT_ROOT}/t3_metrics/sam_refine_mask_rcnn_boxes_class_agnostic"
fi

python tools/make_summary_tables.py \
  --metrics-root "${OUT_ROOT}/t3_metrics" \
  --out-csv "${OUT_ROOT}/t3_metrics_summary.csv" \
  --out-xlsx "${OUT_ROOT}/t3_metrics_summary.xlsx"

echo "[DONE] Task 3 clean pipeline complete."
