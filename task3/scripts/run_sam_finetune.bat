@echo off
setlocal

REM Edit these paths for the local machine.
set PREP_ROOT=E:\Project Data\ChatGPT-Urban ImageNet\t3_1k_class_agnostic
set OUT_ROOT=E:\Project Data\ChatGPT-Urban ImageNet\t3_rerun_outputs
set SAM_CHECKPOINT=E:\checkpoints\sam_vit_b_01ec64.pth

set EPOCHS=3
set BATCH_SIZE=1
set LR=0.0001
set SAM_FT_DIR=%OUT_ROOT%\sam_ft_class_agnostic

cd /d "%~dp0\.."

python models\sam_ft.py ^
  --train-ann "%PREP_ROOT%\annotations\train.json" ^
  --img-root "%PREP_ROOT%\images" ^
  --sam-checkpoint "%SAM_CHECKPOINT%" ^
  --model-type vit_b ^
  --epochs %EPOCHS% ^
  --batch-size %BATCH_SIZE% ^
  --lr %LR% ^
  --save-dir "%SAM_FT_DIR%"

python models\sam_ft_infer.py ^
  --ann "%PREP_ROOT%\annotations\test.json" ^
  --img-root "%PREP_ROOT%\images" ^
  --checkpoint "%SAM_FT_DIR%\best.pth" ^
  --model-type vit_b ^
  --out-json "%SAM_FT_DIR%\sam_ft_predictions.json" ^
  --device cuda

python tools\eval_sam_predictions.py ^
  --ann "%PREP_ROOT%\annotations\test.json" ^
  --pred "%SAM_FT_DIR%\sam_ft_predictions.json" ^
  --model-name "SAM fine-tuned class-agnostic" ^
  --out-dir "%OUT_ROOT%\t3_metrics\sam_ft_class_agnostic"

endlocal
