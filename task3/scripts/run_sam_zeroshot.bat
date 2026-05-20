@echo off
setlocal

REM Edit these paths for the local machine.
set PREP_ROOT=E:\Project Data\ChatGPT-Urban ImageNet\t3_1k_class_agnostic
set OUT_ROOT=E:\Project Data\ChatGPT-Urban ImageNet\t3_rerun_outputs
set SAM_CHECKPOINT=E:\checkpoints\sam_vit_b_01ec64.pth

cd /d "%~dp0\.."

python models\sam_zero_shot.py ^
  --ann "%PREP_ROOT%\annotations\test.json" ^
  --img-root "%PREP_ROOT%\images" ^
  --sam-checkpoint "%SAM_CHECKPOINT%" ^
  --model-type vit_b ^
  --out-json "%OUT_ROOT%\sam_zeroshot_class_agnostic\sam_predictions.json" ^
  --device cuda

python tools\eval_sam_predictions.py ^
  --ann "%PREP_ROOT%\annotations\test.json" ^
  --pred "%OUT_ROOT%\sam_zeroshot_class_agnostic\sam_predictions.json" ^
  --model-name "SAM zero-shot class-agnostic" ^
  --out-dir "%OUT_ROOT%\t3_metrics\sam_zeroshot_class_agnostic"

endlocal
