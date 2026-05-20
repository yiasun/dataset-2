@echo off
setlocal

REM Edit these paths for the local machine.
set PREP_ROOT=E:\Project Data\ChatGPT-Urban ImageNet\t3_1k_class_agnostic
set OUT_ROOT=E:\Project Data\ChatGPT-Urban ImageNet\t3_rerun_outputs

REM Optional but recommended: set this to a COCO pretrained Mask2Former checkpoint.
REM Example: set PRETRAIN=E:\checkpoints\mask2former_r50_8xb2-lsj-50e_coco.pth
set PRETRAIN=

set MODEL=mask2former
set EPOCHS=6
set BATCH_SIZE=1
set LR=0.0001
set WORK_DIR=%OUT_ROOT%\%MODEL%_class_agnostic

cd /d "%~dp0\.."

if "%PRETRAIN%"=="" (
  set LOAD_FROM_ARGS=
) else (
  set LOAD_FROM_ARGS=--load-from "%PRETRAIN%"
)

python run_t3.py ^
  --model %MODEL% ^
  --data-root "%PREP_ROOT%\annotations" ^
  --img-root "%PREP_ROOT%\images" ^
  --work-dir "%WORK_DIR%" ^
  --epochs %EPOCHS% ^
  --batch-size %BATCH_SIZE% ^
  --auto-classes ^
  --lr %LR% ^
  --warmup-iters 50 ^
  --num-workers 2 ^
  %LOAD_FROM_ARGS%

python tools\eval_t3_metrics.py ^
  --config "%WORK_DIR%\%MODEL%_auto.py" ^
  --checkpoint "%WORK_DIR%\epoch_%EPOCHS%.pth" ^
  --ann "%PREP_ROOT%\annotations\test.json" ^
  --img-root "%PREP_ROOT%\images" ^
  --model-name "Mask2Former class-agnostic" ^
  --out-dir "%OUT_ROOT%\t3_metrics\mask2former_class_agnostic" ^
  --score-thr 0.001

endlocal
