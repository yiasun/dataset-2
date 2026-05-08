@echo off
setlocal enabledelayedexpansion

REM ======================================================
REM T2 Multi-positive Retrieval Benchmark - BLIP / BLIP2
REM Expected structure:
REM D:\data2\
REM ├── 10K Dataset\
REM │   ├── 01 Images with labels
REM │   └── 02 Text-Image Pairs
REM ├── run_task2_multipositive_vlm.py
REM └── run_t2_blip_blip2_multipositive_10k.bat
REM ======================================================

cd /d D:\data2

set PY_FILE=run_task2_multipositive_vlm.py
set DATASET_ROOT=D:\data2
set DATASET_SIZE=10K Dataset
set OUT_BASE=outputs_t2_multipositive_10k_vlm

set SPLIT=test
set TRAIN_SPLIT=train
set DEVICE=cuda
set SEED=42

REM 0 = full dataset, 200 = quick debug
set MAX_SAMPLES=0

REM BLIP settings
set BLIP_MODEL=Salesforce/blip-itm-base-coco
set BLIP_BATCH_ZS=8
set BLIP_BATCH_FT=4

REM BLIP2 settings
set BLIP2_MODEL=Salesforce/blip2-opt-2.7b
set BLIP2_BATCH_ZS=1
set BLIP2_BATCH_FT=1

REM Fine-tune settings
set EPOCHS=5
set EPOCH=3
set LR=1e-5
set WEIGHT_DECAY=0.01

echo ==========================================
echo T2 Multi-positive BLIP / BLIP2 Benchmark
echo ==========================================
echo DATASET_ROOT=%DATASET_ROOT%
echo DATASET_SIZE=%DATASET_SIZE%
echo PY_FILE=%PY_FILE%
echo OUT_BASE=%OUT_BASE%
echo ==========================================

if not exist "%PY_FILE%" (
    echo [ERROR] Cannot find %PY_FILE%
    pause
    exit /b 1
)

if not exist "%DATASET_ROOT%\%DATASET_SIZE%" (
    echo [ERROR] Missing dataset folder:
    echo %DATASET_ROOT%\%DATASET_SIZE%
    echo.
    echo If your real folders are:
    echo   D:\data2\10K Dataset_Labelled
    echo   D:\data2\10K Dataset_Text-Image Pairs
    echo then create junctions first:
    echo   mkdir "D:\data2\10K Dataset"
    echo   cmd /c mklink /J "D:\data2\10K Dataset\01 Images with labels" "D:\data2\10K Dataset_Labelled"
    echo   cmd /c mklink /J "D:\data2\10K Dataset\02 Text-Image Pairs" "D:\data2\10K Dataset_Text-Image Pairs"
    pause
    exit /b 1
)

if not exist "%DATASET_ROOT%\%DATASET_SIZE%\01 Images with labels" (
    echo [ERROR] Missing image folder:
    echo %DATASET_ROOT%\%DATASET_SIZE%\01 Images with labels
    pause
    exit /b 1
)

if not exist "%DATASET_ROOT%\%DATASET_SIZE%\02 Text-Image Pairs\train.xlsx" (
    echo [ERROR] Missing train.xlsx:
    echo %DATASET_ROOT%\%DATASET_SIZE%\02 Text-Image Pairs\train.xlsx
    pause
    exit /b 1
)

mkdir "%OUT_BASE%" 2>nul

REM ======================================================
REM BLIP ZERO-SHOT
REM ======================================================

echo.
echo ==========================================
echo Step 1/6: BLIP zero-shot
echo ==========================================

echo [RUN] BLIP zero-shot text_source=label
python %PY_FILE% ^
  --dataset-root "%DATASET_ROOT%" ^
  --dataset-size "%DATASET_SIZE%" ^
  --split %SPLIT% ^
  --train-split %TRAIN_SPLIT% ^
  --text-source label ^
  --group-mode label ^
  --model-family blip ^
  --model-name "%BLIP_MODEL%" ^
  --output-dir "%OUT_BASE%\blip_label_zs" ^
  --batch-size %BLIP_BATCH_ZS% ^
  --device %DEVICE% ^
  --max-samples %MAX_SAMPLES% ^
  --seed %SEED%

echo [RUN] BLIP zero-shot text_source=post
python %PY_FILE% ^
  --dataset-root "%DATASET_ROOT%" ^
  --dataset-size "%DATASET_SIZE%" ^
  --split %SPLIT% ^
  --train-split %TRAIN_SPLIT% ^
  --text-source post ^
  --group-mode auto ^
  --model-family blip ^
  --model-name "%BLIP_MODEL%" ^
  --output-dir "%OUT_BASE%\blip_post_zs" ^
  --batch-size %BLIP_BATCH_ZS% ^
  --device %DEVICE% ^
  --max-samples %MAX_SAMPLES% ^
  --seed %SEED%

echo [RUN] BLIP zero-shot text_source=label_plus_post
python %PY_FILE% ^
  --dataset-root "%DATASET_ROOT%" ^
  --dataset-size "%DATASET_SIZE%" ^
  --split %SPLIT% ^
  --train-split %TRAIN_SPLIT% ^
  --text-source label_plus_post ^
  --group-mode auto ^
  --model-family blip ^
  --model-name "%BLIP_MODEL%" ^
  --output-dir "%OUT_BASE%\blip_label_plus_post_zs" ^
  --batch-size %BLIP_BATCH_ZS% ^
  --device %DEVICE% ^
  --max-samples %MAX_SAMPLES% ^
  --seed %SEED%

REM ======================================================
REM BLIP FINE-TUNE
REM ======================================================

echo.
echo ==========================================
echo Step 2/6: BLIP fine-tune
echo ==========================================

echo [RUN] BLIP fine-tune text_source=label
python %PY_FILE% ^
  --dataset-root "%DATASET_ROOT%" ^
  --dataset-size "%DATASET_SIZE%" ^
  --split %SPLIT% ^
  --train-split %TRAIN_SPLIT% ^
  --text-source label ^
  --group-mode label ^
  --model-family blip ^
  --model-name "%BLIP_MODEL%" ^
  --do-finetune ^
  --epochs %EPOCHS% ^
  --lr %LR% ^
  --weight-decay %WEIGHT_DECAY% ^
  --output-dir "%OUT_BASE%\blip_label_ft" ^
  --batch-size %BLIP_BATCH_FT% ^
  --device %DEVICE% ^
  --max-samples %MAX_SAMPLES% ^
  --seed %SEED%

echo [RUN] BLIP fine-tune text_source=post
python %PY_FILE% ^
  --dataset-root "%DATASET_ROOT%" ^
  --dataset-size "%DATASET_SIZE%" ^
  --split %SPLIT% ^
  --train-split %TRAIN_SPLIT% ^
  --text-source post ^
  --group-mode auto ^
  --model-family blip ^
  --model-name "%BLIP_MODEL%" ^
  --do-finetune ^
  --epochs %EPOCHS% ^
  --lr %LR% ^
  --weight-decay %WEIGHT_DECAY% ^
  --output-dir "%OUT_BASE%\blip_post_ft" ^
  --batch-size %BLIP_BATCH_FT% ^
  --device %DEVICE% ^
  --max-samples %MAX_SAMPLES% ^
  --seed %SEED%

echo [RUN] BLIP fine-tune text_source=label_plus_post
python %PY_FILE% ^
  --dataset-root "%DATASET_ROOT%" ^
  --dataset-size "%DATASET_SIZE%" ^
  --split %SPLIT% ^
  --train-split %TRAIN_SPLIT% ^
  --text-source label_plus_post ^
  --group-mode auto ^
  --model-family blip ^
  --model-name "%BLIP_MODEL%" ^
  --do-finetune ^
  --epochs %EPOCHS% ^
  --lr %LR% ^
  --weight-decay %WEIGHT_DECAY% ^
  --output-dir "%OUT_BASE%\blip_label_plus_post_ft" ^
  --batch-size %BLIP_BATCH_FT% ^
  --device %DEVICE% ^
  --max-samples %MAX_SAMPLES% ^
  --seed %SEED%

REM ======================================================
REM BLIP2 ZERO-SHOT
REM ======================================================

echo.
echo ==========================================
echo Step 3/6: BLIP2 zero-shot
echo ==========================================

echo [RUN] BLIP2 zero-shot text_source=label
python %PY_FILE% ^
  --dataset-root "%DATASET_ROOT%" ^
  --dataset-size "%DATASET_SIZE%" ^
  --split %SPLIT% ^
  --train-split %TRAIN_SPLIT% ^
  --text-source label ^
  --group-mode label ^
  --model-family blip2 ^
  --model-name "%BLIP2_MODEL%" ^
  --output-dir "%OUT_BASE%\blip2_label_zs" ^
  --batch-size %BLIP2_BATCH_ZS% ^
  --device %DEVICE% ^
  --max-samples %MAX_SAMPLES% ^
  --seed %SEED%

echo [RUN] BLIP2 zero-shot text_source=post
python %PY_FILE% ^
  --dataset-root "%DATASET_ROOT%" ^
  --dataset-size "%DATASET_SIZE%" ^
  --split %SPLIT% ^
  --train-split %TRAIN_SPLIT% ^
  --text-source post ^
  --group-mode auto ^
  --model-family blip2 ^
  --model-name "%BLIP2_MODEL%" ^
  --output-dir "%OUT_BASE%\blip2_post_zs" ^
  --batch-size %BLIP2_BATCH_ZS% ^
  --device %DEVICE% ^
  --max-samples %MAX_SAMPLES% ^
  --seed %SEED%

echo [RUN] BLIP2 zero-shot text_source=label_plus_post
python %PY_FILE% ^
  --dataset-root "%DATASET_ROOT%" ^
  --dataset-size "%DATASET_SIZE%" ^
  --split %SPLIT% ^
  --train-split %TRAIN_SPLIT% ^
  --text-source label_plus_post ^
  --group-mode auto ^
  --model-family blip2 ^
  --model-name "%BLIP2_MODEL%" ^
  --output-dir "%OUT_BASE%\blip2_label_plus_post_zs" ^
  --batch-size %BLIP2_BATCH_ZS% ^
  --device %DEVICE% ^
  --max-samples %MAX_SAMPLES% ^
  --seed %SEED%

REM ======================================================
REM BLIP2 FINE-TUNE
REM ======================================================

echo.
echo ==========================================
echo Step 4/6: BLIP2 fine-tune
echo ==========================================
echo [WARN] BLIP2 fine-tune is very slow and memory-heavy. Batch size is set to 1.

echo [RUN] BLIP2 fine-tune text_source=label
python %PY_FILE% ^
  --dataset-root "%DATASET_ROOT%" ^
  --dataset-size "%DATASET_SIZE%" ^
  --split %SPLIT% ^
  --train-split %TRAIN_SPLIT% ^
  --text-source label ^
  --group-mode label ^
  --model-family blip2 ^
  --model-name "%BLIP2_MODEL%" ^
  --do-finetune ^
  --epochs %EPOCH% ^
  --lr %LR% ^
  --weight-decay %WEIGHT_DECAY% ^
  --output-dir "%OUT_BASE%\blip2_label_ft" ^
  --batch-size %BLIP2_BATCH_FT% ^
  --device %DEVICE% ^
  --max-samples %MAX_SAMPLES% ^
  --seed %SEED%

echo [RUN] BLIP2 fine-tune text_source=post
python %PY_FILE% ^
  --dataset-root "%DATASET_ROOT%" ^
  --dataset-size "%DATASET_SIZE%" ^
  --split %SPLIT% ^
  --train-split %TRAIN_SPLIT% ^
  --text-source post ^
  --group-mode auto ^
  --model-family blip2 ^
  --model-name "%BLIP2_MODEL%" ^
  --do-finetune ^
  --epochs %EPOCH% ^
  --lr %LR% ^
  --weight-decay %WEIGHT_DECAY% ^
  --output-dir "%OUT_BASE%\blip2_post_ft" ^
  --batch-size %BLIP2_BATCH_FT% ^
  --device %DEVICE% ^
  --max-samples %MAX_SAMPLES% ^
  --seed %SEED%

echo [RUN] BLIP2 fine-tune text_source=label_plus_post
python %PY_FILE% ^
  --dataset-root "%DATASET_ROOT%" ^
  --dataset-size "%DATASET_SIZE%" ^
  --split %SPLIT% ^
  --train-split %TRAIN_SPLIT% ^
  --text-source label_plus_post ^
  --group-mode auto ^
  --model-family blip2 ^
  --model-name "%BLIP2_MODEL%" ^
  --do-finetune ^
  --epochs %EPOCH% ^
  --lr %LR% ^
  --weight-decay %WEIGHT_DECAY% ^
  --output-dir "%OUT_BASE%\blip2_label_plus_post_ft" ^
  --batch-size %BLIP2_BATCH_FT% ^
  --device %DEVICE% ^
  --max-samples %MAX_SAMPLES% ^
  --seed %SEED%

REM ======================================================
REM MERGE
REM ======================================================

echo.
echo ==========================================
echo Step 5/6: Merge all result CSV files
echo ==========================================

python -c "import pandas as pd; from pathlib import Path; base=Path(r'%OUT_BASE%'); csvs=sorted(base.rglob('task2_multipositive_results.csv')); dfs=[]; [dfs.append(pd.read_csv(p).assign(run_dir=p.parent.name)) for p in csvs]; out=base/'T2_multipositive_blip_blip2_results_merged.csv'; pd.concat(dfs, ignore_index=True).to_csv(out, index=False, encoding='utf-8-sig') if dfs else print('[WARN] no result csv found'); print('[OK] merged:', out) if dfs else None"

echo.
echo ==========================================
echo DONE
echo Final file:
echo %OUT_BASE%\T2_multipositive_blip_blip2_results_merged.csv
echo ==========================================

pause
