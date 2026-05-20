@echo off
setlocal enabledelayedexpansion

REM ======================================================
REM T2 Multi-positive Retrieval Benchmark
REM Dataset layout:
REM   D:\data2\01 Images with labels
REM   D:\data2\02 Text-Image Pairs
REM   D:\data2\03 Instance Segmentation
REM ======================================================

cd /d D:\data2

REM ------------------------------------------------------
REM Basic paths
REM ------------------------------------------------------
set PY_FILE=run_task2_multipositive.py
set DATASET_ROOT=D:\data2

REM Important:
REM The 1K dataset folders are directly under D:\data2:
REM   D:\data2\01 Images with labels
REM   D:\data2\02 Text-Image Pairs
REM Therefore DATASET_SIZE must be "."
set DATASET_SIZE=.

set OUT_BASE=outputs_t2_multipositive_1k

REM ------------------------------------------------------
REM Common parameters
REM ------------------------------------------------------
set SPLIT=test
set TRAIN_SPLIT=train
set MODEL_NAME=openai/clip-vit-base-patch32
set DEVICE=cuda
set SEED=42

REM zero-shot batch size
set BATCH_ZS=32

REM fine-tune parameters
set BATCH_FT=16
set EPOCHS=3
set LR=1e-5
set WEIGHT_DECAY=0.01

REM Optional quick debugging.
REM Set MAX_SAMPLES=200 for quick test.
REM Set MAX_SAMPLES=0 for full split.
set MAX_SAMPLES=0

echo ==========================================
echo T2 Multi-positive Benchmark - 1K Dataset
echo ==========================================
echo Dataset root: %DATASET_ROOT%
echo Dataset size: %DATASET_SIZE%
echo Python file : %PY_FILE%
echo Output base : %OUT_BASE%
echo ==========================================

if not exist "%PY_FILE%" (
  echo [ERROR] Cannot find %PY_FILE% under D:\data2
  pause
  exit /b 1
)

if not exist "%DATASET_ROOT%\01 Images with labels" (
  echo [ERROR] Cannot find image folder:
  echo %DATASET_ROOT%\01 Images with labels
  pause
  exit /b 1
)

if not exist "%DATASET_ROOT%\02 Text-Image Pairs\train.xlsx" (
  echo [ERROR] Cannot find train.xlsx:
  echo %DATASET_ROOT%\02 Text-Image Pairs\train.xlsx
  pause
  exit /b 1
)

if not exist "%DATASET_ROOT%\02 Text-Image Pairs\val.xlsx" (
  echo [ERROR] Cannot find val.xlsx:
  echo %DATASET_ROOT%\02 Text-Image Pairs\val.xlsx
  pause
  exit /b 1
)

if not exist "%DATASET_ROOT%\02 Text-Image Pairs\test.xlsx" (
  echo [ERROR] Cannot find test.xlsx:
  echo %DATASET_ROOT%\02 Text-Image Pairs\test.xlsx
  pause
  exit /b 1
)

mkdir "%OUT_BASE%" 2>nul

echo.
echo ==========================================
echo Step 1/4: Label-only retrieval
echo ==========================================

echo.
echo [RUN] CLIP zero-shot | text_source=label
python %PY_FILE% ^
  --dataset-root "%DATASET_ROOT%" ^
  --dataset-size "%DATASET_SIZE%" ^
  --split %SPLIT% ^
  --train-split %TRAIN_SPLIT% ^
  --text-source label ^
  --group-mode label ^
  --model-name "%MODEL_NAME%" ^
  --output-dir "%OUT_BASE%\clip_label_zs" ^
  --batch-size %BATCH_ZS% ^
  --device %DEVICE% ^
  --max-samples %MAX_SAMPLES% ^
  --seed %SEED%

echo.
echo [RUN] CLIP fine-tune | text_source=label
python %PY_FILE% ^
  --dataset-root "%DATASET_ROOT%" ^
  --dataset-size "%DATASET_SIZE%" ^
  --split %SPLIT% ^
  --train-split %TRAIN_SPLIT% ^
  --text-source label ^
  --group-mode label ^
  --model-name "%MODEL_NAME%" ^
  --do-finetune ^
  --epochs %EPOCHS% ^
  --lr %LR% ^
  --weight-decay %WEIGHT_DECAY% ^
  --output-dir "%OUT_BASE%\clip_label_ft" ^
  --batch-size %BATCH_FT% ^
  --device %DEVICE% ^
  --max-samples %MAX_SAMPLES% ^
  --seed %SEED%

echo.
echo ==========================================
echo Step 2/4: Post-text retrieval
echo ==========================================

echo.
echo [RUN] CLIP zero-shot | text_source=post
python %PY_FILE% ^
  --dataset-root "%DATASET_ROOT%" ^
  --dataset-size "%DATASET_SIZE%" ^
  --split %SPLIT% ^
  --train-split %TRAIN_SPLIT% ^
  --text-source post ^
  --group-mode auto ^
  --model-name "%MODEL_NAME%" ^
  --output-dir "%OUT_BASE%\clip_post_zs" ^
  --batch-size %BATCH_ZS% ^
  --device %DEVICE% ^
  --max-samples %MAX_SAMPLES% ^
  --seed %SEED%

echo.
echo [RUN] CLIP fine-tune | text_source=post
python %PY_FILE% ^
  --dataset-root "%DATASET_ROOT%" ^
  --dataset-size "%DATASET_SIZE%" ^
  --split %SPLIT% ^
  --train-split %TRAIN_SPLIT% ^
  --text-source post ^
  --group-mode auto ^
  --model-name "%MODEL_NAME%" ^
  --do-finetune ^
  --epochs %EPOCHS% ^
  --lr %LR% ^
  --weight-decay %WEIGHT_DECAY% ^
  --output-dir "%OUT_BASE%\clip_post_ft" ^
  --batch-size %BATCH_FT% ^
  --device %DEVICE% ^
  --max-samples %MAX_SAMPLES% ^
  --seed %SEED%

echo.
echo ==========================================
echo Step 3/4: Label + post retrieval
echo ==========================================

echo.
echo [RUN] CLIP zero-shot | text_source=label_plus_post
python %PY_FILE% ^
  --dataset-root "%DATASET_ROOT%" ^
  --dataset-size "%DATASET_SIZE%" ^
  --split %SPLIT% ^
  --train-split %TRAIN_SPLIT% ^
  --text-source label_plus_post ^
  --group-mode auto ^
  --model-name "%MODEL_NAME%" ^
  --output-dir "%OUT_BASE%\clip_label_plus_post_zs" ^
  --batch-size %BATCH_ZS% ^
  --device %DEVICE% ^
  --max-samples %MAX_SAMPLES% ^
  --seed %SEED%

echo.
echo [RUN] CLIP fine-tune | text_source=label_plus_post
python %PY_FILE% ^
  --dataset-root "%DATASET_ROOT%" ^
  --dataset-size "%DATASET_SIZE%" ^
  --split %SPLIT% ^
  --train-split %TRAIN_SPLIT% ^
  --text-source label_plus_post ^
  --group-mode auto ^
  --model-name "%MODEL_NAME%" ^
  --do-finetune ^
  --epochs %EPOCHS% ^
  --lr %LR% ^
  --weight-decay %WEIGHT_DECAY% ^
  --output-dir "%OUT_BASE%\clip_label_plus_post_ft" ^
  --batch-size %BATCH_FT% ^
  --device %DEVICE% ^
  --max-samples %MAX_SAMPLES% ^
  --seed %SEED%

echo.
echo ==========================================
echo Step 4/4: Merge result CSV files
echo ==========================================

python -c "import pandas as pd; from pathlib import Path; base=Path(r'%OUT_BASE%'); csvs=sorted(base.rglob('task2_multipositive_results.csv')); dfs=[]; [dfs.append(pd.read_csv(p).assign(run_dir=p.parent.name)) for p in csvs]; out=base/'T2_multipositive_results_merged.csv'; pd.concat(dfs, ignore_index=True).to_csv(out, index=False, encoding='utf-8-sig') if dfs else print('[WARN] no result csv found'); print('[OK] merged:', out) if dfs else None"

echo.
echo ==========================================
echo Done.
echo Final merged table:
echo %OUT_BASE%\T2_multipositive_results_merged.csv
echo ==========================================

pause
