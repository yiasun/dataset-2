@echo off
setlocal enabledelayedexpansion

REM ======================================================
REM T2 Multi-positive Retrieval Benchmark - 10K Direct Mode
REM Your actual structure:
REM D:\data2\
REM ├── 10K Dataset_Labelled
REM ├── 10K Dataset_Text-Image Pairs
REM └── run_task2_multipositive.py
REM
REM This script automatically creates a temporary compatible folder:
REM D:\data2\10K Dataset\
REM ├── 01 Images with labels
REM └── 02 Text-Image Pairs
REM ======================================================

cd /d D:\data2

set PY_FILE=run_task2_multipositive.py
set DATASET_ROOT=D:\data2
set DATASET_SIZE=10K Dataset
set LINK_DIR=D:\data2\10K Dataset

set IMAGE_SRC=D:\data2\10K Dataset_Labelled
set TEXT_SRC=D:\data2\10K Dataset_Text-Image Pairs

set OUT_BASE=outputs_t2_multipositive_10k

set SPLIT=test
set TRAIN_SPLIT=train
set MODEL_NAME=openai/clip-vit-base-patch32
set DEVICE=cuda
set SEED=42

set BATCH_ZS=32
set BATCH_FT=16
set EPOCHS=5
set LR=1e-5
set WEIGHT_DECAY=0.01

REM 0 = full dataset, 200 = quick debug
set MAX_SAMPLES=0

echo ==========================================
echo T2 Multi-positive Benchmark - 10K Dataset
echo ==========================================
echo DATASET_ROOT=%DATASET_ROOT%
echo DATASET_SIZE=%DATASET_SIZE%
echo IMAGE_SRC=%IMAGE_SRC%
echo TEXT_SRC=%TEXT_SRC%
echo OUT_BASE=%OUT_BASE%
echo ==========================================

if not exist "%PY_FILE%" (
    echo [ERROR] Cannot find %PY_FILE%
    pause
    exit /b 1
)

if not exist "%IMAGE_SRC%" (
    echo [ERROR] Missing image folder:
    echo %IMAGE_SRC%
    pause
    exit /b 1
)

if not exist "%TEXT_SRC%\train.xlsx" (
    echo [ERROR] Missing train.xlsx:
    echo %TEXT_SRC%\train.xlsx
    pause
    exit /b 1
)

if not exist "%TEXT_SRC%\val.xlsx" (
    echo [ERROR] Missing val.xlsx:
    echo %TEXT_SRC%\val.xlsx
    pause
    exit /b 1
)

if not exist "%TEXT_SRC%\test.xlsx" (
    echo [ERROR] Missing test.xlsx:
    echo %TEXT_SRC%\test.xlsx
    pause
    exit /b 1
)

REM ------------------------------------------------------
REM Build compatible folder layout for run_task2_multipositive.py
REM ------------------------------------------------------
if not exist "%LINK_DIR%" (
    mkdir "%LINK_DIR%"
)

if not exist "%LINK_DIR%\01 Images with labels" (
    echo [INFO] Creating junction: 01 Images with labels
    cmd /c mklink /J "%LINK_DIR%\01 Images with labels" "%IMAGE_SRC%"
)

if not exist "%LINK_DIR%\02 Text-Image Pairs" (
    echo [INFO] Creating junction: 02 Text-Image Pairs
    cmd /c mklink /J "%LINK_DIR%\02 Text-Image Pairs" "%TEXT_SRC%"
)

if not exist "%LINK_DIR%\01 Images with labels" (
    echo [ERROR] Failed to create image junction.
    echo Please run this BAT as Administrator, or manually copy folders.
    pause
    exit /b 1
)

if not exist "%LINK_DIR%\02 Text-Image Pairs" (
    echo [ERROR] Failed to create text junction.
    echo Please run this BAT as Administrator, or manually copy folders.
    pause
    exit /b 1
)

mkdir "%OUT_BASE%" 2>nul

echo.
echo ==========================================
echo Step 1: label retrieval
echo ==========================================

echo [RUN] CLIP zero-shot text_source=label
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

echo [RUN] CLIP fine-tune text_source=label
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
echo Step 2: post retrieval
echo ==========================================

echo [RUN] CLIP zero-shot text_source=post
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

echo [RUN] CLIP fine-tune text_source=post
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
echo Step 3: label_plus_post retrieval
echo ==========================================

echo [RUN] CLIP zero-shot text_source=label_plus_post
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

echo [RUN] CLIP fine-tune text_source=label_plus_post
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
echo Step 4: merge csv results
echo ==========================================

python -c "import pandas as pd; from pathlib import Path; base=Path(r'%OUT_BASE%'); csvs=sorted(base.rglob('task2_multipositive_results.csv')); dfs=[]; [dfs.append(pd.read_csv(p).assign(run_dir=p.parent.name)) for p in csvs]; out=base/'T2_multipositive_results_merged.csv'; pd.concat(dfs, ignore_index=True).to_csv(out, index=False, encoding='utf-8-sig') if dfs else print('[WARN] no result csv found'); print('[OK] merged:', out) if dfs else None"

echo.
echo ==========================================
echo DONE
echo Final file:
echo %OUT_BASE%\T2_multipositive_results_merged.csv
echo ==========================================

pause