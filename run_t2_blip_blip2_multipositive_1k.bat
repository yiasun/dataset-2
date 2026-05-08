@echo off
setlocal enabledelayedexpansion

REM ======================================================
REM T2 Multi-positive Retrieval Benchmark - 1K BLIP / BLIP2
REM Structure:
REM   D:\data2\01 Images with labels
REM   D:\data2\02 Text-Image Pairs
REM   D:\data2\run_task2_multipositive_vlm.py
REM ======================================================

cd /d D:\data2

set PY_FILE=run_task2_multipositive_vlm.py
set DATASET_ROOT=D:\data2
set DATASET_SIZE=.
set OUT_BASE=outputs_t2_multipositive_1k_vlm

set SPLIT=test
set TRAIN_SPLIT=train
set DEVICE=cuda
set SEED=42
set MAX_SAMPLES=0

set BLIP_MODEL=Salesforce/blip-itm-base-coco
set BLIP2_MODEL=Salesforce/blip2-flan-t5-xl

set BLIP_BATCH_ZS=8
set BLIP_BATCH_FT=4
set BLIP2_BATCH_ZS=1
set BLIP2_BATCH_FT=1

set EPOCHS=5
set EPOCH=3
set LR=1e-5
set WEIGHT_DECAY=0.01
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ==========================================
echo T2 1K BLIP / BLIP2 Benchmark
echo ==========================================
echo DATASET_ROOT=%DATASET_ROOT%
echo DATASET_SIZE=%DATASET_SIZE%
echo OUT_BASE=%OUT_BASE%
echo ==========================================

if not exist "%PY_FILE%" (
    echo [ERROR] Cannot find %PY_FILE%
    pause
    exit /b 1
)

if not exist "%DATASET_ROOT%\01 Images with labels" (
    echo [ERROR] Missing image folder: %DATASET_ROOT%\01 Images with labels
    pause
    exit /b 1
)

if not exist "%DATASET_ROOT%\02 Text-Image Pairs\train.xlsx" (
    echo [ERROR] Missing train.xlsx
    pause
    exit /b 1
)

mkdir "%OUT_BASE%" 2>nul

REM ======================================================
REM BLIP
REM ======================================================

echo.
echo ==========================================
echo Step 1: BLIP zero-shot / fine-tune
echo ==========================================

for %%T in (label post label_plus_post) do (
    echo.
    echo [RUN] BLIP zero-shot text_source=%%T
    if "%%T"=="label" (
        set GM=label
    ) else (
        set GM=auto
    )

    python %PY_FILE% ^
      --dataset-root "%DATASET_ROOT%" ^
      --dataset-size "%DATASET_SIZE%" ^
      --split %SPLIT% ^
      --train-split %TRAIN_SPLIT% ^
      --text-source %%T ^
      --group-mode !GM! ^
      --model-family blip ^
      --model-name "%BLIP_MODEL%" ^
      --output-dir "%OUT_BASE%\blip_%%T_zs" ^
      --batch-size %BLIP_BATCH_ZS% ^
      --device %DEVICE% ^
      --max-samples %MAX_SAMPLES% ^
      --seed %SEED%

    echo.
    echo [RUN] BLIP fine-tune text_source=%%T
    python %PY_FILE% ^
      --dataset-root "%DATASET_ROOT%" ^
      --dataset-size "%DATASET_SIZE%" ^
      --split %SPLIT% ^
      --train-split %TRAIN_SPLIT% ^
      --text-source %%T ^
      --group-mode !GM! ^
      --model-family blip ^
      --model-name "%BLIP_MODEL%" ^
      --do-finetune ^
      --epochs %EPOCHS% ^
      --lr %LR% ^
      --weight-decay %WEIGHT_DECAY% ^
      --output-dir "%OUT_BASE%\blip_%%T_ft" ^
      --batch-size %BLIP_BATCH_FT% ^
      --device %DEVICE% ^
      --max-samples %MAX_SAMPLES% ^
      --seed %SEED%
)

REM ======================================================
REM BLIP2
REM ======================================================

echo.
echo ==========================================
echo Step 2: BLIP2 zero-shot / fine-tune
echo ==========================================

for %%T in (label post label_plus_post) do (
    echo.
    echo [RUN] BLIP2 zero-shot text_source=%%T
    if "%%T"=="label" (
        set GM=label
    ) else (
        set GM=auto
    )

    python %PY_FILE% ^
      --dataset-root "%DATASET_ROOT%" ^
      --dataset-size "%DATASET_SIZE%" ^
      --split %SPLIT% ^
      --train-split %TRAIN_SPLIT% ^
      --text-source %%T ^
      --group-mode !GM! ^
      --model-family blip2 ^
      --model-name "%BLIP2_MODEL%" ^
      --output-dir "%OUT_BASE%\blip2_%%T_zs" ^
      --batch-size %BLIP2_BATCH_ZS% ^
      --device %DEVICE% ^
      --max-samples %MAX_SAMPLES% ^
      --seed %SEED%

    echo.
    echo [RUN] BLIP2 fine-tune text_source=%%T
    python %PY_FILE% ^
      --dataset-root "%DATASET_ROOT%" ^
      --dataset-size "%DATASET_SIZE%" ^
      --split %SPLIT% ^
      --train-split %TRAIN_SPLIT% ^
      --text-source %%T ^
      --group-mode !GM! ^
      --model-family blip2 ^
      --model-name "%BLIP2_MODEL%" ^
      --do-finetune ^
      --freeze-backbone ^
      --epochs %EPOCH% ^
      --lr %LR% ^
      --weight-decay %WEIGHT_DECAY% ^
      --output-dir "%OUT_BASE%\blip2_%%T_ft" ^
      --batch-size %BLIP2_BATCH_FT% ^
      --device %DEVICE% ^
      --max-samples %MAX_SAMPLES% ^
      --seed %SEED%
)

REM ======================================================
REM MERGE
REM ======================================================

echo.
echo ==========================================
echo Step 3: Merge all result CSV files
echo ==========================================

python -c "import pandas as pd; from pathlib import Path; base=Path(r'%OUT_BASE%'); csvs=sorted(base.rglob('task2_multipositive_results.csv')); dfs=[pd.read_csv(p).assign(run_dir=p.parent.name) for p in csvs]; out=base/'T2_multipositive_1k_blip_blip2_results_merged.csv'; pd.concat(dfs, ignore_index=True).to_csv(out, index=False, encoding='utf-8-sig') if dfs else print('[WARN] no result csv found'); print('[OK] merged:', out) if dfs else None"

echo.
echo ==========================================
echo DONE
echo Final file:
echo %OUT_BASE%\T2_multipositive_1k_blip_blip2_results_merged.csv
echo ==========================================

pause
