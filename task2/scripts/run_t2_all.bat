@echo off
setlocal enabledelayedexpansion

REM ======================================================
REM T2 Benchmark Lib v0.1
REM ======================================================

cd /d D:\data2

set PY_FILE=t2_benchmark_v01.py

set TEXT_PAIR_DIR=D:/data2/10K Dataset_Text-Image Pairs
set IMAGE_DIR=D:/data2/10K Dataset_Labelled
set OUT_DIR=./outputs_t2

REM 通用参数
set NUM_WORKERS=4
set EPOCHS=5
set LR=1e-4

echo ==========================================
echo Step 1/3: Build retrieval json from Excel
echo ==========================================

python %PY_FILE% ^
  --build_pairs_from_excel ^
  --input_xlsx "%TEXT_PAIR_DIR%/train.xlsx" ^
  --image_root "%IMAGE_DIR%/train" ^
  --output_json "%TEXT_PAIR_DIR%/train_pairs.json" ^
  --text_mode label_plus_post

python %PY_FILE% ^
  --build_pairs_from_excel ^
  --input_xlsx "%TEXT_PAIR_DIR%/val.xlsx" ^
  --image_root "%IMAGE_DIR%/val" ^
  --output_json "%TEXT_PAIR_DIR%/val_pairs.json" ^
  --text_mode label_plus_post

python %PY_FILE% ^
  --build_pairs_from_excel ^
  --input_xlsx "%TEXT_PAIR_DIR%/test.xlsx" ^
  --image_root "%IMAGE_DIR%/test" ^
  --output_json "%TEXT_PAIR_DIR%/test_pairs.json" ^
  --text_mode label_plus_post

echo.
echo ==========================================
echo Step 2/3: Run T2 baselines
echo ==========================================

REM ------------------------------------------------------
REM 1. CLIP zero-shot
REM ------------------------------------------------------
echo.
echo [RUN] CLIP zero-shot
python %PY_FILE% ^
  --train_json "%TEXT_PAIR_DIR%/train_pairs.json" ^
  --val_json "%TEXT_PAIR_DIR%/val_pairs.json" ^
  --test_json "%TEXT_PAIR_DIR%/test_pairs.json" ^
  --output_dir "%OUT_DIR%" ^
  --model_name clip ^
  --mode zero_shot ^
  --batch_size 16 ^
  --num_workers %NUM_WORKERS%

REM ------------------------------------------------------
REM 2. CLIP fine-tune
REM ------------------------------------------------------
echo.
echo [RUN] CLIP fine-tune
python %PY_FILE% ^
  --train_json "%TEXT_PAIR_DIR%/train_pairs.json" ^
  --val_json "%TEXT_PAIR_DIR%/val_pairs.json" ^
  --test_json "%TEXT_PAIR_DIR%/test_pairs.json" ^
  --output_dir "%OUT_DIR%" ^
  --model_name clip ^
  --mode ft ^
  --batch_size 16 ^
  --epochs %EPOCHS% ^
  --lr %LR% ^
  --num_workers %NUM_WORKERS%

REM ------------------------------------------------------
REM 3. BLIP zero-shot
REM ------------------------------------------------------
echo.
echo [RUN] BLIP zero-shot
python %PY_FILE% ^
  --train_json "%TEXT_PAIR_DIR%/train_pairs.json" ^
  --val_json "%TEXT_PAIR_DIR%/val_pairs.json" ^
  --test_json "%TEXT_PAIR_DIR%/test_pairs.json" ^
  --output_dir "%OUT_DIR%" ^
  --model_name blip ^
  --mode zero_shot ^
  --batch_size 8 ^
  --num_workers %NUM_WORKERS%

REM ------------------------------------------------------
REM 4. BLIP fine-tune
REM ------------------------------------------------------
echo.
echo [RUN] BLIP fine-tune
python %PY_FILE% ^
  --train_json "%TEXT_PAIR_DIR%/train_pairs.json" ^
  --val_json "%TEXT_PAIR_DIR%/val_pairs.json" ^
  --test_json "%TEXT_PAIR_DIR%/test_pairs.json" ^
  --output_dir "%OUT_DIR%" ^
  --model_name blip ^
  --mode ft ^
  --batch_size 8 ^
  --epochs %EPOCHS% ^
  --lr %LR% ^
  --num_workers %NUM_WORKERS%

REM ------------------------------------------------------
REM 5. BLIP-2 zero-shot
REM ------------------------------------------------------
echo.
echo [RUN] BLIP-2 zero-shot
python %PY_FILE% ^
  --train_json "%TEXT_PAIR_DIR%/train_pairs.json" ^
  --val_json "%TEXT_PAIR_DIR%/val_pairs.json" ^
  --test_json "%TEXT_PAIR_DIR%/test_pairs.json" ^
  --output_dir "%OUT_DIR%" ^
  --model_name blip2 ^
  --mode zero_shot ^
  --batch_size 2 ^
  --num_workers 2

REM ------------------------------------------------------
REM 6. BLIP-2 fine-tune
REM ------------------------------------------------------
echo.
echo [RUN] BLIP-2 fine-tune
python %PY_FILE% ^
  --train_json "%TEXT_PAIR_DIR%/train_pairs.json" ^
  --val_json "%TEXT_PAIR_DIR%/val_pairs.json" ^
  --test_json "%TEXT_PAIR_DIR%/test_pairs.json" ^
  --output_dir "%OUT_DIR%" ^
  --model_name blip2 ^
  --mode ft ^
  --batch_size 2 ^
  --epochs 3 ^
  --lr %LR% ^
  --num_workers 2

REM ------------------------------------------------------
REM 7. LLaVA zero-shot
REM ------------------------------------------------------
echo.
echo [RUN] LLaVA-1.5 zero-shot
python %PY_FILE% ^
  --train_json "%TEXT_PAIR_DIR%/train_pairs.json" ^
  --val_json "%TEXT_PAIR_DIR%/val_pairs.json" ^
  --test_json "%TEXT_PAIR_DIR%/test_pairs.json" ^
  --output_dir "%OUT_DIR%" ^
  --model_name llava ^
  --mode zero_shot ^
  --batch_size 1 ^
  --num_workers 1

REM ------------------------------------------------------
REM 8. LLaVA fine-tune
REM ------------------------------------------------------
echo.
echo [RUN] LLaVA-1.5 fine-tune
python %PY_FILE% ^
  --train_json "%TEXT_PAIR_DIR%/train_pairs.json" ^
  --val_json "%TEXT_PAIR_DIR%/val_pairs.json" ^
  --test_json "%TEXT_PAIR_DIR%/test_pairs.json" ^
  --output_dir "%OUT_DIR%" ^
  --model_name llava ^
  --mode ft ^
  --batch_size 1 ^
  --epochs 3 ^
  --lr %LR% ^
  --num_workers 1

echo.
echo ==========================================
echo Step 3/3: Merge summary tables
echo ==========================================

python %PY_FILE% ^
  --merge_summaries_only ^
  --merge_base_dir "%OUT_DIR%"

echo.
echo ==========================================
echo Done.
echo Final merged table:
echo %OUT_DIR%/T2_results_merged.csv
echo ==========================================

pause