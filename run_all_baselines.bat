@echo off
setlocal enabledelayedexpansion

REM ================================
REM 基础配置
REM ================================
set DATA_DIR=D:/data2/10K Dataset_Labelled
set PY_FILE=baseline_image_classification.py
set BASE_OUT=./outputs_10k_all

REM 通用训练参数
set EPOCHS=20
set LR=1e-4
set NUM_WORKERS=4

echo ==========================================
echo Running all baselines on %DATA_DIR%
echo ==========================================

REM ==========================================
REM 1. ResNet-18
REM ==========================================
echo.
echo [RUN] ResNet-18
python %PY_FILE% ^
  --data_dir "%DATA_DIR%" ^
  --output_dir "%BASE_OUT%/resnet18" ^
  --model_name resnet18 ^
  --image_size 224 ^
  --batch_size 32 ^
  --epochs %EPOCHS% ^
  --lr %LR% ^
  --num_workers %NUM_WORKERS% ^
  --pretrained

REM ==========================================
REM 2. ResNet-50
REM ==========================================
echo.
echo [RUN] ResNet-50
python %PY_FILE% ^
  --data_dir "%DATA_DIR%" ^
  --output_dir "%BASE_OUT%/resnet50" ^
  --model_name resnet50 ^
  --image_size 224 ^
  --batch_size 32 ^
  --epochs %EPOCHS% ^
  --lr %LR% ^
  --num_workers %NUM_WORKERS% ^
  --pretrained

REM ==========================================
REM 3. ResNet-152
REM ==========================================
echo.
echo [RUN] ResNet-152
python %PY_FILE% ^
  --data_dir "%DATA_DIR%" ^
  --output_dir "%BASE_OUT%/resnet152" ^
  --model_name resnet152 ^
  --image_size 224 ^
  --batch_size 16 ^
  --epochs %EPOCHS% ^
  --lr %LR% ^
  --num_workers %NUM_WORKERS% ^
  --pretrained

REM ==========================================
REM 4. EfficientNet-B4
REM ==========================================
echo.
echo [RUN] EfficientNet-B4
python %PY_FILE% ^
  --data_dir "%DATA_DIR%" ^
  --output_dir "%BASE_OUT%/efficientnet_b4" ^
  --model_name efficientnet_b4 ^
  --image_size 380 ^
  --batch_size 16 ^
  --epochs %EPOCHS% ^
  --lr %LR% ^
  --num_workers %NUM_WORKERS% ^
  --pretrained

REM ==========================================
REM 5. ViT-B/16
REM ==========================================
echo.
echo [RUN] ViT-B/16
python %PY_FILE% ^
  --data_dir "%DATA_DIR%" ^
  --output_dir "%BASE_OUT%/vit_b_16" ^
  --model_name vit_b_16 ^
  --image_size 224 ^
  --batch_size 16 ^
  --epochs %EPOCHS% ^
  --lr %LR% ^
  --num_workers %NUM_WORKERS% ^
  --pretrained

REM ==========================================
REM 6. DeiT-B
REM ==========================================
echo.
echo [RUN] DeiT-B
python %PY_FILE% ^
  --data_dir "%DATA_DIR%" ^
  --output_dir "%BASE_OUT%/deit_b" ^
  --model_name deit_b ^
  --image_size 224 ^
  --batch_size 16 ^
  --epochs %EPOCHS% ^
  --lr %LR% ^
  --num_workers %NUM_WORKERS% ^
  --pretrained

REM ==========================================
REM 7. CLIP Zero-shot
REM ==========================================
echo.
echo [RUN] CLIP Zero-shot
python %PY_FILE% ^
  --data_dir "%DATA_DIR%" ^
  --output_dir "%BASE_OUT%/clip_zero_shot" ^
  --clip_zero

echo.
echo ==========================================
echo All experiments finished.
echo Outputs saved under %BASE_OUT%
echo ==========================================

pause