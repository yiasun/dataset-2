@echo off
setlocal

REM Edit these three paths for the local machine.
set IMAGE_ROOT=E:\NIPS 2026 Dataset\00 Dataset\1K Dataset\1K Dataset_Labelled
set ANN_ROOT=E:\NIPS 2026 Dataset\00 Dataset\1K Dataset\1K Dataset_Segmentation Labelled
set PREP_ROOT=E:\Project Data\ChatGPT-Urban ImageNet\t3_1k_class_agnostic

cd /d "%~dp0\.."

python tools\prepare_t3_dataset.py ^
  --image-root "%IMAGE_ROOT%" ^
  --ann-root "%ANN_ROOT%" ^
  --out-root "%PREP_ROOT%" ^
  --mode class_agnostic

python tools\check_coco_instance.py --ann "%PREP_ROOT%\annotations\train.json" --img-root "%PREP_ROOT%\images"
python tools\check_coco_instance.py --ann "%PREP_ROOT%\annotations\val.json" --img-root "%PREP_ROOT%\images"
python tools\check_coco_instance.py --ann "%PREP_ROOT%\annotations\test.json" --img-root "%PREP_ROOT%\images"

echo.
echo Prepared class-agnostic dataset:
echo   %PREP_ROOT%
endlocal
