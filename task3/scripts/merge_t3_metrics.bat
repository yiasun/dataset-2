@echo off
setlocal

REM Edit this path for the local machine.
set OUT_ROOT=E:\Project Data\ChatGPT-Urban ImageNet\t3_rerun_outputs

cd /d "%~dp0\.."

python tools\make_summary_tables.py ^
  --metrics-root "%OUT_ROOT%\t3_metrics" ^
  --out-xlsx "%OUT_ROOT%\t3_metrics_summary.xlsx" ^
  --out-csv "%OUT_ROOT%\t3_metrics_summary.csv"

endlocal
