@echo off
if "%~2"=="" (
    echo Usage: %0 ^<raw_dir^> ^<out_dir^>
    exit /b 1
)
python -m src.pipeline --raw-dir "%~1" --out-dir "%~2"

