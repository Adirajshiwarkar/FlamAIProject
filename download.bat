@echo off
set RAW_DIR=%1
if "%RAW_DIR%"=="" set RAW_DIR=raw
if not exist "%RAW_DIR%" mkdir "%RAW_DIR%"
python -m src.download --output-dir "%RAW_DIR%"

