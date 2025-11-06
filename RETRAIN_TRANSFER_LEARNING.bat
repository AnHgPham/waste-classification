@echo off
REM Quick script to retrain transfer learning model with fixes
REM Run this after fixing the preprocessing bugs

echo ============================================================
echo   RETRAIN TRANSFER LEARNING MODEL (WITH FIXES)
echo ============================================================
echo.
echo This script will:
echo 1. Delete old (incorrectly trained) models
echo 2. Retrain with corrected preprocessing
echo 3. Save new models with proper normalization
echo.

REM Change to project directory
cd /d "%~dp0"

echo Checking for old models...
if exist "outputs\models\mobilenetv2_phase1.keras" (
    echo [WARNING] Found old Phase 1 model - deleting...
    del "outputs\models\mobilenetv2_phase1.keras"
)

if exist "outputs\models\mobilenetv2_final.keras" (
    echo [WARNING] Found old final model - deleting...
    del "outputs\models\mobilenetv2_final.keras"
)

echo.
echo ============================================================
echo   STARTING TRAINING WITH FIXED CODE
echo ============================================================
echo.

REM Run training script
python scripts\04_transfer_learning.py

echo.
echo ============================================================
echo   TRAINING COMPLETE!
echo ============================================================
echo.
echo Check outputs\reports\ for training history plots
echo Check outputs\models\ for saved models
echo.
pause

