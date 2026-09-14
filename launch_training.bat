@echo off
echo ============================================
echo  PIA-IVIM Auto-Resume Training
echo  Started at: %date% %time%
echo ============================================

cd /d "%~dp0src"

echo.
echo Starting DL Training (retrain_all.py)...
start "DL_Training" cmd /k "cd /d "%~dp0src" && python retrain_all.py && echo DL TRAINING COMPLETE && pause"

echo Starting NLLS Evaluation (evaluate_nlls_subset.py)...
start "NLLS_Eval" cmd /k "cd /d "%~dp0src" && python evaluate_nlls_subset.py && echo NLLS EVALUATION COMPLETE && pause"

echo.
echo Both processes launched in separate windows.
echo They will auto-resume from their last checkpoint.
echo Close this window - the training windows will keep running.
