@echo off
echo ============================================
echo  PIA-IVIM Training Status Check
echo ============================================
echo.

:: Check DL Training
tasklist /fi "windowtitle eq DL_Training*" 2>nul | find "cmd" >nul
if %errorlevel%==0 (
    echo [RUNNING] DL Training
) else (
    :: Check if all models are done
    if exist "%~dp0checkpoints\refiner_cnn_e2e_best.pt" (
        if not exist "%~dp0checkpoints\*_training.pt" (
            echo [DONE]    DL Training - all models complete!
        ) else (
            echo [STOPPED] DL Training - resuming...
            start "DL_Training" cmd /k "cd /d "%~dp0src" && python retrain_all.py && echo DL TRAINING COMPLETE && pause"
        )
    ) else (
        echo [STOPPED] DL Training - resuming...
        start "DL_Training" cmd /k "cd /d "%~dp0src" && python retrain_all.py && echo DL TRAINING COMPLETE && pause"
    )
)

:: Check NLLS Evaluation
tasklist /fi "windowtitle eq NLLS_Eval*" 2>nul | find "cmd" >nul
if %errorlevel%==0 (
    echo [RUNNING] NLLS Evaluation
) else (
    echo [STOPPED] NLLS Evaluation - resuming...
    start "NLLS_Eval" cmd /k "cd /d "%~dp0src" && python evaluate_nlls_subset.py && echo NLLS EVALUATION COMPLETE && pause"
)

:: Show checkpoint status
echo.
echo --- Checkpoint Status ---
if exist "%~dp0checkpoints\ivim_net_best.pt"         (echo [x] IVIM-NET) else (echo [ ] IVIM-NET)
if exist "%~dp0checkpoints\pia_baseline.pt"           (echo [x] MLP-PIA)  else (echo [ ] MLP-PIA)
if exist "%~dp0checkpoints\pia_cnn_best.pt"           (echo [x] CNN-PIA)  else (echo [ ] CNN-PIA)
if exist "%~dp0checkpoints\refiner_mlp_e2e_best.pt"   (echo [x] Refiner-MLP) else (echo [ ] Refiner-MLP)
if exist "%~dp0checkpoints\refiner_cnn_e2e_best.pt"   (echo [x] Refiner-CNN) else (echo [ ] Refiner-CNN)

echo.
echo --- NLLS Progress ---
python -c "import json,os; f='%~dp0results/nlls_subset_results.json'; d=json.load(open(f)) if os.path.exists(f) else None; print(f'{sum(1 for v in d[\"nlls\"][\"composite_total\"][\"per_patient\"][\"0.05\"] if v is not None)} / 20 patients done') if d else print('Not started yet')" 2>nul || echo Not started yet

echo.
pause
