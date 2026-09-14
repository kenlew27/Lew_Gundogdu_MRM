# -*- coding: utf-8 -*-
"""
High-performance parallel NLLS evaluation using multi-core processing.
Matches evaluate_comprehensive.py and evaluate_nlls_subset.py exactly in:
- Deterministic Rician noise seeding (pat * 10000 + full_noise_idx)
- avg_S0 scaling from training cases 1-600 (0.266081)
- Data paths and rRMSE metric calculations
- Paired Wilcoxon signed-rank tests
"""
import os
import sys
import json
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import rRMSE_per_param, fit_biExponential_model

FULL_NOISE_LEVELS = [1e-5, 1e-4, 1e-3, 0.005, 0.01, 0.02, 0.033, 0.05, 0.1, 0.15, 0.2, 0.25]

def noise_seed(patient_idx, noise_idx):
    return patient_idx * 10000 + noise_idx

def add_rician_noise(signal, noise_std, rng=None):
    if rng is None:
        rng = np.random
    signal_mag = np.abs(signal).astype(np.float64)
    noise1 = rng.normal(0, noise_std, signal_mag.shape)
    noise2 = rng.normal(0, noise_std, signal_mag.shape)
    return np.sqrt((signal_mag + noise1)**2 + noise2**2)

def worker_nlls(args):
    pat, nl, full_noise_idx, avg_S0, data_dir, b_values = args
    try:
        dwi = np.load(os.path.join(data_dir, f"{pat:04d}_gtDWIs.npy"))
        ivim = np.load(os.path.join(data_dir, f"{pat:04d}_IVIMParam.npy"))
        tissue = np.load(os.path.join(data_dir, f"{pat:04d}_TissueType.npy"))
        
        rng = np.random.RandomState(noise_seed(pat, full_noise_idx))
        noisy_dwi = add_rician_noise(dwi, nl * avg_S0, rng=rng)
        
        pred = fit_biExponential_model(noisy_dwi, b_values)
        
        x_f, x_dt, x_ds = pred[:, :, 0], pred[:, :, 1], pred[:, :, 2]
        y_f, y_dt, y_ds = ivim[:, :, 0], ivim[:, :, 1], ivim[:, :, 2]
        
        rrmses = rRMSE_per_param(x_f, x_dt, x_ds, y_f, y_dt, y_ds, tissue)
        return (nl, pat, rrmses, None)
    except Exception as e:
        return (nl, pat, None, str(e))

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, 'results')
    results_path = os.path.join(results_dir, 'nlls_subset_results.json')
    detailed_json_path = os.path.join(results_dir, 'evaluation_detailed.json')
    
    b_values = np.array([0, 5, 50, 100, 200, 500, 800, 1000])
    noise_levels = [0.02, 0.05, 0.10, 0.20]
    patients = list(range(801, 1001))
    
    noise_idx_map = {nl: FULL_NOISE_LEVELS.index(nl) for nl in noise_levels}
    
    # Load or initialize results dictionary
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
    else:
        results = {
            "nlls": {},
            "wilcoxon_tests": {},
            "noise_levels": noise_levels,
            "patient_count": len(patients),
            "patients": patients
        }
        for k in ["composite_total", "f_tumor", "f_nontumor", "Dt_tumor", "Dt_nontumor",
                  "Dstar_tumor", "Dstar_nontumor", "composite_tumor", "composite_nontumor"]:
            results["nlls"][k] = {
                "mean": [],
                "std": [],
                "per_patient": {str(nl): [None] * len(patients) for nl in noise_levels}
            }

    # Verify S0 from training cases 1-600
    avg_S0 = 0.266081
    
    # Find all tasks to run
    tasks = []
    for nl in noise_levels:
        nl_str = str(nl)
        full_idx = noise_idx_map[nl]
        for j, pat in enumerate(patients):
            curr_val = results["nlls"]["composite_total"]["per_patient"][nl_str][j]
            if curr_val is None:
                tasks.append((pat, nl, full_idx, avg_S0, data_dir, b_values))
                
    total_to_run = len(tasks)
    print(f"Total NLLS patient tasks to compute: {total_to_run} across {len(noise_levels)} noise levels.")
    
    if total_to_run > 0:
        max_workers = 28  # Utilize high core count (16 cores / 32 threads)
        print(f"Launching ProcessPoolExecutor with {max_workers} worker processes...")
        
        t0 = time.time()
        completed = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(worker_nlls, task): task for task in tasks}
            
            for future in as_completed(future_to_task):
                nl, pat, rrmses, err = future.result()
                completed += 1
                
                if rrmses is not None:
                    j = pat - 801
                    for k, v in rrmses.items():
                        results["nlls"][k]["per_patient"][str(nl)][j] = v
                else:
                    print(f"Error for patient {pat} at sigma={nl}: {err}")
                    
                if completed % 10 == 0 or completed == total_to_run:
                    elapsed = time.time() - t0
                    rate = completed / elapsed
                    eta = (total_to_run - completed) / rate if rate > 0 else 0
                    print(f"[{completed}/{total_to_run}] ({completed/total_to_run*100:.1f}%) "
                          f"Elapsed: {elapsed/60:.1f}m, ETA: {eta/60:.1f}m, Speed: {rate:.2f} slices/sec", flush=True)
                    
                    with open(results_path, 'w') as f:
                        json.dump(results, f)
                        
    # Compute summary statistics
    for k in ["composite_total", "f_tumor", "f_nontumor", "Dt_tumor", "Dt_nontumor",
              "Dstar_tumor", "Dstar_nontumor", "composite_tumor", "composite_nontumor"]:
        results["nlls"][k]["mean"] = []
        results["nlls"][k]["std"] = []
        for nl in noise_levels:
            vals = [v for v in results["nlls"][k]["per_patient"][str(nl)] if v is not None]
            if len(vals) > 0:
                results["nlls"][k]["mean"].append(float(np.mean(vals)))
                results["nlls"][k]["std"].append(float(np.std(vals)))
            else:
                results["nlls"][k]["mean"].append(None)
                results["nlls"][k]["std"].append(None)

    # Perform paired Wilcoxon signed-rank tests against DL results
    if os.path.exists(detailed_json_path):
        with open(detailed_json_path, 'r') as f:
            dl_results = json.load(f)
            
        dl_methods_data = dl_results['methods']
        dl_noise_levels = dl_results['noise_levels']
        
        results["wilcoxon_tests"] = {}
        for method in dl_methods_data.keys():
            test_key = f"{method}_vs_nlls"
            results["wilcoxon_tests"][test_key] = {"p_values": [], "significant": [], "noise_levels": []}
            for nl in noise_levels:
                nlls_scores = results["nlls"]["composite_total"]["per_patient"][str(nl)]
                if None in nlls_scores:
                    continue
                try:
                    dl_noise_idx = dl_noise_levels.index(nl)
                except ValueError:
                    continue
                    
                dl_scores = dl_methods_data[method]["composite_total"]["per_patient"][dl_noise_idx]
                try:
                    stat, p = wilcoxon(nlls_scores, dl_scores)
                    results["wilcoxon_tests"][test_key]["p_values"].append(float(p))
                    results["wilcoxon_tests"][test_key]["significant"].append(bool(p < 0.05))
                    results["wilcoxon_tests"][test_key]["noise_levels"].append(nl)
                except Exception as e:
                    results["wilcoxon_tests"][test_key]["p_values"].append(None)
                    results["wilcoxon_tests"][test_key]["significant"].append(None)
                    results["wilcoxon_tests"][test_key]["noise_levels"].append(nl)
                    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print("\n[SUCCESS] NLLS parallel evaluation complete and results saved.")

if __name__ == '__main__':
    main()
