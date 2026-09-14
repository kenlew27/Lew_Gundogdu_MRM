"""NLLS subset evaluation with deterministic seeding.

DEPRECATED: Prefer running `evaluate_comprehensive.py` which evaluates all
methods (DL + NLLS) in a single pass with guaranteed identical noise per
patient, enabling valid paired Wilcoxon tests.

This script is kept for standalone NLLS-only runs. It uses the same
deterministic seed formula as evaluate_comprehensive.py so results will
match if both are run on the same patients and noise levels.
"""
import sys
import os
import json
import time
import numpy as np
from tqdm import tqdm
from scipy.stats import wilcoxon

sys.path.insert(0, '.')
from utils import rRMSE_per_param, fit_biExponential_model


# Must match evaluate_comprehensive.py exactly
FULL_NOISE_LEVELS = [1e-5, 1e-4, 1e-3, 0.005, 0.01, 0.02, 0.033, 0.05, 0.1, 0.15, 0.2, 0.25]


def noise_seed(patient_idx, noise_idx):
    """Deterministic seed for reproducible Rician noise per patient+noise combo.

    Uses the noise_idx from the FULL 12-level noise schedule so that seeds
    match evaluate_comprehensive.py regardless of which subset we evaluate.
    """
    return patient_idx * 10000 + noise_idx


def add_rician_noise(signal, noise_std, rng=None):
    """Add Rician noise using optional seeded RNG for reproducibility.
    
    Args:
        signal: ground-truth DWI signal array (may be complex128)
        noise_std: absolute noise standard deviation (= sigma * avg_S0)
        rng: optional seeded numpy RandomState
    """
    if rng is None:
        rng = np.random
    # Take magnitude first — ground truth DWIs are complex128
    signal_mag = np.abs(signal).astype(np.float64)
    noise1 = rng.normal(0, noise_std, signal_mag.shape)
    noise2 = rng.normal(0, noise_std, signal_mag.shape)
    return np.sqrt((signal_mag + noise1)**2 + noise2**2)


def main():
    b_values = np.array([0, 5, 50, 100, 200, 500, 800, 1000])
    noise_levels = [0.02, 0.05, 0.1, 0.2]
    patients = list(range(801, 1001))

    # Map each noise level to its index in the full 12-level schedule
    noise_idx_map = {}
    for nl in noise_levels:
        try:
            noise_idx_map[nl] = FULL_NOISE_LEVELS.index(nl)
        except ValueError:
            raise ValueError(f"Noise level {nl} not found in FULL_NOISE_LEVELS. "
                             f"Available: {FULL_NOISE_LEVELS}")

    results_path = '../results/nlls_subset_results.json'
    detailed_json_path = '../results/evaluation_detailed.json'

    # Initialize or load results
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
        # Check if existing results used deterministic seeding
        if "deterministic_seed" not in results:
            print("WARNING: Existing results were generated WITHOUT deterministic seeding.")
            print("         They will be re-generated with seeded noise for valid pairing.")
            # Reset all per_patient data to force re-evaluation
            for k in results["nlls"]:
                results["nlls"][k]["per_patient"] = {str(nl): [None]*len(patients) for nl in noise_levels}
            results["deterministic_seed"] = True
        # Update config to reflect current noise levels and patients
        results["noise_levels"] = noise_levels
        results["patient_indices"] = patients
        # Backfill any missing noise level keys and resize patient arrays
        for k in results["nlls"]:
            if "per_patient" not in results["nlls"][k]:
                results["nlls"][k]["per_patient"] = {}
            for nl in noise_levels:
                nl_key = str(nl)
                if nl_key not in results["nlls"][k]["per_patient"]:
                    results["nlls"][k]["per_patient"][nl_key] = [None]*len(patients)
                else:
                    # Resize if patient list changed
                    existing = results["nlls"][k]["per_patient"][nl_key]
                    if len(existing) < len(patients):
                        existing.extend([None] * (len(patients) - len(existing)))
                    elif len(existing) > len(patients):
                        existing = existing[:len(patients)]
                    results["nlls"][k]["per_patient"][nl_key] = existing
    else:
        results = {
            "noise_levels": noise_levels,
            "patient_indices": patients,
            "deterministic_seed": True,
            "nlls": {
                "composite_total": {"mean": [], "std": [], "per_patient": {}},
                "f_tumor": {"mean": [], "std": [], "per_patient": {}},
                "f_nontumor": {"mean": [], "std": [], "per_patient": {}},
                "Dt_tumor": {"mean": [], "std": [], "per_patient": {}},
                "Dt_nontumor": {"mean": [], "std": [], "per_patient": {}},
                "Dstar_tumor": {"mean": [], "std": [], "per_patient": {}},
                "Dstar_nontumor": {"mean": [], "std": [], "per_patient": {}},
                "composite_tumor": {"mean": [], "std": [], "per_patient": {}}
            },
            "wilcoxon_tests": {}
        }
        for k in results["nlls"]:
            results["nlls"][k]["per_patient"] = {str(nl): [None]*len(patients) for nl in noise_levels}

    total_tasks = len(noise_levels) * len(patients)
    tasks_done = 0
    start_time = time.time()

    # Compute average S(b=0) from TRAINING set for noise scaling.
    # Using training cases (1-600) avoids test-set information leakage.
    # AAPM convention: noise_std = sigma * avg_S0, so SNR = 1/sigma.
    print("Computing average S0 from training set for noise scaling...")
    s0_sum, s0_count = 0.0, 0
    for pat in range(1, 601):  # training cases only
        try:
            s = np.load(f"../data/{pat:04d}_gtDWIs.npy")
            t = np.load(f"../data/{pat:04d}_TissueType.npy")
            mask = (t != 1)
            s0_sum += np.abs(s[:, :, 0][mask]).sum()
            s0_count += mask.sum()
        except FileNotFoundError:
            continue
    avg_S0 = s0_sum / s0_count if s0_count > 0 else 1.0
    print(f"  avg_S0 = {avg_S0:.6f}")

    for i, nl in enumerate(noise_levels):
        full_noise_idx = noise_idx_map[nl]
        for j, pat in enumerate(patients):
            tasks_done += 1
            if results["nlls"]["composite_total"]["per_patient"][str(nl)][j] is not None:
                continue

            print(f"Processing patient {pat} at noise level {nl}... ({tasks_done}/{total_tasks})")

            dwi = np.load(f"../data/{pat:04d}_gtDWIs.npy")
            ivim = np.load(f"../data/{pat:04d}_IVIMParam.npy")
            tissue = np.load(f"../data/{pat:04d}_TissueType.npy")

            # Deterministic noise matching evaluate_comprehensive.py
            rng = np.random.RandomState(noise_seed(pat, full_noise_idx))
            noisy_dwi = add_rician_noise(dwi, nl * avg_S0, rng=rng)

            pred = fit_biExponential_model(noisy_dwi, b_values)

            x_f = pred[:, :, 0]
            x_dt = pred[:, :, 1]
            x_ds = pred[:, :, 2]

            y_f = ivim[:, :, 0]
            y_dt = ivim[:, :, 1]
            y_ds = ivim[:, :, 2]

            rrmses = rRMSE_per_param(x_f, x_dt, x_ds, y_f, y_dt, y_ds, tissue)

            for k, v in rrmses.items():
                results["nlls"][k]["per_patient"][str(nl)][j] = v

            with open(results_path, 'w') as f:
                json.dump(results, f)

            elapsed = time.time() - start_time
            avg_time = elapsed / (j + 1 + i*len(patients))  # rough if resumed, but ok
            remaining = (total_tasks - tasks_done) * avg_time
            print(f"  Done. Estimated time remaining: {remaining/60:.1f} minutes")

    # Compute means and stds
    for k in results["nlls"]:
        results["nlls"][k]["mean"] = []
        results["nlls"][k]["std"] = []
        for nl in noise_levels:
            vals = results["nlls"][k]["per_patient"][str(nl)]
            if None not in vals:
                results["nlls"][k]["mean"].append(float(np.mean(vals)))
                results["nlls"][k]["std"].append(float(np.std(vals)))
            else:
                results["nlls"][k]["mean"].append(None)
                results["nlls"][k]["std"].append(None)

    # Load DL results to do Wilcoxon tests
    if os.path.exists(detailed_json_path):
        with open(detailed_json_path, 'r') as f:
            dl_results = json.load(f)

        # FIX: methods are nested under 'methods' key, not top-level
        if 'methods' in dl_results:
            dl_methods_data = dl_results['methods']
            dl_noise_levels = dl_results['noise_levels']
        else:
            # Legacy format: methods at top level
            dl_methods_data = {k: v for k, v in dl_results.items()
                               if k not in ('noise_levels', 'nlls', 'statistical_tests', 'nlls_sigmas')}
            dl_noise_levels = dl_results.get('noise_levels', [])

        methods = [m for m in dl_methods_data.keys() if m != 'nlls']

        for method in methods:
            test_key = f"{method}_vs_nlls"
            results["wilcoxon_tests"][test_key] = {"p_values": [], "significant": [], "noise_levels": []}
            for nl in noise_levels:
                nlls_scores = results["nlls"]["composite_total"]["per_patient"][str(nl)]
                if None in nlls_scores:
                    print(f"  Skipping {method} at sigma={nl}: NLLS not complete")
                    continue

                # FIX: Look up the correct DL noise index by matching sigma value
                try:
                    dl_noise_idx = dl_noise_levels.index(nl)
                except ValueError:
                    print(f"  Warning: sigma={nl} not found in DL noise levels, skipping")
                    continue

                dl_scores = dl_methods_data[method]["composite_total"]["per_patient"][dl_noise_idx]
                if len(dl_scores) != len(nlls_scores):
                    print(f"Warning: Patient count mismatch for {method} composite_total: "
                          f"DL={len(dl_scores)}, NLLS={len(nlls_scores)}")
                    min_len = min(len(dl_scores), len(nlls_scores))
                    dl_scores = dl_scores[:min_len]
                    nlls_scores = nlls_scores[:min_len]

                try:
                    stat, p = wilcoxon(nlls_scores, dl_scores)
                    results["wilcoxon_tests"][test_key]["p_values"].append(float(p))
                    results["wilcoxon_tests"][test_key]["significant"].append(bool(p < 0.05))
                    results["wilcoxon_tests"][test_key]["noise_levels"].append(nl)
                except Exception as e:
                    print(f"Wilcoxon failed for {method} at sigma={nl}: {e}")
                    results["wilcoxon_tests"][test_key]["p_values"].append(None)
                    results["wilcoxon_tests"][test_key]["significant"].append(None)
                    results["wilcoxon_tests"][test_key]["noise_levels"].append(nl)
    else:
        print(f"WARNING: {detailed_json_path} not found. Wilcoxon tests skipped.")
        print(f"         Run evaluate_comprehensive.py first to generate DL results.")

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("All done!")

if __name__ == '__main__':
    main()
