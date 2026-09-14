"""Full pipeline: retrain all models with corrected bounds, then evaluate everything.

This script orchestrates the complete pipeline:
  1. Retrain all 5 models (IVIM-NET, MLP-PIA, CNN-PIA, MLP-Refiner, CNN-Refiner)
  2. Run DL evaluation (200 test cases × 12 noise levels with correct S0-scaled noise)
  3. Run NLLS evaluation (200 test cases × 4 noise levels with correct S0-scaled noise)
  4. Print summary comparison

Usage:
    python run_full_pipeline.py           # Full pipeline
    python run_full_pipeline.py --skip-training  # Skip training, just evaluate
    python run_full_pipeline.py --quick   # Quick mode (20 patients, 5 epochs)
"""
import subprocess
import sys
import time
import os
import json
import numpy as np


def run(cmd, desc, cwd=None):
    print(f"\n{'='*70}")
    print(f"STEP: {desc}")
    print(f"CMD:  {cmd}")
    print(f"{'='*70}\n")
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    elapsed = time.time() - t0
    status = "PASS" if result.returncode == 0 else f"FAIL (code {result.returncode})"
    print(f"\n[{status}] {desc} — {elapsed/60:.1f} min\n")
    return result.returncode == 0


def summarize_results():
    """Print a final summary of all results."""
    eval_path = '../results/evaluation_detailed.json'
    nlls_path = '../results/nlls_subset_results.json'

    if not os.path.exists(eval_path):
        print("WARNING: evaluation_detailed.json not found — skipping summary")
        return

    with open(eval_path, 'r') as f:
        data = json.load(f)

    noise_levels = data['noise_levels']
    methods = data['methods']
    benchmark = [0.02, 0.05, 0.1, 0.2]
    labels = {0.02: 'SNR 50', 0.05: 'SNR 20', 0.1: 'SNR 10', 0.2: 'SNR 5'}

    print("\n" + "=" * 80)
    print("FINAL RESULTS: Composite Total rRMSE")
    print("=" * 80)
    header = f"{'Method':<20}"
    for nl in benchmark:
        header += f" {labels.get(nl, str(nl)):>12}"
    print(header)
    print("-" * 68)

    for m in ['cnn_ref', 'mlp_ref', 'cnn', 'mlp', 'ivim_net']:
        if m not in methods:
            continue
        row = f"{m:<20}"
        for nl in benchmark:
            if nl in noise_levels:
                idx = noise_levels.index(nl)
                val = methods[m]['composite_total']['mean'][idx]
                row += f" {val:>12.3f}"
            else:
                row += f" {'N/A':>12}"
        print(row)

    # NLLS results
    if os.path.exists(nlls_path):
        with open(nlls_path, 'r') as f:
            nlls = json.load(f)
        row = f"{'nlls':<20}"
        for nl in benchmark:
            vals = nlls['nlls']['composite_total']['per_patient'].get(str(nl), [])
            vals_clean = [v for v in vals if v is not None]
            if vals_clean:
                row += f" {np.mean(vals_clean):>12.3f}"
            else:
                row += f" {'N/A':>12}"
        print(row)

    print()
    print("=" * 80)
    print("Composite Tumor rRMSE")
    print("=" * 80)
    print(header)
    print("-" * 68)

    for m in ['cnn_ref', 'mlp_ref', 'cnn', 'mlp', 'ivim_net']:
        if m not in methods:
            continue
        row = f"{m:<20}"
        for nl in benchmark:
            if nl in noise_levels:
                idx = noise_levels.index(nl)
                val = methods[m]['composite_tumor']['mean'][idx]
                row += f" {val:>12.3f}"
            else:
                row += f" {'N/A':>12}"
        print(row)

    if os.path.exists(nlls_path):
        row = f"{'nlls':<20}"
        for nl in benchmark:
            vals = nlls['nlls']['composite_tumor']['per_patient'].get(str(nl), [])
            vals_clean = [v for v in vals if v is not None]
            if vals_clean:
                row += f" {np.mean(vals_clean):>12.3f}"
            else:
                row += f" {'N/A':>12}"
        print(row)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training, just evaluate existing checkpoints')
    parser.add_argument('--skip-nlls', action='store_true',
                        help='Skip NLLS evaluation (saves ~39 hours)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick smoke test (20 patients, 5 epochs)')
    args = parser.parse_args()

    py = sys.executable
    quick = " --quick" if args.quick else ""
    results = []
    t_total = time.time()

    # ---- 1. TRAINING ----
    if not args.skip_training:
        ok = run(f"{py} retrain_all.py{quick}",
                 "Retrain all models (IVIM-NET > MLP-PIA > CNN-PIA > Refiners)")
        results.append(("Training", "PASS" if ok else "FAIL"))
        if not ok:
            print("FATAL: Training failed. Cannot proceed.")
            return
    else:
        print("\n*** Skipping training (--skip-training) ***\n")
        results.append(("Training", "SKIP"))

    # ---- 2. DL EVALUATION ----
    ok = run(f"{py} evaluate_comprehensive.py --skip-nlls{quick}",
             "DL evaluation (200 patients × 12 noise levels, corrected noise)")
    results.append(("DL Evaluation", "PASS" if ok else "FAIL"))

    # ---- 3. NLLS EVALUATION ----
    if not args.skip_nlls:
        ok = run(f"{py} evaluate_nlls_subset.py",
                 "NLLS evaluation (200 patients × 4 noise levels, corrected noise)")
        results.append(("NLLS Evaluation", "PASS" if ok else "FAIL"))
    else:
        results.append(("NLLS Evaluation", "SKIP"))

    # ---- 4. SUMMARY ----
    elapsed_total = (time.time() - t_total) / 3600
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE — {elapsed_total:.1f} hours total")
    print(f"{'='*70}")
    for desc, status in results:
        print(f"  [{status}] {desc}")

    summarize_results()


if __name__ == '__main__':
    main()
