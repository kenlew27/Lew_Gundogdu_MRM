"""Master retraining script with checkpoint/resume awareness.

Runs models in sequence, skipping any that have already completed
(have a best checkpoint but no training checkpoint). If a model was
interrupted mid-training, it will auto-resume from its last checkpoint.

Usage:
    python retrain_all.py            # Full training
    python retrain_all.py --quick    # Quick smoke test (20 patients, 5 epochs)
    python retrain_all.py --only 2   # Run only step 2 (MLP-PIA)

Data split: 600 train (1-600) / 200 val (601-800) / 200 test (801-1000)
"""
import subprocess
import sys
import os
import time

STEPS = [
    {
        'cmd_base': 'train_ivim_net.py',
        'desc': '1/5 IVIM-NET baseline (self-supervised)',
        'best_ckpt': '../checkpoints/ivim_net_best.pt',
        'resume_ckpt': '../checkpoints/ivim_net_training.pt',
    },
    {
        'cmd_base': 'train_pia_mlp.py',
        'desc': '2/5 MLP-PIA Stage 1 (self-supervised)',
        'best_ckpt': '../checkpoints/pia_baseline.pt',
        'resume_ckpt': '../checkpoints/pia_baseline_training.pt',
    },
    {
        'cmd_base': 'train_pia_cnn.py',
        'desc': '3/5 CNN-PIA Stage 1 (self-supervised)',
        'best_ckpt': '../checkpoints/pia_cnn_best.pt',
        'resume_ckpt': '../checkpoints/pia_cnn_training.pt',
    },
    {
        'cmd_base': 'train_refiner.py --stage1 mlp',
        'desc': '4/5 UNet Refiner on MLP-PIA (supervised)',
        'best_ckpt': '../checkpoints/refiner_mlp_e2e_best.pt',
        'resume_ckpt': '../checkpoints/refiner_mlp_training.pt',
    },
    {
        'cmd_base': 'train_refiner.py --stage1 cnn',
        'desc': '5/5 UNet Refiner on CNN-PIA (supervised)',
        'best_ckpt': '../checkpoints/refiner_cnn_e2e_best.pt',
        'resume_ckpt': '../checkpoints/refiner_cnn_training.pt',
    },
]

def is_completed(step):
    """A step is completed if it has a best checkpoint but no resume checkpoint."""
    return os.path.exists(step['best_ckpt']) and not os.path.exists(step['resume_ckpt'])

def run(cmd, desc):
    print(f"\n{'='*70}")
    print(f"STARTING: {desc}")
    print(f"Command: {cmd}")
    print(f"{'='*70}\n")
    t0 = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"\n[{status}] {desc} completed in {elapsed/60:.1f} min\n")
    return result.returncode == 0

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--only', type=int, help="Run only step N (1-5)")
    args = parser.parse_args()

    quick = " --quick" if args.quick else ""
    epochs_s1 = " --epochs 5" if args.quick else ""
    epochs_s2 = " --epochs 5" if args.quick else ""

    results = []
    for i, step in enumerate(STEPS, 1):
        if args.only and args.only != i:
            continue

        if is_completed(step):
            print(f"\nSKIPPED: {step['desc']} (already completed)")
            results.append((step['desc'], 'SKIP'))
            continue

        has_resume = os.path.exists(step['resume_ckpt'])
        if has_resume:
            print(f"\nRESUMING: {step['desc']} (found training checkpoint)")

        ep_flag = epochs_s2 if 'refiner' in step['cmd_base'] else epochs_s1
        cmd = f"{sys.executable} {step['cmd_base']}{ep_flag}{quick}"
        ok = run(cmd, step['desc'])
        results.append((step['desc'], 'PASS' if ok else 'FAIL'))

        if not ok:
            print(f"\nWARNING: {step['desc']} failed. Run again to resume.\n")
            break  # Stop on failure since downstream steps depend on this

    print("\n" + "="*70)
    print("RETRAINING SUMMARY")
    print("="*70)
    for desc, status in results:
        print(f"  {status}: {desc}")

    all_done = all(is_completed(s) for s in STEPS)
    if all_done:
        print(f"\nAll models trained! Next: run evaluation pipeline.")
    else:
        remaining = [s['desc'] for s in STEPS if not is_completed(s)]
        print(f"\nRemaining: {len(remaining)} step(s). Run 'python retrain_all.py' to continue.")

if __name__ == '__main__':
    main()
