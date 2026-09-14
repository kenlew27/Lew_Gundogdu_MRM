import os
import json
import argparse
import time
import numpy as np
import torch
from scipy.stats import wilcoxon
from tqdm import tqdm

from utils import rRMSE_per_param, fit_biExponential_model
from PIA import PIA, CNN_PIA, UNetRefiner

try:
    from ivim_net import IVIM_NEToptim
    HAS_IVIM_NET = True
except ImportError:
    HAS_IVIM_NET = False

def noise_seed(patient_idx, noise_idx):
    """Deterministic seed for reproducible Rician noise per patient+noise combo.

    Ensures every method (DL and NLLS) sees identical noisy input for the same
    patient at the same noise level, enabling valid paired statistical tests.
    """
    return patient_idx * 10000 + noise_idx


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate IVIM estimation methods comprehensively")
    parser.add_argument("--quick", action="store_true", help="Use only 20 test patients and 4 noise levels for fast iteration")
    parser.add_argument("--skip-nlls", action="store_true", help="Skip NLLS entirely (it's very slow)")
    parser.add_argument("--nlls-sigmas", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2],
                        help="Noise levels at which to run NLLS (default: 0.02 0.05 0.1 0.2)")
    parser.add_argument("--skip-ivim-net", action="store_true", help="Skip IVIM-NET if checkpoint doesn't exist")
    parser.add_argument("--device", type=str, default="auto", help="Device: cpu or cuda (default: auto-detect)")
    parser.add_argument("--output", type=str, default="../results/evaluation_detailed.json", help="Output JSON path")
    return parser.parse_args()

def evaluate():
    args = get_args()
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        
    print(f"Using device: {device}")
    
    b_values = np.array([0, 5, 50, 100, 200, 500, 800, 1000])
    
    if args.quick:
        noise_levels = [1e-3, 0.01, 0.05, 0.1]
        patient_indices = range(801, 821)
    else:
        noise_levels = [1e-5, 1e-4, 1e-3, 0.005, 0.01, 0.02, 0.033, 0.05, 0.1, 0.15, 0.2, 0.25]
        patient_indices = range(801, 1001)
        
    data_dir = "../data/"
    
    # Compute average S(b=0) from TRAINING set for noise scaling.
    # Using training cases (1-600) avoids test-set information leakage.
    # The AAPM Grand Challenge defines noise as sigma_noise = sigma * avg_S0,
    # so that SNR = avg_S0 / sigma_noise = 1/sigma.
    print("Computing average S0 from training set for noise scaling...")
    s0_sum, s0_count = 0.0, 0
    for p_idx in range(1, 601):  # training cases only
        try:
            s = np.load(f"{data_dir}{p_idx:04d}_gtDWIs.npy")
            t = np.load(f"{data_dir}{p_idx:04d}_TissueType.npy")
            mask = (t != 1)  # non-air
            s0_sum += np.abs(s[:, :, 0][mask]).sum()
            s0_count += mask.sum()
        except FileNotFoundError:
            continue
    avg_S0 = s0_sum / s0_count if s0_count > 0 else 1.0
    print(f"  avg_S0 = {avg_S0:.6f} (over {s0_count:,} training voxels)"
          f" -- sigma=0.05 -> noise_std={0.05*avg_S0:.6f}, SNR={1/0.05:.0f}")
    
    # Load Models
    models = {}
    
    # MLP
    print("Loading MLP-PIA...")
    mlp_model = PIA(device=str(device)).to(device)
    if os.path.exists("../checkpoints/pia_baseline.pt"):
        mlp_model.load_state_dict(torch.load("../checkpoints/pia_baseline.pt", map_location=device, weights_only=False))
    mlp_model.eval()
    models["mlp"] = mlp_model
    
    # CNN
    print("Loading CNN-PIA...")
    cnn_model = CNN_PIA(device=str(device)).to(device)
    if os.path.exists("../checkpoints/pia_cnn_best.pt"):
        cnn_model.load_state_dict(torch.load("../checkpoints/pia_cnn_best.pt", map_location=device, weights_only=False))
    cnn_model.eval()
    models["cnn"] = cnn_model
    
    # MLP Refiner
    print("Loading MLP-Refiner...")
    mlp_refiner = UNetRefiner().to(device)
    if os.path.exists("../checkpoints/refiner_mlp_e2e_best.pt"):
        ckpt = torch.load("../checkpoints/refiner_mlp_e2e_best.pt", map_location=device, weights_only=False)
        state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        mlp_refiner.load_state_dict(state)
    mlp_refiner.eval()
    models["mlp_ref"] = mlp_refiner
    
    # CNN Refiner
    print("Loading CNN-Refiner...")
    cnn_refiner = UNetRefiner().to(device)
    if os.path.exists("../checkpoints/refiner_cnn_e2e_best.pt"):
        ckpt = torch.load("../checkpoints/refiner_cnn_e2e_best.pt", map_location=device, weights_only=False)
        state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        cnn_refiner.load_state_dict(state)
    cnn_refiner.eval()
    models["cnn_ref"] = cnn_refiner
    
    # IVIM-NET
    if HAS_IVIM_NET and not args.skip_ivim_net:
        print("Loading IVIM-NET...")
        ivim_net = IVIM_NEToptim(device=str(device)).to(device)
        if os.path.exists("../checkpoints/ivim_net_best.pt"):
            ivim_net.load_state_dict(torch.load("../checkpoints/ivim_net_best.pt", map_location=device, weights_only=False))
        ivim_net.eval()
        models["ivim_net"] = ivim_net
    
    # Determine which methods run at each noise level
    dl_methods = ["mlp", "cnn", "mlp_ref", "cnn_ref"]
    if HAS_IVIM_NET and not args.skip_ivim_net:
        dl_methods.append("ivim_net")

    nlls_sigmas_set = set(args.nlls_sigmas) if not args.skip_nlls else set()
    all_method_names = (["nlls"] if nlls_sigmas_set else []) + dl_methods

    metrics_keys = ['f_tumor', 'f_nontumor', 'Dt_tumor', 'Dt_nontumor', 'Dstar_tumor', 'Dstar_nontumor', 'composite_total', 'composite_tumor']

    results = {
        "noise_levels": noise_levels,
        "nlls_sigmas": sorted(s for s in nlls_sigmas_set if s in noise_levels) if nlls_sigmas_set else [],
        "methods": {m: {k: {"mean": [], "std": [], "per_patient": []} for k in metrics_keys} for m in all_method_names},
        "statistical_tests": {}
    }

    # Pre-populate statistical_tests for every DL method vs NLLS
    for m in dl_methods:
        results["statistical_tests"][f"{m}_vs_nlls"] = {"wilcoxon_p": [], "significant_005": [], "noise_levels": []}

    for noise_idx, sigma in enumerate(noise_levels):
        run_nlls = sigma in nlls_sigmas_set
        active_methods = (["nlls"] if run_nlls else []) + dl_methods
        print(f"\nProcessing noise level {sigma} ({noise_idx+1}/{len(noise_levels)})"
              + (" [+NLLS]" if run_nlls else ""))

        # Temporary storage for this noise level
        level_results = {m: {k: [] for k in metrics_keys} for m in active_methods}

        for p_idx in tqdm(patient_indices, desc="Patients"):
            # Try to load patient data
            try:
                prefix = f"{data_dir}{p_idx:04d}"
                signal = np.load(f"{prefix}_gtDWIs.npy")
                gt_params = np.load(f"{prefix}_IVIMParam.npy")
                tissue = np.load(f"{prefix}_TissueType.npy")

                # Extract gt params (IVIMParam has 3 channels: f, Dt, D*)
                gt_f = gt_params[:, :, 0]
                gt_dt = gt_params[:, :, 1]
                gt_ds = gt_params[:, :, 2]
            except FileNotFoundError:
                continue

            # Add Rician noise with deterministic seed for reproducibility.
            # Ground truth DWIs are complex128; take magnitude first since
            # the Rician model assumes real-valued (magnitude) input signal.
            # noise_std = sigma * avg_S0, so SNR = avg_S0 / noise_std = 1/sigma.
            signal_mag = np.abs(signal).astype(np.float64)
            rng = np.random.RandomState(noise_seed(p_idx, noise_idx))
            noise_std = sigma * avg_S0
            noise_re = rng.normal(0, noise_std, signal_mag.shape)
            noise_im = rng.normal(0, noise_std, signal_mag.shape)
            noisy_img = np.sqrt((signal_mag + noise_re)**2 + noise_im**2)
            
            # --- Inference for each method ---
            
            preds = {}
            
            if run_nlls:
                # NLLS (only at selected noise levels)
                pred_nlls = fit_biExponential_model(noisy_img, b_values)
                preds["nlls"] = (pred_nlls[:,:,0], pred_nlls[:,:,1], pred_nlls[:,:,2])
                
            with torch.no_grad():
                # MLP
                samples = noisy_img.reshape(-1, 8)
                s0 = np.clip(samples[:, 0:1], 1e-8, None)
                samples = samples / s0
                samples_t = torch.from_numpy(samples).float().to(device)
                
                _, _, f_mlp, Dt_mlp, Dstar_mlp = models["mlp"](samples_t)
                f_mlp_np = f_mlp.cpu().numpy().reshape(200, 200)
                Dt_mlp_np = Dt_mlp.cpu().numpy().reshape(200, 200)
                Dstar_mlp_np = Dstar_mlp.cpu().numpy().reshape(200, 200)
                preds["mlp"] = (f_mlp_np, Dt_mlp_np, Dstar_mlp_np)
                
                # CNN
                noisy_norm = noisy_img / np.clip(noisy_img[:, :, 0:1], 1e-8, None)
                input_4d = torch.from_numpy(noisy_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
                f_cnn, Dt_cnn, Dstar_cnn = models["cnn"](input_4d)
                f_cnn_np = f_cnn.squeeze().cpu().numpy()
                Dt_cnn_np = Dt_cnn.squeeze().cpu().numpy()
                Dstar_cnn_np = Dstar_cnn.squeeze().cpu().numpy()
                preds["cnn"] = (f_cnn_np, Dt_cnn_np, Dstar_cnn_np)
                
                # Refiners
                def refine(f, Dt, Ds, refiner):
                    f_norm = (f - 0.005) / 0.395
                    Dt_norm = (Dt - 0.0001) / 0.0024
                    Ds_norm = (Ds - 0.002) / 0.063
                    param_input = torch.tensor(np.stack([f_norm, Dt_norm, Ds_norm]), dtype=torch.float32).unsqueeze(0).to(device)
                    refined = refiner(param_input).squeeze(0).cpu().numpy()
                    f_ref = np.clip(refined[0] * 0.395 + 0.005, 0.0, 0.45)
                    Dt_ref = np.clip(refined[1] * 0.0024 + 0.0001, 0.0, 0.003)
                    Ds_ref = np.clip(refined[2] * 0.063 + 0.002, 0.0, 0.07)
                    return f_ref, Dt_ref, Ds_ref
                    
                preds["mlp_ref"] = refine(f_mlp_np, Dt_mlp_np, Dstar_mlp_np, models["mlp_ref"])
                preds["cnn_ref"] = refine(f_cnn_np, Dt_cnn_np, Dstar_cnn_np, models["cnn_ref"])
                
                # IVIM-NET
                if "ivim_net" in dl_methods:
                    _, f_ivim, Dt_ivim, Dstar_ivim = models["ivim_net"](samples_t)
                    preds["ivim_net"] = (f_ivim.cpu().numpy().reshape(200, 200), 
                                         Dt_ivim.cpu().numpy().reshape(200, 200), 
                                         Dstar_ivim.cpu().numpy().reshape(200, 200))
                                         
            # Calculate metrics
            for m in active_methods:
                p_f, p_dt, p_ds = preds[m]
                metrics = rRMSE_per_param(p_f, p_dt, p_ds, gt_f, gt_dt, gt_ds, tissue)
                for k in metrics_keys:
                    level_results[m][k].append(metrics[k])
                    
        # Aggregate stats for DL methods (always present)
        for m in dl_methods:
            for k in metrics_keys:
                arr = level_results[m][k]
                results["methods"][m][k]["per_patient"].append(arr)
                results["methods"][m][k]["mean"].append(float(np.mean(arr)) if arr else 0.0)
                results["methods"][m][k]["std"].append(float(np.std(arr)) if arr else 0.0)

        # Aggregate NLLS (only at noise levels where it ran)
        if run_nlls and "nlls" in results["methods"]:
            for k in metrics_keys:
                arr = level_results["nlls"][k]
                results["methods"]["nlls"][k]["per_patient"].append(arr)
                results["methods"]["nlls"][k]["mean"].append(float(np.mean(arr)) if arr else 0.0)
                results["methods"]["nlls"][k]["std"].append(float(np.std(arr)) if arr else 0.0)

        # Wilcoxon signed-rank tests (only at noise levels with NLLS)
        if run_nlls:
            nlls_composite = level_results["nlls"]["composite_total"]
            if nlls_composite:
                for m in dl_methods:
                    m_composite = level_results[m]["composite_total"]
                    test_key = f"{m}_vs_nlls"
                    try:
                        stat, p = wilcoxon(m_composite, nlls_composite)
                        results["statistical_tests"][test_key]["wilcoxon_p"].append(float(p))
                        results["statistical_tests"][test_key]["significant_005"].append(bool(p < 0.05))
                        results["statistical_tests"][test_key]["noise_levels"].append(sigma)
                    except ValueError:
                        results["statistical_tests"][test_key]["wilcoxon_p"].append(1.0)
                        results["statistical_tests"][test_key]["significant_005"].append(False)
                        results["statistical_tests"][test_key]["noise_levels"].append(sigma)

    # --- Save results ---
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Also write NLLS subset results in standalone format for backward compatibility
    if nlls_sigmas_set and "nlls" in results["methods"]:
        # Only include sigmas that were actually in the noise schedule
        actual_nlls_sigmas = sorted(s for s in nlls_sigmas_set if s in noise_levels)
        nlls_subset = {
            "noise_levels": actual_nlls_sigmas,
            "patient_indices": list(patient_indices),
            "deterministic_seed": True,
            "nlls": {},
            "wilcoxon_tests": {}
        }
        for k in metrics_keys:
            nlls_data = results["methods"]["nlls"][k]
            nlls_subset["nlls"][k] = {
                "mean": nlls_data["mean"],
                "std": nlls_data["std"],
                "per_patient": {str(s): nlls_data["per_patient"][i]
                                for i, s in enumerate(actual_nlls_sigmas)}
            }
        # Copy Wilcoxon tests
        for test_key, test_data in results["statistical_tests"].items():
            nlls_subset["wilcoxon_tests"][test_key] = {
                "p_values": test_data["wilcoxon_p"],
                "significant": test_data["significant_005"],
                "noise_levels": test_data["noise_levels"]
            }
        nlls_path = os.path.join(os.path.dirname(args.output) or '.', "nlls_subset_results.json")
        with open(nlls_path, "w") as f:
            json.dump(nlls_subset, f, indent=2)
        print(f"NLLS subset results saved to {nlls_path}")

    print(f"\nEvaluation complete. Results saved to {args.output}")

if __name__ == "__main__":
    evaluate()
