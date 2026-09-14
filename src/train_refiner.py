"""Train UNet Refiner (Stage 2) with checkpoint/resume.

Supervised: requires ground truth + pre-trained Stage 1. Auto-resumes.

Usage:
    python train_refiner.py --stage1 mlp [--epochs 200] [--lr 1e-4] [--quick]
    python train_refiner.py --stage1 cnn [--epochs 200] [--lr 1e-4] [--quick]
"""
import argparse
import os
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from PIA import PIA, CNN_PIA, UNetRefiner
from utils import read_data, rRMSE_per_case

def main():
    parser = argparse.ArgumentParser(description="Train UNet Refiner (Stage 2)")
    parser.add_argument('--stage1', type=str, required=True, choices=['mlp', 'cnn'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--tumor_weight', type=float, default=5.0)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    num_train = 20 if args.quick else 600
    num_val = 20 if args.quick else 200

    if args.stage1 == 'mlp':
        CKPT_BEST = '../checkpoints/refiner_mlp_e2e_best.pt'
        CKPT_RESUME = '../checkpoints/refiner_mlp_training.pt'
        STAGE1_CKPT = '../checkpoints/pia_baseline.pt'
    else:
        CKPT_BEST = '../checkpoints/refiner_cnn_e2e_best.pt'
        CKPT_RESUME = '../checkpoints/refiner_cnn_training.pt'
        STAGE1_CKPT = '../checkpoints/pia_cnn_best.pt'

    file_dir = '../data/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    b_values = [0, 5, 50, 100, 200, 500, 800, 1000]

    # Load pre-trained Stage 1 (frozen)
    if args.stage1 == 'mlp':
        stage1 = PIA(b_values=b_values, device=device).to(device)
    else:
        stage1 = CNN_PIA(b_values=b_values, device=device).to(device)
    stage1.load_state_dict(torch.load(STAGE1_CKPT, map_location=device, weights_only=False))
    stage1.eval()
    for p in stage1.parameters():
        p.requires_grad = False

    # Refiner
    refiner = UNetRefiner().to(device)
    optimizer = optim.Adam(refiner.parameters(), lr=args.lr)
    NORM = UNetRefiner.NORM

    os.makedirs('../checkpoints', exist_ok=True)
    best_val_rrmse = float('inf')
    start_epoch = 0

    # Resume from checkpoint
    if os.path.exists(CKPT_RESUME):
        ckpt = torch.load(CKPT_RESUME, map_location=device, weights_only=False)
        refiner.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_rrmse = ckpt['best_val_rrmse']
        print(f"Resumed from epoch {start_epoch} (best_val_rrmse={best_val_rrmse:.4f})")

    def normalize_params(f, Dt, Dstar):
        f_n = (f - NORM['f']['min']) / NORM['f']['range']
        Dt_n = (Dt - NORM['Dt']['min']) / NORM['Dt']['range']
        Ds_n = (Dstar - NORM['Dstar']['min']) / NORM['Dstar']['range']
        return torch.cat([f_n, Dt_n, Ds_n], dim=1)

    def denormalize_params(refined):
        f_out = torch.clamp(refined[:, 0:1] * NORM['f']['range'] + NORM['f']['min'], 0.0, 0.45)
        Dt_out = torch.clamp(refined[:, 1:2] * NORM['Dt']['range'] + NORM['Dt']['min'], 0.0, 0.003)
        Ds_out = torch.clamp(refined[:, 2:3] * NORM['Dstar']['range'] + NORM['Dstar']['min'], 0.0, 0.07)
        return f_out, Dt_out, Ds_out

    def run_stage1(noisy_norm):
        with torch.no_grad():
            if args.stage1 == 'cnn':
                signal = torch.from_numpy(noisy_norm.real.transpose(2, 0, 1)[np.newaxis]).to(device).float()
                return stage1(signal)
            else:
                samples = noisy_norm.real.reshape(-1, 8)
                signal_t = torch.from_numpy(samples).to(device).float()
                _, _, f_flat, Dt_flat, Ds_flat = stage1(signal_t)
                H, W = noisy_norm.shape[0], noisy_norm.shape[1]
                return f_flat.view(1, 1, H, W), Dt_flat.view(1, 1, H, W), Ds_flat.view(1, 1, H, W)

    for ep in range(start_epoch, args.epochs):
        refiner.train()
        train_loss = 0.0
        train_count = 0

        for case_idx in range(1, num_train + 1):
            tissue = read_data(file_dir, '_TissueType.npy', case_idx)
            k = read_data(file_dir, '_NoisyDWIk.npy', case_idx)
            gt_params = read_data(file_dir, '_IVIMParam.npy', case_idx)
            noisy = np.abs(np.fft.ifft2(k, axes=(0, 1), norm='ortho'))

            body_mask = (tissue != 1).astype(np.float32)
            tumor_mask = (tissue == 8).astype(np.float32)
            weight_map = body_mask + (args.tumor_weight - 1.0) * tumor_mask
            weight_tensor = torch.from_numpy(weight_map[np.newaxis, np.newaxis]).to(device).float()

            gt_f = torch.from_numpy(gt_params[:,:,0].real[np.newaxis, np.newaxis]).to(device).float()
            gt_Dt = torch.from_numpy(gt_params[:,:,1].real[np.newaxis, np.newaxis]).to(device).float()
            gt_Ds = torch.from_numpy(gt_params[:,:,2].real[np.newaxis, np.newaxis]).to(device).float()
            gt_norm = normalize_params(gt_f, gt_Dt, gt_Ds)

            S0_img = noisy[:, :, 0:1].copy()
            S0_img[S0_img == 0] = 1.0
            noisy_norm = noisy / S0_img

            f_s1, Dt_s1, Ds_s1 = run_stage1(noisy_norm)
            s1_norm = normalize_params(f_s1, Dt_s1, Ds_s1)
            refined = refiner(s1_norm)

            loss = torch.mean(weight_tensor * (refined - gt_norm) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_count += 1

        avg_train = train_loss / train_count if train_count > 0 else 0
        print(f"Epoch {ep+1}/{args.epochs}, Train Loss: {avg_train:.6f}")

        # Validation every 5 epochs
        if (ep + 1) % 5 == 0:
            refiner.eval()
            val_rrmse_sum = 0.0
            val_count = 0

            with torch.no_grad():
                for case_idx in range(601, 601 + num_val):
                    try:
                        tissue = read_data(file_dir, '_TissueType.npy', case_idx)
                        k = read_data(file_dir, '_NoisyDWIk.npy', case_idx)
                        gt_params = read_data(file_dir, '_IVIMParam.npy', case_idx)
                    except Exception:
                        continue

                    noisy = np.abs(np.fft.ifft2(k, axes=(0, 1), norm='ortho'))
                    S0_img = noisy[:, :, 0:1].copy()
                    S0_img[S0_img == 0] = 1.0
                    noisy_norm = noisy / S0_img

                    f_s1, Dt_s1, Ds_s1 = run_stage1(noisy_norm)
                    s1_norm = normalize_params(f_s1, Dt_s1, Ds_s1)
                    refined = refiner(s1_norm)
                    f_out, Dt_out, Ds_out = denormalize_params(refined)

                    rrmse, _ = rRMSE_per_case(
                        f_out[0, 0].cpu().numpy(), Dt_out[0, 0].cpu().numpy(),
                        Ds_out[0, 0].cpu().numpy(),
                        gt_params[:,:,0].real, gt_params[:,:,1].real,
                        gt_params[:,:,2].real, tissue
                    )
                    val_rrmse_sum += rrmse
                    val_count += 1

            avg_rrmse = val_rrmse_sum / val_count if val_count > 0 else float('inf')
            print(f"  Val rRMSE: {avg_rrmse:.4f}")

            if avg_rrmse < best_val_rrmse:
                best_val_rrmse = avg_rrmse
                torch.save(refiner.state_dict(), CKPT_BEST)
                print(f"  Saved new best refiner (rRMSE={avg_rrmse:.4f})")

        # Save training state for resume
        torch.save({
            'epoch': ep,
            'model_state_dict': refiner.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_rrmse': best_val_rrmse,
        }, CKPT_RESUME)

    # Ensure checkpoint exists and cleanup
    if not os.path.exists(CKPT_BEST):
        torch.save(refiner.state_dict(), CKPT_BEST)
        print("Saved final refiner (no validation improvement)")
    if os.path.exists(CKPT_RESUME):
        os.remove(CKPT_RESUME)
    print("Training complete.")

if __name__ == '__main__':
    main()
