"""Train IVIM-NEToptim on 600-case training set with checkpoint/resume support.

Self-supervised: physics decoder loss. Auto-resumes from last checkpoint if interrupted.

Usage:
    python train_ivim_net.py [--epochs 200] [--lr 3e-4] [--quick]
"""
import argparse
import os
import numpy as np
import torch
from torch import optim
from ivim_net import IVIM_NEToptim
from utils import read_data, rRMSE_per_case

CKPT_BEST = '../checkpoints/ivim_net_best.pt'
CKPT_RESUME = '../checkpoints/ivim_net_training.pt'

def save_training_state(model, optimizer, epoch, best_val):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_score': best_val,
    }, CKPT_RESUME)

def main():
    parser = argparse.ArgumentParser(description="Train IVIM-NEToptim")
    parser.add_argument('--epochs', type=int, default=200, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate")
    parser.add_argument('--quick', action='store_true', help="Use quick mode (20 patients instead of 600)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    num_train = 20 if args.quick else 600
    num_val = 20 if args.quick else 200

    file_dir = '../data/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    b_values = [0, 5, 50, 100, 200, 500, 800, 1000]

    model = IVIM_NEToptim(b_values=b_values, device=device).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs('../checkpoints', exist_ok=True)
    best_val_score = float('inf')
    start_epoch = 0

    # Resume from checkpoint if available
    if os.path.exists(CKPT_RESUME):
        ckpt = torch.load(CKPT_RESUME, map_location=device, weights_only=False)
        actual = model.module if isinstance(model, torch.nn.DataParallel) else model
        actual.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_score = ckpt['best_val_score']
        print(f"Resumed from epoch {start_epoch} (best_val={best_val_score:.4f})")

    for ep in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        train_count = 0

        for case_idx in range(1, num_train + 1):
            tissue = read_data(file_dir, '_TissueType.npy', case_idx)
            k = read_data(file_dir, '_NoisyDWIk.npy', case_idx)
            noisy = np.abs(np.fft.ifft2(k, axes=(0, 1), norm='ortho'))

            coordBody = np.argwhere(tissue != 1)
            if coordBody.shape[0] == 0:
                continue

            ix, jx = coordBody[:, 0], coordBody[:, 1]
            signal = noisy[ix, jx, :]
            S0 = signal[:, 0:1].copy()
            S0[S0 == 0] = 1.0
            signal = signal / S0

            signal_tensor = torch.from_numpy(signal).to(device).float()
            actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            S_pred, _, _, _ = actual_model(signal_tensor)
            loss = actual_model.loss_function(S_pred, signal_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * signal_tensor.shape[0]
            train_count += signal_tensor.shape[0]

        print(f"Epoch {ep+1}/{args.epochs}, Train Loss: {train_loss/train_count:.6f}")

        # Validation
        model.eval()
        tumor_rrmse_sum = 0
        non_tumor_rrmse_sum = 0
        test_cases_count = 0

        with torch.no_grad():
            for case_idx in range(601, 601 + num_val):
                try:
                    params = read_data(file_dir, '_IVIMParam.npy', case_idx)
                    tissue = read_data(file_dir, '_TissueType.npy', case_idx)
                    k = read_data(file_dir, '_NoisyDWIk.npy', case_idx)
                except Exception:
                    continue

                noisy = np.abs(np.fft.ifft2(k, axes=(0, 1), norm='ortho'))
                S0_img = noisy[:, :, 0:1].copy()
                S0_img[S0_img == 0] = 1.0
                noisy_norm = noisy / S0_img

                samples = noisy_norm.reshape(-1, 8)
                samples_tensor = torch.from_numpy(samples).to(device).float()
                actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model

                batch_size = 10000
                f_list, Dt_list, Ds_list = [], [], []
                for i in range(0, samples_tensor.shape[0], batch_size):
                    batch = samples_tensor[i:i+batch_size]
                    _, f_b, Dt_b, Ds_b = actual_model(batch)
                    f_list.append(f_b.detach().cpu())
                    Dt_list.append(Dt_b.detach().cpu())
                    Ds_list.append(Ds_b.detach().cpu())

                H, W = noisy.shape[0], noisy.shape[1]
                f_pred = torch.cat(f_list).numpy().reshape(H, W)
                Dt_pred = torch.cat(Dt_list).numpy().reshape(H, W)
                Ds_pred = torch.cat(Ds_list).numpy().reshape(H, W)

                t, nt = rRMSE_per_case(f_pred, Dt_pred, Ds_pred,
                                       params[:,:,0], params[:,:,1], params[:,:,2], tissue)
                tumor_rrmse_sum += t
                non_tumor_rrmse_sum += nt
                test_cases_count += 1

        if test_cases_count > 0:
            avg_tumor = tumor_rrmse_sum / test_cases_count
            avg_nt = non_tumor_rrmse_sum / test_cases_count
            print(f"Val rRMSE - Tumor: {avg_tumor:.4f}, Non-tumor: {avg_nt:.4f}")

            if avg_tumor < best_val_score:
                best_val_score = avg_tumor
                actual = model.module if isinstance(model, torch.nn.DataParallel) else model
                torch.save(actual.state_dict(), CKPT_BEST)
                print(f"Saved new best model to {CKPT_BEST}")

        # Save training state for resume
        save_training_state(model, optimizer, ep, best_val_score)

    # Cleanup resume checkpoint on successful completion
    if os.path.exists(CKPT_RESUME):
        os.remove(CKPT_RESUME)
    print("Training complete.")

if __name__ == '__main__':
    main()
