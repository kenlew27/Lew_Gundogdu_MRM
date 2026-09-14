"""Train CNN-PIA (Stage 1) on 600-case training set with checkpoint/resume.

Self-supervised: physics decoder loss. Auto-resumes from last checkpoint.

Usage:
    python train_pia_cnn.py [--epochs 500] [--lr 3e-4] [--quick]
"""
import argparse
import os
import numpy as np
import torch
from torch import optim
from PIA import CNN_PIA
from utils import read_data

CKPT_BEST = '../checkpoints/pia_cnn_best.pt'
CKPT_RESUME = '../checkpoints/pia_cnn_training.pt'

def save_training_state(model, optimizer, epoch, best_val):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val,
    }, CKPT_RESUME)

def main():
    parser = argparse.ArgumentParser(description="Train CNN-PIA (Stage 1)")
    parser.add_argument('--epochs', type=int, default=500, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate")
    parser.add_argument('--quick', action='store_true', help="Quick mode (20 patients, fewer epochs)")
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

    model = CNN_PIA(b_values=b_values, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs('../checkpoints', exist_ok=True)
    best_val_loss = float('inf')
    start_epoch = 0

    # Resume from checkpoint
    if os.path.exists(CKPT_RESUME):
        ckpt = torch.load(CKPT_RESUME, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']
        print(f"Resumed from epoch {start_epoch} (best_val_loss={best_val_loss:.6f})")

    for ep in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        train_count = 0

        for case_idx in range(1, num_train + 1):
            tissue = read_data(file_dir, '_TissueType.npy', case_idx)
            k = read_data(file_dir, '_NoisyDWIk.npy', case_idx)
            noisy = np.abs(np.fft.ifft2(k, axes=(0, 1), norm='ortho'))

            body_mask = (tissue != 1).astype(np.float32)
            S0_img = noisy[:, :, 0:1].copy()
            S0_img[S0_img == 0] = 1.0
            noisy_norm = noisy / S0_img

            signal = torch.from_numpy(noisy_norm.real.transpose(2, 0, 1)[np.newaxis]).to(device).float()
            mask = torch.from_numpy(body_mask[np.newaxis, np.newaxis]).to(device).float()

            f, Dt, Dstar = model(signal)
            S_pred = model.decode(f, Dt, Dstar)
            loss = torch.mean(mask * (S_pred - signal) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_count += 1

        avg_train = train_loss / train_count if train_count > 0 else 0
        print(f"Epoch {ep+1}/{args.epochs}, Train Loss: {avg_train:.6f}")

        # Validation every 10 epochs
        if (ep + 1) % 10 == 0:
            model.eval()
            val_loss = 0.0
            val_count = 0

            with torch.no_grad():
                for case_idx in range(601, 601 + num_val):
                    try:
                        tissue = read_data(file_dir, '_TissueType.npy', case_idx)
                        k = read_data(file_dir, '_NoisyDWIk.npy', case_idx)
                    except Exception:
                        continue
                    noisy = np.abs(np.fft.ifft2(k, axes=(0, 1), norm='ortho'))
                    body_mask = (tissue != 1).astype(np.float32)

                    S0_img = noisy[:, :, 0:1].copy()
                    S0_img[S0_img == 0] = 1.0
                    noisy_norm = noisy / S0_img

                    signal = torch.from_numpy(noisy_norm.real.transpose(2, 0, 1)[np.newaxis]).to(device).float()
                    mask = torch.from_numpy(body_mask[np.newaxis, np.newaxis]).to(device).float()

                    f, Dt, Dstar = model(signal)
                    S_pred = model.decode(f, Dt, Dstar)
                    loss = torch.mean(mask * (S_pred - signal) ** 2)

                    val_loss += loss.item()
                    val_count += 1

            avg_val = val_loss / val_count if val_count > 0 else float('inf')
            print(f"  Val Loss: {avg_val:.6f}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(model.state_dict(), CKPT_BEST)
                print(f"  Saved new best model (val_loss={avg_val:.6f})")

        # Save training state for resume
        save_training_state(model, optimizer, ep, best_val_loss)

    # Ensure checkpoint exists and cleanup
    if not os.path.exists(CKPT_BEST):
        torch.save(model.state_dict(), CKPT_BEST)
        print("Saved final model (no validation improvement)")
    if os.path.exists(CKPT_RESUME):
        os.remove(CKPT_RESUME)
    print("Training complete.")

if __name__ == '__main__':
    main()
