# PIA-IVIM: Physics-Informed Deep Learning for Breast IVIM Parameter Estimation

Two-stage physics-informed deep learning pipeline for noise-robust intravoxel incoherent motion (IVIM) parameter estimation in breast diffusion-weighted MRI.

> **Paper:** *Physics-informed deep learning for noise-robust intravoxel incoherent motion estimation in breast MRI*
>
> Ken Lew, Batuhan Gundogdu

Evaluated on 200 held-out test cases (801–1000) from the [AAPM 2024 IVIM-dMRI Grand Challenge](https://www.aapm.org/GrandChallenge/IVIM-dMRI/) VICTRE digital breast phantom dataset.

## Results

| Method | Composite Total rRMSE (SNR 20) | Composite Tumor rRMSE (SNR 20) | GPU Latency |
|--------|-------------------------------|-------------------------------|-------------|
| **CNN-PIA + Refiner** | **0.087 ± 0.039** | **0.052 ± 0.038** | **5.88 ms** |
| MLP-PIA + Refiner | 0.095 ± 0.039 | 0.050 ± 0.037 | 18.36 ms |
| CNN-PIA | 0.616 ± 0.051 | 0.386 ± 0.046 | 1.56 ms |
| MLP-PIA | 0.624 ± 0.050 | 0.384 ± 0.047 | 16.58 ms |
| NLLS (baseline) | 0.892 ± 0.091 | 0.529 ± 0.086 | N/A (187.5 s CPU) |

## Repository Structure

```
├── src/                     Source code
│   ├── PIA.py               Stage 1 autoencoders (MLP-PIA, CNN-PIA) + Stage 2 UNet Refiner
│   ├── ivim_net.py           IVIM-NET baseline (Barbieri et al.)
│   ├── utils.py              Data loading, rRMSE metrics, NLLS fitting, signal model
│   ├── train_pia_mlp.py      Train MLP-PIA (Stage 1)
│   ├── train_pia_cnn.py      Train CNN-PIA (Stage 1)
│   ├── train_ivim_net.py     Train IVIM-NET baseline
│   ├── train_refiner.py      Train UNet Refiner (Stage 2)
│   ├── retrain_all.py        Master training orchestrator
│   ├── evaluate_comprehensive.py   Full evaluation (12 noise levels, 200 test cases)
│   ├── evaluate_nlls_subset.py     NLLS benchmark (30 cases, 4 noise levels)
│   ├── plot_main_figures.py        Main paper figures
│   └── plot_supplementary_figures.py  Supplementary figures
├── checkpoints/             Trained model weights (included)
├── results/                 Evaluation outputs (JSON)
├── figures/                 Generated figures (PNG + PDF)
├── data/                    VICTRE phantom data (NOT included — see below)
├── docs/                    Additional documentation
├── requirements.txt         Python dependencies
└── LICENSE                  MIT License
```

## Setup

### Prerequisites

- Python ≥ 3.10
- NVIDIA GPU recommended (CUDA-compatible); CPU-only is supported but slower

### Installation

```bash
git clone https://github.com/kenlew27/Lew_Gundogdu_MRM.git
cd Lew_Gundogdu_MRM
pip install -r requirements.txt
```

### Data

The VICTRE phantom dataset is provided by the AAPM 2024 IVIM-dMRI Grand Challenge and is **not redistributed** in this repository per the challenge data-use terms.

1. Register at [https://www.aapm.org/GrandChallenge/IVIM-dMRI/](https://www.aapm.org/GrandChallenge/IVIM-dMRI/)
2. Download the 1,000-case training dataset
3. Place the `.npy` files in the `data/` directory:
   - `XXXX_IVIMParam.npy` — Ground truth IVIM parameters (f, Dt, D*)
   - `XXXX_NoisyDWIk.npy` — Noisy k-space DWI data
   - `XXXX_TissueType.npy` — Tissue segmentation masks
   - `XXXX_gtDWIs.npy` — Ground truth clean DWI signals

### Data Split

| Split | Cases | Purpose |
|-------|-------|---------|
| Training | 0001–0600 | Model training |
| Validation | 0601–0800 | Checkpoint selection |
| Test | 0801–1000 | Held-out evaluation (all reported metrics) |

## Reproducing Results

All commands should be run from the `src/` directory:

```bash
cd src
```

### 1. Training (optional — pretrained weights included in `checkpoints/`)

```bash
# Train all models sequentially (MLP-PIA → CNN-PIA → Refiners)
python retrain_all.py

# Or train individual models:
python train_pia_mlp.py --epochs 500 --seed 42
python train_pia_cnn.py --epochs 500 --seed 42
python train_refiner.py --stage1 mlp --epochs 200 --seed 42
python train_refiner.py --stage1 cnn --epochs 200 --seed 42
```

### 2. Evaluation

```bash
# Full deep learning evaluation (200 test cases × 12 noise levels)
python evaluate_comprehensive.py

# NLLS benchmark (200 test cases × 4 noise levels) — CPU-intensive
python evaluate_nlls_subset.py
```

Results are saved to `results/evaluation_detailed.json` and `results/nlls_subset_results.json`.

### 3. Figure Generation

```bash
python generate_all_figures.py
```

Generates all main and supplementary figures to `figures/` and `new_paper/figures/`.

## Trained Checkpoints

All weights are included in `checkpoints/`:

| File | Model | Size |
|------|-------|------|
| `ivim_net_best.pt` | IVIM-NET (baseline) | 24 KB |
| `pia_baseline.pt` | MLP-PIA (Stage 1) | 3.9 MB |
| `pia_cnn_best.pt` | CNN-PIA (Stage 1) | 763 KB |
| `refiner_mlp_e2e_best.pt` | MLP-PIA + UNet Refiner | 13.4 MB |
| `refiner_cnn_e2e_best.pt` | CNN-PIA + UNet Refiner | 13.4 MB |

All models were trained with random seed 42 (`torch.manual_seed(42)`, `numpy.random.seed(42)`, `torch.cuda.manual_seed_all(42)`).

## Reproducibility

- **Seeds**: All training scripts default to `--seed 42`
- **Evaluation noise**: Deterministic per-case seeding (`seed = patient_idx × 10000 + noise_idx`)
- **avg S₀**: Computed from training cases 1–600 only (0.266081) to avoid test-set leakage

## Citation

```bibtex
@article{lew2026pia_ivim,
  title={Physics-informed deep learning for noise-robust intravoxel incoherent motion estimation in breast {MRI}},
  author={Lew, Ken and Gundogdu, Batuhan},
  journal={Magnetic Resonance in Medicine},
  year={2026}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- VICTRE digital breast phantom platform (U.S. FDA)
- AAPM 2024 Quantitative IVIM-dMRI Reconstruction Grand Challenge
- Original PIA framework: [batuhan-gundogdu/PIA_IVIM](https://github.com/batuhan-gundogdu/PIA_IVIM)
