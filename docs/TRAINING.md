# Training Documentation

> Comprehensive hyperparameter and training procedure documentation for all models
> in the PIA-IVIM breast DW-MRI pipeline.

---

## Dataset

- **Source**: VICTRE (FDA Virtual Imaging Clinical Trials) digital breast phantoms
- **Total cases**: 1000 (200×200 pixels, 8 b-value DWI volumes)
- **Train/Val/Test split**: Cases 1–600 (training), Cases 601–800 (validation / checkpoint selection), Cases 801–1000 (held-out testing)
- **b-values**: [0, 5, 50, 100, 200, 500, 800, 1000] s/mm²
- **Ground truth**: Per-voxel IVIM parameters (f, Dt, D*) and tissue type labels
- **Noise**: Complex k-space Rician noise → magnitude via 2D iFFT

### Ground Truth Parameter Ranges

| Parameter | Description | Range | Unit |
|-----------|-------------|-------|------|
| f | Perfusion fraction | [0.009, 0.330] | — |
| Dt | Tissue diffusion coefficient | [0.000135, 0.00209] | mm²/s |
| D* | Pseudo-diffusion coefficient | [0.003, 0.060] | mm²/s |

---

## Model 1: MLP-PIA (Self-Supervised)

**File**: [`PIA.py`](../src/PIA.py) — `PIA` class  
**Checkpoint**: `checkpoints/pia_baseline.pt`  
**Parameters**: ~965K  

### Architecture
- **Encoder**: 5-layer MLP [8 → 32 → 64 → 128 → 256 → 512] with LeakyReLU
- **Prediction heads**: 3 separate branches (f, Dt, D*), each with 1 hidden layer (512→512→1)
- **Output activation**: Sigmoid-scaled bounded outputs
  - f: mean ± delta·tanh → [0.005, 0.400] (mean=0.2025, delta=0.1975)
  - Dt: mean ± delta·tanh → [0.0001, 0.0025] (mean=0.0013, delta=0.0012)
  - D*: mean ± delta·tanh → [0.002, 0.065] (mean=0.0335, delta=0.0315)
- **Decoder**: Physics-based bi-exponential model S(b) = (1-f)·exp(-b·Dt) + f·exp(-b·D*)

### Training
- **Training mode**: Self-supervised (signal reconstruction loss, no ground truth labels needed)
- **Loss**: MSE between input signal and physics-decoded reconstruction
- **Optimizer**: Adam, lr=3×10⁻⁴
- **Epochs**: 500
- **Batch size**: Voxel-wise (one voxel at a time per training step)
- **Preprocessing**: Normalize by S(b=0), mask non-body voxels (tissue ≠ 1)
- **Training data**: k-space → magnitude via 2D iFFT → normalize
- **Training time**: ~12 hours on Quadro T2000

---

## Model 2: CNN-PIA (Self-Supervised)

**File**: [`PIA.py`](../src/PIA.py) — `CNN_PIA` class  
**Checkpoint**: `checkpoints/pia_cnn_best.pt`  
**Parameters**: ~189K  

### Architecture
- **Encoder**: 4-layer Conv2d [8→64→64→128→64] with 3×3 kernels, padding=1, LeakyReLU
- **Prediction heads**: Three 1×1 Conv2d layers (64→1) for f, Dt, D*
- **Output activation**: Sigmoid-scaled bounded outputs (same ranges as MLP-PIA)
- **Decoder**: Same physics-based bi-exponential model

### Training
- **Training mode**: Self-supervised (signal reconstruction loss)
- **Loss**: MSE signal reconstruction + optional tumor-weighted MSE
- **Optimizer**: Adam, lr=3×10⁻⁴
- **Epochs**: 500
- **Input shape**: (B, 8, 200, 200) — full 2D slice
- **Key difference from MLP-PIA**: Operates on spatial context via convolutions rather than individual voxels

---

## Model 3: UNet Refiner (Supervised, Stage 2)

**File**: [`PIA.py`](../src/PIA.py) — `UNetRefiner` class  
**Checkpoints**:
- `checkpoints/refiner_mlp_e2e_best.pt` (trained on MLP-PIA outputs)
- `checkpoints/refiner_cnn_e2e_best.pt` (trained on CNN-PIA outputs)
- `checkpoints/refiner_cnn_best.pt` (initial training, before end-to-end fine-tuning)  

**Parameters**: ~3.49M

### Architecture
- **Type**: 3-level U-Net encoder-decoder with ResBlock stages
- **Encoder**: ResBlock(3→64) → Pool → ResBlock(64→128) → Pool → ResBlock(128→256) → Pool
- **Bottleneck**: ResBlock(256→256)
- **Decoder**: Skip-connected upsampling with bilinear interpolation
  - ResBlock(512→128) → ResBlock(256→64) → ResBlock(128→64)
- **ResBlock**: 2× Conv2d(3×3) + GroupNorm + LeakyReLU + learned 1×1 shortcut
- **Output**: 1×1 Conv2d (64→3), producing refined [f, Dt, D*] maps

### Input/Output Normalization
```
# Input normalization (Stage 1 → Refiner)
f_norm     = (f - 0.005) / 0.395     # f to ~[0, 1]
Dt_norm    = (Dt - 0.0001) / 0.0024  # Dt to ~[0, 1]
Dstar_norm = (D* - 0.002) / 0.063    # D* to ~[0, 1]

# Output denormalization (Refiner → final)
f_out     = clip(refined[0] × 0.395 + 0.005,   0.0,    0.45)
Dt_out    = clip(refined[1] × 0.0024 + 0.0001, 0.0,    0.003)
Dstar_out = clip(refined[2] × 0.063 + 0.002,   0.0,    0.1)
```

### Training
- **Training mode**: Supervised (ground truth IVIM parameter labels required)
- **Loss**: MSE between refined parameters and ground truth, with tumor-weighted upsampling
- **Optimizer**: Adam, lr=1×10⁻⁴
- **Epochs**: 200 (with early stopping on validation rRMSE)
- **Input**: 3-channel parameter maps from Stage 1 (MLP-PIA or CNN-PIA)
- **End-to-end fine-tuning**: After initial refiner training, Stage 1 + Refiner are jointly fine-tuned

---

## Baseline: IVIM-NEToptim (Self-Supervised)

**File**: [`ivim_net.py`](../src/ivim_net.py) — `IVIM_NEToptim` class  
**Checkpoint**: `checkpoints/ivim_net_best.pt`  
**Reference**: Barbieri et al., MRM 2020; Kaandorp et al., MRM 2021

### Architecture
- **Type**: Parallel sub-network MLP (separate branch per parameter)
- **Branches**: 2 hidden layers per branch, width=8, ELU activation
- **Regularization**: BatchNorm + 10% Dropout in each branch
- **Output**: Sigmoid-scaled bounds matching our parameter ranges
- **Decoder**: Same bi-exponential physics model

### Training
- **Training mode**: Self-supervised (signal reconstruction loss)
- **Loss**: MSE between input and reconstructed signal
- **Optimizer**: Adam, lr=3×10⁻⁴
- **Epochs**: 200
- **Input**: Voxel-wise (same as MLP-PIA)

---

## Baseline: NLLS (Classical)

**File**: [`utils.py`](../src/utils.py) — `fit_biExponential_model()`  
**No checkpoint** (non-parametric)

### Method
- Voxel-wise nonlinear least-squares curve fitting using `scipy.optimize.curve_fit`
- Fits the bi-exponential IVIM model S(b) = S₀[(1-f)·exp(-b·Dt) + f·exp(-b·D*)]
- **Runtime**: ~187.5 seconds per patient (CPU-only)

---

## Hardware

| Resource | Specification |
|----------|--------------|
| GPU | NVIDIA Quadro T2000 (4GB VRAM) |
| CPU | Intel Core i7 |
| Framework | PyTorch 2.x |
| Python | 3.10+ |

---

## Evaluation Metric

### rRMSE (Relative Root Mean Square Error) — AAPM Grand Challenge Eq. 4

For each IVIM parameter θ ∈ {Dt, D*}, the per-voxel relative RMSE is:

$$R_\theta = \sqrt{\frac{1}{N}\sum_{j=1}^{N}\left(\frac{\hat{\theta}_j - \theta^*_j}{\theta^*_j}\right)^2}$$

where $\theta^*_j$ is the ground truth value at voxel $j$ (normalization constant $C^*_{i,j} = p^*_{i,j}$). This treats every voxel's relative error equally, matching the AAPM 2024 IVIM Grand Challenge evaluation metric.

**Special case for f**: Since f is a dimensionless ratio, $C^*_{i,j} = 1$ (absolute RMSE):

$$R_f = \sqrt{\frac{1}{N}\sum_{j=1}^{N}(\hat{f}_j - f^*_j)^2}$$

This is implemented in `utils.py` via the `is_f=True` flag.

### Composite Score

$$\text{Composite} = \underbrace{\frac{R_f^T + R_{D_t}^T + R_{D^*}^T}{3}}_{\text{Tumor}} + \underbrace{\frac{R_f^{NT} + R_{D_t}^{NT}}{2}}_{\text{Non-tumor}}$$

### Design Justification

1. **Separate tumor/non-tumor evaluation**: Clinical significance of IVIM parameters differs between malignant lesions and healthy tissue. Tumor regions are the primary diagnostic target, justifying separate reporting.

2. **D* excluded from non-tumor composite**: The pseudo-diffusion coefficient D* represents blood microcirculation (perfusion). In non-tumor fibroglandular tissue, D* has minimal clinical relevance and high biological variability, making it a poor discriminator. Including D* in the non-tumor composite would inflate error for all methods without adding diagnostic information. This exclusion follows conventions in the IVIM literature (Barbieri et al., MRM 2020; Kaandorp et al., MRM 2021).

3. **Equal weighting within regions**: Within each region, parameters are equally weighted (1/3 for tumor, 1/2 for non-tumor). This avoids introducing arbitrary parameter importance that could favor specific methods.

4. **Additive (not averaged) region combination**: The tumor and non-tumor scores are summed rather than averaged to ensure both regions contribute meaningfully. Since the non-tumor region has fewer parameters (2 vs 3), this effectively gives ~60% weight to tumor accuracy and ~40% to non-tumor accuracy.
