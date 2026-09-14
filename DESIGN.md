# DESIGN.md - Codebase Overview for LLM Context

This codebase implements a physics-informed deep learning framework for non-invasive quantitative Intravoxel Incoherent Motion (IVIM) magnetic resonance imaging (MRI) parameter estimation in breast tissue. The 8 key bullet points below detail the software design, biophysical math, model architecture, dataset processing, and evaluation pipelines:

1. **Biophysical Forward Model & Physics Integration**:
   The codebase maps 8 diffusion-weighted MRI (DW-MRI) signals ($b = [0, 5, 50, 100, 200, 500, 800, 1000]\text{ s/mm}^2$) into three biophysical parameters ($f$: perfusion fraction, $D_t$: tissue diffusivity, $D^*$: capillary pseudo-diffusivity) using the bi-exponential signal decay law:
   $$S(b) = (1 - f) \cdot e^{-b \cdot D_t} + f \cdot e^{-b \cdot D^*}$$
   The analytical physics equation is integrated into PyTorch (`src/utils.py`, `src/PIA.py`) as a non-differentiable/differentiable decoder.

2. **Stage 1 Self-Supervised Physics Autoencoder (PIA)**:
   Stage 1 (`src/PIA.py`) contains MLP (`PIA`) and 2D spatial CNN (`CNN_PIA`) encoders that transform raw noisy DW-MRI signals into latent parameter space. It enforces strict physiological parameter boundaries ($f \in [0.005, 0.40]$, $D_t \in [0.0001, 0.0025]$, $D^* \in [0.002, 0.065]$) via custom $\text{mean} \pm \Delta \cdot \tanh(\cdot)$ activation functions, enabling self-supervised parameter estimation without expert ground-truth annotations.

3. **Stage 2 Multi-Scale Spatial U-Net Refiner**:
   Stage 2 (`src/PIA.py`, class `UNetRefiner`) ingests all three Stage 1 parameter maps simultaneously into a 3.49M-parameter residual U-Net architecture. It leverages multi-scale 2D spatial context across neighboring voxels to remove salt-and-pepper noise artifacts and applies tumor-weighted supervision to preserve sharp boundaries around microvascular breast malignancies.

4. **Voxel-Wise NLLS Optimization Baseline**:
   The module `src/NLLS_solution.py` implements a parallelized CPU non-linear least squares (`scipy.optimize.curve_fit` with Truncated Newton Bounds) baseline. It evaluates classical optimization across test patient cases with physiologically bounded parameters ($D^* \leq 0.1\text{ mm}^2/\text{s}$), providing empirical comparison metrics for speed, parameter variance, and convergence failures under noise.

5. **VICTRE Digital Breast Phantom Dataset Handling**:
   The dataset loading pipeline (`src/utils.py`, `data/`) processes 1000 synthetic patient cases ($200 \times 200$ resolution slices) generated via the FDA VICTRE platform, split into 600 training / 200 validation / 200 held-out test cases. Files include complex k-space noisy signals (`XXXX_NoisyDWIk.npy`), clean ground-truth signals (`XXXX_gtDWIs.npy`), ground-truth parameter maps (`XXXX_IVIMParam.npy`), and tissue segmentation masks (`XXXX_TissueType.npy`).

6. **Automated CPU vs. GPU Speed Benchmarking**:
   The benchmarking module evaluates execution time across CPU and GPU hardware, recording slice-level latency metrics. The CNN-PIA + Refiner pipeline achieves a 31,888× speedup (5.88 ms per slice on GPU vs. 187.5 seconds for CPU NLLS fitting).

7. **Multi-Noise Robustness & Metric Evaluation**:
   The evaluation workflow assesses parameter estimation accuracy across 12 noise standard deviation levels ($\sigma = 0.00$ to $0.25$, corresponding to SNR 4 to 100). Relative Root Mean Square Error ($\text{rRMSE}$) is computed separately across total tissue, tumor regions, and per-parameter ($f, D_t, D^*$), saving structured quantitative results to `results/noise_evaluation_results_e2e.json`.

8. **Publication-Ready Visualization & Reporting Suite**:
   The visualization engine (`figures/plot_paper_figures.py`, `figures/plot_noise_clean.py`, `figures/plot_tables.py`) generates monochromatic publication-quality figure artifacts, including SNR stability line charts, qualitative parameter map overlays, refiner ablation comparisons, and summary tables rendered in PNG format.
