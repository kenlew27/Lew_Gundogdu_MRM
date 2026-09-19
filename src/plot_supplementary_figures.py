# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams

rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 11
rcParams['axes.titlesize'] = 13
rcParams['axes.labelsize'] = 11
rcParams['xtick.labelsize'] = 9.5
rcParams['ytick.labelsize'] = 9.5
rcParams['legend.fontsize'] = 9.5
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['axes.linewidth'] = 1.0

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')
NEW_PAPER_FIGS = os.path.join(BASE_DIR, 'new_paper', 'figures')
ROOT_FIGS = os.path.join(BASE_DIR, 'figures')

os.makedirs(NEW_PAPER_FIGS, exist_ok=True)
os.makedirs(ROOT_FIGS, exist_ok=True)

def save_fig(fig, name):
    for out_dir in [NEW_PAPER_FIGS, ROOT_FIGS]:
        fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(out_dir, f"{name}.pdf"), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {name}.png and {name}.pdf")

def load_data():
    dl_path = os.path.join(RESULTS_DIR, 'evaluation_detailed.json')
    nlls_path = os.path.join(RESULTS_DIR, 'nlls_subset_results.json')
    with open(dl_path, 'r') as f:
        dl_data = json.load(f)
    nlls_data = json.load(open(nlls_path)) if os.path.exists(nlls_path) else None
    return dl_data, nlls_data

def generate_figure_s1(dl_data, nlls_data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    eval_sigmas = [0.02, 0.033, 0.05, 0.10, 0.15, 0.20, 0.25]
    x_indices = np.arange(len(eval_sigmas))
    xtick_labels = [f"$\\sigma$={s:.3g}\nSNR {int(round(1/s))}" for s in eval_sigmas]
    noise_levels = dl_data['noise_levels']
    
    params = [
        ('f_tumor', 'Perfusion Fraction $f$ (Tumor)', axes[0]),
        ('Dt_tumor', 'Diffusion Coeff. $D_t$ (Tumor)', axes[1]),
        ('Dstar_tumor', 'Pseudo-Diffusion $D^*$ (Tumor)', axes[2])
    ]
    
    methods_config = [
        ('mlp', 'MLP-PIA', ':^', '#888888', 1.4, 6, 'white'),
        ('mlp_ref', 'MLP-PIA + Ref', '-.v', '#444444', 1.6, 6, '#444444'),
        ('cnn', 'CNN-PIA', ':d', '#aaaaaa', 1.4, 6, 'white'),
        ('cnn_ref', 'CNN-PIA + Ref', '-o', 'black', 2.2, 7, 'black'),
    ]
    
    for metric, title, ax in params:
        for m_key, label, fmt, color, lw, ms, mfc in methods_config:
            means = [dl_data['methods'][m_key][metric]['mean'][noise_levels.index(s)] for s in eval_sigmas]
            stds = [dl_data['methods'][m_key][metric]['std'][noise_levels.index(s)] for s in eval_sigmas]
            means, stds = np.array(means), np.array(stds)
            
            ls = fmt[0] if fmt[1] not in ['-', '.', ':'] else fmt[:2]
            marker = fmt[-1]
            ax.plot(x_indices, means, ls=ls, marker=marker, color=color, lw=lw, 
                    markersize=ms, markerfacecolor=mfc, markeredgecolor=color, label=label)
            ax.fill_between(x_indices, np.maximum(0, means - stds), means + stds, color='#e0e0e0', alpha=0.2)
            
        ax.set_title(title, fontsize=12, weight='bold')
        ax.set_ylabel("rRMSE", fontsize=11, weight='bold')
        ax.set_xlabel("Noise Condition", fontsize=11, weight='bold')
        ax.set_xticks(x_indices)
        ax.set_xticklabels(xtick_labels, fontsize=9)
        ax.grid(True, linestyle='-', color='#e8e8e8', alpha=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    axes[0].legend(loc='upper left', fontsize=8.5, framealpha=0.95)
    plt.tight_layout()
    save_fig(fig, 'Figure_S1')

def generate_figure_s2(dl_data):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)
    bench_sigmas = [0.02, 0.05, 0.10, 0.20]
    snr_titles = ['SNR 50 ($\\sigma=0.02$)', 'SNR 20 ($\\sigma=0.05$)', 'SNR 10 ($\\sigma=0.10$)', 'SNR 5 ($\\sigma=0.20$)']
    
    methods = ['mlp', 'cnn', 'mlp_ref', 'cnn_ref']
    labels = ['MLP-PIA', 'CNN-PIA', 'MLP+Ref', 'CNN+Ref']
    colors = ['#cccccc', '#bbbbbb', '#aaaaaa', '#666666', '#222222']
    noise_levels = dl_data['noise_levels']
    
    for ax_idx, (s, title) in enumerate(zip(bench_sigmas, snr_titles)):
        ax = axes[ax_idx]
        s_idx = noise_levels.index(s)
        
        box_data = []
        for m in methods:
            per_pat = dl_data['methods'][m]['composite_total']['per_patient'][s_idx]
            box_data.append(per_pat)
            
        bp = ax.boxplot(box_data, patch_artist=True, tick_labels=labels, 
                        showfliers=False, widths=0.6,
                        medianprops=dict(color='black', lw=1.5),
                        whiskerprops=dict(color='black', lw=1.0),
                        capprops=dict(color='black', lw=1.0))
        
        for patch_b, c in zip(bp['boxes'], colors):
            patch_b.set_facecolor(c)
            patch_b.set_alpha(0.7)
            patch_b.set_edgecolor('black')
            
        ax.set_title(title, fontsize=11.5, weight='bold')
        ax.grid(True, linestyle=':', color='#cccccc', alpha=0.7, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', rotation=30)
        
    axes[0].set_ylabel("Composite rRMSE (Total)", fontsize=11, weight='bold')
    plt.tight_layout()
    save_fig(fig, 'Figure_S2')

def generate_figure_s3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    
    methods = ['NLLS\n(Voxel-wise)', 'MLP-PIA', 'CNN-PIA', 'MLP+Refiner', 'CNN+Refiner']
    # CPU latency values from Table 2 of manuscript (seconds per 200x200 slice)
    # MLP+Refiner = MLP-PIA (0.164) + Refiner (0.042) stage times
    cpu_times = [187.511, 0.164, 0.015, 0.206, 0.075]
    
    y_pos = np.arange(len(methods))
    ax1.barh(y_pos, cpu_times, color=['#333333', '#888888', '#aaaaaa', '#666666', '#222222'], height=0.6, edgecolor='black')
    ax1.set_xscale('log')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(methods, fontsize=10)
    ax1.set_xlabel("CPU Execution Time (seconds / 200x200 slice)", fontsize=10.5, weight='bold')
    ax1.set_title("CPU Latency (Intel Core i7)", fontsize=12, weight='bold')
    ax1.grid(True, linestyle=':', axis='x', color='#cccccc')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    for i, t in enumerate(cpu_times):
        ax1.text(t * 1.3, i, f"{t:.3f} s" if t < 1 else f"{t:.1f} s", va='center', fontsize=9.5, weight='bold')
        
    # Speedups from Table 2: 187.511 / cpu_time for each DL method
    dl_methods = ['MLP-PIA', 'CNN-PIA', 'MLP+Refiner', 'CNN+Refiner']
    dl_cpu = [0.164, 0.015, 0.206, 0.075]
    dl_speedups = [187.511 / t for t in dl_cpu]
    
    ax2.bar(np.arange(len(dl_methods)), dl_speedups, color=['#888888', '#aaaaaa', '#666666', '#222222'], width=0.55, edgecolor='black')
    ax2.set_xticks(np.arange(len(dl_methods)))
    ax2.set_xticklabels(dl_methods, fontsize=10)
    ax2.set_ylabel("Speedup Factor vs CPU NLLS", fontsize=10.5, weight='bold')
    ax2.set_title("Inference Acceleration (CPU)", fontsize=12, weight='bold')
    ax2.grid(True, linestyle=':', axis='y', color='#cccccc')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    for i, sp in enumerate(dl_speedups):
        ax2.text(i, sp + 80, f"{int(round(sp))}x", ha='center', fontsize=10, weight='bold')
        
    plt.tight_layout()
    save_fig(fig, 'Figure_S3')

def generate_figure_s4(dl_data, nlls_data):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axis('off')
    
    ax.text(0.5, 0.92, "Per-Parameter Tumor rRMSE Breakdown at SNR = 10 (\u03c3 = 0.10)", 
            ha='center', va='center', fontsize=14, weight='bold')
    ax.text(0.5, 0.84, "Mean +/- SD across 200 test cases | Bold indicates lowest error", 
            ha='center', va='center', fontsize=10.5, color='#555555')
    
    headers = ["Method", "f (Tumor)", "Dt (Tumor)", "D* (Tumor)", "Composite (Tumor)"]
    col_x = [0.03, 0.32, 0.51, 0.70, 0.89]
    
    y_start = 0.74
    row_h = 0.07
    
    ax.plot([0.03, 0.97], [y_start + 0.02, y_start + 0.02], color='black', lw=1.8)
    for cx, h in zip(col_x, headers):
        ax.text(cx, y_start, h, ha='left' if cx < 0.2 else 'center', va='center', fontsize=11, weight='bold')
    ax.plot([0.03, 0.97], [y_start - 0.02, y_start - 0.02], color='black', lw=1.2)
    
    methods_list = [
        ('NLLS', 'nlls'),
        ('MLP-PIA', 'mlp'),
        ('CNN-PIA', 'cnn'),
        ('MLP-PIA + Refiner', 'mlp_ref'),
        ('CNN-PIA + Refiner', 'cnn_ref')
    ]
    
    dl_noise = dl_data['noise_levels']
    snr10_idx = dl_noise.index(0.10)
    
    curr_y = y_start - 0.05
    for m_label, m_key in methods_list:
        ax.text(col_x[0], curr_y, m_label, ha='left', va='center', fontsize=10.5, weight='bold')
        
        metrics = ['f_tumor', 'Dt_tumor', 'Dstar_tumor', 'composite_tumor']
        for c_idx, met in enumerate(metrics):
            if m_key == 'nlls':
                val_str = "[TBD]"
                if nlls_data and 'nlls' in nlls_data and met in nlls_data['nlls']:
                    arr = [v for v in nlls_data['nlls'][met]['per_patient'].get('0.1', []) if v is not None]
                    if arr:
                        val_str = f"{np.mean(arr):.3f} +/- {np.std(arr):.3f}"
            else:
                m_val = dl_data['methods'][m_key][met]['mean'][snr10_idx]
                s_val = dl_data['methods'][m_key][met]['std'][snr10_idx]
                val_str = f"{m_val:.3f} +/- {s_val:.3f}"
                
            is_best = False
            if m_key == 'mlp_ref' and met in ['f_tumor', 'Dt_tumor', 'composite_tumor']:
                is_best = True
            elif m_key == 'cnn_ref' and met in ['Dstar_tumor']:
                is_best = True
                
            ax.text(col_x[c_idx + 1], curr_y, val_str, ha='center', va='center', 
                    fontsize=10, weight='bold' if is_best else 'normal')
            
        curr_y -= row_h
        
    ax.plot([0.03, 0.97], [curr_y + 0.03, curr_y + 0.03], color='black', lw=1.8)
    save_fig(fig, 'Figure_S4')

def generate_figure_s5():
    case_idx = 803
    dwi = np.abs(np.load(os.path.join(DATA_DIR, f"{case_idx:04d}_gtDWIs.npy")))
    tissue = np.load(os.path.join(DATA_DIR, f"{case_idx:04d}_TissueType.npy"))
    b_values = [0, 5, 50, 100, 200, 500, 800, 1000]
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.2))
    axes = axes.ravel()
    
    for i in range(8):
        axes[i].imshow(dwi[:, :, i], cmap='gray')
        axes[i].set_title(f"b = {b_values[i]} s/mm$^2$", fontsize=10.5, weight='bold')
        axes[i].axis('off')
        
    axes[8].imshow(tissue, cmap='tab10', vmin=1, vmax=10)
    axes[8].set_title("Tissue Segmentation", fontsize=10.5, weight='bold')
    axes[8].axis('off')
    
    tumor_mask = (tissue == 8)
    axes[9].imshow(dwi[:, :, 0], cmap='gray')
    axes[9].contour(tumor_mask, colors='red', linewidths=1.5)
    axes[9].set_title("Tumor ROI (Red)", fontsize=10.5, weight='bold')
    axes[9].axis('off')
    
    plt.suptitle(f"Representative Test DW-MRI Input Data (Phantom Case {case_idx:04d})", 
                 fontsize=14, weight='bold', y=0.98)
    plt.tight_layout()
    save_fig(fig, 'Figure_S5')

def generate_figure_s6():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.axis('off')
    ax.text(0.5, 0.94, "MLP-PIA Architecture (Voxel-Wise Stage 1)", ha='center', fontsize=14, weight='bold')
    
    x_pos = [0.08, 0.23, 0.38, 0.53, 0.68]
    w = 0.12
    h = 0.46
    layers = [
        ("Input\nDW-MRI\n8 b-values", '#ffffff'),
        ("Linear\n8→64\n+ LeakyReLU", '#e8e8e8'),
        ("Linear\n64→128\n+ LeakyReLU", '#d0d0d0'),
        ("Linear\n128→256\n+ LeakyReLU", '#b8b8b8'),
        ("Linear\n256→512\n+ LeakyReLU", '#a0a0a0'),
    ]
    
    for i, ((text, c), xp) in enumerate(zip(layers, x_pos)):
        rect = patches.Rectangle((xp - w/2, 0.44 - h/2), w, h, edgecolor='black', facecolor=c, lw=1.2)
        ax.add_patch(rect)
        ax.text(xp, 0.44, text, ha='center', va='center', fontsize=8.5, weight='bold',
                clip_on=False)
        
    for i in range(len(x_pos) - 1):
        ax.annotate('', xy=(x_pos[i+1] - w/2 - 0.005, 0.44), 
                    xytext=(x_pos[i] + w/2 + 0.005, 0.44),
                    arrowprops=dict(facecolor='black', width=1, headwidth=4))
        
    head_y = [0.72, 0.44, 0.16]
    head_names = ["$f$: mean-delta-tanh\n[0.005, 0.400]", 
                  "$D_t$: mean-delta-tanh\n[0.0001, 0.0025]", 
                  "$D^*$: mean-delta-tanh\n[0.002, 0.065]"]
    
    for hy, hname in zip(head_y, head_names):
        ax.annotate('', xy=(0.80, hy), xytext=(x_pos[-1] + w/2 + 0.005, 0.44),
                    arrowprops=dict(facecolor='black', width=1, headwidth=4))
        hrect = patches.Rectangle((0.80, hy - 0.10), 0.18, 0.20, edgecolor='black', facecolor='#eaeaea', lw=1.2)
        ax.add_patch(hrect)
        ax.text(0.89, hy, hname, ha='center', va='center', fontsize=8.5, weight='bold',
                clip_on=False)
        
    save_fig(fig, 'Figure_S6')

def generate_figure_s7():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.axis('off')
    ax.text(0.5, 0.94, "CNN-PIA Architecture (Spatially Aware Stage 1)", ha='center', fontsize=14, weight='bold')
    
    x_pos = [0.08, 0.23, 0.38, 0.53, 0.68]
    w = 0.12
    h = 0.48
    layers = [
        ("Input Map\n8×200×200", '#ffffff'),
        ("Conv2D 8→64\n3×3, pad 1\n+ LeakyReLU", '#e8e8e8'),
        ("Conv2D 64→64\n3×3, pad 1\n+ LeakyReLU", '#d0d0d0'),
        ("Conv2D 64→128\n3×3, pad 1\n+ LeakyReLU", '#b8b8b8'),
        ("Conv2D 128→64\n3×3, pad 1\n+ LeakyReLU", '#a0a0a0'),
    ]
    
    for i, ((text, c), xp) in enumerate(zip(layers, x_pos)):
        rect = patches.Rectangle((xp - w/2, 0.44 - h/2), w, h, edgecolor='black', facecolor=c, lw=1.2)
        ax.add_patch(rect)
        ax.text(xp, 0.44, text, ha='center', va='center', fontsize=8.5, weight='bold',
                clip_on=False)
        
    for i in range(len(x_pos) - 1):
        ax.annotate('', xy=(x_pos[i+1] - w/2 - 0.005, 0.44), 
                    xytext=(x_pos[i] + w/2 + 0.005, 0.44),
                    arrowprops=dict(facecolor='black', width=1, headwidth=4))
        
    head_y = [0.72, 0.44, 0.16]
    head_names = ["1×1 Conv → $f$\n[0.005, 0.400]", 
                  "1×1 Conv → $D_t$\n[0.0001, 0.0025]", 
                  "1×1 Conv → $D^*$\n[0.002, 0.065]"]
    
    for hy, hname in zip(head_y, head_names):
        ax.annotate('', xy=(0.80, hy), xytext=(x_pos[-1] + w/2 + 0.005, 0.44),
                    arrowprops=dict(facecolor='black', width=1, headwidth=4))
        hrect = patches.Rectangle((0.80, hy - 0.10), 0.18, 0.20, edgecolor='black', facecolor='#eaeaea', lw=1.2)
        ax.add_patch(hrect)
        ax.text(0.89, hy, hname, ha='center', va='center', fontsize=8.5, weight='bold',
                clip_on=False)
        
    save_fig(fig, 'Figure_S7')

def generate_figure_s8():
    fig, ax = plt.subplots(figsize=(14, 6.0))
    ax.axis('off')
    ax.text(0.5, 0.96, "Stage 2: Residual U-Net Refiner Architecture", ha='center', fontsize=14, weight='bold')
    
    # (text, cx, cy, w, h, color)
    blocks = [
        ("Input Stage 1\n(f, Dt, D*)\n3×200×200", 0.08, 0.50, 0.11, 0.55, '#ffffff'),
        ("ResBlock 1\n64 ch\n200×200",             0.22, 0.50, 0.09, 0.55, '#d0d0d0'),
        ("MaxPool 2×2\nResBlock 2\n128 ch\n100×100", 0.36, 0.38, 0.10, 0.38, '#b0b0b0'),
        ("MaxPool 2×2\nBottleneck\n256 ch\n50×50",   0.50, 0.28, 0.10, 0.26, '#909090'),
        ("Upsample 2×2\n+ Skip Cat\nResBlock 128 ch", 0.64, 0.38, 0.10, 0.38, '#b0b0b0'),
        ("Upsample 2×2\n+ Skip Cat\nResBlock 64 ch",  0.78, 0.50, 0.09, 0.55, '#d0d0d0'),
        ("1×1 Conv\nRefined Maps\n3×200×200",          0.92, 0.50, 0.11, 0.55, '#ffffff')
    ]
    
    for text, cx, cy, w, h, c in blocks:
        rect = patches.Rectangle((cx - w/2, cy - h/2), w, h, edgecolor='black', facecolor=c, lw=1.2)
        ax.add_patch(rect)
        ax.text(cx, cy, text, ha='center', va='center', fontsize=7.5, weight='bold',
                clip_on=False)
        
    for i in range(len(blocks) - 1):
        ax.annotate('', xy=(blocks[i+1][1] - blocks[i+1][3]/2, blocks[i+1][2]), 
                    xytext=(blocks[i][1] + blocks[i][3]/2, blocks[i][2]),
                    arrowprops=dict(facecolor='black', width=1, headwidth=4))

    # Skip connection arrows — placed well above the tallest block tops (~0.775)
    ax.annotate('', xy=(0.783, 0.86), xytext=(0.225, 0.86),
                arrowprops=dict(facecolor='black', edgecolor='black', width=1.1, headwidth=4, ls='--'))
    ax.text(0.504, 0.89, "Skip Connection (Concat 64 ch)", ha='center', fontsize=8.5, weight='bold')
    
    ax.annotate('', xy=(0.645, 0.80), xytext=(0.405, 0.80),
                arrowprops=dict(facecolor='black', edgecolor='black', width=1.1, headwidth=4, ls='--'))
    ax.text(0.525, 0.83, "Skip Connection (Concat 128 ch)", ha='center', fontsize=8.5, weight='bold')

    save_fig(fig, 'Figure_S8')

def generate_figure_s9():
    import sys
    import torch
    sys.path.append(os.path.join(BASE_DIR, 'src'))
    from PIA import CNN_PIA, UNetRefiner
    from utils import fit_biExponential_model
    
    case_idx = 803
    gt_params = np.load(os.path.join(DATA_DIR, f"{case_idx:04d}_IVIMParam.npy"))
    gt_dwi = np.abs(np.load(os.path.join(DATA_DIR, f"{case_idx:04d}_gtDWIs.npy")))
    tissue = np.load(os.path.join(DATA_DIR, f"{case_idx:04d}_TissueType.npy"))
    
    np.random.seed(42 + case_idx)
    avg_S0 = 0.266081
    noise_std = 0.05 * avg_S0
    n_re = np.random.normal(0, noise_std, gt_dwi.shape)
    n_im = np.random.normal(0, noise_std, gt_dwi.shape)
    noisy_dwi = np.sqrt((gt_dwi + n_re)**2 + n_im**2)
    dwi_norm = noisy_dwi / np.clip(noisy_dwi[:, :, 0:1], 1e-8, None)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    b_values = [0, 5, 50, 100, 200, 500, 800, 1000]
    
    cnn_model = CNN_PIA(b_values=b_values, device=device).to(device)
    cnn_model.load_state_dict(torch.load(os.path.join(CHECKPOINTS_DIR, 'pia_cnn_best.pt'), map_location=device, weights_only=False))
    cnn_model.eval()
    
    refiner_model = UNetRefiner().to(device)
    ref_ckpt = torch.load(os.path.join(CHECKPOINTS_DIR, 'refiner_cnn_e2e_best.pt'), map_location=device, weights_only=False)
    ref_state = ref_ckpt if isinstance(ref_ckpt, dict) and 'model_state_dict' not in ref_ckpt else ref_ckpt['model_state_dict']
    refiner_model.load_state_dict(ref_state)
    refiner_model.eval()
    
    inp_t = torch.from_numpy(dwi_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        f_s1, dt_s1, ds_s1 = cnn_model(inp_t)
        NORM = UNetRefiner.NORM
        fn = (f_s1 - NORM['f']['min']) / NORM['f']['range']
        dtn = (dt_s1 - NORM['Dt']['min']) / NORM['Dt']['range']
        dsn = (ds_s1 - NORM['Dstar']['min']) / NORM['Dstar']['range']
        s1_cat = torch.cat([fn, dtn, dsn], dim=1)
        
        refined = refiner_model(s1_cat)
        f_s2 = torch.clamp(refined[:, 0:1] * NORM['f']['range'] + NORM['f']['min'], 0.0, 0.45)
        dt_s2 = torch.clamp(refined[:, 1:2] * NORM['Dt']['range'] + NORM['Dt']['min'], 0.0, 0.003)
        ds_s2 = torch.clamp(refined[:, 2:3] * NORM['Dstar']['range'] + NORM['Dstar']['min'], 0.0, 0.07)
        
    f_cnn = f_s1.squeeze().cpu().numpy()
    dt_cnn = dt_s1.squeeze().cpu().numpy()
    ds_cnn = ds_s1.squeeze().cpu().numpy()
    f_ref = f_s2.squeeze().cpu().numpy()
    dt_ref = dt_s2.squeeze().cpu().numpy()
    ds_ref = ds_s2.squeeze().cpu().numpy()
    
    dwi_nlls_in = np.where(tissue[:, :, None] != 1, dwi_norm, 0)
    nlls_fitted = fit_biExponential_model(dwi_nlls_in, np.array(b_values))
    f_nlls = nlls_fitted[:, :, 0]
    dt_nlls = nlls_fitted[:, :, 1]
    ds_nlls = nlls_fitted[:, :, 2]
    
    mask = (tissue != 1)
    
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    v_ranges = [
        (0.0, 0.35, 'viridis', '$f$ (fraction)'),
        (0.0001, 0.0022, 'magma', '$D_t$ (mm$^2$/s)'),
        (0.002, 0.060, 'plasma', '$D^*$ (mm$^2$/s)')
    ]
    row_data = [
        (gt_params[:, :, 0], f_nlls, f_cnn, f_ref),
        (gt_params[:, :, 1], dt_nlls, dt_cnn, dt_ref),
        (gt_params[:, :, 2], ds_nlls, ds_cnn, ds_ref)
    ]
    col_titles = ["Ground Truth", "NLLS", "Stage 1 (CNN-PIA)", "Stage 2 (CNN-PIA + Refiner)"]
    
    for row_idx, ((gt_map, nlls_map, s1_map, s2_map), (vmin, vmax, cmap, p_label)) in enumerate(zip(row_data, v_ranges)):
        maps = [gt_map * mask, nlls_map * mask, s1_map * mask, s2_map * mask]
        for col_idx, m_arr in enumerate(maps):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(m_arr, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.axis('off')
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=11, weight='bold')
            if col_idx == 0:
                ax.text(-0.15, 0.5, p_label, transform=ax.transAxes, ha='center', va='center', fontsize=12, weight='bold', rotation=90)
        cbar = fig.colorbar(im, ax=axes[row_idx, :], shrink=0.7, aspect=20, pad=0.03)
        cbar.ax.tick_params(labelsize=9)
        
    plt.suptitle(f"Qualitative IVIM Parameter Reconstructions at SNR 20 (Test Case {case_idx:04d})", 
                 fontsize=13.5, weight='bold', y=0.98)
    save_fig(fig, 'Figure_S9')

def generate_figure_s10():
    import sys
    import torch
    sys.path.append(os.path.join(BASE_DIR, 'src'))
    from PIA import CNN_PIA, UNetRefiner
    
    case_idx = 803
    gt_params = np.load(os.path.join(DATA_DIR, f"{case_idx:04d}_IVIMParam.npy"))
    gt_dwi = np.abs(np.load(os.path.join(DATA_DIR, f"{case_idx:04d}_gtDWIs.npy")))
    tissue = np.load(os.path.join(DATA_DIR, f"{case_idx:04d}_TissueType.npy"))
    
    np.random.seed(42 + case_idx)
    avg_S0 = 0.266081
    noise_std = 0.05 * avg_S0
    n_re = np.random.normal(0, noise_std, gt_dwi.shape)
    n_im = np.random.normal(0, noise_std, gt_dwi.shape)
    noisy_dwi = np.sqrt((gt_dwi + n_re)**2 + n_im**2)
    dwi_norm = noisy_dwi / np.clip(noisy_dwi[:, :, 0:1], 1e-8, None)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    b_values = [0, 5, 50, 100, 200, 500, 800, 1000]
    
    cnn_model = CNN_PIA(b_values=b_values, device=device).to(device)
    cnn_model.load_state_dict(torch.load(os.path.join(CHECKPOINTS_DIR, 'pia_cnn_best.pt'), map_location=device, weights_only=False))
    cnn_model.eval()
    
    refiner_model = UNetRefiner().to(device)
    ref_ckpt = torch.load(os.path.join(CHECKPOINTS_DIR, 'refiner_cnn_e2e_best.pt'), map_location=device, weights_only=False)
    ref_state = ref_ckpt if isinstance(ref_ckpt, dict) and 'model_state_dict' not in ref_ckpt else ref_ckpt['model_state_dict']
    refiner_model.load_state_dict(ref_state)
    refiner_model.eval()
    
    inp_t = torch.from_numpy(dwi_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        f_s1, dt_s1, ds_s1 = cnn_model(inp_t)
        NORM = UNetRefiner.NORM
        fn = (f_s1 - NORM['f']['min']) / NORM['f']['range']
        dtn = (dt_s1 - NORM['Dt']['min']) / NORM['Dt']['range']
        dsn = (ds_s1 - NORM['Dstar']['min']) / NORM['Dstar']['range']
        s1_cat = torch.cat([fn, dtn, dsn], dim=1)
        
        refined = refiner_model(s1_cat)
        f_s2 = torch.clamp(refined[:, 0:1] * NORM['f']['range'] + NORM['f']['min'], 0.0, 0.45)
        dt_s2 = torch.clamp(refined[:, 1:2] * NORM['Dt']['range'] + NORM['Dt']['min'], 0.0, 0.003)
        ds_s2 = torch.clamp(refined[:, 2:3] * NORM['Dstar']['range'] + NORM['Dstar']['min'], 0.0, 0.07)
        
    s1_maps = [f_s1.squeeze().cpu().numpy(), dt_s1.squeeze().cpu().numpy(), ds_s1.squeeze().cpu().numpy()]
    s2_maps = [f_s2.squeeze().cpu().numpy(), dt_s2.squeeze().cpu().numpy(), ds_s2.squeeze().cpu().numpy()]
    mask = (tissue != 1)
    
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    p_names = ['$f$', '$D_t$', '$D^*$']
    
    for row in range(3):
        gt = gt_params[:, :, row]
        s1 = s1_maps[row]
        s2 = s2_maps[row]
        
        err_s1 = np.abs(s1 - gt) * mask
        err_s2 = np.abs(s2 - gt) * mask
        err_diff = (err_s1 - err_s2) * mask
        
        axes[row, 0].imshow(gt * mask, cmap='viridis' if row==0 else ('magma' if row==1 else 'plasma'))
        axes[row, 0].set_title(f"Ground Truth {p_names[row]}", fontsize=10, weight='bold')
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(err_s1, cmap='hot_r', vmin=0, vmax=np.percentile(err_s1[mask], 98))
        axes[row, 1].set_title(f"Stage 1 Error |S1 - GT|", fontsize=10, weight='bold')
        axes[row, 1].axis('off')
        
        axes[row, 2].imshow(err_s2, cmap='hot_r', vmin=0, vmax=np.percentile(err_s1[mask], 98))
        axes[row, 2].set_title(f"Stage 2 Error |S2 - GT|", fontsize=10, weight='bold')
        axes[row, 2].axis('off')
        
        max_d = max(np.percentile(np.abs(err_diff[mask]), 99), 1e-6)
        im_diff = axes[row, 3].imshow(err_diff, cmap='coolwarm', vmin=-max_d, vmax=max_d)
        axes[row, 3].set_title("Improvement (Red = Reduced Error,\nBlue = Increased Error)", fontsize=9.5, weight='bold')
        axes[row, 3].axis('off')
        
    plt.suptitle(f"Stage 2 Spatial Refinement Denoising & Error Reduction (Case {case_idx:04d})", 
                 fontsize=13.5, weight='bold', y=0.98)
    save_fig(fig, 'Figure_S10')

def main():
    dl_data, nlls_data = load_data()
    print('Generating Figure S1...')
    generate_figure_s1(dl_data, nlls_data)
    print('Generating Figure S2...')
    generate_figure_s2(dl_data)
    print('Generating Figure S3...')
    generate_figure_s3()
    print('Generating Figure S4...')
    generate_figure_s4(dl_data, nlls_data)
    print('Generating Figure S5...')
    generate_figure_s5()
    print('Generating Figure S6...')
    generate_figure_s6()
    print('Generating Figure S7...')
    generate_figure_s7()
    print('Generating Figure S8...')
    generate_figure_s8()
    print('Generating Figure S9...')
    generate_figure_s9()
    print('Generating Figure S10...')
    generate_figure_s10()

if __name__ == '__main__':
    main()
