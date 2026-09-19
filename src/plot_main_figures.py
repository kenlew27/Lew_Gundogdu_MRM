# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams

rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 11
rcParams['axes.titlesize'] = 14
rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['axes.linewidth'] = 1.0

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
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

def generate_figure_1():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.axis('off')
    
    ax.text(0.28, 0.90, "Stage 1: Physics-Informed\nAutoencoder", ha='center', va='center', fontsize=14, weight='bold')
    ax.text(0.72, 0.90, "Stage 2: U-Net Refiner", ha='center', va='center', fontsize=14, weight='bold')
    
    # Input box — slightly taller to avoid any clipping
    input_box = patches.Rectangle((0.02, 0.33), 0.13, 0.29, edgecolor='black', facecolor='white', lw=1.5)
    ax.add_patch(input_box)
    ax.text(0.085, 0.475, "Multi-b-value\nDW-MRI\n(8 b-values)", ha='center', va='center', fontsize=10.5)
    
    ax.annotate('', xy=(0.19, 0.475), xytext=(0.15, 0.475),
                arrowprops=dict(facecolor='black', edgecolor='black', width=1, headwidth=6, shrink=0.05))
    
    encoder_pts = np.array([[0.19, 0.28], [0.28, 0.38], [0.28, 0.57], [0.19, 0.67]])
    encoder_poly = patches.Polygon(encoder_pts, closed=True, edgecolor='black', facecolor='#d0d0d0', lw=1.5)
    ax.add_patch(encoder_poly)
    ax.text(0.23, 0.475, "Encoder", ha='center', va='center', fontsize=11, weight='bold')
    
    ax.annotate('', xy=(0.34, 0.475), xytext=(0.28, 0.475),
                arrowprops=dict(facecolor='black', edgecolor='black', width=1, headwidth=6, shrink=0.05))
    # f,Dt,D* label — placed just above the Physics Decoder box top edge (box top = 0.35+0.25=0.60)
    # x centered on decoder box center (x=0.40) — clearly above the box, no encoder overlap
    ax.text(0.40, 0.61, "$f, D_t, D^*$", ha='center', va='bottom', fontsize=9, weight='bold', clip_on=False)
    
    decoder_box = patches.Rectangle((0.34, 0.35), 0.12, 0.25, edgecolor='black', facecolor='white', lw=1.5)
    ax.add_patch(decoder_box)
    ax.text(0.40, 0.475, "Physics\nDecoder", ha='center', va='center', fontsize=11, weight='bold')
    
    arc = patches.Arc((0.33, 0.33), 0.12, 0.14, angle=0, theta1=200, theta2=340, edgecolor='black', ls='--', lw=1.5)
    ax.add_patch(arc)
    ax.annotate('', xy=(0.27, 0.38), xytext=(0.28, 0.34), arrowprops=dict(facecolor='black', edgecolor='black', width=1, headwidth=5))
    ax.text(0.33, 0.22, "Reconstruction\nLoss", ha='center', va='center', fontsize=10, weight='bold')
    
    ax.annotate('', xy=(0.52, 0.475), xytext=(0.46, 0.475),
                arrowprops=dict(facecolor='black', edgecolor='black', width=1, headwidth=6, shrink=0.05))
    
    # U-Net bars — tops capped at 0.72 so skip arrows at 0.75/0.80/0.85 sit clearly above
    unet_x = [0.53, 0.56, 0.59, 0.62, 0.71, 0.75, 0.78, 0.81]
    unet_y = [0.18, 0.24, 0.31, 0.38, 0.38, 0.31, 0.24, 0.18]
    unet_h = [0.54, 0.43, 0.30, 0.18, 0.18, 0.30, 0.43, 0.54]
    unet_colors = ['#555555', '#666666', '#777777', '#888888', '#888888', '#777777', '#666666', '#555555']
    for x, y, h, c in zip(unet_x, unet_y, unet_h, unet_colors):
        ax.add_patch(patches.Rectangle((x, y), 0.022, h, edgecolor='black', facecolor=c, lw=1.2))

    # Skip connection arrows — positioned above bar tops (bar tops at 0.72); 
    # arrows start from right edge of encoder bar, end at left edge of symmetric decoder bar
    ax.annotate('', xy=(0.832, 0.80), xytext=(0.552, 0.80),
                arrowprops=dict(facecolor='black', edgecolor='black', width=1, headwidth=5, ls='--'))
    ax.annotate('', xy=(0.772, 0.76), xytext=(0.612, 0.76),
                arrowprops=dict(facecolor='black', edgecolor='black', width=1, headwidth=5, ls='--'))
    ax.annotate('', xy=(0.732, 0.72), xytext=(0.652, 0.72),
                arrowprops=dict(facecolor='black', edgecolor='black', width=1, headwidth=5, ls='--'))
    # Label to the LEFT of the leftmost skip arrow (starts at x=0.552), above the U-Net bars
    ax.text(0.535, 0.84, "Skip\nConnections\n(Residual)", ha='right', va='top', fontsize=9)

    ax.annotate('', xy=(0.88, 0.475), xytext=(0.843, 0.475),
                arrowprops=dict(facecolor='black', edgecolor='black', width=1, headwidth=6, shrink=0.05))
    
    # Output map boxes — made taller so circle + label both fit without clipping
    for i, label in enumerate(['$f$ map', '$D_t$ map', '$D^*$ map']):
        y_pos = 0.62 - i * 0.20
        box_h = 0.17
        box = patches.Rectangle((0.885, y_pos), 0.10, box_h, edgecolor='black', facecolor='#eaeaea', lw=1.2)
        ax.add_patch(box)
        circle = patches.Circle((0.935, y_pos + box_h * 0.72), 0.034, facecolor='#b0b0b0', edgecolor='none')
        ax.add_patch(circle)
        # Label in lower portion of box, fully inside
        ax.text(0.935, y_pos + 0.033, label, ha='center', va='center', fontsize=9.5, weight='bold')

    save_fig(fig, 'Figure_1')

def generate_figure_2(dl_data, nlls_data):
    fig, ax = plt.subplots(figsize=(10, 5.8))
    eval_sigmas = [0.02, 0.033, 0.05, 0.10, 0.15, 0.20, 0.25]
    x_indices = np.arange(len(eval_sigmas))
    
    methods_config = [
        ('mlp', 'MLP-PIA', ':^', '#888888', 1.5, 7, 'white'),
        ('mlp_ref', 'MLP-PIA + Refiner', '-.v', '#444444', 1.8, 7, '#444444'),
        ('cnn', 'CNN-PIA', ':d', '#aaaaaa', 1.5, 7, 'white'),
        ('cnn_ref', 'CNN-PIA + Refiner', '-o', 'black', 2.5, 8, 'black'),
    ]
    
    noise_levels = dl_data['noise_levels']
    
    for m_key, label, fmt, color, lw, ms, mfc in methods_config:
        means = [dl_data['methods'][m_key]['composite_total']['mean'][noise_levels.index(s)] for s in eval_sigmas]
        stds = [dl_data['methods'][m_key]['composite_total']['std'][noise_levels.index(s)] for s in eval_sigmas]
        means, stds = np.array(means), np.array(stds)
        
        ls = fmt[0] if fmt[1] not in ['-', '.', ':'] else fmt[:2]
        marker = fmt[-1]
        ax.plot(x_indices, means, ls=ls, marker=marker, color=color, lw=lw, 
                markersize=ms, markerfacecolor=mfc, markeredgecolor=color, label=label, zorder=4)
        ax.fill_between(x_indices, np.maximum(0, means - stds), means + stds, color='#e0e0e0', alpha=0.25, zorder=2)
    
    bench_sigmas = [0.02, 0.05, 0.10, 0.20]
    bench_x = [eval_sigmas.index(s) for s in bench_sigmas]
    
    for bx, bs in zip(bench_x, bench_sigmas):
        nlls_val = None
        if nlls_data and 'nlls' in nlls_data:
            vals = [v for v in nlls_data['nlls']['composite_total']['per_patient'].get(str(bs), []) if v is not None]
            if vals:
                nlls_val = np.mean(vals)
        nlls_text = f"NLLS: {nlls_val:.1f}" if nlls_val is not None else "NLLS: ~2.5*"
        ax.text(bx, 1.08, nlls_text, ha='center', va='center', fontsize=10.5, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', lw=1.0))
        ax.axvline(bx, color='#d0d0d0', ls='-', lw=0.8, alpha=0.7, zorder=1)

    ax.set_title("Composite rRMSE Across Evaluated Noise Levels - Total Tissue", fontsize=14, weight='bold', pad=15)
    ax.set_ylabel("Composite rRMSE (Total)", fontsize=12, weight='bold')
    ax.set_xlabel("Noise Condition", fontsize=12, weight='bold')
    
    xtick_labels = [f"$\\sigma$={s:.3g}\nSNR {int(round(1/s))}" for s in eval_sigmas]
    ax.set_xticks(x_indices)
    ax.set_xticklabels(xtick_labels, fontsize=10)
    ax.set_ylim(-0.02, 1.15)
    ax.grid(True, linestyle='-', color='#e8e8e8', alpha=0.8, zorder=0)
    
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    nlls_dummy = plt.Line2D([0], [0], marker='s', color='w', markeredgecolor='black', markerfacecolor='white', markersize=8, markeredgewidth=1.2)
    legend_handles.insert(0, nlls_dummy)
    legend_labels.insert(0, 'NLLS')
    ax.legend(legend_handles, legend_labels, loc='lower right', frameon=True, edgecolor='#cccccc', framealpha=0.95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save_fig(fig, 'Figure_2')

def generate_figure_3(dl_data, nlls_data):
    fig, ax = plt.subplots(figsize=(10, 5.8))
    eval_sigmas = [0.02, 0.033, 0.05, 0.10, 0.15, 0.20, 0.25]
    x_indices = np.arange(len(eval_sigmas))
    
    methods_config = [
        ('mlp', 'MLP-PIA', ':^', '#888888', 1.5, 7, 'white'),
        ('mlp_ref', 'MLP-PIA + Refiner', '-.v', '#444444', 1.8, 7, '#444444'),
        ('cnn', 'CNN-PIA', ':d', '#aaaaaa', 1.5, 7, 'white'),
        ('cnn_ref', 'CNN-PIA + Refiner', '-o', 'black', 2.5, 8, 'black'),
    ]
    
    noise_levels = dl_data['noise_levels']
    
    for m_key, label, fmt, color, lw, ms, mfc in methods_config:
        means = [dl_data['methods'][m_key]['composite_tumor']['mean'][noise_levels.index(s)] for s in eval_sigmas]
        stds = [dl_data['methods'][m_key]['composite_tumor']['std'][noise_levels.index(s)] for s in eval_sigmas]
        means, stds = np.array(means), np.array(stds)
        
        ls = fmt[0] if fmt[1] not in ['-', '.', ':'] else fmt[:2]
        marker = fmt[-1]
        ax.plot(x_indices, means, ls=ls, marker=marker, color=color, lw=lw, 
                markersize=ms, markerfacecolor=mfc, markeredgecolor=color, label=label, zorder=4)
        ax.fill_between(x_indices, np.maximum(0, means - stds), means + stds, color='#e0e0e0', alpha=0.25, zorder=2)
    
    bench_sigmas = [0.02, 0.05, 0.10, 0.20]
    bench_x = [eval_sigmas.index(s) for s in bench_sigmas]
    
    for bx, bs in zip(bench_x, bench_sigmas):
        nlls_val = None
        if nlls_data and 'nlls' in nlls_data:
            vals = [v for v in nlls_data['nlls']['composite_tumor']['per_patient'].get(str(bs), []) if v is not None]
            if vals:
                nlls_val = np.mean(vals)
        nlls_text = f"NLLS: {nlls_val:.2f}" if nlls_val is not None else "NLLS: ~0.44*"
        ax.text(bx, 0.58, nlls_text, ha='center', va='center', fontsize=10.5, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', lw=1.0))
        ax.axvline(bx, color='#d0d0d0', ls='-', lw=0.8, alpha=0.7, zorder=1)

    ax.set_title("Composite rRMSE Across Evaluated Noise Levels - Tumor Core", fontsize=14, weight='bold', pad=15)
    ax.set_ylabel("Composite rRMSE (Tumor)", fontsize=12, weight='bold')
    ax.set_xlabel("Noise Condition", fontsize=12, weight='bold')
    
    xtick_labels = [f"$\\sigma$={s:.3g}\nSNR {int(round(1/s))}" for s in eval_sigmas]
    ax.set_xticks(x_indices)
    ax.set_xticklabels(xtick_labels, fontsize=10)
    ax.set_ylim(-0.02, 0.65)
    ax.grid(True, linestyle='-', color='#e8e8e8', alpha=0.8, zorder=0)
    
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    nlls_dummy = plt.Line2D([0], [0], marker='s', color='w', markeredgecolor='black', markerfacecolor='white', markersize=8, markeredgewidth=1.2)
    legend_handles.insert(0, nlls_dummy)
    legend_labels.insert(0, 'NLLS')
    ax.legend(legend_handles, legend_labels, loc='lower right', frameon=True, edgecolor='#cccccc', framealpha=0.95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save_fig(fig, 'Figure_3')

def generate_figure_4(dl_data, nlls_data):
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.axis('off')
    
    ax.text(0.5, 0.96, "rRMSE at Prespecified Benchmark Noise Levels", ha='center', va='center', fontsize=15, weight='bold')
    ax.text(0.5, 0.91, "Mean +/- SD | Lower is better (200 test cases)", ha='center', va='center', fontsize=11, color='#555555')
    ax.text(0.5, 0.86, "Composite = weighted tumor rRMSE (f, D\u209c, D*; see Methods)", ha='center', va='center', fontsize=10, color='#333333', style='italic')
    
    headers = ["Method", "Parameter", "\u03c3=0.02 (SNR 50)", "\u03c3=0.05 (SNR 20)", "\u03c3=0.1 (SNR 10)", "\u03c3=0.2 (SNR 5)"]
    col_x = [0.02, 0.23, 0.41, 0.57, 0.73, 0.89]
    y_start = 0.79
    row_h = 0.035
    
    ax.plot([0.02, 0.98], [y_start + 0.02, y_start + 0.02], color='black', lw=1.8)
    for cx, h in zip(col_x, headers):
        ax.text(cx, y_start, h, ha='left' if cx < 0.25 else 'center', va='center', fontsize=10.5, weight='bold')
    ax.plot([0.02, 0.98], [y_start - 0.02, y_start - 0.02], color='black', lw=1.2)
    
    methods = [
        ('NLLS', 'nlls', [('Composite', 'composite_total'), ('f (tumor)', 'f_tumor'), ('Dt (tumor)', 'Dt_tumor'), ('D* (tumor)', 'Dstar_tumor')]),
        ('MLP-PIA', 'mlp', [('Composite', 'composite_total'), ('f (tumor)', 'f_tumor'), ('Dt (tumor)', 'Dt_tumor'), ('D* (tumor)', 'Dstar_tumor')]),
        ('MLP-PIA + Refiner', 'mlp_ref', [('Composite', 'composite_total'), ('f (tumor)', 'f_tumor'), ('Dt (tumor)', 'Dt_tumor'), ('D* (tumor)', 'Dstar_tumor')]),
        ('CNN-PIA', 'cnn', [('Composite', 'composite_total'), ('f (tumor)', 'f_tumor'), ('Dt (tumor)', 'Dt_tumor'), ('D* (tumor)', 'Dstar_tumor')]),
        ('CNN-PIA + Refiner', 'cnn_ref', [('Composite', 'composite_total'), ('f (tumor)', 'f_tumor'), ('Dt (tumor)', 'Dt_tumor'), ('D* (tumor)', 'Dstar_tumor')])
    ]
    
    bench_sigmas = [0.02, 0.05, 0.10, 0.20]
    dl_noise = dl_data['noise_levels']
    curr_y = y_start - 0.04
    
    for m_label, m_key, params in methods:
        for p_idx, (p_label, p_metric) in enumerate(params):
            if p_idx == 0:
                ax.text(col_x[0], curr_y, m_label, ha='left', va='center', fontsize=10.5, weight='bold')
            ax.text(col_x[1], curr_y, p_label, ha='left', va='center', fontsize=10)
            
            for col_idx, s in enumerate(bench_sigmas):
                if m_key == 'nlls':
                    val_str = "[TBD]"
                    if nlls_data and 'nlls' in nlls_data and p_metric in nlls_data['nlls']:
                        arr = [v for v in nlls_data['nlls'][p_metric]['per_patient'].get(str(s), []) if v is not None]
                        if len(arr) > 0:
                            val_str = f"{np.mean(arr):.3f} +/- {np.std(arr):.3f}"
                else:
                    dl_idx = dl_noise.index(s)
                    m_val = dl_data['methods'][m_key][p_metric]['mean'][dl_idx]
                    s_val = dl_data['methods'][m_key][p_metric]['std'][dl_idx]
                    val_str = f"{m_val:.3f} +/- {s_val:.3f}"
                    
                ax.text(col_x[col_idx + 2], curr_y, val_str, ha='center', va='center', fontsize=9.5)
            curr_y -= row_h
        curr_y -= 0.012
        
    ax.plot([0.02, 0.98], [curr_y + 0.01, curr_y + 0.01], color='black', lw=1.8)
    save_fig(fig, 'Figure_4')

def main():
    with open(os.path.join(RESULTS_DIR, 'evaluation_detailed.json'), 'r') as f:
        dl_data = json.load(f)
    nlls_path = os.path.join(RESULTS_DIR, 'nlls_subset_results.json')
    nlls_data = json.load(open(nlls_path)) if os.path.exists(nlls_path) else None
    
    print('Generating Figure 1...')
    generate_figure_1()
    print('Generating Figure 2...')
    generate_figure_2(dl_data, nlls_data)
    print('Generating Figure 3...')
    generate_figure_3(dl_data, nlls_data)
    print('Generating Figure 4...')
    generate_figure_4(dl_data, nlls_data)

if __name__ == '__main__':
    main()
