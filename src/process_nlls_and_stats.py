# -*- coding: utf-8 -*-
import json
import os
import numpy as np
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    dl_path = os.path.join(BASE_DIR, 'results', 'evaluation_detailed.json')
    nlls_path = os.path.join(BASE_DIR, 'results', 'nlls_subset_results.json')
    
    with open(dl_path) as f:
        dl_data = json.load(f)
    with open(nlls_path) as f:
        nlls_data = json.load(f)
        
    sigmas = ['0.02', '0.05', '0.1', '0.2']
    snrs = [50, 20, 10, 5]
    dl_sigmas = dl_data['noise_levels']
    
    # Calculate composite_nontumor if missing
    if 'composite_nontumor' not in nlls_data['nlls']:
        nlls_data['nlls']['composite_nontumor'] = {'per_patient': {}, 'mean': [], 'std': []}
    for s in sigmas:
        f_nt = nlls_data['nlls']['f_nontumor']['per_patient'][s]
        dt_nt = nlls_data['nlls']['Dt_nontumor']['per_patient'][s]
        nlls_data['nlls']['composite_nontumor']['per_patient'][s] = [
            (f + dt)/2.0 if (f is not None and dt is not None) else None
            for f, dt in zip(f_nt, dt_nt)
        ]
            
    # Calculate means and stds for all nlls metrics
    for k, v in nlls_data['nlls'].items():
        v['mean'] = []
        v['std'] = []
        for s in sigmas:
            arr = [x for x in v['per_patient'][s] if x is not None]
            v['mean'].append(float(np.mean(arr)) if arr else None)
            v['std'].append(float(np.std(arr)) if arr else None)
            
    # Run Wilcoxon tests vs DL
    dl_methods = dl_data['methods']
    nlls_data['wilcoxon_tests'] = {}
    for m in dl_methods.keys():
        test_key = f"{m}_vs_nlls"
        nlls_data['wilcoxon_tests'][test_key] = {"p_values": [], "significant": [], "noise_levels": [float(s) for s in sigmas]}
        for s in sigmas:
            s_float = float(s)
            dl_idx = dl_sigmas.index(s_float)
            dl_scores = dl_methods[m]['composite_total']['per_patient'][dl_idx]
            nlls_scores = nlls_data['nlls']['composite_total']['per_patient'][s]
            
            stat, p = stats.wilcoxon(dl_scores, nlls_scores)
            nlls_data['wilcoxon_tests'][test_key]['p_values'].append(float(p))
            nlls_data['wilcoxon_tests'][test_key]['significant'].append(bool(p < 0.05))
            
    with open(nlls_path, 'w') as f:
        json.dump(nlls_data, f, indent=2)

    print("=" * 115)
    print(" FULL STATISTICAL SUMMARY ACROSS 200 TEST PATIENTS (0801-1000)")
    print("=" * 115)
    
    metrics = ['f_tumor', 'Dt_tumor', 'Dstar_tumor', 'composite_tumor', 
               'f_nontumor', 'Dt_nontumor', 'composite_nontumor', 'composite_total']
               
    for s_str, snr in zip(sigmas, snrs):
        s_float = float(s_str)
        dl_idx = dl_sigmas.index(s_float)
        
        print(f"\n" + "-" * 115)
        print(f" NOISE LEVEL: sigma = {s_str} (SNR = {snr})")
        print("-" * 115)
        
        methods = ['nlls', 'ivim_net', 'mlp', 'cnn', 'mlp_ref', 'cnn_ref']
        
        print(f"{'Method':<20} | {'Total rRMSE':<18} | {'Tumor Comp':<18} | {'NonTumor Comp':<18} | {'f_tumor':<10} | {'Dt_tumor':<10} | {'D*_tumor':<10}")
        print("-" * 125)
        
        for m in methods:
            res = {}
            if m == 'nlls':
                for met in metrics:
                    arr = np.array(nlls_data['nlls'][met]['per_patient'][s_str])
                    res[met] = (np.mean(arr), np.std(arr))
            else:
                for met in metrics:
                    if met == 'composite_nontumor':
                        f_arr = np.array(dl_data['methods'][m]['f_nontumor']['per_patient'][dl_idx])
                        dt_arr = np.array(dl_data['methods'][m]['Dt_nontumor']['per_patient'][dl_idx])
                        nt_arr = (f_arr + dt_arr) / 2.0
                        res[met] = (np.mean(nt_arr), np.std(nt_arr))
                    else:
                        res[met] = (dl_data['methods'][m][met]['mean'][dl_idx], dl_data['methods'][m][met]['std'][dl_idx])
                    
            print(f"{m:<20} | {res['composite_total'][0]:.4f} +/- {res['composite_total'][1]:.4f} | "
                  f"{res['composite_tumor'][0]:.4f} +/- {res['composite_tumor'][1]:.4f} | "
                  f"{res['composite_nontumor'][0]:.4f} +/- {res['composite_nontumor'][1]:.4f} | "
                  f"{res['f_tumor'][0]:.4f}   | {res['Dt_tumor'][0]:.4f}   | {res['Dstar_tumor'][0]:.4f}")
                  
    print("\n" + "=" * 115)
    print(" PAIRED WILCOXON SIGNED-RANK TESTS (Method vs NLLS)")
    print("=" * 115)
    
    for s_str, snr in zip(sigmas, snrs):
        s_float = float(s_str)
        dl_idx = dl_sigmas.index(s_float)
        
        nlls_tot = np.array(nlls_data['nlls']['composite_total']['per_patient'][s_str])
        
        print(f"\n>>> SNR {snr} (sigma={s_str}):")
        for m in ['ivim_net', 'mlp', 'cnn', 'mlp_ref', 'cnn_ref']:
            m_tot = np.array(dl_data['methods'][m]['composite_total']['per_patient'][dl_idx])
            stat, pval = stats.wilcoxon(m_tot, nlls_tot)
            print(f"  {m:<10} vs NLLS: W={stat:.1f}, p={pval:.2e} ({'p < 0.001' if pval < 0.001 else f'p={pval:.4f}'})")

if __name__ == '__main__':
    main()
