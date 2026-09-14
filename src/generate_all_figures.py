# -*- coding: utf-8 -*-
"""
Master runner to generate all Main Figures (1-4) and Supplementary Figures (S1-S10)
for the MRM paper.
"""
import time
import plot_main_figures
import plot_supplementary_figures

def main():
    t0 = time.time()
    print("==================================================")
    print(" GENERATING ALL PUBLICATION FIGURES (MAIN & SUPP)")
    print("==================================================")
    
    print("\n>>> [1/2] Generating Main Figures (1, 2, 3, 4)...")
    plot_main_figures.main()
    
    print("\n>>> [2/2] Generating Supplementary Figures (S1 - S10)...")
    plot_supplementary_figures.main()
    
    print(f"\n[SUCCESS] All 14 figures generated in {time.time()-t0:.1f}s.")
    print("Figures saved in both 'new_paper/figures/' and 'figures/'.")

if __name__ == '__main__':
    main()
