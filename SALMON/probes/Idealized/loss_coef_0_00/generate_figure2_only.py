#!/usr/bin/env python3
"""
Generate Figure 2 only - for testing modifications
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path
import glob
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

# Define colors and markers for each locus
LOCUS_STYLES = {
    'L1_fea': {'color': '#2E86AB', 'marker': 'o', 'label': 'L1', 'linestyle': '-'},
    'L2_fea': {'color': '#A23B72', 'marker': 's', 'label': 'L2', 'linestyle': '--'},
    'L3_fea': {'color': '#F18F01', 'marker': '^', 'label': 'L3', 'linestyle': '-.'}
}

def load_data(data_dir):
    """Load all CSV files and combine into dataframes"""
    task_files = sorted(glob.glob(f"{data_dir}/epoch_*_task_centric.csv"))
    fp32_files = sorted(glob.glob(f"{data_dir}/epoch_*_fp32_alignment.csv"))
    
    task_dfs = []
    fp32_dfs = []
    
    for tf, ff in zip(task_files, fp32_files):
        task_df = pd.read_csv(tf)
        fp32_df = pd.read_csv(ff)
        task_dfs.append(task_df)
        fp32_dfs.append(fp32_df)
    
    task_data = pd.concat(task_dfs, ignore_index=True)
    fp32_data = pd.concat(fp32_dfs, ignore_index=True)
    
    # Merge on common columns
    merged_data = pd.merge(task_data, fp32_data, 
                           on=['epoch', 'batch', 'seed', 'branch', 'locus'],
                           suffixes=('_task', '_fp32'))
    
    return merged_data

def calculate_confidence_interval(data, confidence=0.95):
    """Calculate mean and confidence interval"""
    n = len(data)
    if n == 0:
        return np.nan, np.nan, np.nan
    
    mean = np.mean(data)
    if n == 1:
        return mean, mean, mean
    
    sem = stats.sem(data)
    ci = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
    
    return mean, mean - ci, mean + ci

def get_alpha_columns(alpha, prefix='cos_S_FP'):
    """Get the correct column name for a given alpha value"""
    alpha_map = {
        0.00: f'{prefix}@0.00',
        0.01: f'{prefix}@0.01',
        0.02: f'{prefix}@0.02',
        0.05: f'{prefix}@0.05',
        0.10: f'{prefix}@0.10',
        0.25: f'{prefix}@0.25',
        0.50: f'{prefix}@0.50',
        1.00: f'{prefix}@1.00'
    }
    
    # Find closest alpha value
    alphas = list(alpha_map.keys())
    closest_alpha = min(alphas, key=lambda x: abs(x - alpha))
    return alpha_map[closest_alpha]

def figure2_fp32_alignment(data, output_dir):
    """Figure 2: Small-α FP32 alignment improvement (Fact 2)"""
    
    epochs = sorted(data['epoch'].unique())
    branches = ['Dsum']
    alpha = 0.10  # Fixed small α
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    for branch in branches:
        branch_data = data[data['branch'] == branch]
        
        for locus in ['L1_fea', 'L2_fea', 'L3_fea']:
            style = LOCUS_STYLES[locus]
            locus_data = branch_data[branch_data['locus'] == locus]
            
            delta_cos_means = []
            delta_cos_lower = []
            delta_cos_upper = []
            
            for epoch in epochs:
                epoch_data = locus_data[locus_data['epoch'] == epoch]
                
                # Δcos_i^FP(α) = cos(S_i(α), A_i^FP) - cos(A_i, A_i^FP)
                col_name = get_alpha_columns(alpha, 'cos_S_FP')
                cos_S_FP = epoch_data[col_name].values
                cos_A_FP = epoch_data['cos_A_FP'].values
                delta_cos = cos_S_FP - cos_A_FP
                
                mean, lower, upper = calculate_confidence_interval(delta_cos)
                delta_cos_means.append(mean)
                delta_cos_lower.append(lower)
                delta_cos_upper.append(upper)
            
            # Plot Δcos
            ax.plot(epochs, delta_cos_means,
                   color=style['color'], marker=style['marker'],
                   linestyle=style['linestyle'], label=style['label'],
                   markersize=6, linewidth=1.5)
            ax.fill_between(epochs, delta_cos_lower, delta_cos_upper,
                           color=style['color'], alpha=0.2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'$\Delta\cos^{FP}_i(\alpha) = \cos(S_i(\alpha), A_i^{FP}) - \cos(A_i, A_i^{FP})$')
    ax.set_title(f'Figure 2: Small-α FP32 Alignment Improvement (α = {alpha})')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure2_fp32_alignment.pdf", format='pdf')
    plt.savefig(f"{output_dir}/figure2_fp32_alignment.png", format='png')
    plt.show()
    
    print(f"Figure 2 saved to {output_dir}/")

def main():
    # Setup paths
    data_dir = Path("/root/SALMON/probes/Idealized/loss_coef_0_00")
    output_dir = data_dir / "plots"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading data from {data_dir}...")
    data = load_data(data_dir)
    
    print(f"Data shape: {data.shape}")
    print(f"Unique epochs: {sorted(data['epoch'].unique())}")
    
    print("\nGenerating Figure 2: FP32 alignment improvement...")
    figure2_fp32_alignment(data, output_dir)
    
    print("\nFigure 2 generated successfully!")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()