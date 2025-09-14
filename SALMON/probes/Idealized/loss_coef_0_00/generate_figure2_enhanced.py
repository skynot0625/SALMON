#!/usr/bin/env python3
"""
Generate Enhanced Figure 2 - FP32 Alignment with Relative Improvement
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

def figure2_fp32_alignment_enhanced(data, output_dir):
    """Enhanced Figure 2: FP32 alignment improvement with relative improvement rate"""
    
    epochs = sorted(data['epoch'].unique())
    branches = ['Dsum']
    alpha = 0.10  # Fixed small α
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for branch in branches:
        branch_data = data[data['branch'] == branch]
        
        for locus in ['L1_fea', 'L2_fea', 'L3_fea']:
            style = LOCUS_STYLES[locus]
            locus_data = branch_data[branch_data['locus'] == locus]
            
            delta_cos_means = []
            delta_cos_lower = []
            delta_cos_upper = []
            relative_improvement_means = []
            relative_improvement_lower = []
            relative_improvement_upper = []
            
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
                
                # Calculate relative improvement: Δcos / (1 - cos_A_FP)
                # This shows what percentage of the FP32 gap was filled
                relative_improvement = delta_cos / (1 - cos_A_FP + 1e-10) * 100  # as percentage
                rel_mean, rel_lower, rel_upper = calculate_confidence_interval(relative_improvement)
                relative_improvement_means.append(rel_mean)
                relative_improvement_lower.append(rel_lower)
                relative_improvement_upper.append(rel_upper)
            
            # Plot Δcos (left panel)
            ax1.plot(epochs, delta_cos_means,
                    color=style['color'], marker=style['marker'],
                    linestyle=style['linestyle'], label=style['label'],
                    markersize=6, linewidth=1.5)
            ax1.fill_between(epochs, delta_cos_lower, delta_cos_upper,
                            color=style['color'], alpha=0.2)
            
            # Plot relative improvement (right panel)
            ax2.plot(epochs, relative_improvement_means,
                    color=style['color'], marker=style['marker'],
                    linestyle=style['linestyle'], label=style['label'],
                    markersize=6, linewidth=1.5)
            ax2.fill_between(epochs, relative_improvement_lower, relative_improvement_upper,
                            color=style['color'], alpha=0.2)
    
    # Format left panel (Δcos)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(r'$\Delta\cos^{FP}_i(\alpha) = \cos(S_i(\alpha), A_i^{FP}) - \cos(A_i, A_i^{FP})$')
    ax1.set_title(f'(a) Absolute FP32 Alignment Improvement (α = {alpha})')
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.legend(loc='best')
    ax1.set_ylim([-0.01, 0.08])
    
    # Add reference lines for typical Δcos values
    for delta_val in [0.02, 0.04, 0.06]:
        ax1.axhline(y=delta_val, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        # Calculate angle change for c0=0.6 as example
        c0 = 0.6
        angle_change = np.degrees(np.arccos(c0) - np.arccos(c0 + delta_val))
        ax1.text(epochs[-1] + 5, delta_val, f'≈{angle_change:.1f}°', 
                fontsize=8, color='gray', va='center')
    
    # Format right panel (relative improvement)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(r'Relative FP32 Gap Reduction (%)')
    ax2.set_title(f'(b) Relative Improvement: $\Delta\cos^{{FP}}_i / (1 - \cos(A_i, A_i^{{FP}}))$ × 100%')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.legend(loc='best')
    
    # Add percentage reference lines
    for pct in [5, 10, 15, 20]:
        ax2.axhline(y=pct, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        ax2.text(epochs[-1] + 5, pct, f'{pct}%', fontsize=8, color='gray', va='center')
    
    plt.suptitle('Figure 2: FP32 Alignment Improvement - Absolute and Relative Metrics', y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure2_fp32_alignment_enhanced.pdf", format='pdf')
    plt.savefig(f"{output_dir}/figure2_fp32_alignment_enhanced.png", format='png')
    plt.show()
    
    # Print summary statistics
    print("\n=== Summary Statistics at Final Epoch ===")
    final_epoch = epochs[-1]
    final_data = data[data['epoch'] == final_epoch]
    
    for locus in ['L1_fea', 'L2_fea', 'L3_fea']:
        locus_data = final_data[final_data['locus'] == locus]
        col_name = get_alpha_columns(alpha, 'cos_S_FP')
        cos_S_FP = locus_data[col_name].values
        cos_A_FP = locus_data['cos_A_FP'].values
        delta_cos = cos_S_FP - cos_A_FP
        relative_improvement = delta_cos / (1 - cos_A_FP + 1e-10) * 100
        
        mean_delta = np.mean(delta_cos)
        mean_relative = np.mean(relative_improvement)
        mean_cos_A_FP = np.mean(cos_A_FP)
        
        # Calculate angle change
        angle_change = np.degrees(np.arccos(mean_cos_A_FP) - np.arccos(mean_cos_A_FP + mean_delta))
        
        print(f"\n{locus}:")
        print(f"  Base cos(A, A^FP) = {mean_cos_A_FP:.3f}")
        print(f"  Δcos = {mean_delta:.4f}")
        print(f"  Angle improvement ≈ {angle_change:.2f}°")
        print(f"  Relative FP32 gap reduction = {mean_relative:.1f}%")
    
    print(f"\nFigure 2 (enhanced) saved to {output_dir}/")

def main():
    # Setup paths
    data_dir = Path("/root/SALMON/probes/Idealized/loss_coef_0_00")
    output_dir = data_dir / "plots"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading data from {data_dir}...")
    data = load_data(data_dir)
    
    print(f"Data shape: {data.shape}")
    print(f"Unique epochs: {sorted(data['epoch'].unique())}")
    
    print("\nGenerating Enhanced Figure 2: FP32 alignment improvement...")
    figure2_fp32_alignment_enhanced(data, output_dir)
    
    print("\nFigure 2 (enhanced) generated successfully!")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()