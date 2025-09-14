#!/usr/bin/env python3
"""
Generate fresh plots directly from the latest data files
All plots with uniform figsize (10, 8)
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

# UNIFORM FIGURE SIZE FOR ALL PLOTS
UNIFORM_FIGSIZE = (10, 8)

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
    
    print(f"Found {len(task_files)} task files and {len(fp32_files)} fp32 files")
    
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
    
    return merged_data, task_data, fp32_data

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

# ============== FIGURE 1: Effective Descent Coefficient (Beta) ==============
def figure1_beta_fresh(data, output_dir):
    """Figure 1: Effective Descent Coefficient (beta = r * cos(D, A))"""
    
    epochs = sorted(data['epoch'].unique())
    print(f"Epochs for Figure 1: {epochs}")
    
    fig, ax = plt.subplots(1, 1, figsize=UNIFORM_FIGSIZE)
    
    # Focus on Dsum branch
    branch_data = data[data['branch'] == 'Dsum']
    
    for locus in ['L1_fea', 'L2_fea', 'L3_fea']:
        style = LOCUS_STYLES[locus]
        locus_data = branch_data[branch_data['locus'] == locus]
        
        beta_means = []
        beta_lower = []
        beta_upper = []
        
        for epoch in epochs:
            epoch_data = locus_data[locus_data['epoch'] == epoch]
            
            # Check if beta_task column exists, otherwise calculate it
            if 'beta_task' in epoch_data.columns:
                beta_values = epoch_data['beta_task'].values
            else:
                # Calculate beta = r * cos(D, A)
                r = epoch_data['r_norm_ratio_task_task'].values
                cos_D_A = epoch_data['cos_D_task_task'].values
                beta_values = r * cos_D_A
            
            mean, lower, upper = calculate_confidence_interval(beta_values)
            beta_means.append(mean)
            beta_lower.append(lower)
            beta_upper.append(upper)
        
        ax.plot(epochs, beta_means,
               color=style['color'], marker=style['marker'],
               linestyle=style['linestyle'], label=style['label'],
               markersize=6, linewidth=1.5)
        ax.fill_between(epochs, beta_lower, beta_upper,
                       color=style['color'], alpha=0.2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'$\beta_l = r_l \cdot \cos(D_l, A_l)$')
    ax.set_title('Effective Descent Coefficient')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.legend(loc='right')
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure1_beta_fresh.png", dpi=300)
    plt.close()
    print(f"Figure 1 saved to {output_dir}/")

# ============== FIGURE 2: Small-α FP32 Alignment Improvement ==============
def figure2_fp32_improvement(data, output_dir):
    """Figure 2: Small-α FP32 Alignment Improvement"""
    
    epochs = sorted(data['epoch'].unique())
    alpha = 0.10  # Fixed small α
    
    fig, ax = plt.subplots(1, 1, figsize=UNIFORM_FIGSIZE)
    
    branch_data = data[data['branch'] == 'Dsum']
    
    for locus in ['L1_fea', 'L2_fea', 'L3_fea']:
        style = LOCUS_STYLES[locus]
        locus_data = branch_data[branch_data['locus'] == locus]
        
        delta_cos_means = []
        delta_cos_lower = []
        delta_cos_upper = []
        
        for epoch in epochs:
            epoch_data = locus_data[locus_data['epoch'] == epoch]
            
            # Δcos = cos(S(α), FP32) - cos(A, FP32)
            # Find the right column for alpha=0.10
            cos_S_FP_col = None
            for col in epoch_data.columns:
                if 'cos_S_FP@0.10' in col or (col.startswith('cos_S_FP') and '0.10' in col):
                    cos_S_FP_col = col
                    break
            
            if cos_S_FP_col:
                cos_S_FP = epoch_data[cos_S_FP_col].values
            else:
                # Try alternative column name
                cos_S_FP = epoch_data['cos_S_FP'].values if 'cos_S_FP' in epoch_data.columns else np.zeros(len(epoch_data))
            
            cos_A_FP = epoch_data['cos_A_FP'].values
            delta_cos = cos_S_FP - cos_A_FP
            
            mean, lower, upper = calculate_confidence_interval(delta_cos)
            delta_cos_means.append(mean)
            delta_cos_lower.append(lower)
            delta_cos_upper.append(upper)
        
        ax.plot(epochs, delta_cos_means,
               color=style['color'], marker=style['marker'],
               linestyle=style['linestyle'], label=style['label'],
               markersize=6, linewidth=1.5)
        ax.fill_between(epochs, delta_cos_lower, delta_cos_upper,
                       color=style['color'], alpha=0.2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'$\Delta\cos^{FP}(\alpha) = \cos(S(\alpha), A^{FP}) - \cos(A, A^{FP})$')
    ax.set_title(f'Small-α FP32 Alignment Improvement (α = {alpha})')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.legend(loc='best')
    ax.set_xlim(0, 300)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure2_fp32_fresh.png", dpi=300)
    plt.close()
    print(f"Figure 2 saved to {output_dir}/")

# ============== FIGURE 3: Safe Mixing Window ==============
def figure3_safe_window(task_data, output_dir):
    """Figure 3: Safe Mixing Window"""
    
    epochs = sorted(task_data['epoch'].unique())
    alphas = [0.00, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00]
    
    fig, ax = plt.subplots(1, 1, figsize=UNIFORM_FIGSIZE)
    
    branch_data = task_data[task_data['branch'] == 'Dsum']
    
    for locus in ['L1_fea', 'L2_fea', 'L3_fea']:
        style = LOCUS_STYLES[locus]
        locus_data = branch_data[branch_data['locus'] == locus]
        
        max_safe_alphas = []
        lower_bounds = []
        upper_bounds = []
        
        for epoch in epochs:
            epoch_data = locus_data[locus_data['epoch'] == epoch]
            
            # Find maximum α where descent gain is still positive
            safe_alphas = []
            for _, row in epoch_data.iterrows():
                max_safe = 0.0
                for alpha in alphas:
                    col_name = f'descent_gain_task@{alpha:.2f}'
                    if col_name in row and row[col_name] > 0:
                        max_safe = alpha
                safe_alphas.append(max_safe)
            
            mean, lower, upper = calculate_confidence_interval(safe_alphas)
            max_safe_alphas.append(mean)
            lower_bounds.append(lower)
            upper_bounds.append(upper)
        
        ax.plot(epochs, max_safe_alphas,
               color=style['color'], marker=style['marker'],
               linestyle=style['linestyle'], label=style['label'],
               markersize=6, linewidth=1.5)
        ax.fill_between(epochs, lower_bounds, upper_bounds,
                       color=style['color'], alpha=0.2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'Maximum Safe $\alpha$')
    ax.set_title('Safe Mixing Window')
    ax.legend(loc='best')
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure3_safe_fresh.png", dpi=300)
    plt.close()
    print(f"Figure 3 saved to {output_dir}/")

# ============== FIGURE 4: Device Accuracy ==============
def figure4_accuracy(output_dir):
    """Figure 4: Device Accuracy vs Alpha"""
    
    # Data from original accuracy plot
    alpha = [0.01, 0.1, 0.3, 0.5, 1, 2, 10]
    device1_ideal = [87.1, 85.53, 87.54, 88.29, 92.1, 91.6, 92.05]
    device2_idealized = [76.58, 74.28, 81.13, 81.48, 84.32, 84.87, 86.8]
    device3_ecram = [52.13, 47.46, 52.57, 56.23, 57.92, 56.87, 54.64]
    
    x_positions = np.arange(len(alpha))
    
    fig, ax = plt.subplots(1, 1, figsize=UNIFORM_FIGSIZE)
    
    ax.plot(x_positions, device1_ideal, 'o-', color='green', linewidth=2, markersize=8, 
            label='Device 1 (Ideal)')
    ax.plot(x_positions, device2_idealized, 's-', color='blue', linewidth=2, markersize=8,
            label='Device 2 (Idealized)')
    ax.plot(x_positions, device3_ecram, '^-', color='red', linewidth=2, markersize=8,
            label='Device 3 (EcramDevice)')
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(alpha)
    ax.set_xlabel('Alpha', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Device Accuracy vs Alpha Parameter', fontsize=14, fontweight='bold')
    ax.grid(True, ls="-", alpha=0.2)
    ax.legend(loc='best', fontsize=10)
    ax.set_ylim(45, 95)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure4_accuracy_fresh.png", dpi=300)
    plt.close()
    print(f"Figure 4 saved to {output_dir}/")

def main():
    # Setup paths
    data_dir = Path("/root/SALMON/probes/Idealized/loss_coef_0_00")
    output_dir = data_dir / "plots_fresh"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading fresh data from {data_dir}...")
    merged_data, task_data, fp32_data = load_data(data_dir)
    
    print(f"Merged data shape: {merged_data.shape}")
    print(f"Task data shape: {task_data.shape}")
    print(f"FP32 data shape: {fp32_data.shape}")
    print(f"Epochs available: {sorted(merged_data['epoch'].unique())}")
    print(f"\nGenerating all plots with uniform size {UNIFORM_FIGSIZE}...\n")
    
    # Generate all 4 plots
    print("1. Generating Figure 1 (Effective Descent Coefficient)...")
    figure1_beta_fresh(merged_data, output_dir)
    
    print("2. Generating Figure 2 (Small-α FP32 Alignment Improvement)...")
    figure2_fp32_improvement(merged_data, output_dir)
    
    print("3. Generating Figure 3 (Safe Mixing Window)...")
    figure3_safe_window(task_data, output_dir)
    
    print("4. Generating Figure 4 (Device Accuracy)...")
    figure4_accuracy(output_dir)
    
    print(f"\nAll fresh plots generated with uniform size in: {output_dir}")
    print("All plots created directly from the latest data files (Sep 11-12)")

if __name__ == "__main__":
    main()