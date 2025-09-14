#!/usr/bin/env python3
"""
Generate plots from ORIGINAL data (from tar file)
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
    """Load all CSV files from ORIGINAL data"""
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
def figure1_beta_original(task_data, output_dir):
    """Figure 1: Effective Descent Coefficient using ORIGINAL data"""
    
    epochs = sorted(task_data['epoch'].unique())
    print(f"Epochs for Figure 1: {epochs}")
    
    fig, ax = plt.subplots(1, 1, figsize=UNIFORM_FIGSIZE)
    
    # Focus on Dsum branch
    branch_data = task_data[task_data['branch'] == 'Dsum']
    
    for locus in ['L1_fea', 'L2_fea', 'L3_fea']:
        style = LOCUS_STYLES[locus]
        locus_data = branch_data[branch_data['locus'] == locus]
        
        beta_means = []
        beta_lower = []
        beta_upper = []
        
        for epoch in epochs:
            epoch_data = locus_data[locus_data['epoch'] == epoch]
            
            # Use beta_task directly from original data
            if 'beta_task' in epoch_data.columns:
                beta_values = epoch_data['beta_task'].values
                mean, lower, upper = calculate_confidence_interval(beta_values)
                beta_means.append(mean)
                beta_lower.append(lower)
                beta_upper.append(upper)
                
                # Debug: print values for epoch 50
                if epoch == 50:
                    print(f"{locus} at epoch 50: mean={mean:.4f}")
        
        ax.plot(epochs, beta_means,
               color=style['color'], marker=style['marker'],
               linestyle=style['linestyle'], label=style['label'],
               markersize=6, linewidth=1.5)
        ax.fill_between(epochs, beta_lower, beta_upper,
                       color=style['color'], alpha=0.2)
    
    ax.set_xlabel('Epoch', fontsize=18)  # 1.5x from 12
    ax.set_ylabel(r'$\beta_l = r_l \cdot \cos(D_l, A_l)$', fontsize=18)  # 1.5x from 12
    ax.set_title('Effective Descent Coefficient')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=15)  # 1.5x from 10
    ax.set_xlim(0, 310)  # Extended to 310 for small buffer after 300
    ax.set_ylim(-0.05, 1.0)  # Adjusted to match Figure 2's zero line proportion
    ax.tick_params(axis='both', which='major', labelsize=13.5)  # 1.5x from 9
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure1_beta_original.png", dpi=300)
    plt.close()
    print(f"Figure 1 saved to {output_dir}/")

# ============== FIGURE 2: Small-α FP32 Alignment Improvement ==============
def figure2_fp32_original(merged_data, output_dir):
    """Figure 2: Small-α FP32 Alignment Improvement from ORIGINAL data"""
    
    epochs = sorted(merged_data['epoch'].unique())
    alpha = 0.10  # Fixed small α
    
    fig, ax = plt.subplots(1, 1, figsize=UNIFORM_FIGSIZE)
    
    branch_data = merged_data[merged_data['branch'] == 'Dsum']
    
    for locus in ['L1_fea', 'L2_fea', 'L3_fea']:
        style = LOCUS_STYLES[locus]
        locus_data = branch_data[branch_data['locus'] == locus]
        
        delta_cos_means = []
        delta_cos_lower = []
        delta_cos_upper = []
        
        for epoch in epochs:
            epoch_data = locus_data[locus_data['epoch'] == epoch]
            
            # Calculate Δcos from original data columns
            # In original data, the column should be cos_S_FP@0.10
            cos_S_FP = epoch_data['cos_S_FP@0.10'].values if 'cos_S_FP@0.10' in epoch_data.columns else np.zeros(len(epoch_data))
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
    
    ax.set_xlabel('Epoch', fontsize=18)  # 1.5x from 12
    ax.set_ylabel(r'$\Delta\cos^{FP}(\alpha) = \cos(S(\alpha), A^{FP}) - \cos(A, A^{FP})$', fontsize=18)  # 1.5x from 12
    ax.set_title(f'Small-α FP32 Alignment Improvement (α = {alpha})')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=15)  # 1.5x from 10
    ax.set_xlim(0, 310)  # Extended to 310 for small buffer after 300
    ax.tick_params(axis='both', which='major', labelsize=13.5)  # 1.5x from 9
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure2_fp32_original.png", dpi=300)
    plt.close()
    print(f"Figure 2 saved to {output_dir}/")

# Helper function for alpha columns
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

def calculate_magnitude_gain(data, alpha):
    """Calculate M_i(α) = ||S_i(α)|| cos(S_i(α), A_i^FP) / (||A_i|| cos(A_i, A_i^FP))"""
    
    # Get necessary values - after merging, columns have _task suffix
    col_name = get_alpha_columns(alpha, 'cos_S_FP')
    cos_S_FP = data[col_name].values
    cos_A_FP = data['cos_A_FP'].values
    r = data['r_norm_ratio_task_task'].values  # After merge, has _task suffix
    cos_D_A = data['cos_D_task_task'].values    # After merge, has _task suffix
    
    # Calculate ||S(α)||/||A|| using closed form
    # ||S(α)||^2 = ||A||^2 + 2α||A||||D||cos(D,A) + α^2||D||^2
    # ||S(α)||/||A|| = sqrt(1 + 2α*r*cos(D,A) + α^2*r^2)
    norm_ratio_S_A = np.sqrt(1 + 2*alpha*r*cos_D_A + alpha**2 * r**2)
    
    # Calculate M(α)
    # Avoid division by zero
    denominator = cos_A_FP
    denominator[np.abs(denominator) < 1e-10] = 1e-10
    
    M = norm_ratio_S_A * cos_S_FP / denominator
    
    return M

# ============== FIGURE 3: Safe Mixing Window ==============
def figure3_safe_window_original(merged_data, output_dir):
    """Figure 3: Safe mixing window - curvature-safety vs. FP32-gain (trajectory plot)"""
    
    epoch = 300  # Use epoch 300 data
    branches = ['Dsum']
    alpha_values = [0.00, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00]
    
    fig, ax = plt.subplots(1, 1, figsize=UNIFORM_FIGSIZE)
    
    epoch_data = merged_data[merged_data['epoch'] == epoch]
    
    for branch in branches:
        branch_data = epoch_data[epoch_data['branch'] == branch]
        
        for locus in ['L1_fea', 'L2_fea', 'L3_fea']:
            style = LOCUS_STYLES[locus]
            locus_data = branch_data[branch_data['locus'] == locus]
            
            if len(locus_data) == 0:
                continue
            
            M_values_all = []
            rho_values_all = []
            
            for alpha in alpha_values:
                if alpha == 0:
                    M_values = np.ones(len(locus_data))
                    rho_values = np.ones(len(locus_data))
                else:
                    # Use correct column names from merged data
                    M_values = calculate_magnitude_gain(locus_data, alpha)
                    # ρ_i(α) = cos²(S_i(α), A_i)
                    # cos_S_task columns don't get suffix as they're only in task_centric file
                    col_name = f'cos_S_task@{alpha:.2f}'
                    cos_S_task = locus_data[col_name].values
                    rho_values = cos_S_task ** 2
                
                M_mean, _, _ = calculate_confidence_interval(M_values)
                rho_mean, _, _ = calculate_confidence_interval(rho_values)
                
                M_values_all.append(M_mean)
                rho_values_all.append(rho_mean)
            
            # Plot trajectory
            ax.plot(M_values_all, rho_values_all,
                   color=style['color'], marker=style['marker'],
                   linestyle=style['linestyle'], label=style['label'],
                   markersize=4, linewidth=1.5)
            
            # Add alpha labels for selected points
            labels_to_show = [0.10, 0.25, 0.50]
            for alpha_label in labels_to_show:
                if alpha_label in alpha_values:
                    idx = alpha_values.index(alpha_label)
                    if idx < len(M_values_all):
                        ax.annotate(f'α={alpha_label:.2f}', 
                                  xy=(M_values_all[idx], rho_values_all[idx]),
                                  xytext=(5, 5), textcoords='offset points',
                                  fontsize=10, fontweight='bold', alpha=0.9)
    
    # Add safe region shading
    ax.axvspan(1.0, 3.0, alpha=0.1, color='green', label='M > 1 (Gain region)')
    ax.axhspan(0.8, 1.0, alpha=0.1, color='blue', label='ρ ≥ 0.8 (Safe curvature)')
    
    # Add reference lines
    ax.axvline(x=1, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(y=0.8, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    
    ax.set_xlabel(r'$M(\alpha)$ (FP32-Parallel Magnitude Gain)', fontsize=18)  # 1.5x from 12
    ax.set_ylabel(r'$\rho(\alpha) = \cos^2(S(\alpha), A)$ (Curvature Safety)', fontsize=18)  # 1.5x from 12
    ax.set_title(f'Safe Mixing Window (Epoch {epoch})', fontsize=13, pad=15)  # Added padding to move title up
    ax.legend(loc='upper right', fontsize=15, frameon=True, fancybox=True, shadow=True)  # Top-right corner
    ax.tick_params(axis='both', which='major', labelsize=15)  # 1.5x from 10
    ax.set_xlim([0.8, 3.0])
    ax.set_ylim([0.2, 1.0])
    
    # Add arrow to show α increase direction
    ax.annotate('', xy=(2.5, 0.3), xytext=(1.5, 0.8),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.5))
    ax.text(2.0, 0.55, 'α increase', fontsize=10, color='gray', rotation=-60, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure3_safe_original.png", dpi=300)
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
            label='Device 1')
    ax.plot(x_positions, device2_idealized, 's-', color='blue', linewidth=2, markersize=8,
            label='Device 2')
    ax.plot(x_positions, device3_ecram, '^-', color='red', linewidth=2, markersize=8,
            label='Device 3')
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(alpha, fontsize=13.5)  # 1.5x from 9
    ax.set_xlabel(r'$\alpha$', fontsize=18)  # 1.5x from 12
    ax.set_ylabel('Accuracy (%)', fontsize=18)  # 1.5x from 12
    ax.set_title('Device Accuracy vs α Parameter', fontsize=14, fontweight='bold', pad=15)  # Added padding to move title up
    ax.grid(True, ls="-", alpha=0.2)
    ax.legend(loc='upper left', fontsize=14, frameon=True, fancybox=True, shadow=True)  # Upper left with slight offset to avoid data
    ax.tick_params(axis='y', which='major', labelsize=13.5)  # 1.5x from 9
    ax.set_ylim(45, 95)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure4_accuracy_original.png", dpi=300)
    plt.close()
    print(f"Figure 4 saved to {output_dir}/")

def main():
    # Use ORIGINAL data from tar extract
    data_dir = Path("/tmp/loss_coef_0_00_original/loss_coef_0_00")
    output_dir = Path("/root/SALMON/probes/Idealized/loss_coef_0_00/plots_original")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading ORIGINAL data from {data_dir}...")
    merged_data, task_data, fp32_data = load_data(data_dir)
    
    print(f"Merged data shape: {merged_data.shape}")
    print(f"Task data shape: {task_data.shape}")
    print(f"FP32 data shape: {fp32_data.shape}")
    print(f"Epochs available: {sorted(merged_data['epoch'].unique())}")
    print(f"\nGenerating all plots with uniform size {UNIFORM_FIGSIZE} from ORIGINAL data...\n")
    
    # Generate all 4 plots
    print("1. Generating Figure 1 (Effective Descent Coefficient)...")
    figure1_beta_original(task_data, output_dir)
    
    print("\n2. Generating Figure 2 (Small-α FP32 Alignment Improvement)...")
    figure2_fp32_original(merged_data, output_dir)
    
    print("\n3. Generating Figure 3 (Safe Mixing Window)...")
    figure3_safe_window_original(merged_data, output_dir)
    
    print("\n4. Generating Figure 4 (Device Accuracy)...")
    figure4_accuracy(output_dir)
    
    print(f"\nAll plots generated from ORIGINAL data with uniform size in: {output_dir}")

if __name__ == "__main__":
    main()