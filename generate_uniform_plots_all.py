#!/usr/bin/env python3
"""
Generate all 4 plots with uniform figure size (10, 8) while maintaining original scales
Based on the original plot generation code
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
# IMPORTANT: Don't use tight bbox to ensure uniform size
# plt.rcParams['savefig.bbox'] = 'tight'  # Comment this out  
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

# ============== FIGURE 1: Effective Descent Coefficient Only ==============
def figure1_edc_only(data, output_dir):
    """Generate ONLY Effective Descent Coefficient plot"""
    
    epochs = sorted(data['epoch'].unique())
    branches = ['Dsum']
    
    fig, ax = plt.subplots(1, 1, figsize=UNIFORM_FIGSIZE)
    
    for branch in branches:
        branch_data = data[data['branch'] == branch]
        
        for locus in ['L1_fea', 'L2_fea', 'L3_fea']:
            style = LOCUS_STYLES[locus]
            locus_data = branch_data[branch_data['locus'] == locus]
            
            beta_means = []
            beta_lower = []
            beta_upper = []
            
            for epoch in epochs:
                epoch_data = locus_data[locus_data['epoch'] == epoch]
                # β_i = r_i * cos(D_i, A_i)
                beta_values = epoch_data['r_norm_ratio_task_task'] * epoch_data['cos_D_task_task']
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
    ax.set_ylabel(r'$\beta_i = r_i \cdot \cos(D_i, A_i)$')
    ax.set_title('Effective Descent Coefficient')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.legend(loc='right')
    ax.set_xlim(0, 300)
    # Keep original y-axis scale
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'figure1_edc_uniform.png'
    plt.savefig(output_path, format='png', dpi=300)
    print(f"Figure 1 (EDC only) saved to: {output_path}")
    plt.close()

# ============== FIGURE 2: Small-α FP32 Alignment Improvement ==============
def figure2_fp32_alignment(data, output_dir):
    """Figure 2: Small-α FP32 alignment improvement"""
    
    epochs = sorted(data['epoch'].unique())
    branches = ['Dsum']
    alpha = 0.10  # Fixed small α
    
    fig, ax = plt.subplots(1, 1, figsize=UNIFORM_FIGSIZE)
    
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
            
            ax.plot(epochs, delta_cos_means,
                   color=style['color'], marker=style['marker'],
                   linestyle=style['linestyle'], label=style['label'],
                   markersize=6, linewidth=1.5)
            ax.fill_between(epochs, delta_cos_lower, delta_cos_upper,
                           color=style['color'], alpha=0.2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'$\Delta\cos^{FP}_i(\alpha) = \cos(S_i(\alpha), A_i^{FP}) - \cos(A_i, A_i^{FP})$')
    ax.set_title(f'Small-α FP32 Alignment Improvement (α = {alpha})')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.legend(loc='best')
    ax.set_xlim(0, 300)
    # Keep original y-axis scale from the plot
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'figure2_fp32_uniform.png'
    plt.savefig(output_path, format='png', dpi=300)
    print(f"Figure 2 saved to: {output_path}")
    plt.close()

# ============== FIGURE 4: Safe Mixing Window ==============
def figure4_safe_mixing_window(data, output_dir):
    """Figure 4: Safe mixing window"""
    
    epochs = sorted(data['epoch'].unique())
    branches = ['Dsum']
    alphas = [0.00, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00]
    
    fig, ax = plt.subplots(1, 1, figsize=UNIFORM_FIGSIZE)
    
    for branch in branches:
        branch_data = data[data['branch'] == branch]
        
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
    ax.set_ylabel(r'Max Safe $\alpha$')
    ax.set_title('Safe Mixing Window')
    ax.legend(loc='best')
    ax.set_xlim(0, 300)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'figure4_safe_uniform.png'
    plt.savefig(output_path, format='png', dpi=300)
    print(f"Figure 4 saved to: {output_path}")
    plt.close()

# ============== ACCURACY PLOT ==============
def accuracy_plot(output_dir):
    """Generate accuracy plot with uniform size"""
    
    # Data
    alpha = [0.01, 0.1, 0.3, 0.5, 1, 2, 10]
    device1_ideal = [87.1, 85.53, 87.54, 88.29, 92.1, 91.6, 92.05]
    device2_idealized = [76.58, 74.28, 81.13, 81.48, 84.32, 84.87, 86.8]
    device3_ecram = [52.13, 47.46, 52.57, 56.23, 57.92, 56.87, 54.64]
    
    # Create x positions with equal spacing
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
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Device Accuracy vs Alpha Parameter')
    ax.grid(True, ls="-", alpha=0.2)
    ax.legend(loc='best')
    ax.set_ylim(45, 95)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'accuracy_uniform.png'
    plt.savefig(output_path, format='png', dpi=300)
    print(f"Accuracy plot saved to: {output_path}")
    plt.close()

# ============== MAIN EXECUTION ==============
def main():
    # Setup paths
    data_dir = Path("/root/SALMON/probes/Idealized/loss_coef_0_00")
    output_dir = data_dir / "plots_uniform"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading data from {data_dir}...")
    data = load_data(data_dir)
    
    print(f"Data shape: {data.shape}")
    print(f"Generating all plots with uniform size {UNIFORM_FIGSIZE}...\n")
    
    # Generate all 4 plots
    print("1. Generating Figure 1 (Effective Descent Coefficient)...")
    figure1_edc_only(data, output_dir)
    
    print("2. Generating Figure 2 (FP32 Alignment Improvement)...")
    figure2_fp32_alignment(data, output_dir)
    
    print("3. Generating Figure 4 (Safe Mixing Window)...")
    figure4_safe_mixing_window(data, output_dir)
    
    print("4. Generating Accuracy Plot...")
    accuracy_plot(output_dir)
    
    print(f"\nAll plots generated with uniform size in: {output_dir}")
    print("All plots have the same dimensions for perfect alignment in combined figure.")

if __name__ == "__main__":
    main()