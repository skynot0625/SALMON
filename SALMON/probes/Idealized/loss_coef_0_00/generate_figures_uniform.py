#!/usr/bin/env python3
"""
Generate Figures with UNIFORM SIZE - Modified from original generate_figures.py
Only changing figsize to (10, 8) for all plots
Figure 1: Only beta score (Effective Descent Coefficient) plot
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

def figure1_beta_only(data, output_dir):
    """Figure 1: ONLY Effective Descent Coefficient (beta score)"""
    
    epochs = sorted(data['epoch'].unique())
    branches = ['Dsum']  # Focus on Dsum for main paper
    
    # Changed to single plot with UNIFORM_FIGSIZE
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
                beta_values = epoch_data['beta_task'].values
                mean, lower, upper = calculate_confidence_interval(beta_values)
                beta_means.append(mean)
                beta_lower.append(lower)
                beta_upper.append(upper)
            
            # Plot β
            ax.plot(epochs, beta_means,
                   color=style['color'], marker=style['marker'],
                   linestyle=style['linestyle'], label=style['label'],
                   markersize=6, linewidth=1.5)
            ax.fill_between(epochs, beta_lower, beta_upper,
                           color=style['color'], alpha=0.2)
    
    # Formatting - keeping original scale and labels
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'$\beta_i = r_i \cos(D_i, A_i)$')
    ax.set_title('Effective Descent Coefficient')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.legend(loc='best')
    # Keep original y-axis limits if needed
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure1_beta_uniform.pdf", format='pdf')
    plt.savefig(f"{output_dir}/figure1_beta_uniform.png", format='png')
    plt.close()
    
    print(f"Figure 1 (beta only) saved to {output_dir}/")

def figure2_fp32_alignment(data, output_dir):
    """Figure 2: Small-α FP32 alignment improvement (Fact 2) - ORIGINAL CODE"""
    
    epochs = sorted(data['epoch'].unique())
    branches = ['Dsum']
    alpha = 0.1  # Fixed small α
    
    # Changed figsize to UNIFORM_FIGSIZE
    fig, ax = plt.subplots(1, 1, figsize=UNIFORM_FIGSIZE)
    
    for branch in branches:
        branch_data = data[data['branch'] == branch]
        
        for locus in ['L1_fea', 'L2_fea', 'L3_fea']:
            style = LOCUS_STYLES[locus]
            locus_data = branch_data[branch_data['locus'] == locus]
            
            delta_cos_means = []
            delta_cos_lower = []
            delta_cos_upper = []
            significance = []
            
            for epoch in epochs:
                epoch_data = locus_data[locus_data['epoch'] == epoch]
                
                # Δcos_i^FP(α) = cos(S_i(α), A_i^FP) - cos(A_i, A_i^FP)
                cos_S_FP = epoch_data['cos_S_FP@0.10'].values  # Using fixed alpha=0.1
                cos_A_FP = epoch_data['cos_A_FP'].values
                delta_cos = cos_S_FP - cos_A_FP
                
                mean, lower, upper = calculate_confidence_interval(delta_cos)
                delta_cos_means.append(mean)
                delta_cos_lower.append(lower)
                delta_cos_upper.append(upper)
                
                # Statistical test for significance
                if len(delta_cos) > 1:
                    t_stat, p_val = stats.ttest_1samp(delta_cos, 0, alternative='greater')
                    if p_val < 0.001:
                        significance.append('***')
                    elif p_val < 0.01:
                        significance.append('**')
                    elif p_val < 0.05:
                        significance.append('*')
                    else:
                        significance.append('')
                else:
                    significance.append('')
            
            # Plot Δcos
            ax.plot(epochs, delta_cos_means,
                   color=style['color'], marker=style['marker'],
                   linestyle=style['linestyle'], label=style['label'],
                   markersize=6, linewidth=1.5)
            ax.fill_between(epochs, delta_cos_lower, delta_cos_upper,
                           color=style['color'], alpha=0.2)
            
            # Add significance markers
            for i, (epoch, sig) in enumerate(zip(epochs, significance)):
                if sig:
                    ax.text(epoch, delta_cos_means[i] + 0.005, sig,
                           ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'$\Delta\cos^{FP}_i(\alpha) = \cos(S_i(\alpha), A_i^{FP}) - \cos(A_i, A_i^{FP})$')
    ax.set_title(f'Small-α FP32 Alignment Improvement (α = {alpha})')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure2_fp32_uniform.pdf", format='pdf')
    plt.savefig(f"{output_dir}/figure2_fp32_uniform.png", format='png')
    plt.close()
    
    print(f"Figure 2 saved to {output_dir}/")

# Copy figure4 function from generate_figure4_only.py
def figure4_safe_mixing_window(data, output_dir):
    """Figure 4: Safe mixing window - from original code"""
    
    epochs = sorted(data['epoch'].unique())
    branches = ['Dsum']
    alphas = [0.00, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00]
    
    # Changed figsize to UNIFORM_FIGSIZE
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
    plt.savefig(f"{output_dir}/figure4_safe_uniform.pdf", format='pdf')
    plt.savefig(f"{output_dir}/figure4_safe_uniform.png", format='png')
    plt.close()
    
    print(f"Figure 4 saved to {output_dir}/")

def accuracy_plot_uniform(output_dir):
    """Accuracy plot with uniform size - from original code"""
    
    # Data from original plot_accuracy_square_fixed.py
    alpha = [0.01, 0.1, 0.3, 0.5, 1, 2, 10]
    device1_ideal = [87.1, 85.53, 87.54, 88.29, 92.1, 91.6, 92.05]
    device2_idealized = [76.58, 74.28, 81.13, 81.48, 84.32, 84.87, 86.8]
    device3_ecram = [52.13, 47.46, 52.57, 56.23, 57.92, 56.87, 54.64]
    
    # Create x positions with equal spacing
    x_positions = np.arange(len(alpha))
    
    # Changed figsize to UNIFORM_FIGSIZE
    plt.figure(figsize=UNIFORM_FIGSIZE)
    
    # Plot each device with specified colors using equal-spaced positions
    plt.plot(x_positions, device1_ideal, 'o-', color='green', linewidth=2, markersize=8, 
             label='Device 1 (Ideal)')
    plt.plot(x_positions, device2_idealized, 's-', color='blue', linewidth=2, markersize=8,
             label='Device 2 (Idealized)')
    plt.plot(x_positions, device3_ecram, '^-', color='red', linewidth=2, markersize=8,
             label='Device 3 (EcramDevice)')
    
    # Set x-axis labels at the positions
    plt.xticks(x_positions, alpha)
    
    # Labels and title
    plt.xlabel('Alpha', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Device Accuracy vs Alpha Parameter', fontsize=14, fontweight='bold')
    
    # Grid
    plt.grid(True, ls="-", alpha=0.2)
    
    # Legend
    plt.legend(loc='best', fontsize=10)
    
    # Set y-axis limits for better visualization
    plt.ylim(45, 95)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_path = f'{output_dir}/accuracy_uniform.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/accuracy_uniform.pdf', bbox_inches='tight')
    print(f"Accuracy plot saved to: {output_path}")
    
    plt.close()

def main():
    # Setup paths
    data_dir = Path("/root/SALMON/probes/Idealized/loss_coef_0_00")
    output_dir = data_dir / "plots_uniform_final"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading data from {data_dir}...")
    data = load_data(data_dir)
    
    print(f"Data shape: {data.shape}")
    print(f"Generating all plots with uniform size {UNIFORM_FIGSIZE}...\n")
    
    # Generate all 4 plots with uniform size
    print("1. Generating Figure 1 (Beta score - Effective Descent Coefficient only)...")
    figure1_beta_only(data, output_dir)
    
    print("2. Generating Figure 2 (Small-α FP32 Alignment Improvement)...")
    figure2_fp32_alignment(data, output_dir)
    
    print("3. Generating Figure 4 (Safe Mixing Window)...")
    figure4_safe_mixing_window(data, output_dir)
    
    print("4. Generating Accuracy Plot...")
    accuracy_plot_uniform(output_dir)
    
    print(f"\nAll plots generated with uniform size in: {output_dir}")
    print("All plots use the EXACT ORIGINAL CODE with only figsize changed to (10, 8)")

if __name__ == "__main__":
    main()