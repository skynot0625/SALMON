#!/usr/bin/env python3
"""
Generate ALL figures with EXACT ORIGINAL CODE - only changing figsize to (10, 8)
Figure 1: Beta score only (from generate_figures.py)
Figure 2: From generate_figure2_only.py
Figure 4: From generate_figure4_only.py
Accuracy: From plot_accuracy_square_fixed.py
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

# ============== FIGURE 1: Beta Score Only (from generate_figures.py) ==============
def figure1_beta_only(data, output_dir):
    """Figure 1: ONLY Beta score (Effective Descent Coefficient) - EXACT from generate_figures.py"""
    
    epochs = sorted(data['epoch'].unique())
    branches = ['Dsum']  # Focus on Dsum for main paper
    
    # Changed from (12, 5) with 2 subplots to single plot with UNIFORM_FIGSIZE
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
                
                # β_i = r_i * cos(D_i, A_i) - EXACT from original
                beta_values = epoch_data['beta_task'].values
                mean, lower, upper = calculate_confidence_interval(beta_values)
                beta_means.append(mean)
                beta_lower.append(lower)
                beta_upper.append(upper)
            
            # Plot β - EXACT from original
            ax.plot(epochs, beta_means,
                   color=style['color'], marker=style['marker'],
                   linestyle=style['linestyle'], label=style['label'],
                   markersize=6, linewidth=1.5)
            ax.fill_between(epochs, beta_lower, beta_upper,
                           color=style['color'], alpha=0.2)
    
    # Formatting - EXACT from original
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'$\beta_i = r_i \cos(D_i, A_i)$')
    ax.set_title('Effective Descent Coefficient')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.legend(loc='best')
    # Note: original doesn't set y-axis limits for beta plot
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure1_beta_uniform.pdf", format='pdf')
    plt.savefig(f"{output_dir}/figure1_beta_uniform.png", format='png')
    plt.close()
    
    print(f"Figure 1 (beta only) saved to {output_dir}/")

# ============== FIGURE 2: From generate_figure2_only.py ==============
def figure2_fp32_alignment(data, output_dir):
    """Figure 2: Small-α FP32 alignment improvement - EXACT from generate_figure2_only.py"""
    
    epochs = sorted(data['epoch'].unique())
    branches = ['Dsum']
    alpha = 0.10  # Fixed small α
    
    # Changed from (8, 6) to UNIFORM_FIGSIZE
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
    plt.savefig(f"{output_dir}/figure2_fp32_uniform.pdf", format='pdf')
    plt.savefig(f"{output_dir}/figure2_fp32_uniform.png", format='png')
    plt.close()
    
    print(f"Figure 2 saved to {output_dir}/")

# ============== Helper function for Figure 4 ==============
def calculate_magnitude_gain(data, alpha):
    """Calculate M_i(α) = ||S_i(α)||/||A_i|| * cos(S_i(α), A_i^FP) / cos(A_i, A_i^FP)"""
    
    # Get necessary columns
    col_name = get_alpha_columns(alpha, 'cos_S_FP')
    cos_S_FP = data[col_name].values
    cos_A_FP = data['cos_A_FP'].values
    r = data['r_norm_ratio_task_task'].values
    cos_D_A = data['cos_D_task_task'].values
    
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

# ============== FIGURE 4: From generate_figure4_only.py ==============
def figure4_safe_mixing_window(data, output_dir):
    """Figure 4: Safe mixing window - EXACT from generate_figure4_only.py"""
    
    epoch = 300
    branches = ['Dsum']
    alpha_values = [0.00, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00]
    
    # Changed from (10, 8) to UNIFORM_FIGSIZE (which is (10, 8) anyway)
    fig, ax = plt.subplots(1, 1, figsize=UNIFORM_FIGSIZE)
    
    epoch_data = data[data['epoch'] == epoch]
    
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
                    M_values = calculate_magnitude_gain(locus_data, alpha)
                    # ρ_i(α) = cos²(S_i(α), A_i)
                    col_name = get_alpha_columns(alpha, 'cos_S_task')
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
            
            # Add alpha labels for selected points with larger font
            labels_to_show = [0.1, 0.25, 0.5]
            for alpha_label in labels_to_show:
                if alpha_label in alpha_values:
                    idx = alpha_values.index(alpha_label)
                    if idx < len(M_values_all):
                        ax.annotate(f'α={alpha_label:.2f}', 
                                  xy=(M_values_all[idx], rho_values_all[idx]),
                                  xytext=(5, 5), textcoords='offset points',
                                  fontsize=12, fontweight='bold', alpha=0.9)
    
    # Add safe region shading
    ax.axvspan(1.0, 3.0, alpha=0.1, color='green', label='M > 1 (Gain region)')
    ax.axhspan(0.8, 1.0, alpha=0.1, color='blue', label='ρ ≥ 0.8 (Safe curvature)')
    
    # Add reference lines
    ax.axvline(x=1, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(y=0.8, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    
    ax.set_xlabel(r'$M_i(\alpha)$ (FP32-Parallel Magnitude Gain)', fontsize=13)
    ax.set_ylabel(r'$\rho_i(\alpha) = \cos^2(S_i(\alpha), A_i)$ (Curvature Safety)', fontsize=13)
    ax.set_title(f'Figure 4: Safe Mixing Window (Epoch {epoch})', fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.set_xlim([0.8, 3.0])
    ax.set_ylim([0.2, 1.0])
    
    # Add arrow to show α increase direction
    ax.annotate('', xy=(2.5, 0.3), xytext=(1.5, 0.8),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.5))
    ax.text(2.0, 0.55, 'α increase', fontsize=11, color='gray', rotation=-60, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure4_safe_uniform.pdf", format='pdf')
    plt.savefig(f"{output_dir}/figure4_safe_uniform.png", format='png')
    plt.close()
    
    print(f"Figure 4 saved to {output_dir}/")

# ============== ACCURACY PLOT (from plot_accuracy_square_fixed.py) ==============
def accuracy_plot_uniform(output_dir):
    """Accuracy plot - EXACT from plot_accuracy_square_fixed.py"""
    
    # Data
    alpha = [0.01, 0.1, 0.3, 0.5, 1, 2, 10]
    device1_ideal = [87.1, 85.53, 87.54, 88.29, 92.1, 91.6, 92.05]
    device2_idealized = [76.58, 74.28, 81.13, 81.48, 84.32, 84.87, 86.8]
    device3_ecram = [52.13, 47.46, 52.57, 56.23, 57.92, 56.87, 54.64]
    
    # Create x positions with equal spacing
    x_positions = np.arange(len(alpha))
    
    # Changed from (8, 8) to UNIFORM_FIGSIZE
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
    print(f"Accuracy plot saved to: {output_path}")
    
    plt.close()

def main():
    # Setup paths
    data_dir = Path("/root/SALMON/probes/Idealized/loss_coef_0_00")
    output_dir = data_dir / "plots_uniform_exact"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading data from {data_dir}...")
    data = load_data(data_dir)
    
    print(f"Data shape: {data.shape}")
    print(f"Generating all plots with uniform size {UNIFORM_FIGSIZE}...\n")
    print("Using EXACT ORIGINAL CODE - only changing figsize\n")
    
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
    print("All plots use EXACT ORIGINAL CODE with only figsize changed to (10, 8)")

if __name__ == "__main__":
    main()