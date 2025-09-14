#!/usr/bin/env python3
"""
Generate Figures for Task-Consistency Analysis of Auxiliary Gradients
Dataset: probes/Idealized/loss_coef_0_00
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

def figure1_task_consistency(data, output_dir):
    """Figure 1: Task-consistency of auxiliary gradients (Fact 1)"""
    
    epochs = sorted(data['epoch'].unique())
    branches = ['Dsum']  # Focus on Dsum for main paper
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for branch in branches:
        branch_data = data[data['branch'] == branch]
        
        for locus in ['L1_fea', 'L2_fea', 'L3_fea']:
            style = LOCUS_STYLES[locus]
            locus_data = branch_data[branch_data['locus'] == locus]
            
            cos_means = []
            cos_lower = []
            cos_upper = []
            beta_means = []
            beta_lower = []
            beta_upper = []
            positive_ratios = []
            
            for epoch in epochs:
                epoch_data = locus_data[locus_data['epoch'] == epoch]
                
                # cos(D_i, A_i)
                cos_values = epoch_data['cos_D_task_task'].values
                mean, lower, upper = calculate_confidence_interval(cos_values)
                cos_means.append(mean)
                cos_lower.append(lower)
                cos_upper.append(upper)
                
                # β_i = r_i * cos(D_i, A_i)
                beta_values = epoch_data['beta_task'].values
                mean, lower, upper = calculate_confidence_interval(beta_values)
                beta_means.append(mean)
                beta_lower.append(lower)
                beta_upper.append(upper)
                
                # Pr[cos > 0]
                positive_ratio = np.mean(cos_values > 0)
                positive_ratios.append(positive_ratio)
            
            # Plot cos(D, A)
            ax1.plot(epochs, cos_means, 
                    color=style['color'], marker=style['marker'], 
                    linestyle=style['linestyle'], label=style['label'],
                    markersize=6, linewidth=1.5)
            ax1.fill_between(epochs, cos_lower, cos_upper, 
                            color=style['color'], alpha=0.2)
            
            # Plot β
            ax2.plot(epochs, beta_means,
                    color=style['color'], marker=style['marker'],
                    linestyle=style['linestyle'], label=style['label'],
                    markersize=6, linewidth=1.5)
            ax2.fill_between(epochs, beta_lower, beta_upper,
                            color=style['color'], alpha=0.2)
    
    # Formatting
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(r'$\cos(D_i, A_i)$')
    ax1.set_title('(a) Task-Alignment of Auxiliary Gradients')
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.legend(loc='best')
    ax1.set_ylim([-0.1, 1.0])
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(r'$\beta_i = r_i \cos(D_i, A_i)$')
    ax2.set_title('(b) Effective Descent Coefficient')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.legend(loc='best')
    
    plt.suptitle('Figure 1: Task-Consistency of Auxiliary Gradients (Proposition 1)', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure1_task_consistency.pdf", format='pdf')
    plt.savefig(f"{output_dir}/figure1_task_consistency.png", format='png')
    plt.show()
    
    print(f"Figure 1 saved to {output_dir}/")

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
                
                # Statistical test for significance (removed markers)
                # Significance testing removed per user request
            
            # Plot Δcos
            ax.plot(epochs, delta_cos_means,
                   color=style['color'], marker=style['marker'],
                   linestyle=style['linestyle'], label=style['label'],
                   markersize=6, linewidth=1.5)
            ax.fill_between(epochs, delta_cos_lower, delta_cos_upper,
                           color=style['color'], alpha=0.2)
            
            # Significance markers removed per user request
    
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

def calculate_magnitude_gain(data, alpha):
    """Calculate M_i(α) = ||S_i(α)|| cos(S_i(α), A_i^FP) / (||A_i|| cos(A_i, A_i^FP))"""
    
    # Get necessary values
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

def figure3_magnitude_gain(data, output_dir):
    """Figure 3: FP32-parallel magnitude gain vs. α"""
    
    # Use the available alpha values from the CSV columns
    alpha_values = [0.00, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00]
    selected_epochs = [10, 150, 300]
    branches = ['Dsum']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, epoch in enumerate(selected_epochs):
        ax = axes[idx]
        epoch_data = data[data['epoch'] == epoch]
        
        for branch in branches:
            branch_data = epoch_data[epoch_data['branch'] == branch]
            
            for locus in ['L1_fea', 'L2_fea', 'L3_fea']:
                style = LOCUS_STYLES[locus]
                locus_data = branch_data[branch_data['locus'] == locus]
                
                if len(locus_data) == 0:
                    continue
                
                M_means = []
                M_lower = []
                M_upper = []
                
                for alpha in alpha_values:
                    if alpha == 0:
                        M_values = np.ones(len(locus_data))
                    else:
                        M_values = calculate_magnitude_gain(locus_data, alpha)
                    
                    mean, lower, upper = calculate_confidence_interval(M_values)
                    M_means.append(mean)
                    M_lower.append(lower)
                    M_upper.append(upper)
                
                ax.plot(alpha_values, M_means,
                       color=style['color'], marker=style['marker'],
                       linestyle=style['linestyle'], label=style['label'],
                       markersize=4, linewidth=1.5)
                ax.fill_between(alpha_values, M_lower, M_upper,
                               color=style['color'], alpha=0.2)
        
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel(r'$M_i(\alpha)$')
        ax.set_title(f'Epoch {epoch}')
        ax.axhline(y=1, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.legend(loc='best')
        ax.set_ylim([0.5, 3.0])
    
    plt.suptitle('Figure 3: FP32-Parallel Magnitude Gain vs. α', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure3_magnitude_gain.pdf", format='pdf')
    plt.savefig(f"{output_dir}/figure3_magnitude_gain.png", format='png')
    plt.show()
    
    print(f"Figure 3 saved to {output_dir}/")

def figure4_safe_mixing_window(data, output_dir):
    """Figure 4: Safe mixing window - curvature-safety vs. FP32-gain"""
    
    epoch = 300
    branches = ['Dsum']
    alpha_values = [0.00, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
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
    plt.savefig(f"{output_dir}/figure4_safe_mixing_window.pdf", format='pdf')
    plt.savefig(f"{output_dir}/figure4_safe_mixing_window.png", format='png')
    plt.show()
    
    print(f"Figure 4 saved to {output_dir}/")

def main():
    # Setup paths
    data_dir = Path("/root/SALMON/probes/Idealized/loss_coef_0_00")
    output_dir = data_dir / "plots"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading data from {data_dir}...")
    data = load_data(data_dir)
    
    print(f"Data shape: {data.shape}")
    print(f"Unique epochs: {sorted(data['epoch'].unique())}")
    print(f"Unique branches: {data['branch'].unique()}")
    print(f"Unique loci: {data['locus'].unique()}")
    
    # Generate figures
    print("\nGenerating Figure 1: Task-consistency of auxiliary gradients...")
    figure1_task_consistency(data, output_dir)
    
    print("\nGenerating Figure 2: Small-α FP32 alignment improvement...")
    figure2_fp32_alignment(data, output_dir)
    
    print("\nGenerating Figure 3: FP32-parallel magnitude gain...")
    figure3_magnitude_gain(data, output_dir)
    
    print("\nGenerating Figure 4: Safe mixing window...")
    figure4_safe_mixing_window(data, output_dir)
    
    print("\nAll figures generated successfully!")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()