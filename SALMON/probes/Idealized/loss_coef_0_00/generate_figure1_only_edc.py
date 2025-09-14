#!/usr/bin/env python3
"""
Generate Figure 1 with ONLY Effective Descent Coefficient
Using same figure size as figure4 (10, 8)
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
    
    task_dfs = []
    for tf in task_files:
        task_df = pd.read_csv(tf)
        task_dfs.append(task_df)
    
    task_data = pd.concat(task_dfs, ignore_index=True)
    return task_data

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

def figure1_effective_descent_only(data, output_dir):
    """Generate ONLY the Effective Descent Coefficient plot"""
    
    epochs = sorted(data['epoch'].unique())
    branches = ['Dsum']  # Focus on Dsum for main paper
    
    # Create figure with same size as figure4
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    for branch in branches:
        branch_data = data[data['branch'] == branch]
        
        for locus in ['L1_fea', 'L2_fea', 'L3_fea']:
            locus_data = branch_data[branch_data['locus'] == locus]
            style = LOCUS_STYLES[locus]
            
            # Arrays to store statistics for beta_l = r*cos(D,A)
            beta_means = []
            beta_lower = []
            beta_upper = []
            
            for epoch in epochs:
                epoch_data = locus_data[locus_data['epoch'] == epoch]
                # Calculate effective descent coefficient
                beta_values = epoch_data['r_norm_ratio_task'] * epoch_data['cos_D_task']
                mean, lower, upper = calculate_confidence_interval(beta_values)
                beta_means.append(mean)
                beta_lower.append(lower)
                beta_upper.append(upper)
            
            # Plot with confidence intervals
            ax.plot(epochs, beta_means, 
                   color=style['color'],
                   marker=style['marker'],
                   linestyle=style['linestyle'],
                   label=style['label'],
                   markersize=6,
                   linewidth=2)
            
            ax.fill_between(epochs, beta_lower, beta_upper, 
                           color=style['color'], alpha=0.2)
    
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel(r'$\beta_l = r \cdot \cos(D, A)$', fontsize=11)
    ax.set_title('Effective Descent Coefficient', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='right', frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, alpha=0.3)
    
    # Save the figure
    output_path = Path(output_dir) / 'plots' / 'figure1_effective_descent_only.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure 1 (Effective Descent only) saved to: {output_path}")
    
    # Also save PDF version
    output_path_pdf = Path(output_dir) / 'plots' / 'figure1_effective_descent_only.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"PDF version saved to: {output_path_pdf}")
    
    plt.close()

# Main execution
if __name__ == "__main__":
    data_dir = Path(__file__).parent
    output_dir = Path(__file__).parent
    
    print("Loading data...")
    data = load_data(data_dir)
    
    print("Generating Figure 1 (Effective Descent Coefficient only)...")
    figure1_effective_descent_only(data, output_dir)