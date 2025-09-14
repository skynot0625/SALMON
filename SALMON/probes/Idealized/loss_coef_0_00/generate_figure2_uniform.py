#!/usr/bin/env python3
"""
Generate Figure 2 with uniform size (10, 8) to match figure4
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
    """Load FP32 alignment CSV files"""
    fp32_files = sorted(glob.glob(f"{data_dir}/epoch_*_fp32_alignment.csv"))
    
    fp32_dfs = []
    for ff in fp32_files:
        fp32_df = pd.read_csv(ff)
        fp32_dfs.append(fp32_df)
    
    fp32_data = pd.concat(fp32_dfs, ignore_index=True)
    return fp32_data

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

def figure2_fp32_alignment(data, output_dir):
    """Generate Figure 2: FP32 Alignment with uniform size"""
    
    epochs = sorted(data['epoch'].unique())
    branches = ['Dsum']
    
    # Create figure with same size as figure4
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    for branch in branches:
        branch_data = data[data['branch'] == branch]
        
        for locus in ['L1_fea', 'L2_fea', 'L3_fea']:
            locus_data = branch_data[branch_data['locus'] == locus]
            style = LOCUS_STYLES[locus]
            
            cos_means = []
            cos_lower = []
            cos_upper = []
            
            for epoch in epochs:
                epoch_data = locus_data[locus_data['epoch'] == epoch]
                cos_values = epoch_data['cos_A_FP']
                mean, lower, upper = calculate_confidence_interval(cos_values)
                cos_means.append(mean)
                cos_lower.append(lower)
                cos_upper.append(upper)
            
            ax.plot(epochs, cos_means,
                   color=style['color'],
                   marker=style['marker'],
                   linestyle=style['linestyle'],
                   label=style['label'],
                   markersize=6,
                   linewidth=2)
            
            ax.fill_between(epochs, cos_lower, cos_upper,
                           color=style['color'], alpha=0.2)
    
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel(r'$\cos(A, FP32)$', fontsize=11)
    ax.set_title('FP32 Alignment', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, alpha=0.3)
    
    # Save the figure
    output_path = Path(output_dir) / 'plots' / 'figure2_fp32_alignment_uniform.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure 2 (FP32 Alignment) saved to: {output_path}")
    
    # Also save PDF version
    output_path_pdf = Path(output_dir) / 'plots' / 'figure2_fp32_alignment_uniform.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"PDF version saved to: {output_path_pdf}")
    
    plt.close()

# Main execution
if __name__ == "__main__":
    data_dir = Path(__file__).parent
    output_dir = Path(__file__).parent
    
    print("Loading FP32 alignment data...")
    data = load_data(data_dir)
    
    print("Generating Figure 2 (FP32 Alignment)...")
    figure2_fp32_alignment(data, output_dir)