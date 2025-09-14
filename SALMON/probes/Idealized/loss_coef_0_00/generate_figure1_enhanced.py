#!/usr/bin/env python3
"""
Generate Enhanced Figure 1: Three-way cosine similarity analysis
Shows cos(D,A), cos(A,FP32), and cos(D,FP32) together
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

def figure1_three_way_cosines(data, output_dir):
    """Enhanced Figure 1: Three-way cosine similarity (D-A, A-FP32, D-FP32)"""
    
    epochs = sorted(data['epoch'].unique())
    branches = ['Dsum']  # Focus on Dsum for main paper
    
    # Create figure with 3 subplots (one for each locus)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for branch in branches:
        branch_data = data[data['branch'] == branch]
        
        for idx, locus in enumerate(['L1_fea', 'L2_fea', 'L3_fea']):
            ax = axes[idx]
            locus_data = branch_data[branch_data['locus'] == locus]
            
            # Arrays to store statistics
            cos_DA_means = []
            cos_DA_lower = []
            cos_DA_upper = []
            
            cos_AFP_means = []
            cos_AFP_lower = []
            cos_AFP_upper = []
            
            cos_DFP_means = []
            cos_DFP_lower = []
            cos_DFP_upper = []
            
            for epoch in epochs:
                epoch_data = locus_data[locus_data['epoch'] == epoch]
                
                # cos(D, A) - Auxiliary to Task alignment
                cos_DA_values = epoch_data['cos_D_task_task'].values
                mean, lower, upper = calculate_confidence_interval(cos_DA_values)
                cos_DA_means.append(mean)
                cos_DA_lower.append(lower)
                cos_DA_upper.append(upper)
                
                # cos(A, FP32) - Task to FP32 alignment
                cos_AFP_values = epoch_data['cos_A_FP'].values
                mean, lower, upper = calculate_confidence_interval(cos_AFP_values)
                cos_AFP_means.append(mean)
                cos_AFP_lower.append(lower)
                cos_AFP_upper.append(upper)
                
                # cos(D, FP32) - Auxiliary to FP32 alignment
                cos_DFP_values = epoch_data['cos_D_FP'].values
                mean, lower, upper = calculate_confidence_interval(cos_DFP_values)
                cos_DFP_means.append(mean)
                cos_DFP_lower.append(lower)
                cos_DFP_upper.append(upper)
            
            # Plot cos(D, A) - Blue line
            ax.plot(epochs, cos_DA_means, 
                   color='#2E86AB', marker='o', linestyle='-', 
                   label=r'$\cos(D, A)$ (Aux-Task)', 
                   markersize=6, linewidth=2)
            ax.fill_between(epochs, cos_DA_lower, cos_DA_upper, 
                           color='#2E86AB', alpha=0.2)
            
            # Plot cos(A, FP32) - Green line
            ax.plot(epochs, cos_AFP_means,
                   color='#27AE60', marker='s', linestyle='--',
                   label=r'$\cos(A, FP32)$ (Task-FP32)',
                   markersize=6, linewidth=2)
            ax.fill_between(epochs, cos_AFP_lower, cos_AFP_upper,
                           color='#27AE60', alpha=0.2)
            
            # Plot cos(D, FP32) - Red line
            ax.plot(epochs, cos_DFP_means,
                   color='#E74C3C', marker='^', linestyle='-.',
                   label=r'$\cos(D, FP32)$ (Aux-FP32)',
                   markersize=6, linewidth=2)
            ax.fill_between(epochs, cos_DFP_lower, cos_DFP_upper,
                           color='#E74C3C', alpha=0.2)
            
            # Formatting
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Cosine Similarity')
            ax.set_title(f'{LOCUS_STYLES[locus]["label"]} Layer')
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax.legend(loc='best')
            ax.set_ylim([-0.1, 1.0])
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Three-way Gradient Alignment: Auxiliary (D), Task (A), and FP32', y=1.02, fontsize=12)
    plt.tight_layout()
    
    # Save figures
    plt.savefig(f"{output_dir}/figure1_three_way_cosines.pdf", format='pdf')
    plt.savefig(f"{output_dir}/figure1_three_way_cosines.png", format='png')
    plt.show()
    
    print(f"Enhanced Figure 1 saved to {output_dir}/")

def figure1_combined_view(data, output_dir):
    """Alternative view: All loci in one plot with three cosine metrics"""
    
    epochs = sorted(data['epoch'].unique())
    branches = ['Dsum']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Define what to plot in each subplot
    plot_configs = [
        {'metric': 'cos_D_task_task', 'title': r'$\cos(D, A)$ - Auxiliary to Task Alignment', 'ylabel': r'$\cos(D, A)$'},
        {'metric': 'cos_A_FP', 'title': r'$\cos(A, FP32)$ - Task to FP32 Alignment', 'ylabel': r'$\cos(A, FP32)$'},
        {'metric': 'cos_D_FP', 'title': r'$\cos(D, FP32)$ - Auxiliary to FP32 Alignment', 'ylabel': r'$\cos(D, FP32)$'}
    ]
    
    for idx, config in enumerate(plot_configs):
        ax = axes[idx]
        
        for branch in branches:
            branch_data = data[data['branch'] == branch]
            
            for locus in ['L1_fea', 'L2_fea', 'L3_fea']:
                style = LOCUS_STYLES[locus]
                locus_data = branch_data[branch_data['locus'] == locus]
                
                means = []
                lower = []
                upper = []
                
                for epoch in epochs:
                    epoch_data = locus_data[locus_data['epoch'] == epoch]
                    values = epoch_data[config['metric']].values
                    
                    mean, low, up = calculate_confidence_interval(values)
                    means.append(mean)
                    lower.append(low)
                    upper.append(up)
                
                # Plot with locus-specific style
                ax.plot(epochs, means,
                       color=style['color'], marker=style['marker'],
                       linestyle=style['linestyle'], label=style['label'],
                       markersize=6, linewidth=1.5)
                ax.fill_between(epochs, lower, upper,
                               color=style['color'], alpha=0.2)
        
        # Formatting
        ax.set_xlabel('Epoch')
        ax.set_ylabel(config['ylabel'])
        ax.set_title(config['title'])
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.legend(loc='best')
        ax.set_ylim([-0.1, 1.0])
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Gradient Alignment Analysis Across Layers (Dsum)', y=1.02, fontsize=12)
    plt.tight_layout()
    
    # Save figures
    plt.savefig(f"{output_dir}/figure1_combined_cosines.pdf", format='pdf')
    plt.savefig(f"{output_dir}/figure1_combined_cosines.png", format='png')
    plt.show()
    
    print(f"Combined view saved to {output_dir}/")

def figure1_compact_view(data, output_dir):
    """Compact view: Single plot with all three cosines for Dsum across loci"""
    
    epochs = sorted(data['epoch'].unique())
    branch = 'Dsum'
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Line styles for different metrics
    metric_styles = {
        'cos_D_task_task': {'linestyle': '-', 'linewidth': 2.5, 'alpha': 1.0},
        'cos_A_FP': {'linestyle': '--', 'linewidth': 2.0, 'alpha': 0.8},
        'cos_D_FP': {'linestyle': '-.', 'linewidth': 2.0, 'alpha': 0.8}
    }
    
    branch_data = data[data['branch'] == branch]
    
    for locus in ['L1_fea', 'L2_fea', 'L3_fea']:
        style = LOCUS_STYLES[locus]
        locus_data = branch_data[branch_data['locus'] == locus]
        
        for metric, metric_name in [('cos_D_task_task', 'D-A'), 
                                    ('cos_A_FP', 'A-FP32'), 
                                    ('cos_D_FP', 'D-FP32')]:
            means = []
            
            for epoch in epochs:
                epoch_data = locus_data[locus_data['epoch'] == epoch]
                values = epoch_data[metric].values
                mean, _, _ = calculate_confidence_interval(values)
                means.append(mean)
            
            # Create label combining locus and metric
            label = f'{style["label"]} {metric_name}'
            
            # Use locus color but different line styles for metrics
            ax.plot(epochs, means,
                   color=style['color'], 
                   marker=style['marker'] if metric == 'cos_D_task_task' else None,
                   linestyle=metric_styles[metric]['linestyle'],
                   linewidth=metric_styles[metric]['linewidth'],
                   alpha=metric_styles[metric]['alpha'],
                   label=label,
                   markersize=5)
    
    # Formatting
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('Three-way Gradient Alignment: D (Auxiliary), A (Task), FP32', fontsize=13)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Custom legend with two columns
    ax.legend(loc='lower right', ncol=3, fontsize=9, framealpha=0.9)
    ax.set_ylim([-0.1, 1.0])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figures
    plt.savefig(f"{output_dir}/figure1_compact_three_way.pdf", format='pdf')
    plt.savefig(f"{output_dir}/figure1_compact_three_way.png", format='png')
    plt.show()
    
    print(f"Compact three-way view saved to {output_dir}/")

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
    
    # Generate different views
    print("\nGenerating three-way cosine similarity figures...")
    
    print("1. Creating separate plots for each locus...")
    figure1_three_way_cosines(data, output_dir)
    
    print("2. Creating combined view (all loci, separate metrics)...")
    figure1_combined_view(data, output_dir)
    
    print("3. Creating compact view (single plot, all combinations)...")
    figure1_compact_view(data, output_dir)
    
    print("\nAll enhanced Figure 1 variants generated successfully!")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()