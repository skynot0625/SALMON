#!/usr/bin/env python3
"""
Task-centered gradient metrics for SALMON evaluation.
Evaluates if attention gradients help the actual task gradient, not FP32 reference.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List

# Attention gradient reachability based on architecture
REACHABLE = {
    'D1': {'L1_fea'},                    # attention1(x1) → L1 only
    'D2': {'L1_fea', 'L2_fea'},         # attention2(x2) → L1, L2
    'D3': {'L1_fea', 'L2_fea', 'L3_fea'}, # attention3(x3) → L1, L2, L3
    'Dsum': {'L1_fea', 'L2_fea', 'L3_fea'}  # Combined reaches all
}

def compute_task_metrics(g_task: Dict[str, torch.Tensor], 
                         g_D: Dict[str, torch.Tensor],
                         branch: str,
                         alpha_list: List[float],
                         loss_coefficient: float = 0.0) -> List[Dict]:
    """
    Compute task-centered metrics for gradient alignment.
    
    Args:
        g_task: Task gradient (analog backbone CE loss gradient)
        g_D: Digital attention gradient for specific branch
        branch: Branch name (D1, D2, D3, or Dsum)
        alpha_list: List of alpha values to evaluate
        loss_coefficient: Training loss coefficient (alpha_train = 1 - loss_coefficient)
    
    Returns:
        List of metric dictionaries for each valid locus
    """
    metrics = []
    alpha_train = 1.0 - loss_coefficient  # Actual training weight
    
    # Get reachable loci for this branch
    reachable_loci = REACHABLE.get(branch, set())
    
    for locus in g_task.keys():
        # Skip if branch cannot reach this locus
        if locus not in reachable_loci:
            continue
            
        gA = g_task[locus]
        gD = g_D.get(locus, torch.zeros_like(gA))
        
        # Skip if gradients are too small
        norm_gA = torch.norm(gA)
        norm_gD = torch.norm(gD)
        
        if norm_gA < 1e-12:
            continue
            
        # Core task-centered metrics
        dot_AD = torch.dot(gA, gD).item()  # Direct contribution
        cos_D_task = F.cosine_similarity(gA.unsqueeze(0), gD.unsqueeze(0)).item() if norm_gD > 1e-12 else 0.0
        r_norm_ratio = (norm_gD / norm_gA).item() if norm_gA > 1e-12 else 0.0
        
        # Critical alpha (where loss reduction becomes zero)
        if dot_AD >= 0:
            alpha_crit = float('inf')  # gD helps, no critical point
        else:
            alpha_crit = (norm_gA ** 2 / (-dot_AD)).item()  # gD hurts beyond this
        
        # Expected loss reduction for different alphas
        # ΔL ≈ -η * gA^T * (gA + α*gD) = -η * (||gA||^2 + α * gA^T*gD)
        descent_scores = {}
        for alpha in alpha_list + [alpha_train]:
            gS = gA + alpha * gD
            # Positive score = loss reduction (good)
            score = torch.dot(gA, gS).item()
            descent_scores[f'descent_α{alpha:.2f}'] = score
        
        # Determine if gD helps or hurts
        helps_task = dot_AD > 0
        
        metric_dict = {
            'branch': branch,
            'locus': locus,
            'dot_AD': dot_AD,
            'cos_D_task': cos_D_task,
            'r_norm_ratio': r_norm_ratio,
            'alpha_crit': alpha_crit,
            'helps_task': int(helps_task),
            'norm_gA': norm_gA.item(),
            'norm_gD': norm_gD.item(),
            **descent_scores
        }
        
        metrics.append(metric_dict)
    
    return metrics

def analyze_gradient_contribution(metrics_list: List[Dict]) -> Dict:
    """
    Analyze overall gradient contribution patterns.
    
    Returns summary statistics about whether attention helps or hurts.
    """
    if not metrics_list:
        return {}
    
    total = len(metrics_list)
    helps_count = sum(1 for m in metrics_list if m['helps_task'])
    hurts_count = total - helps_count
    
    avg_dot_AD = sum(m['dot_AD'] for m in metrics_list) / total
    avg_cos = sum(m['cos_D_task'] for m in metrics_list) / total
    avg_ratio = sum(m['r_norm_ratio'] for m in metrics_list) / total
    
    # Check critical alpha values
    finite_alpha_crits = [m['alpha_crit'] for m in metrics_list 
                          if m['alpha_crit'] != float('inf')]
    avg_alpha_crit = (sum(finite_alpha_crits) / len(finite_alpha_crits) 
                      if finite_alpha_crits else float('inf'))
    
    return {
        'total_pairs': total,
        'helps_count': helps_count,
        'hurts_count': hurts_count,
        'helps_percentage': 100 * helps_count / total,
        'avg_dot_AD': avg_dot_AD,
        'avg_cos_D_task': avg_cos,
        'avg_norm_ratio': avg_ratio,
        'avg_alpha_crit': avg_alpha_crit,
    }

def print_task_metric_summary(metrics_list: List[Dict]):
    """Print human-readable summary of task-centered metrics."""
    
    summary = analyze_gradient_contribution(metrics_list)
    
    print("\n" + "="*60)
    print("TASK-CENTERED GRADIENT ANALYSIS")
    print("="*60)
    
    print(f"\n1. GRADIENT CONTRIBUTION:")
    print(f"   • Helps task: {summary['helps_count']}/{summary['total_pairs']} "
          f"({summary['helps_percentage']:.1f}%)")
    print(f"   • Hurts task: {summary['hurts_count']}/{summary['total_pairs']} "
          f"({100-summary['helps_percentage']:.1f}%)")
    
    print(f"\n2. AVERAGE METRICS:")
    print(f"   • Mean dot(gD, g_task): {summary['avg_dot_AD']:.6f}")
    if summary['avg_dot_AD'] > 0:
        print(f"     → On average, gD HELPS the task gradient")
    else:
        print(f"     → On average, gD HURTS the task gradient")
    
    print(f"   • Mean cos(gD, g_task): {summary['avg_cos_D_task']:.4f}")
    print(f"   • Mean ||gD||/||g_task||: {summary['avg_norm_ratio']:.4f}")
    
    if summary['avg_alpha_crit'] != float('inf'):
        print(f"\n3. CRITICAL ALPHA:")
        print(f"   • Average α_crit: {summary['avg_alpha_crit']:.4f}")
        print(f"     (gD becomes harmful beyond this α)")
    
    # Per-branch analysis
    print(f"\n4. PER-BRANCH BREAKDOWN:")
    branches = {}
    for m in metrics_list:
        b = m['branch']
        if b not in branches:
            branches[b] = []
        branches[b].append(m['dot_AD'])
    
    for branch, dots in branches.items():
        avg = sum(dots) / len(dots)
        helps = sum(1 for d in dots if d > 0)
        print(f"   {branch:6}: avg dot={avg:+.6f}, helps {helps}/{len(dots)} times")

# Example usage in exp7_6_module.py:
"""
def _compute_metrics_task_centered(self, inputs, labels, alpha_list):
    # Get task gradient (backbone only)
    g_task = self._get_gA_act(inputs, labels)
    
    # Get attention gradients for each branch
    all_metrics = []
    for branch in ['D1', 'D2', 'D3', 'Dsum']:
        g_D = self._get_gD_act(inputs, labels, which=branch)
        metrics = compute_task_metrics(
            g_task, g_D, branch, 
            alpha_list, self.loss_coefficient
        )
        all_metrics.extend(metrics)
    
    return all_metrics
"""
