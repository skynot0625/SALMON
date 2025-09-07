#!/usr/bin/env python3
"""
Test script to verify task-centered probe implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Test the task-centered metrics
def test_task_metrics():
    """Test the task-centered metric computation logic."""
    
    # Simulate gradients
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create sample gradients
    dim = 100
    g_task = torch.randn(dim, device=device)  # Task gradient (analog backbone)
    g_D = torch.randn(dim, device=device) * 0.1  # Attention gradient (smaller)
    
    # Normalize for testing
    g_task = g_task / g_task.norm()
    g_D = g_D / g_D.norm() * 0.1  # Make gD smaller
    
    # Core task-centered metrics
    dot_AD = torch.dot(g_task, g_D).item()
    cos_D_task = F.cosine_similarity(g_task.unsqueeze(0), g_D.unsqueeze(0)).item()
    r_norm_ratio = (g_D.norm() / g_task.norm()).item()
    
    # Critical alpha
    if dot_AD >= 0:
        alpha_crit = float('inf')  # gD helps
        helps = True
    else:
        alpha_crit = (g_task.norm() ** 2 / (-dot_AD)).item()
        helps = False
    
    # Test different alpha values
    alpha_list = [0.0, 0.5, 1.0, 2.0]
    descent_scores = {}
    
    for alpha in alpha_list:
        g_combined = g_task + alpha * g_D
        # Descent score: how much g_task agrees with combined gradient
        # Positive = good (reduces loss), negative = bad
        descent_score = torch.dot(g_task, g_combined).item()
        descent_scores[f'alpha_{alpha}'] = descent_score
    
    print("Task-Centered Metrics Test")
    print("=" * 50)
    print(f"dot(gD, g_task): {dot_AD:.6f}")
    print(f"cos(gD, g_task): {cos_D_task:.6f}")
    print(f"||gD||/||g_task||: {r_norm_ratio:.6f}")
    print(f"Critical alpha: {alpha_crit:.6f}" if alpha_crit != float('inf') else "Critical alpha: inf")
    print(f"gD helps task: {helps}")
    print("\nDescent scores (higher is better):")
    for key, score in descent_scores.items():
        print(f"  {key}: {score:.6f}")
    
    # Test with aligned gradients (should help)
    print("\n" + "=" * 50)
    print("Test with aligned gradients (gD helps):")
    g_D_aligned = g_task * 0.1 + torch.randn_like(g_task) * 0.01
    g_D_aligned = g_D_aligned / g_D_aligned.norm() * 0.1
    
    dot_AD_aligned = torch.dot(g_task, g_D_aligned).item()
    cos_aligned = F.cosine_similarity(g_task.unsqueeze(0), g_D_aligned.unsqueeze(0)).item()
    
    print(f"dot(gD, g_task): {dot_AD_aligned:.6f} (should be positive)")
    print(f"cos(gD, g_task): {cos_aligned:.6f} (should be close to 1)")
    
    # Test with opposing gradients (should hurt)
    print("\nTest with opposing gradients (gD hurts):")
    g_D_opposed = -g_task * 0.1 + torch.randn_like(g_task) * 0.01
    g_D_opposed = g_D_opposed / g_D_opposed.norm() * 0.1
    
    dot_AD_opposed = torch.dot(g_task, g_D_opposed).item()
    cos_opposed = F.cosine_similarity(g_task.unsqueeze(0), g_D_opposed.unsqueeze(0)).item()
    
    print(f"dot(gD, g_task): {dot_AD_opposed:.6f} (should be negative)")
    print(f"cos(gD, g_task): {cos_opposed:.6f} (should be close to -1)")

if __name__ == "__main__":
    test_task_metrics()
    print("\n✓ Task-centered metrics test completed successfully!")