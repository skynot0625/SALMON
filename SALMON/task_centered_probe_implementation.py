#!/usr/bin/env python3
"""
Complete task-centered probe implementation for exp7_6_module.py
This code should be integrated into the SalmonLitModule class.
"""

import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
from contextlib import contextmanager

# ============= CONSTANTS =============

# Branch-to-layer reachability mapping based on architecture
REACHABLE = {
    'D1': {'L1_fea'},                    # attention1(x1) -> L1 only
    'D2': {'L1_fea', 'L2_fea'},          # attention2(x2) -> L1, L2  
    'D3': {'L1_fea', 'L2_fea', 'L3_fea'}, # attention3(x3) -> L1, L2, L3
    'Dsum': {'L1_fea', 'L2_fea', 'L3_fea'}  # Combined reaches L1, L2, L3
}

class TaskCenteredProbes:
    """Add these methods to exp7_6_module.SalmonLitModule"""
    
    # ============= 1. EXTENDED LAYER HANDLES =============
    
    def _setup_layer_handles_extended(self):
        """Setup layer handles including L1, L2, L3 for complete gradient capture."""
        self.layer_handles = {}
        
        # Helper to find first conv in a block
        def find_first_conv(module):
            for m in module.modules():
                if isinstance(m, (nn.Conv2d, AnalogConv2d)):
                    return m
            return None
        
        # Get handles for L1, L2, L3 (required for attention gradients)
        try:
            # Layer 1: First residual block's first conv
            l1_block = self.features.layer1[-1]
            if hasattr(l1_block, 'residual_function'):
                l1_conv = find_first_conv(l1_block.residual_function)
            else:
                l1_conv = find_first_conv(l1_block)
            self.layer_handles['L1_fea'] = l1_conv
            
            # Layer 2: Similar approach
            l2_block = self.features.layer2[-1]
            if hasattr(l2_block, 'residual_function'):
                l2_conv = find_first_conv(l2_block.residual_function)
            else:
                l2_conv = find_first_conv(l2_block)
            self.layer_handles['L2_fea'] = l2_conv
            
            # Layer 3: Similar approach
            l3_block = self.features.layer3[-1]
            if hasattr(l3_block, 'residual_function'):
                l3_conv = find_first_conv(l3_block.residual_function)
            else:
                l3_conv = find_first_conv(l3_block)
            self.layer_handles['L3_fea'] = l3_conv
            
            # FB (Feature Backbone) - not reachable by attention but keep for reference
            self.layer_handles['FB'] = None  # Will use activation gradients for FB
            
            print(f"Extended layer handles set up: {list(self.layer_handles.keys())}")
            
        except Exception as e:
            print(f"Warning: Failed to setup extended layer handles: {e}")
            # Fallback to basic handles
            self._setup_layer_handles()
    
    # ============= 2. BRANCH-SEPARATED GRADIENT CAPTURE =============
    
    def _get_gD_split_act(self, inputs, labels):
        """Get attention gradients for each branch separately using activation gradients."""
        grads = {}
        
        # Forward pass to get activations
        taps = self._get_taps(inputs, use_fp32=False)
        
        # D1: attention1(x1) gradient
        self.zero_grad(set_to_none=True)
        out1, _ = self.attention1(taps['L1_fea'])
        loss1 = self.criterion(out1, labels)
        grad1 = self._backward_and_capture(taps, loss1)
        grads['D1'] = grad1
        
        # D2: attention2(x2) gradient  
        self.zero_grad(set_to_none=True)
        out2, _ = self.attention2(taps['L2_fea'])
        loss2 = self.criterion(out2, labels)
        grad2 = self._backward_and_capture(taps, loss2)
        grads['D2'] = grad2
        
        # D3: attention3(x3) gradient
        self.zero_grad(set_to_none=True)
        out3, _ = self.attention3(taps['L3_fea'])
        loss3 = self.criterion(out3, labels)
        grad3 = self._backward_and_capture(taps, loss3)
        grads['D3'] = grad3
        
        # Dsum: Combined gradient (as used in training)
        self.zero_grad(set_to_none=True)
        out1, _ = self.attention1(taps['L1_fea'])
        out2, _ = self.attention2(taps['L2_fea'])
        out3, _ = self.attention3(taps['L3_fea'])
        loss_sum = (self.criterion(out1, labels) + 
                    self.criterion(out2, labels) + 
                    self.criterion(out3, labels))
        grad_sum = self._backward_and_capture(taps, loss_sum)
        grads['Dsum'] = grad_sum
        
        return grads
    
    # ============= 3. TASK-CENTERED METRICS =============
    
    def _compute_metrics_task_centered(self, g_task, gD_split, alpha_list, loss_coefficient=0.0):
        """
        Compute task-centered metrics for gradient alignment.
        
        Args:
            g_task: Task gradient (analog backbone CE loss) 
            gD_split: Dict of attention gradients per branch
            alpha_list: List of alpha values to evaluate
            loss_coefficient: Training loss coefficient
        
        Returns:
            List of metric dictionaries
        """
        metrics = []
        alpha_train = 1.0 - loss_coefficient  # Actual training weight
        
        # Ensure alpha_train is in the list
        alpha_eval = sorted(set(list(alpha_list) + [alpha_train]))
        
        for branch, gD_dict in gD_split.items():
            # Get reachable loci for this branch
            reachable_loci = REACHABLE.get(branch, set())
            
            for locus in g_task.keys():
                # Skip if branch cannot reach this locus
                if locus not in reachable_loci:
                    continue
                    
                vA = g_task[locus]
                vD = gD_dict.get(locus, torch.zeros_like(vA))
                
                # Skip if gradients are too small
                norm_gA = torch.norm(vA)
                norm_gD = torch.norm(vD)
                
                if norm_gA < 1e-12:
                    continue
                
                # Core task-centered metrics
                dot_AD = torch.dot(vA, vD).item()
                cos_D_task = F.cosine_similarity(vA.unsqueeze(0), vD.unsqueeze(0)).item() if norm_gD > 1e-12 else 0.0
                r_norm_ratio = (norm_gD / norm_gA).item()
                
                # Critical alpha (where loss reduction becomes zero)
                if dot_AD >= 0:
                    alpha_crit = float('inf')  # gD helps, no critical point
                else:
                    alpha_crit = (norm_gA ** 2 / (-dot_AD)).item()
                
                # Compute metrics for each alpha
                for alpha in alpha_eval:
                    # Combined gradient
                    vS = vA + alpha * vD
                    norm_gS = torch.norm(vS)
                    
                    # Descent score: gA^T * gS (positive = good)
                    descent_score = torch.dot(vA, vS).item()
                    
                    # Also keep FP32 metrics if available (as supplementary)
                    # These would come from existing FP32 comparison
                    
                    metric = {
                        'branch': branch,
                        'locus': locus,
                        'alpha': alpha,
                        'is_train_alpha': int(abs(alpha - alpha_train) < 1e-6),
                        'descent_score': descent_score,
                        'cos_D_task': cos_D_task,
                        'dot_AD': dot_AD,
                        'r_norm_ratio': r_norm_ratio,
                        'alpha_crit': alpha_crit,
                        'helps_task': int(dot_AD > 0),
                        'norm_gA': norm_gA.item(),
                        'norm_gD': norm_gD.item(),
                        'norm_gS': norm_gS.item(),
                    }
                    
                    metrics.append(metric)
        
        return metrics
    
    # ============= 4. CSV LOGGING WITH NEW SCHEMA =============
    
    def _setup_csv_logging_task_centered(self):
        """Setup CSV logging with task-centered schema."""
        os.makedirs('./probes', exist_ok=True)
        
        # Include run ID or seed in filename
        run_id = f"seed_{self.probe_seed}_task"
        csv_path = f'./probes/device2_resnet18_cifar10_{run_id}.csv'
        
        self.csv_file = open(csv_path, 'w', newline='')
        print(f"[probe] Task-centered CSV output: {csv_path}")
        
        fieldnames = [
            # Basic info
            'epoch', 'batch', 'branch', 'locus', 'alpha', 'is_train_alpha',
            # Task-centered metrics
            'descent_score', 'cos_D_task', 'dot_AD', 'r_norm_ratio', 'alpha_crit', 'helps_task',
            # Gradient norms
            'norm_gA', 'norm_gD', 'norm_gS',
            # Optional: FP32 comparison metrics (kept for reference)
            'cos_A_FP', 'cos_S_FP', 'cos_D_FP', 'angle_improve_deg'
        ]
        
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()
        self.csv_file.flush()
    
    # ============= 5. MAIN PROBE RUNNER WITH TASK FOCUS =============
    
    def run_gradient_probes_task_centered(self, epoch: int):
        """Run task-centered gradient probes."""
        if not self.probe_enabled or epoch not in self.probe_epochs:
            return
        
        print(f"\n=== Running Task-Centered Gradient Probes at Epoch {epoch} ===")
        print("Evaluating: Does gD help or hurt g_task?")
        print(f"α_train = {1.0 - self.loss_coefficient}")
        
        # Track statistics
        stats = {'helps': 0, 'hurts': 0, 'total': 0}
        
        with self.bn_eval_both():
            actual_batches = max(self.probe_batches_per_epoch, 8)
            
            for batch_idx in range(actual_batches):
                try:
                    inputs, labels = self.get_probe_batch()
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                except Exception as e:
                    print(f"Error getting batch {batch_idx}: {e}")
                    break
                
                # Get task gradient (backbone only)
                try:
                    g_task = self._get_gA_act(inputs, labels)
                except Exception as e:
                    print(f"Error computing task gradient: {e}")
                    continue
                
                # Get attention gradients for each branch
                try:
                    gD_split = self._get_gD_split_act(inputs, labels)
                except Exception as e:
                    print(f"Error computing attention gradients: {e}")
                    continue
                
                # Compute task-centered metrics
                metrics = self._compute_metrics_task_centered(
                    g_task, gD_split, 
                    self.probe_alpha_eval,
                    self.loss_coefficient
                )
                
                # Also compute FP32 metrics if needed (optional)
                if self.fp32_clone is not None:
                    try:
                        gFP32 = self._get_gFP32_act(inputs, labels)
                        # Add FP32 comparison to metrics
                        for m in metrics:
                            # Simple comparison at matching locus
                            if m['locus'] in gFP32:
                                g_FP = gFP32[m['locus']]
                                g_A = g_task[m['locus']]
                                g_S = g_A + m['alpha'] * gD_split[m['branch']].get(m['locus'], torch.zeros_like(g_A))
                                
                                # Compute FP32 alignment
                                cos_A_FP = F.cosine_similarity(g_A.unsqueeze(0), g_FP.unsqueeze(0)).item()
                                cos_S_FP = F.cosine_similarity(g_S.unsqueeze(0), g_FP.unsqueeze(0)).item()
                                
                                m['cos_A_FP'] = cos_A_FP
                                m['cos_S_FP'] = cos_S_FP
                                m['angle_improve_deg'] = np.arccos(np.clip(cos_A_FP, -1, 1)) - np.arccos(np.clip(cos_S_FP, -1, 1))
                                m['angle_improve_deg'] *= 180 / np.pi
                    except Exception as e:
                        print(f"Warning: Failed to compute FP32 metrics: {e}")
                
                # Write metrics to CSV
                for m in metrics:
                    # Fill in missing fields
                    m['epoch'] = epoch
                    m['batch'] = batch_idx
                    
                    # Add default values for optional fields
                    for field in ['cos_A_FP', 'cos_S_FP', 'cos_D_FP', 'angle_improve_deg']:
                        if field not in m:
                            m[field] = 0.0
                    
                    self.csv_writer.writerow(m)
                    
                    # Update statistics
                    if m['is_train_alpha']:
                        stats['total'] += 1
                        if m['helps_task']:
                            stats['helps'] += 1
                        else:
                            stats['hurts'] += 1
                
                # Flush after each batch
                self.csv_file.flush()
                
                # Clear memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Print summary
        if stats['total'] > 0:
            help_pct = 100 * stats['helps'] / stats['total']
            print(f"Summary at α_train: {stats['helps']}/{stats['total']} "
                  f"({help_pct:.1f}%) cases where gD helps g_task")
        
        print(f"Completed task-centered gradient probes for epoch {epoch}")

# ============= INTEGRATION NOTES =============
"""
To integrate into exp7_6_module.py:

1. Add REACHABLE constant at module level

2. Replace/add these methods in SalmonLitModule:
   - _setup_layer_handles_extended() 
   - _get_gD_split_act()
   - _compute_metrics_task_centered()
   - _setup_csv_logging_task_centered()
   - run_gradient_probes_task_centered()

3. In __init__, call:
   self._setup_layer_handles_extended()
   self._setup_csv_logging_task_centered()

4. In on_train_epoch_start, call:
   self.run_gradient_probes_task_centered(self.current_epoch)

5. Ensure loss_coefficient is accessible:
   self.loss_coefficient = hparams.get('loss_coefficient', 0.0)
"""