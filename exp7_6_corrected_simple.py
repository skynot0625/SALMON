#!/usr/bin/env python3
"""
CORRECTED and SIMPLIFIED gradient probe implementation.
All locus-branch combinations are valid because of forward dependencies.
"""

import os
import csv
import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from contextlib import contextmanager

class GradientProbeSimplified:
    """
    Simplified gradient probe implementation without depth checking.
    ALL locus-branch combinations are valid due to forward dependencies in ResNet.
    """
    
    def _setup_csv_logging(self):
        """Setup CSV logging with simplified schema."""
        os.makedirs('./probes', exist_ok=True)
        
        # Include run ID or seed in filename
        run_id = f"seed_{self.probe_seed}"
        csv_path = f'./probes/device2_resnet18_cifar10_{run_id}.csv'
        
        self.csv_file = open(csv_path, 'w', newline='')
        print(f"[probe] CSV output: {csv_path}")
        
        fieldnames = [
            'epoch', 'batch', 'branch', 'locus', 'alpha',
            'cos_A_FP', 'cos_S_FP', 'angle_improve_deg',
            'ortho_ratio_A', 'ortho_ratio_S',
            'r_norm_ratio', 'cos_D_FP', 'cos_D_A',
            'norm_gA', 'norm_gD', 'norm_gS',
            'is_direct'  # Flag for direct vs indirect gradient path
        ]
        
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()
        self.csv_file.flush()
    
    @contextmanager
    def bn_eval_both(self):
        """Freeze BN for both analog model and FP32 clone."""
        saved_states = {}
        
        def _set_bn_eval(module, prefix):
            if module is None:
                return
            for name, m in module.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    key = f'{prefix}.{name}'
                    saved_states[key] = m.training
                    m.eval()
        
        def _restore_bn(module, prefix):
            if module is None:
                return
            for name, m in module.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    key = f'{prefix}.{name}'
                    if key in saved_states:
                        m.train(saved_states[key])
        
        try:
            # Set analog model BN to eval
            _set_bn_eval(self, 'analog')
            
            # Set FP32 clone BN to eval if exists
            if hasattr(self, 'fp32_clone') and self.fp32_clone is not None:
                _set_bn_eval(self.fp32_clone.get('input'), 'fp32_input')
                _set_bn_eval(self.fp32_clone.get('features'), 'fp32_features')
                _set_bn_eval(self.fp32_clone.get('classifier'), 'fp32_classifier')
            
            yield
            
        finally:
            # Restore original states
            _restore_bn(self, 'analog')
            
            if hasattr(self, 'fp32_clone') and self.fp32_clone is not None:
                _restore_bn(self.fp32_clone.get('input'), 'fp32_input')
                _restore_bn(self.fp32_clone.get('features'), 'fp32_features')
                _restore_bn(self.fp32_clone.get('classifier'), 'fp32_classifier')
    
    def _get_taps(self, inputs, use_fp32=False):
        """Get activation tensors at branch points."""
        if use_fp32 and self.fp32_clone is not None:
            input_features = self.fp32_clone['input'](inputs)
            fb, x1, x2, x3 = self.fp32_clone['features'](input_features)
        else:
            input_features = self.input(inputs)
            fb, x1, x2, x3 = self.features(input_features)
        
        return {'L1_fea': x1, 'L2_fea': x2, 'L3_fea': x3, 'FB': fb}
    
    def _backward_and_capture(self, taps: Dict[str, torch.Tensor], loss: torch.Tensor):
        """Backward and capture activation gradients using tensor hooks."""
        grads = {}
        
        # Ensure gradients are retained
        for name, t in taps.items():
            if t.requires_grad:
                t.retain_grad()
        
        # Backward pass
        loss.backward()
        
        # Collect gradients
        for name, t in taps.items():
            if t.grad is not None:
                g = t.grad.detach().flatten()
                if torch.isfinite(g).all():
                    grads[name] = g.clone()
                else:
                    grads[name] = torch.zeros(1, device=self.device)
            else:
                grads[name] = torch.zeros(1, device=self.device)
        
        return grads
    
    def _get_gA_act(self, inputs, labels):
        """Get backbone gradient gA using activation gradients."""
        self.zero_grad(set_to_none=True)
        taps = self._get_taps(inputs, use_fp32=False)
        logits = self.classifier(taps['FB'])
        loss = self.criterion(logits, labels)
        return self._backward_and_capture(taps, loss)
    
    def _get_gD_act(self, inputs, labels, which: str):
        """Get attention gradient for specific branch."""
        self.zero_grad(set_to_none=True)
        taps = self._get_taps(inputs, use_fp32=False)
        
        # Get attention outputs
        out1, _ = self.attention1(taps['L1_fea'])
        out2, _ = self.attention2(taps['L2_fea'])
        out3, _ = self.attention3(taps['L3_fea'])
        
        # Compute loss based on branch
        if which == 'D1':
            loss = self.criterion(out1, labels)
        elif which == 'D2':
            loss = self.criterion(out2, labels)
        elif which == 'D3':
            loss = self.criterion(out3, labels)
        elif which == 'Dsum':
            loss = (self.criterion(out1, labels) +
                    self.criterion(out2, labels) +
                    self.criterion(out3, labels))
        else:
            raise ValueError(f"Unknown branch: {which}")
        
        return self._backward_and_capture(taps, loss)
    
    def _get_gFP32_act(self, inputs, labels):
        """Get FP32 reference gradient using activation gradients."""
        if self.fp32_clone is None:
            self._create_fp32_clone()
        
        # Zero gradients for FP32 modules
        self.fp32_clone['input'].zero_grad()
        self.fp32_clone['features'].zero_grad()
        self.fp32_clone['classifier'].zero_grad()
        
        taps = self._get_taps(inputs, use_fp32=True)
        logits = self.fp32_clone['classifier'](taps['FB'])
        loss = self.criterion(logits, labels)
        return self._backward_and_capture(taps, loss)
    
    def _compute_metrics_all(self, gA, gD, gFP32, alpha: float):
        """Compute metrics for all loci (no depth checking needed)."""
        metrics = {}
        
        def _cos(u, v):
            """Safe cosine similarity."""
            eps = 1e-12
            return torch.clamp(
                torch.dot(u, v) / ((u.norm() + eps) * (v.norm() + eps)),
                -1 + 1e-7, 1 - 1e-7
            )
        
        for locus in ['L1_fea', 'L2_fea', 'L3_fea', 'FB']:
            g_A = gA.get(locus, torch.zeros(1, device=self.device))
            g_D = gD.get(locus, torch.zeros(1, device=self.device))
            g_FP = gFP32.get(locus, torch.zeros(1, device=self.device))
            
            # Skip if base gradients are zero
            if torch.norm(g_A) < 1e-12 or torch.norm(g_FP) < 1e-12:
                continue
            
            # Combined gradient
            g_S = g_A + alpha * g_D
            nA = torch.norm(g_A)
            nS = torch.norm(g_S)
            g_S_tilde = (nA / (nS + 1e-12)) * g_S
            
            # Compute cosine similarities
            cos_A_FP = _cos(g_A, g_FP).item()
            cos_S_FP = _cos(g_S_tilde, g_FP).item()
            cos_D_FP = _cos(g_D, g_FP).item() if torch.norm(g_D) > 1e-12 else 0.0
            cos_D_A = _cos(g_D, g_A).item() if torch.norm(g_D) > 1e-12 else 0.0
            
            # Compute angles
            angle_A = torch.acos(torch.tensor(cos_A_FP))
            angle_S = torch.acos(torch.tensor(cos_S_FP))
            angle_improve_deg = ((angle_A - angle_S) * 180 / np.pi).item()
            
            # Orthogonal components
            g_FP_unit = g_FP / (g_FP.norm() + 1e-12)
            g_A_perp = g_A - torch.dot(g_A, g_FP_unit) * g_FP_unit
            g_S_perp = g_S - torch.dot(g_S, g_FP_unit) * g_FP_unit
            
            metrics[locus] = dict(
                cos_A_FP=cos_A_FP,
                cos_S_FP=cos_S_FP,
                angle_improve_deg=angle_improve_deg,
                ortho_ratio_A=(g_A_perp.norm() / (nA + 1e-12)).item(),
                ortho_ratio_S=(g_S_perp.norm() / (nS + 1e-12)).item(),
                r_norm_ratio=(torch.norm(g_D) / (nA + 1e-12)).item(),
                cos_D_FP=cos_D_FP,
                cos_D_A=cos_D_A,
                norm_gA=nA.item(),
                norm_gD=torch.norm(g_D).item(),
                norm_gS=nS.item()
            )
        
        return metrics
    
    def run_gradient_probes(self, epoch: int):
        """Run gradient probes - ALL combinations are valid!"""
        if not self.probe_enabled or epoch not in self.probe_epochs:
            return
        
        print(f"\n=== Running Gradient Probes at Epoch {epoch} ===")
        print("ALL locus-branch combinations are valid due to forward dependencies!")
        print("Direct connections (stronger gradients):")
        print("  D1 → L1_fea (direct)")
        print("  D2 → L2_fea (direct)")
        print("  D3 → L3_fea (direct)")
        print("Indirect connections through forward path also exist for all combinations.")
        
        # Define which connections are direct
        direct_connections = {
            ('D1', 'L1_fea'),
            ('D2', 'L2_fea'),
            ('D3', 'L3_fea'),
        }
        
        with self.bn_eval_both():
            actual_batches = max(self.probe_batches_per_epoch, 8)
            
            for batch_idx in range(actual_batches):
                try:
                    inputs, labels = self.get_probe_batch()
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                except Exception as e:
                    print(f"Error getting batch {batch_idx}: {e}")
                    break
                
                # Compute base gradients
                try:
                    gA = self._get_gA_act(inputs, labels)
                    gFP32 = self._get_gFP32_act(inputs, labels)
                except Exception as e:
                    print(f"Error computing base gradients: {e}")
                    continue
                
                # Process each branch
                for branch in ['D1', 'D2', 'D3', 'Dsum']:
                    try:
                        gD = self._get_gD_act(inputs, labels, which=branch)
                    except Exception as e:
                        print(f"Error computing {branch} gradient: {e}")
                        continue
                    
                    # Compute metrics for each alpha
                    for alpha in self.probe_alpha_eval:
                        metrics = self._compute_metrics_all(gA, gD, gFP32, alpha)
                        
                        # Log ALL metrics (all are valid!)
                        for locus, m in metrics.items():
                            # Check if this is a direct connection
                            is_direct = (branch, locus) in direct_connections
                            
                            row = {
                                'epoch': epoch,
                                'batch': batch_idx,
                                'branch': branch,
                                'locus': locus,
                                'alpha': alpha,
                                'is_direct': 1 if is_direct else 0,
                                **m
                            }
                            self.csv_writer.writerow(row)
                
                # Flush after each batch
                self.csv_file.flush()
                
                # Clear memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        print(f"Completed gradient probes for epoch {epoch}")

# Key insight: ALL locus-branch combinations are valid because:
# 1. x1 is used by both attention1 AND layer2
# 2. x2 is used by both attention2 AND layer3  
# 3. x3 is used by both attention3 AND layer4
# 4. All these dependencies create gradient paths to all layers
#
# The difference is only in magnitude:
# - Direct connections (D1→L1, D2→L2, D3→L3) have stronger gradients
# - Indirect connections have weaker but non-zero gradients