#!/usr/bin/env python3
"""
Complete fix implementation for exp7_6_module.py
This file contains all necessary methods to fix gradient flow issues.
"""

import os
import csv
import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from contextlib import contextmanager

# Constants for gradient flow depth checking
LOCUS_DEPTH = {
    'L1_fea': 1,  # Layer 1 output (x1)
    'L2_fea': 2,  # Layer 2 output (x2)  
    'L3_fea': 3,  # Layer 3 output (x3)
    'FB': 4       # Feature backbone (classifier input)
}

BRANCH_DEPTH = {
    'D1': 1,      # Attention-1 reaches up to L1
    'D2': 2,      # Attention-2 reaches up to L2
    'D3': 3,      # Attention-3 reaches up to L3
    'Dsum': 3     # Sum reaches up to L3
}

class GradientProbeFixComplete:
    """Complete implementation of gradient probe fixes.
    
    Add these methods to exp7_6_module.SalmonLitModule class.
    """
    
    # ============= 1. CSV Logging Schema Update =============
    
    def _setup_csv_logging(self):
        """Setup CSV logging with corrected schema."""
        os.makedirs('./probes', exist_ok=True)
        
        # Include run ID or seed in filename
        run_id = f"seed_{self.probe_seed}"
        csv_path = f'./probes/device2_resnet18_cifar10_{run_id}.csv'
        
        self.csv_file = open(csv_path, 'w', newline='')
        print(f"[probe] CSV output: {csv_path}")
        
        fieldnames = [
            'epoch', 'batch', 'branch', 'locus', 'reachable', 'alpha',
            'cos_A_FP', 'cos_S_FP', 'angle_improve_deg',
            'ortho_ratio_A', 'ortho_ratio_S',
            'r_norm_ratio', 'cos_D_FP', 'cos_D_A',
            'norm_gA', 'norm_gD', 'norm_gS'
        ]
        
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()
        self.csv_file.flush()  # Ensure header is written
    
    # ============= 2. BN Eval for Both Models =============
    
    @contextmanager
    def bn_eval_both(self):
        """Freeze BN for both analog model and FP32 clone."""
        saved_states = {}
        
        def _set_bn_eval(module, prefix):
            """Set all BN layers to eval and save states."""
            if module is None:
                return
            for name, m in module.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    key = f'{prefix}.{name}'
                    saved_states[key] = m.training
                    m.eval()
        
        def _restore_bn(module, prefix):
            """Restore BN training states."""
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
    
    # ============= 3. Activation Tap Points =============
    
    def _get_taps(self, inputs, use_fp32=False):
        """Get activation tensors at branch points.
        Returns fb (feature backbone), x1, x2, x3 from layer outputs.
        """
        if use_fp32 and self.fp32_clone is not None:
            input_features = self.fp32_clone['input'](inputs)
            fb, x1, x2, x3 = self.fp32_clone['features'](input_features)
        else:
            input_features = self.input(inputs)
            fb, x1, x2, x3 = self.features(input_features)
        
        return {'L1_fea': x1, 'L2_fea': x2, 'L3_fea': x3, 'FB': fb}
    
    # ============= 4. Activation Gradient Capture =============
    
    def _backward_and_capture(self, taps: Dict[str, torch.Tensor], loss: torch.Tensor):
        """Backward and capture activation gradients using tensor hooks."""
        grads = {}
        EPS = 1e-12
        
        # Register hooks to capture gradients
        def _make(name):
            def _hook(g):
                if g is None:
                    grads[name] = torch.zeros(1, device=self.device)
                else:
                    v = g.detach().flatten()
                    # Check for finite values
                    if torch.isfinite(v).all():
                        grads[name] = v.clone()
                    else:
                        grads[name] = torch.zeros(1, device=self.device)
            return _hook
        
        # Register hooks
        for name, t in taps.items():
            if t.requires_grad:
                t.register_hook(_make(name))
        
        # Backward pass
        loss.backward()
        
        # Fill missing keys
        for k in taps.keys():
            if k not in grads:
                grads[k] = torch.zeros(1, device=self.device)
        
        return grads
    
    # ============= 5. Gradient Computation Methods =============
    
    def _get_gA_act(self, inputs, labels):
        """Get backbone gradient gA using activation gradients."""
        self.zero_grad(set_to_none=True)
        taps = self._get_taps(inputs, use_fp32=False)
        logits = self.classifier(taps['FB'])
        loss = self.criterion(logits, labels)
        return self._backward_and_capture(taps, loss)
    
    def _get_gD_act(self, inputs, labels, which: str):
        """Get attention gradient for specific branch.
        Args:
            which: Branch name in {'D1','D2','D3','Dsum'}
        """
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
    
    # ============= 6. Depth-Aware Metrics Computation =============
    
    def _compute_metrics_depth_aware(self, gA, gD, gFP32, alpha: float, branch: str):
        """Compute metrics with depth-aware validity checking."""
        metrics = {}
        
        def _cos(u, v):
            """Safe cosine similarity."""
            eps = 1e-12
            return torch.clamp(
                torch.dot(u, v) / ((u.norm() + eps) * (v.norm() + eps)),
                -1 + 1e-7, 1 - 1e-7
            )
        
        for locus in ['L1_fea', 'L2_fea', 'L3_fea', 'FB']:
            # Skip if branch doesn't reach this locus
            if LOCUS_DEPTH[locus] > BRANCH_DEPTH[branch]:
                continue
            
            g_A = gA[locus]
            g_D = gD[locus]
            g_FP = gFP32[locus]
            
            # Skip if zero vectors
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
                ortho_ratio_A=(g_A_perp.norm() / (g_A.norm() + 1e-12)).item(),
                ortho_ratio_S=(g_S_perp.norm() / (g_S.norm() + 1e-12)).item(),
                r_norm_ratio=(g_D.norm() / (g_A.norm() + 1e-12)).item(),
                cos_D_FP=cos_D_FP,
                cos_D_A=cos_D_A,
                norm_gA=g_A.norm().item(),
                norm_gD=g_D.norm().item(),
                norm_gS=g_S.norm().item()
            )
        
        return metrics
    
    # ============= 7. Main Probe Runner =============
    
    def run_gradient_probes(self, epoch: int):
        """Run gradient probes with correct gradient flow understanding."""
        if not self.probe_enabled or epoch not in self.probe_epochs:
            return
        
        print(f"\n=== Running Gradient Probes at Epoch {epoch} ===")
        print("Valid locus-branch combinations:")
        print("  L1_fea: D1, D2, D3, Dsum (all valid)")
        print("  L2_fea: D2, D3, Dsum (D1 unreachable)")
        print("  L3_fea: D3, Dsum (D1, D2 unreachable)")
        print("  FB: None (all unreachable)")
        
        with self.bn_eval_both():
            # Use actual configured batches or at least 8
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
                        metrics = self._compute_metrics_depth_aware(
                            gA, gD, gFP32, alpha, branch
                        )
                        
                        # Log valid metrics (reachable=1)
                        for locus, m in metrics.items():
                            row = {
                                'epoch': epoch,
                                'batch': batch_idx,
                                'branch': branch,
                                'locus': locus,
                                'reachable': 1,
                                'alpha': alpha,
                                **m
                            }
                            self.csv_writer.writerow(row)
                        
                        # Optional: Log unreachable combinations (reachable=0)
                        # This helps with analysis to explicitly mark invalid pairs
                        for locus in ['L1_fea', 'L2_fea', 'L3_fea', 'FB']:
                            if locus not in metrics and LOCUS_DEPTH[locus] > BRANCH_DEPTH[branch]:
                                row = {
                                    'epoch': epoch,
                                    'batch': batch_idx,
                                    'branch': branch,
                                    'locus': locus,
                                    'reachable': 0,
                                    'alpha': alpha,
                                    'cos_A_FP': np.nan,
                                    'cos_S_FP': np.nan,
                                    'angle_improve_deg': np.nan,
                                    'ortho_ratio_A': np.nan,
                                    'ortho_ratio_S': np.nan,
                                    'r_norm_ratio': np.nan,
                                    'cos_D_FP': np.nan,
                                    'cos_D_A': np.nan,
                                    'norm_gA': np.nan,
                                    'norm_gD': np.nan,
                                    'norm_gS': np.nan
                                }
                                self.csv_writer.writerow(row)
                
                # Flush after each batch
                self.csv_file.flush()
                
                # Log summary metrics to tensorboard (optional)
                if hasattr(self, 'log'):
                    alpha_05 = 0.5
                    for branch in ['D1', 'D2', 'D3', 'Dsum']:
                        try:
                            gD_log = self._get_gD_act(inputs, labels, which=branch)
                            m_05 = self._compute_metrics_depth_aware(
                                gA, gD_log, gFP32, alpha_05, branch
                            )
                            for locus, m in m_05.items():
                                for k, v in m.items():
                                    self.log(f"probe/{branch}/{locus}/{k}", v, on_epoch=True)
                        except:
                            pass
                
                # Clear memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

# Implementation instructions:
"""
To apply these fixes to exp7_6_module.py:

1. Add the constants LOCUS_DEPTH and BRANCH_DEPTH at module level
2. Replace the existing methods with these implementations
3. Ensure bn_eval_both() is used instead of bn_eval()
4. Update probe configuration in YAML if needed:
   - batches_per_epoch: 8-16
   - alpha_eval: [0.0, 0.25, 0.5, 1.0, 2.0]
5. Run with different seeds for statistical robustness
"""