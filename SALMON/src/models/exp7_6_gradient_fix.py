"""
Fixed gradient probe implementation with proper gradient flow understanding.
This module contains the corrected methods for exp7_6_module.py
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List
from contextlib import contextmanager

# Depth mapping for gradient reachability
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

class GradientProbeFix:
    """Fixed gradient probe methods with proper gradient flow."""
    
    @contextmanager
    def bn_eval_both(self):
        """Context manager to set BN layers to eval for both analog and FP32 models."""
        saved_states = []
        
        # Collect all modules to set to eval
        modules_to_eval = [self]
        if hasattr(self, 'fp32_clone') and self.fp32_clone is not None:
            modules_to_eval.extend([
                self.fp32_clone['input'],
                self.fp32_clone['features'],
                self.fp32_clone['classifier']
            ])
        
        # Save current states and set to eval
        for module in modules_to_eval:
            for m in module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    saved_states.append((m, m.training))
                    m.eval()
        
        try:
            yield
        finally:
            # Restore original states
            for m, training in saved_states:
                m.train(training)
    
    def _get_activation_taps(self, inputs, use_fp32=False):
        """Get activation tensors at key points for gradient capture."""
        if use_fp32 and self.fp32_clone is not None:
            input_features = self.fp32_clone['input'](inputs)
            fb, x1, x2, x3 = self.fp32_clone['features'](input_features)
        else:
            input_features = self.input(inputs)
            fb, x1, x2, x3 = self.features(input_features)
        
        return {
            'L1_fea': x1,
            'L2_fea': x2,
            'L3_fea': x3,
            'FB': fb
        }
    
    def _backward_and_capture_grads(self, taps: Dict[str, torch.Tensor], loss: torch.Tensor):
        """Backward pass and capture activation gradients using tensor hooks."""
        grads = {}
        handles = []
        
        # Register hooks to capture gradients
        for name, tensor in taps.items():
            if tensor.requires_grad:
                tensor.retain_grad()
        
        # Backward pass
        loss.backward()
        
        # Collect gradients
        for name, tensor in taps.items():
            if tensor.grad is not None:
                g = tensor.grad.detach().flatten()
                # Check for finite values
                if torch.isfinite(g).all():
                    grads[name] = g.clone()
                else:
                    grads[name] = torch.zeros(1, device=self.device)
            else:
                grads[name] = torch.zeros(1, device=self.device)
        
        return grads
    
    def _get_gA_activation(self, inputs, labels):
        """Get backbone gradient gA using activation gradients."""
        self.zero_grad(set_to_none=True)
        with self.bn_eval_both():
            taps = self._get_activation_taps(inputs, use_fp32=False)
            logits = self.classifier(taps['FB'])
            loss = self.criterion(logits, labels)
            return self._backward_and_capture_grads(taps, loss)
    
    def _get_gD_activation(self, inputs, labels, branch='Dsum'):
        """Get attention gradient for specific branch using activation gradients."""
        self.zero_grad(set_to_none=True)
        with self.bn_eval_both():
            taps = self._get_activation_taps(inputs, use_fp32=False)
            
            # Get attention outputs
            out1, _ = self.attention1(taps['L1_fea'])
            out2, _ = self.attention2(taps['L2_fea'])
            out3, _ = self.attention3(taps['L3_fea'])
            
            # Compute loss based on branch
            if branch == 'D1':
                loss = self.criterion(out1, labels)
            elif branch == 'D2':
                loss = self.criterion(out2, labels)
            elif branch == 'D3':
                loss = self.criterion(out3, labels)
            elif branch == 'Dsum':
                loss = (self.criterion(out1, labels) + 
                       self.criterion(out2, labels) + 
                       self.criterion(out3, labels))
            else:
                raise ValueError(f"Unknown branch: {branch}")
            
            return self._backward_and_capture_grads(taps, loss)
    
    def _get_gFP32_activation(self, inputs, labels):
        """Get FP32 reference gradient using activation gradients."""
        if self.fp32_clone is None:
            self._create_fp32_clone()
        
        self.zero_grad(set_to_none=True)
        # Zero grad for FP32 modules
        for module in [self.fp32_clone['input'], 
                      self.fp32_clone['features'], 
                      self.fp32_clone['classifier']]:
            module.zero_grad()
        
        with self.bn_eval_both():
            taps = self._get_activation_taps(inputs, use_fp32=True)
            logits = self.fp32_clone['classifier'](taps['FB'])
            loss = self.criterion(logits, labels)
            return self._backward_and_capture_grads(taps, loss)
    
    def _safe_cosine(self, u, v, eps=1e-12):
        """Compute cosine similarity with numerical safety."""
        u_norm = torch.norm(u) + eps
        v_norm = torch.norm(v) + eps
        cos_val = torch.dot(u, v) / (u_norm * v_norm)
        return torch.clamp(cos_val, -1 + 1e-7, 1 - 1e-7).item()
    
    def _compute_metrics_with_depth_check(self, gA, gD, gFP32, alpha, branch):
        """Compute metrics only for valid locus-branch pairs based on gradient flow."""
        metrics = {}
        
        for locus in ['L1_fea', 'L2_fea', 'L3_fea', 'FB']:
            # Skip if gradient doesn't reach this locus from this branch
            if LOCUS_DEPTH[locus] > BRANCH_DEPTH.get(branch, 0):
                # Mark as invalid/unreachable
                metrics[locus] = {
                    'valid': False,
                    'reason': f"{branch} doesn't reach {locus}"
                }
                continue
            
            g_A = gA.get(locus, torch.zeros(1, device=self.device))
            g_D = gD.get(locus, torch.zeros(1, device=self.device))
            g_FP = gFP32.get(locus, torch.zeros(1, device=self.device))
            
            # Skip if any gradient is zero
            if torch.norm(g_A) < 1e-12 or torch.norm(g_FP) < 1e-12:
                metrics[locus] = {
                    'valid': False,
                    'reason': 'Zero gradient'
                }
                continue
            
            # Compute combined gradient
            g_S = g_A + alpha * g_D
            
            # Norm-preserving scaling
            norm_A = torch.norm(g_A)
            norm_S = torch.norm(g_S) + 1e-12
            g_S_tilde = (norm_A / norm_S) * g_S
            
            # Compute cosine similarities
            cos_A_FP = self._safe_cosine(g_A, g_FP)
            cos_S_FP = self._safe_cosine(g_S_tilde, g_FP)
            cos_D_FP = self._safe_cosine(g_D, g_FP) if torch.norm(g_D) > 1e-12 else 0.0
            cos_D_A = self._safe_cosine(g_D, g_A) if torch.norm(g_D) > 1e-12 else 0.0
            
            # Compute angles
            angle_A = torch.acos(torch.tensor(cos_A_FP))
            angle_S = torch.acos(torch.tensor(cos_S_FP))
            angle_improve_deg = ((angle_A - angle_S) * 180 / np.pi).item()
            
            # Compute orthogonal components
            g_FP_unit = g_FP / (torch.norm(g_FP) + 1e-12)
            g_A_perp = g_A - torch.dot(g_A, g_FP_unit) * g_FP_unit
            g_S_perp = g_S - torch.dot(g_S, g_FP_unit) * g_FP_unit
            ortho_ratio_A = (torch.norm(g_A_perp) / (norm_A + 1e-12)).item()
            ortho_ratio_S = (torch.norm(g_S_perp) / (norm_S + 1e-12)).item()
            
            metrics[locus] = {
                'valid': True,
                'cos_A_FP': cos_A_FP,
                'cos_S_FP': cos_S_FP,
                'angle_improve_deg': angle_improve_deg,
                'ortho_ratio_A': ortho_ratio_A,
                'ortho_ratio_S': ortho_ratio_S,
                'r_norm_ratio': (torch.norm(g_D) / (norm_A + 1e-12)).item(),
                'cos_D_FP': cos_D_FP,
                'cos_D_A': cos_D_A,
                'norm_gA': norm_A.item(),
                'norm_gD': torch.norm(g_D).item(),
                'norm_gS': torch.norm(g_S).item()
            }
        
        return metrics
    
    def run_fixed_gradient_probes(self, epoch: int):
        """Run gradient probes with proper gradient flow understanding."""
        if not self.probe_enabled or epoch not in self.probe_epochs:
            return
        
        print(f"\n=== Running Fixed Gradient Probes at Epoch {epoch} ===")
        print(f"Understanding gradient flow:")
        print(f"  - D1 reaches: L1_fea only")
        print(f"  - D2 reaches: L1_fea, L2_fea")
        print(f"  - D3 reaches: L1_fea, L2_fea, L3_fea")
        print(f"  - FB is reference only (no attention gradients reach here)")
        
        branches = ['D1', 'D2', 'D3', 'Dsum']
        
        with self.bn_eval_both():
            actual_batches = min(self.probe_batches_per_epoch, 4)  # Limit for memory
            
            for batch_idx in range(actual_batches):
                inputs, labels = self.get_probe_batch()
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Get gradients
                gA = self._get_gA_activation(inputs, labels)
                gFP32 = self._get_gFP32_activation(inputs, labels)
                
                # Process each branch separately
                for branch in branches:
                    gD = self._get_gD_activation(inputs, labels, branch)
                    
                    # Compute metrics for each alpha
                    for alpha in self.probe_alpha_eval:
                        metrics = self._compute_metrics_with_depth_check(
                            gA, gD, gFP32, alpha, branch
                        )
                        
                        # Log valid metrics
                        for locus, m in metrics.items():
                            if m.get('valid', False):
                                row = {
                                    'epoch': epoch,
                                    'batch': batch_idx,
                                    'space': 'activation',
                                    'locus': locus,
                                    'branch': branch,
                                    'alpha': alpha,
                                    'valid': 1,
                                    **{k: v for k, v in m.items() if k != 'valid'}
                                }
                                self.csv_writer.writerow(row)
                
                self.csv_file.flush()
                
                # Clear memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

# Add these methods to the exp7_6_module.SalmonLitModule class