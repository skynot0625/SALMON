"""
Methods to add to exp7_6_module.py for correct gradient flow implementation.
These methods should replace or be added to the SalmonLitModule class.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from contextlib import contextmanager

# Add these constants at module level
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

# Add these methods to SalmonLitModule class:

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
    handles = []
    EPS = 1e-12
    
    # Register hooks to capture gradients
    for name, t in taps.items():
        def _make(name=name):
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
        
        if t.requires_grad:
            handles.append(t.register_hook(_make()))
    
    # Backward pass
    loss.backward()
    
    # No need to remove tensor hooks (automatically cleaned up)
    
    # Fill missing keys
    for k in taps.keys():
        if k not in grads:
            grads[k] = torch.zeros(1, device=self.device)
    
    return grads

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
    for module in [self.fp32_clone['input'], 
                   self.fp32_clone['features'],
                   self.fp32_clone['classifier']]:
        module.zero_grad()
    
    taps = self._get_taps(inputs, use_fp32=True)
    logits = self.fp32_clone['classifier'](taps['FB'])
    loss = self.criterion(logits, labels)
    return self._backward_and_capture(taps, loss)

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
        if module is not None:
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

def _compute_metrics_depth_aware(self, gA, gD, gFP32, alpha, branch):
    """Compute metrics with depth-aware validity checking."""
    metrics = {}
    
    def safe_cos(u, v):
        """Safe cosine computation."""
        eps = 1e-12
        return torch.clamp(
            torch.dot(u, v) / ((u.norm() + eps) * (v.norm() + eps)),
            -1 + 1e-7, 1 - 1e-7
        )
    
    for locus in ['L1_fea', 'L2_fea', 'L3_fea', 'FB']:
        # Check if gradient reaches this locus from this branch
        if LOCUS_DEPTH[locus] > BRANCH_DEPTH[branch]:
            # Skip invalid locus-branch pair
            continue
        
        g_A = gA[locus]
        g_D = gD[locus]
        g_FP = gFP32[locus]
        
        # Skip if zero vectors
        if torch.norm(g_A) < 1e-12 or torch.norm(g_FP) < 1e-12:
            continue
        
        # Combined gradient
        g_S = g_A + alpha * g_D
        
        # Norm-preserving scaling
        nA = torch.norm(g_A)
        nS = torch.norm(g_S)
        g_S_tilde = (nA / (nS + 1e-12)) * g_S
        
        # Compute metrics
        cos_A_FP = safe_cos(g_A, g_FP).item()
        cos_S_FP = safe_cos(g_S_tilde, g_FP).item()
        cos_D_FP = safe_cos(g_D, g_FP).item() if torch.norm(g_D) > 1e-12 else 0.0
        cos_D_A = safe_cos(g_D, g_A).item() if torch.norm(g_D) > 1e-12 else 0.0
        
        # Angles
        angle_A = torch.acos(torch.tensor(cos_A_FP))
        angle_S = torch.acos(torch.tensor(cos_S_FP))
        angle_improve_deg = ((angle_A - angle_S) * 180 / np.pi).item()
        
        # Orthogonal components
        g_FP_unit = g_FP / (g_FP.norm() + 1e-12)
        g_A_perp = g_A - torch.dot(g_A, g_FP_unit) * g_FP_unit
        g_S_perp = g_S - torch.dot(g_S, g_FP_unit) * g_FP_unit
        ortho_A = (g_A_perp.norm() / (g_A.norm() + 1e-12)).item()
        ortho_S = (g_S_perp.norm() / (g_S.norm() + 1e-12)).item()
        
        metrics[locus] = dict(
            cos_A_FP=cos_A_FP,
            cos_S_FP=cos_S_FP,
            angle_improve_deg=angle_improve_deg,
            ortho_ratio_A=ortho_A,
            ortho_ratio_S=ortho_S,
            r_norm_ratio=(g_D.norm() / (g_A.norm() + 1e-12)).item(),
            cos_D_FP=cos_D_FP,
            cos_D_A=cos_D_A,
            norm_gA=g_A.norm().item(),
            norm_gD=g_D.norm().item(),
            norm_gS=g_S.norm().item()
        )
    
    return metrics

def run_gradient_probes_fixed(self, epoch: int):
    """Run gradient probes with correct gradient flow understanding."""
    if not self.probe_enabled or epoch not in self.probe_epochs:
        return
    
    print(f"\n=== Running Fixed Gradient Probes at Epoch {epoch} ===")
    print("Gradient flow understanding:")
    print("  D1 → L1_fea only")
    print("  D2 → L1_fea, L2_fea")
    print("  D3 → L1_fea, L2_fea, L3_fea")
    print("  FB → Reference only (no attention gradients)")
    
    branches = ['D1', 'D2', 'D3', 'Dsum']
    
    with self.bn_eval_both():
        actual_batches = max(self.probe_batches_per_epoch, 8)
        
        for batch_idx in range(actual_batches):
            try:
                inputs, labels = self.get_probe_batch()
                inputs, labels = inputs.to(self.device), labels.to(self.device)
            except RuntimeError as e:
                print(f"Error getting batch {batch_idx}: {e}")
                break
            
            # Compute gradients
            try:
                gA = self._get_gA_act(inputs, labels)
                gFP32 = self._get_gFP32_act(inputs, labels)
            except RuntimeError as e:
                print(f"Error computing base gradients: {e}")
                continue
            
            # Process each branch
            for branch in branches:
                try:
                    gD = self._get_gD_act(inputs, labels, branch)
                except RuntimeError as e:
                    print(f"Error computing {branch} gradient: {e}")
                    continue
                
                # Compute metrics for each alpha
                for alpha in self.probe_alpha_eval:
                    metrics = self._compute_metrics_depth_aware(
                        gA, gD, gFP32, alpha, branch
                    )
                    
                    # Log metrics
                    for locus, m in metrics.items():
                        row = {
                            'epoch': epoch,
                            'batch': batch_idx,
                            'space': 'activation',
                            'locus': locus,
                            'branch': branch,
                            'alpha': alpha,
                            'reachable': 1,  # Only logging reachable pairs
                            **m
                        }
                        
                        # Fill any missing fields
                        for field in ['cos_D12', 'cos_D13', 'cos_D23', 
                                     'mask_mean', 'mask_std', 'mask_sparsity']:
                            if field not in row:
                                row[field] = 0.0
                        
                        self.csv_writer.writerow(row)
            
            # Flush after each batch
            self.csv_file.flush()
            
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# Usage instructions:
"""
1. Add the constants (LOCUS_DEPTH, BRANCH_DEPTH) at module level
2. Add all these methods to SalmonLitModule class
3. Replace run_gradient_probes with run_gradient_probes_fixed
4. Update CSV schema to include 'reachable' field
5. Use bn_eval_both() instead of bn_eval() in probe methods
"""