from typing import Any, Dict, Tuple, List, Optional
import functools
import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from src.models.components.exp7_1_model import IntegratedResNet
from aihwkit.optim import AnalogSGD, AnalogAdam
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.presets import TikiTakaEcRamPreset, IdealizedPreset, EcRamPreset
from aihwkit.simulator.configs import MappingParameter
from aihwkit.simulator.rpu_base import cuda
from aihwkit.simulator.presets import GokmenVlasovPreset
from aihwkit.simulator.configs import (
    InferenceRPUConfig,
    UnitCellRPUConfig,
    SingleRPUConfig,
    BufferedTransferCompound,
    SoftBoundsDevice,
    ConstantStepDevice,
    MappingParameter,
    IOParameters,
    UpdateParameters,IdealDevice
)
from aihwkit.simulator.configs import SoftBoundsPmaxDevice
from aihwkit.simulator.configs import SingleRPUConfig, ConstantStepDevice, IdealDevice
from aihwkit.simulator.configs import SoftBoundsDevice, SoftBoundsPmaxDevice
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms

# Branch-to-layer reachability mapping for gradient flow
REACHABLE = {
    'D1': {'L1_fea'},                    # attention1(x1) -> L1 only
    'D2': {'L1_fea', 'L2_fea'},          # attention2(x2) -> L1, L2  
    'D3': {'L1_fea', 'L2_fea', 'L3_fea'}, # attention3(x3) -> L1, L2, L3
    'Dsum': {'L1_fea', 'L2_fea', 'L3_fea'}  # Combined reaches L1, L2, L3
}

class SalmonLitModule(LightningModule):
    def __init__(
        self,
        model: str,
        integrated_resnet: IntegratedResNet,
        compile: bool,
        optimizer: dict,
        dataset: str,
        epoch: int,
        loss_coefficient: float,
        feature_loss_coefficient: float,
        dataset_path: str,
        autoaugment: bool,
        temperature: float,
        batchsize: int,
        init_lr: float,
        N_CLASSES: int,
        block: str,
        alpha: float,
        p_max: int,
        opt_config: str,
        sd_config: str,
        FC_Digit: str,
        sch_config: str,
        scheduler: dict,
        probe: Optional[Dict] = None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['integrated_resnet'])

        # Initialize components from integrated_resnet
        self.input = integrated_resnet.input_module
        self.features = integrated_resnet.features
        self.classifier = integrated_resnet.classifier
        self.attention1 = integrated_resnet.attention1
        self.attention2 = integrated_resnet.attention2
        self.attention3 = integrated_resnet.attention3

        # Initialize other properties and metrics
        self.compile = compile
        self.model = model
        self.dataset = dataset
        self.epoch = epoch
        self.loss_coefficient = loss_coefficient
        self.feature_loss_coefficient = feature_loss_coefficient
        self.dataset_path = dataset_path
        self.autoaugment = autoaugment
        self.temperature = temperature
        self.batchsize = batchsize
        self.init_lr = init_lr
        self.N_CLASSES = N_CLASSES
        self.block = block
        self.alpha = alpha
        self.p_max = p_max
        self.opt_config = opt_config
        self.sd_config = sd_config
        self.FC_Digit = FC_Digit

        # Initialize loss functions and metrics
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc_0_best = MaxMetric()
        self.adaptation_layers = torch.nn.ModuleList()
        self.init_adaptation_layers = False

        # Gradient alignment probe configuration
        self.probe_config = probe or {}
        self.probe_enabled = self.probe_config.get('enabled', False)
        self.probe_epochs = self.probe_config.get('epochs', [])
        self.probe_batches_per_epoch = self.probe_config.get('batches_per_epoch', 2)
        self.probe_alpha_eval = self.probe_config.get('alpha_eval', [0.0, 0.5, 1.0])
        self.probe_seed = self.probe_config.get('seed', 1234)
        
        # Probe data structures
        self.probe_loader = None
        self.probe_iterator = None
        # FP32 clone removed - using task-centered approach
        self.csv_file = None
        self.csv_writer = None
        self.probe_batch_size = self.probe_config.get('batch_size', 32)  # Reduced default batch size
        
        # Layer handles for gradient probing
        self.layer_handles = {}
        
        # Initialize probe run tracking
        self._probed_this_epoch = False

    def setup(self, stage: str = None):
        """Setup probe data loader and layer handles."""
        if stage == "fit" and self.probe_enabled:
            self._setup_probe_loader()
            self._setup_layer_handles()
            # CSV logging will be set up per epoch in run_gradient_probes

    def _setup_probe_loader(self):
        """Create deterministic probe data loader without augmentation."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = datasets.CIFAR10(
            root=self.dataset_path, 
            train=True, 
            download=True,  # Auto-download if needed
            transform=transform
        )
        
        # Increase batches_per_epoch for better statistics
        actual_batches = max(self.probe_batches_per_epoch, 8)  # At least 8 batches
        
        # Fixed indices for deterministic probing
        torch.manual_seed(self.probe_seed)
        indices = torch.randperm(len(train_dataset))[:actual_batches * self.probe_batch_size]
        
        self.probe_loader = DataLoader(
            train_dataset,
            batch_size=self.probe_batch_size,  # Use smaller batch size for memory efficiency
            sampler=SubsetRandomSampler(indices),
            num_workers=0,  # Deterministic
            pin_memory=False  # Don't pin memory to reduce allocation pressure
        )
        self.probe_iterator = iter(self.probe_loader)

    def _setup_layer_handles(self):
        """Setup handles to L1, L2, L3 layers only (reachable by gD)."""
        from aihwkit.nn import AnalogConv2d
        
        def find_first_conv(block):
            """Find first conv in a residual block."""
            if hasattr(block, 'residual_function'):
                for m in block.residual_function.modules():
                    if isinstance(m, (AnalogConv2d, nn.Conv2d)):
                        return m
            else:
                for m in block.modules():
                    if isinstance(m, (AnalogConv2d, nn.Conv2d)):
                        return m
            return None
        
        # L1: First conv in layer1's last block
        l1_block = self.features.layer1[-1]
        self.layer_handles['L1_fea'] = find_first_conv(l1_block)
        
        # L2: First conv in layer2's last block
        l2_block = self.features.layer2[-1]
        self.layer_handles['L2_fea'] = find_first_conv(l2_block)
        
        # L3: First conv in layer3's last block
        l3_block = self.features.layer3[-1]
        self.layer_handles['L3_fea'] = find_first_conv(l3_block)
        
        print("Layer handles (task-centered, L1/L2/L3 only):")
        print({k: type(v).__name__ if v else 'None' for k, v in self.layer_handles.items()})
        
    def _find_first_leaf(self, module, find_linear=False):
        """Find the first leaf conv/linear layer in a module."""
        from aihwkit.nn import AnalogConv2d, AnalogLinear
        # Search through all modules to find first leaf
        for m in module.modules():
            if find_linear:
                if isinstance(m, (AnalogLinear, nn.Linear)):
                    return m
            else:
                if isinstance(m, (AnalogConv2d, nn.Conv2d)):
                    return m
        return None

    # FP32 clone removed - using task-centered approach

    def _setup_csv_logging(self, epoch=None):
        """Setup CSV logging for probe results - creates new file for each epoch."""
        out_dir = os.path.join(os.getcwd(), 'probes')
        os.makedirs(out_dir, exist_ok=True)
        
        # Create epoch-specific filename
        run_id = f"seed_{self.probe_seed}_task"
        if epoch is not None:
            csv_filename = f'device2_resnet18_cifar10_{run_id}_epoch{epoch:03d}.csv'
        else:
            csv_filename = f'device2_resnet18_cifar10_{run_id}.csv'
        
        csv_path = os.path.join(out_dir, csv_filename)
        
        # Close previous file if exists
        if hasattr(self, 'csv_file') and self.csv_file is not None:
            self.csv_file.close()
        
        self.csv_file = open(csv_path, 'w', newline='')
        print(f"[probe] Task-centered CSV output: {csv_path}")
        
        # Pure task-centered fieldnames (no FP32)
        alpha_train = 1.0 - float(self.loss_coefficient)
        alpha_list = sorted(set(list(self.probe_alpha_eval) + [alpha_train]))
        
        fieldnames = [
            'epoch', 'batch', 'branch', 'locus',
            'dot_AD', 'cos_D_task', 'r_norm_ratio', 'alpha_crit', 'helps_task',
            'norm_gA', 'norm_gD'
        ]
        # Add descent scores for each alpha
        for a in alpha_list:
            fieldnames.append(f'descent_score_a{a:.2f}')
        fieldnames.append('is_train_alpha')  # Mark training alpha
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()
        self.csv_file.flush()  # Ensure header is written

    def _safe_norm(self, t, eps=1e-12):
        """Safe norm computation with epsilon."""
        if t.numel() == 0:
            return torch.tensor(eps, device=t.device)
        return torch.norm(t) + eps

    def _safe_cos(self, a, b, eps=1e-12):
        """Safe cosine similarity computation."""
        # Handle size mismatch (when one is zeros fallback)
        if a.numel() <= 1 or b.numel() <= 1:
            return 0.0
        # Truncate to minimum size if needed
        min_size = min(a.numel(), b.numel())
        if a.numel() != b.numel():
            a = a[:min_size]
            b = b[:min_size]
        na = self._safe_norm(a, eps)
        nb = self._safe_norm(b, eps)
        if na <= eps or nb <= eps:
            return 0.0
        dot_product = torch.dot(a, b)
        cos_sim = dot_product / (na * nb)
        return torch.clamp(cos_sim, -1.0, 1.0).item()

    def _sum_grads(self, dicts: list):
        """Sum gradient dictionaries element-wise."""
        keys = dicts[0].keys()
        out = {}
        for k in keys:
            out[k] = sum(d[k] for d in dicts)
        return out

    def _compute_inter_branch_cosines(self, gD1, gD2, gD3):
        """Compute cosine similarities between digital branches."""
        cosines = {}
        for locus in gD1.keys():
            cosines[locus] = {
                'cos_D12': self._safe_cos(gD1[locus], gD2[locus]),
                'cos_D13': self._safe_cos(gD1[locus], gD3[locus]),
                'cos_D23': self._safe_cos(gD2[locus], gD3[locus])
            }
        return cosines

    @contextmanager
    def bn_eval_both(self):
        """Context manager to freeze BN layers for analog model."""
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
            
            # FP32 clone removed - pure task-centered approach
            
            yield
            
        finally:
            # Restore original states
            _restore_bn(self, 'analog')
            
            # FP32 clone restore removed

    def flatten_grads(self, modules: Dict[str, nn.Module]) -> Dict[str, torch.Tensor]:
        """Flatten gradients from modules into 1D tensors.
        
        For analog modules: uses stored output gradients from analog_ctx
        For FP32 modules: uses weight gradients only (no bias)
        """
        from aihwkit.nn import AnalogConv2d, AnalogLinear
        
        flattened = {}
        for locus, module in modules.items():
            if module is None:
                flattened[locus] = torch.zeros(1, device=self.device)
                continue
            
            vec = []
            
            # For analog modules, try to get gradients from analog_ctx
            if isinstance(module, (AnalogConv2d, AnalogLinear)):
                if hasattr(module, 'analog_module'):
                    analog_module = module.analog_module
                    
                    # Check if analog_module has analog_ctx with stored gradients
                    if hasattr(analog_module, 'analog_ctx'):
                        ctx = analog_module.analog_ctx
                        
                        # Check if gradients were stored during backward pass
                        if hasattr(ctx, 'analog_grad_output') and ctx.analog_grad_output:
                            # Get the last gradient output (most recent backward pass)
                            grad_output = ctx.analog_grad_output[-1]
                            
                            if grad_output is not None:
                                # Get weight tensor to match size
                                try:
                                    weights = analog_module.tile.get_weights()
                                    if weights is not None and len(weights) > 0:
                                        weight_tensor = weights[0]
                                        
                                        # For Conv2d layers with analog tiles, weights are stored as 2D [out_features, in_features]
                                        # We need to match this size for gradient alignment
                                        if isinstance(module, AnalogConv2d) and weight_tensor.dim() == 2:
                                            # Create a pseudo weight gradient of matching size
                                            # This represents the gradient information flow through weights
                                            out_features, in_features = weight_tensor.shape
                                            
                                            # Use a portion of output gradient reshaped to match weight size
                                            grad_flat = grad_output.detach().flatten()
                                            
                                            # Create weight-sized gradient proxy
                                            if len(grad_flat) >= out_features * in_features:
                                                # Take first part matching weight size
                                                weight_grad_proxy = grad_flat[:out_features * in_features]
                                            else:
                                                # Repeat to match weight size
                                                repeat_factor = (out_features * in_features + len(grad_flat) - 1) // len(grad_flat)
                                                weight_grad_proxy = grad_flat.repeat(repeat_factor)[:out_features * in_features]
                                            
                                            vec.append(weight_grad_proxy)
                                        elif isinstance(module, AnalogLinear):
                                            # For Linear layers, compute actual weight gradient
                                            if hasattr(ctx, 'analog_input') and ctx.analog_input:
                                                input_activation = ctx.analog_input[-1]
                                                if input_activation is not None:
                                                    weight_grad = grad_output.t() @ input_activation / grad_output.shape[0]
                                                    vec.append(weight_grad.detach().flatten())
                                                else:
                                                    # Fallback: match weight size
                                                    grad_flat = grad_output.detach().flatten()
                                                    weight_size = weight_tensor.numel()
                                                    if len(grad_flat) != weight_size:
                                                        weight_grad_proxy = grad_flat.repeat((weight_size + len(grad_flat) - 1) // len(grad_flat))[:weight_size]
                                                        vec.append(weight_grad_proxy)
                                                    else:
                                                        vec.append(grad_flat)
                                        else:
                                            # Default: use flattened gradient
                                            vec.append(grad_output.detach().flatten())
                                except Exception as e:
                                    print(f"  Error processing analog gradient for {locus}: {e}")
                                    vec.append(grad_output.detach().flatten())
                                    
                                if vec:
                                    print(f"  Got analog gradient for {locus}: shape {vec[-1].shape}, norm {vec[-1].norm()}")
            
            # Fallback to standard parameter gradients (FP32 modules)
            if not vec:
                # Only use weight gradients, not bias
                if hasattr(module, 'weight') and module.weight is not None and module.weight.grad is not None:
                    vec.append(module.weight.grad.detach().flatten())
                else:
                    # Try to find weight in submodules
                    for n, p in module.named_parameters(recurse=True):
                        if 'weight' in n and p.grad is not None:
                            vec.append(p.grad.detach().flatten())
                            break
            
            if vec:
                flattened[locus] = torch.cat(vec)
                print(f"[probe] Flattened gradients for {locus}: {flattened[locus].shape}, norm={flattened[locus].norm():.6f}")
            else:
                print(f"[probe] WARNING: No gradients found for {locus}, using zeros")
                flattened[locus] = torch.zeros(1, device=self.device)
        
        return flattened

    def _clear_analog_gradients(self):
        """Clear gradients stored in analog contexts."""
        from aihwkit.nn import AnalogConv2d, AnalogLinear
        
        for module in self.modules():
            if isinstance(module, (AnalogConv2d, AnalogLinear)):
                if hasattr(module, 'analog_module'):
                    analog_module = module.analog_module
                    if hasattr(analog_module, 'analog_ctx'):
                        ctx = analog_module.analog_ctx
                        ctx.analog_input = []
                        ctx.analog_grad_output = []
    
    def get_probe_batch(self):
        """Get next probe batch, resetting iterator if needed."""
        try:
            return next(self.probe_iterator)
        except StopIteration:
            self.probe_iterator = iter(self.probe_loader)
            return next(self.probe_iterator)
    
    def _get_taps(self, inputs, use_fp32=False):
        """Get activation tensors at branch points (fb, x1, x2, x3)."""
        # Always use analog model (no FP32 option)
        input_features = self.input(inputs)
        fb, x1, x2, x3 = self.features(input_features)
        
        # Ensure all taps have requires_grad=True for gradient capture
        taps = {'L1_fea': x1, 'L2_fea': x2, 'L3_fea': x3, 'FB': fb}
        for name, t in taps.items():
            if not t.requires_grad:
                t.requires_grad_(True)
        
        return taps
    
    def _backward_and_capture(self, taps: Dict[str, torch.Tensor], loss: torch.Tensor):
        """Backward and capture activation gradients using tensor hooks."""
        grads = {}
        
        # Ensure gradients are retained (all taps should have requires_grad=True now)
        for name, t in taps.items():
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
    
    def _get_gD_split_act(self, inputs, labels):
        """Get attention gradients for each branch separately using activation gradients."""
        grads = {}
        
        # Get all branch gradients
        for branch in ['D1', 'D2', 'D3', 'Dsum']:
            grads[branch] = self._get_gD_act(inputs, labels, which=branch)
        
        return grads
    
    # FP32 methods removed - using pure task-centered approach

    def _compute_metrics_simple(self, gA, gD, gFP32, alpha: float):
        """Compute metrics for all loci without depth checking."""
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
    
    def _compute_metrics_task(self, g_task, gD_split, alpha_list, alpha_train):
        """
        Compute task-centered metrics without FP32 reference.
        
        Args:
            g_task: Task gradient (backbone CE)
            gD_split: Dict of branch gradients
            alpha_list: Alphas to evaluate
            alpha_train: Training alpha value
            
        Returns:
            List of metric rows
        """
        rows = []
        
        for branch, gD in gD_split.items():
            # Get reachable loci for this branch
            reachable_loci = REACHABLE.get(branch, set())
            
            for locus, vA in g_task.items():
                # Skip if branch cannot reach this locus
                if locus not in reachable_loci:
                    continue
                    
                vD = gD.get(locus, torch.zeros_like(vA))
                
                # Skip if gradients too small
                if vA.numel() <= 1 or torch.norm(vA) < 1e-12:
                    continue
                
                norm_gA = torch.norm(vA).item()
                norm_gD = torch.norm(vD).item()
                
                # Core metrics
                dot_AD = torch.dot(vA, vD).item()
                cos_DA = F.cosine_similarity(vA.unsqueeze(0), vD.unsqueeze(0)).item() if norm_gD > 1e-12 else 0.0
                r = norm_gD / norm_gA if norm_gA > 1e-12 else 0.0
                
                # Critical alpha
                if dot_AD >= 0:
                    alpha_crit = float('inf')
                else:
                    alpha_crit = (norm_gA ** 2 / (-dot_AD))
                
                # Descent scores for each alpha
                scores = {}
                for a in sorted(set(alpha_list + [alpha_train])):
                    vS = vA + a * vD
                    scores[f'descent_score@{a}'] = torch.dot(vA, vS).item()
                
                rows.append({
                    'branch': branch,
                    'locus': locus,
                    'dot_AD': dot_AD,
                    'cos_D_task': cos_DA,
                    'r': r,
                    'alpha_crit': alpha_crit,
                    'norm_gA': norm_gA,
                    'norm_gD': norm_gD,
                    **scores
                })
        
        return rows
    
    def run_gradient_probes(self, epoch: int):
        """Run task-centered gradient probes (no FP32 reference)."""
        if not self.probe_enabled or epoch not in self.probe_epochs:
            return
        
        print(f"\n=== Running Task-Centered Gradient Probes at Epoch {epoch} ===")
        print(f"Evaluating: Does gD help g_task (backbone CE gradient)?")
        print(f"α_train = {1.0 - self.loss_coefficient}")
        print("Reachability constraints enforced:")
        print("  D1 → L1_fea only")
        print("  D2 → L1_fea, L2_fea")
        print("  D3 → L1_fea, L2_fea, L3_fea")
        print("  Dsum → L1_fea, L2_fea, L3_fea")
        
        # Create new CSV file for this epoch
        self._setup_csv_logging(epoch=epoch)
        
        # Statistics tracking
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
                
                # Get task gradient (backbone CE)
                try:
                    g_task = self._get_gA_act(inputs, labels)
                except Exception as e:
                    print(f"Error computing task gradient: {e}")
                    continue
                
                # Get branch-separated attention gradients
                try:
                    gD_split = self._get_gD_split_act(inputs, labels)
                except Exception as e:
                    print(f"Error computing attention gradients: {e}")
                    continue
                
                # Compute task-centered metrics
                alpha_train = 1.0 - self.loss_coefficient
                alpha_list = list(self.probe_alpha_eval)
                
                rows = self._compute_metrics_task(g_task, gD_split, alpha_list, alpha_train)
                
                # Write to CSV
                for row in rows:
                    row_out = {
                        'epoch': epoch,
                        'batch': batch_idx,
                        'branch': row['branch'],
                        'locus': row['locus'],
                        'dot_AD': row['dot_AD'],
                        'cos_D_task': row['cos_D_task'],
                        'r_norm_ratio': row['r'],
                        'alpha_crit': row['alpha_crit'],
                        'helps_task': int(row['dot_AD'] > 0),
                        'norm_gA': row.get('norm_gA', 0),
                        'norm_gD': row.get('norm_gD', 0)
                    }
                    
                    # Add descent scores for each alpha
                    for a in sorted(set(alpha_list + [alpha_train])):
                        key = f'descent_score_a{a:.2f}'
                        row_out[key] = row.get(f'descent_score@{a}', 0)
                    
                    # Mark training alpha
                    row_out['is_train_alpha'] = int(abs(alpha_train - row.get('alpha', -1)) < 1e-6)
                    
                    self.csv_writer.writerow(row_out)
                    
                    # Update stats
                    if row['dot_AD'] > 0:
                        stats['helps'] += 1
                    else:
                        stats['hurts'] += 1
                    stats['total'] += 1
                
                # Flush after each batch
                self.csv_file.flush()
                
                # Clear memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Print summary
        if stats['total'] > 0:
            help_pct = 100 * stats['helps'] / stats['total']
            print(f"\nSummary: {stats['helps']}/{stats['total']} ({help_pct:.1f}%) "
                  f"cases where gD helps g_task")
        
        print(f"Completed task-centered gradient probes for epoch {epoch}")
        
        # Close the CSV file for this epoch
        if hasattr(self, 'csv_file') and self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
            print(f"CSV file closed for epoch {epoch}")
    
    def run_gradient_probes_task_centered(self, epoch: int):
        """Run task-centered gradient probes evaluating if gD helps g_task."""
        if not self.probe_enabled or epoch not in self.probe_epochs:
            return
        
        print(f"\n=== Running Task-Centered Gradient Probes at Epoch {epoch} ===")
        print(f"Evaluating: Does gD help or hurt g_task?")
        print(f"α_train = {1.0 - self.loss_coefficient} (actual training weight)")
        
        # Create new CSV file for this epoch
        self._setup_csv_logging(epoch=epoch)
        
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
                
                # Get task gradient (backbone only - this is g_task)
                try:
                    g_task = self._get_gA_act(inputs, labels)
                except Exception as e:
                    print(f"Error computing task gradient: {e}")
                    continue
                
                # Get attention gradients for each branch separately
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
                
                # FP32 comparison removed - using pure task-centered metrics
                
                # Write metrics to CSV
                for m in metrics:
                    # Fill in batch/epoch info
                    m['epoch'] = epoch
                    m['batch'] = batch_idx
                    
                    # Determine if direct connection
                    direct_connections = {
                        ('D1', 'L1_fea'),
                        ('D2', 'L2_fea'),
                        ('D3', 'L3_fea'),
                    }
                    m['is_direct'] = int((m['branch'], m['locus']) in direct_connections)
                    
                    # Fill missing optional fields with defaults
                    for field in ['cos_A_FP', 'cos_S_FP', 'cos_D_FP', 'cos_D_A', 
                                 'angle_improve_deg', 'ortho_ratio_A', 'ortho_ratio_S']:
                        if field not in m:
                            m[field] = 0.0
                    
                    # cos_D_A can be computed from cos_D_task
                    if 'cos_D_A' not in m or m['cos_D_A'] == 0.0:
                        m['cos_D_A'] = m['cos_D_task']  # Since gA = g_task
                    
                    self.csv_writer.writerow(m)
                    
                    # Update statistics for training alpha
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
            print(f"Summary at α_train={1.0-self.loss_coefficient}: "
                  f"{stats['helps']}/{stats['total']} ({help_pct:.1f}%) cases where gD helps g_task")
        
        print(f"Completed task-centered gradient probes for epoch {epoch}")
        
        # Close the CSV file for this epoch
        if hasattr(self, 'csv_file') and self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
            print(f"CSV file closed for epoch {epoch}")

    def _get_feature_grads_A(self, inputs, labels):
        """Get backbone gradient gA using feature-space gradients."""
        self.zero_grad(set_to_none=True)
        with self.bn_eval():
            x0 = self.input(inputs)
            fb, x1, x2, x3 = self.features(x0)
            for t in (x1, x2, x3, fb):
                t.retain_grad()
            logits = self.classifier(fb)
            loss = self.criterion(logits, labels)
            loss.backward()  # No retain_graph - this is a fresh computation
            result = {
                'L1_fea': x1.grad.detach().flatten().clone() if x1.grad is not None else torch.zeros(1, device=self.device),
                'L2_fea': x2.grad.detach().flatten().clone() if x2.grad is not None else torch.zeros(1, device=self.device),
                'L3_fea': x3.grad.detach().flatten().clone() if x3.grad is not None else torch.zeros(1, device=self.device),
                'FB': fb.grad.detach().flatten().clone() if fb.grad is not None else torch.zeros(1, device=self.device),
            }
            # Clear intermediate tensors to free memory
            del x0, fb, x1, x2, x3, logits, loss
            return result

    def _get_feature_grads_Dj(self, inputs, labels, j: int):
        """Get attention gradient for branch j (1, 2, or 3)."""
        assert j in (1, 2, 3)
        self.zero_grad(set_to_none=True)
        with self.bn_eval():
            x0 = self.input(inputs)
            fb, x1, x2, x3 = self.features(x0)
            for t in (x1, x2, x3, fb):
                t.retain_grad()
            outs = [
                self.attention1(x1)[0],
                self.attention2(x2)[0],
                self.attention3(x3)[0],
            ]
            loss = self.criterion(outs[j-1], labels)
            loss.backward()  # No retain_graph - fresh computation
            result = {
                'L1_fea': x1.grad.detach().flatten().clone() if x1.grad is not None else torch.zeros(1, device=self.device),
                'L2_fea': x2.grad.detach().flatten().clone() if x2.grad is not None else torch.zeros(1, device=self.device),
                'L3_fea': x3.grad.detach().flatten().clone() if x3.grad is not None else torch.zeros(1, device=self.device),
                'FB': fb.grad.detach().flatten().clone() if fb.grad is not None else torch.zeros(1, device=self.device),
            }
            # Clear intermediate tensors
            del x0, fb, x1, x2, x3, outs, loss
            return result

    def _get_feature_grads_FP(self, inputs, labels):
        """Get FP32 reference feature-space gradients."""
        if self.fp32_clone is None:
            self._create_fp32_clone()
        
        # Set BN to eval mode for ALL modules
        for module in [self.fp32_clone['input'], self.fp32_clone['features'], self.fp32_clone['classifier']]:
            for m in module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        
        # Zero gradients
        self.fp32_clone['input'].zero_grad()
        self.fp32_clone['features'].zero_grad()
        self.fp32_clone['classifier'].zero_grad()
        
        x0 = self.fp32_clone['input'](inputs)
        fb, x1, x2, x3 = self.fp32_clone['features'](x0)
        for t in (x1, x2, x3, fb):
            t.retain_grad()
        logits = self.fp32_clone['classifier'](fb)
        loss = self.criterion(logits, labels)
        loss.backward()
        return {
            'L1_fea': x1.grad.detach().flatten() if x1.grad is not None else torch.zeros(1, device=self.device),
            'L2_fea': x2.grad.detach().flatten() if x2.grad is not None else torch.zeros(1, device=self.device),
            'L3_fea': x3.grad.detach().flatten() if x3.grad is not None else torch.zeros(1, device=self.device),
            'FB': fb.grad.detach().flatten() if fb.grad is not None else torch.zeros(1, device=self.device),
        }

    def _get_gA(self, inputs, labels):
        """Get backbone gradient gA for weight-space (backward compatibility)."""
        self.zero_grad(set_to_none=True)
        
        with self.bn_eval():
            # Forward pass through backbone
            input_features = self.input(inputs)
            feature_backbone, x1, x2, x3 = self.features(input_features)
            
            # Continue forward
            logits = self.classifier(feature_backbone)
            loss = self.criterion(logits, labels)
            loss.backward()  # No retain_graph - fresh computation
            
            # Get weight-space gradients from layer handles
            result = self.flatten_grads(self.layer_handles)
            
            # Clear intermediate tensors
            del input_features, feature_backbone, x1, x2, x3, logits, loss
            return result

    def _get_gD(self, inputs, labels):
        """Get attention gradient gD for weight-space (backward compatibility)."""
        self.zero_grad(set_to_none=True)
        
        with self.bn_eval():
            # Forward pass through full network
            input_features = self.input(inputs)
            feature_backbone, x1, x2, x3 = self.features(input_features)
            
            # Get attention outputs from intermediate features
            out_attention1, _ = self.attention1(x1)
            out_attention2, _ = self.attention2(x2)
            out_attention3, _ = self.attention3(x3)
            
            # Sum of attention losses
            attention_outputs = [out_attention1, out_attention2, out_attention3]
            total_att_loss = 0
            for att_out in attention_outputs:
                total_att_loss += self.criterion(att_out, labels)
            
            total_att_loss.backward()  # No retain_graph - fresh computation
            
            # Get weight-space gradients from layer handles
            result = self.flatten_grads(self.layer_handles)
            
            # Clear intermediate tensors
            del input_features, feature_backbone, x1, x2, x3
            del out_attention1, out_attention2, out_attention3
            del attention_outputs, total_att_loss
            return result

    # All FP32-related methods removed - using pure task-centered approach

    def _compute_metrics(self, gA, gD, gFP32, alpha):
        """Compute directional metrics for given alpha (backward compatibility)."""
        metrics = {}
        
        for locus in gA.keys():
            g_A = gA[locus]
            g_D = gD[locus]  
            g_FP32 = gFP32[locus]
            
            # Skip if any gradient is effectively zero (size 1 fallback)
            if g_A.numel() == 1 or g_D.numel() == 1 or g_FP32.numel() == 1:
                # Set all metrics to zero for this locus
                if locus not in metrics:
                    for key in ['cos_A_FP', 'cos_S_FP', 'angle_improve_deg', 'ortho_ratio_A', 
                               'ortho_ratio_S', 'r_norm_ratio', 'cos_D_FP', 'cos_D_A', 
                               'norm_gA', 'norm_gD', 'norm_gS']:
                        if key not in metrics:
                            metrics[key] = {}
                        metrics[key][locus] = 0.0
                continue
            
            # Handle size mismatch by truncating to minimum size
            min_size = min(g_A.numel(), g_D.numel(), g_FP32.numel())
            if min_size > 1:
                g_A = g_A[:min_size]
                g_D = g_D[:min_size]
                g_FP32 = g_FP32[:min_size]
            
            # Combined gradient with alpha
            g_S = g_A + alpha * g_D
            
            # Norm-preserved version with epsilon for safety
            eps = 1e-12
            norm_gA = torch.norm(g_A) + eps
            norm_gS = torch.norm(g_S) + eps
            g_S_tilde = (norm_gA / norm_gS) * g_S
            
            # Cosine similarities
            cos_A_FP = F.cosine_similarity(g_A.unsqueeze(0), g_FP32.unsqueeze(0)).item()
            cos_S_FP = F.cosine_similarity(g_S_tilde.unsqueeze(0), g_FP32.unsqueeze(0)).item()
            cos_D_FP = F.cosine_similarity(g_D.unsqueeze(0), g_FP32.unsqueeze(0)).item()
            cos_D_A = F.cosine_similarity(g_D.unsqueeze(0), g_A.unsqueeze(0)).item()
            
            # Angle improvement
            angle_A = torch.acos(torch.clamp(torch.tensor(cos_A_FP), -1, 1))
            angle_S = torch.acos(torch.clamp(torch.tensor(cos_S_FP), -1, 1))
            angle_improve_deg = ((angle_A - angle_S) * 180 / np.pi).item()
            
            # Orthogonal ratios with epsilon for safety
            norm_gFP32 = torch.norm(g_FP32) + eps
            if norm_gFP32 > eps and norm_gA > eps and norm_gS > eps:
                g_FP32_unit = g_FP32 / norm_gFP32
                # Project onto FP32 direction
                proj_A = torch.dot(g_A, g_FP32_unit) * g_FP32_unit
                proj_S = torch.dot(g_S, g_FP32_unit) * g_FP32_unit
                
                # Perpendicular components
                g_A_perp = g_A - proj_A
                g_S_perp = g_S - proj_S
                
                ortho_ratio_A = (torch.norm(g_A_perp) / norm_gA).item()
                ortho_ratio_S = (torch.norm(g_S_perp) / norm_gS).item()
            else:
                ortho_ratio_A = 0
                ortho_ratio_S = 0
            
            # Store metrics
            if locus not in metrics:
                for key in ['cos_A_FP', 'cos_S_FP', 'angle_improve_deg', 'ortho_ratio_A', 
                           'ortho_ratio_S', 'r_norm_ratio', 'cos_D_FP', 'cos_D_A', 
                           'norm_gA', 'norm_gD', 'norm_gS']:
                    metrics[key] = {}
            
            metrics['cos_A_FP'][locus] = cos_A_FP
            metrics['cos_S_FP'][locus] = cos_S_FP
            metrics['angle_improve_deg'][locus] = angle_improve_deg
            metrics['ortho_ratio_A'][locus] = ortho_ratio_A
            metrics['ortho_ratio_S'][locus] = ortho_ratio_S
            metrics['r_norm_ratio'][locus] = (torch.norm(g_D) / torch.norm(g_A)).item()
            metrics['cos_D_FP'][locus] = cos_D_FP
            metrics['cos_D_A'][locus] = cos_D_A
            metrics['norm_gA'][locus] = norm_gA.item()
            metrics['norm_gD'][locus] = torch.norm(g_D).item()
            metrics['norm_gS'][locus] = norm_gS.item()
            
        return metrics

    def on_train_epoch_start(self):
        """Initialize probe flag at start of epoch."""
        self._probed_this_epoch = False

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Forward pass through the network."""
        # Input processing
        input_features = self.input(x)
        
        # Feature extraction - returns multiple outputs
        feature_backbone, x1, x2, x3 = self.features(input_features)
        
        # Get attention outputs from intermediate features
        out_attention1, feature_attention1 = self.attention1(x1)
        out_attention2, feature_attention2 = self.attention2(x2)
        out_attention3, feature_attention3 = self.attention3(x3)
        
        # Backbone classifier
        out_backbone = self.classifier(feature_backbone)
        
        # Return outputs and features for self-distillation
        outputs = [out_backbone, out_attention3, out_attention2, out_attention1]
        features = [feature_backbone, feature_attention3, feature_attention2, feature_attention1]
        
        return outputs, features

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """Perform a single model step on a batch of data."""
        x, y = batch
        outputs, features = self.forward(x)
        return outputs, features, y

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        outputs, features, labels = self.model_step(batch)
        
        # Compute self-distillation loss similar to exp7_2
        loss = self.criterion(outputs[0], labels)
        teacher_output = outputs[0].detach()
        
        for idx, output in enumerate(outputs[1:]):
            # Student network losses with self-distillation
            loss += self.criterion(output, labels) * (1 - self.loss_coefficient)
        
        # Run probes at configured epochs
        current_epoch = self.current_epoch + 1  # Lightning uses 0-indexed epochs
        if (self.probe_enabled 
            and not self._probed_this_epoch 
            and batch_idx == 0 
            and current_epoch in self.probe_epochs):
            print(f"\n>>> Running probes at epoch {current_epoch} (batch {batch_idx})")
            self.run_gradient_probes(current_epoch)
            self._probed_this_epoch = True
            print(f">>> Probe completed for epoch {current_epoch}")
        
        # Log metrics
        self.train_acc(torch.argmax(outputs[0], dim=1), labels)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set."""
        outputs, features, labels = self.model_step(batch)
        
        total_loss = 0
        total_acc = []
        
        for i, output in enumerate(outputs):
            loss = self.criterion(output, labels)
            acc = self.val_acc(torch.argmax(output, dim=1), labels)
            total_loss += loss
            total_acc.append(acc)
            
            self.log(f"val/loss_{i}", loss, on_step=False, on_epoch=True)
            if i == 0:
                self.log(f"val/acc_{i}", acc, on_step=False, on_epoch=True, prog_bar=True)
            else:
                self.log(f"val/acc_{i}", acc, on_step=False, on_epoch=True)

        avg_loss = total_loss / len(outputs)
        avg_acc = sum(total_acc) / len(total_acc)
        return {"val_loss": avg_loss, "val_acc": avg_acc}

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set."""
        outputs, features, labels = self.model_step(batch)
        
        total_loss = 0
        total_acc = []
        
        for i, output in enumerate(outputs):
            loss = self.criterion(output, labels)
            acc = self.test_acc(torch.argmax(output, dim=1), labels)
            total_loss += loss
            total_acc.append(acc)
            
            self.log(f"test/loss_{i}", loss, on_step=False, on_epoch=True)
            self.log(f"test/acc_{i}", acc, on_step=False, on_epoch=True)

        avg_loss = total_loss / len(outputs)
        avg_acc = sum(total_acc) / len(total_acc)
        return {"test_loss": avg_loss, "test_acc": avg_acc}

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_0_best(acc)  # update best so far val acc
        self.log("val/acc_0_best", self.val_acc_0_best.compute(), sync_dist=True, prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training."""
        optimizer = AnalogSGD(
            self.parameters(),
            lr=self.hparams.optimizer["lr"],
            momentum=self.hparams.optimizer["momentum"],
            weight_decay=self.hparams.optimizer["weight_decay"],
        )
        
        if self.hparams.scheduler is not None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.hparams.scheduler["T_max"],
                eta_min=self.hparams.scheduler["eta_min"]
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss_0",
                },
            }
        return {"optimizer": optimizer}

    def sanity_check_gradients(self):
        """Sanity check for gradient computation - for debugging only."""
        if not self.probe_enabled or self.probe_loader is None:
            print("Probe not enabled or loader not initialized")
            return
            
        print("\n=== Gradient Sanity Check ===")
        
        # Get one batch
        inputs, labels = next(iter(self.probe_loader))
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        
        # Test gA (backbone gradient)
        self.zero_grad(set_to_none=True)
        with self.bn_eval():
            x = self.input(inputs)
            fb, x1, x2, x3 = self.features(x)
            for t in (x1, x2, x3, fb):
                t.retain_grad()
            logits = self.classifier(fb)
            loss = self.criterion(logits, labels)
            loss.backward()  # No retain_graph
            print(f"gA - Loss: {loss.item():.4f}")
            print(f"gA - ‖x1‖: {x1.grad.norm().item():.4f}, ‖x2‖: {x2.grad.norm().item():.4f}, "
                  f"‖x3‖: {x3.grad.norm().item():.4f}, ‖fb‖: {fb.grad.norm().item():.4f}")
        
        # Test gD (attention gradients)
        self.zero_grad(set_to_none=True)
        with self.bn_eval():
            x = self.input(inputs)
            fb, x1, x2, x3 = self.features(x)
            for t in (x1, x2, x3, fb):
                t.retain_grad()
            out1, _ = self.attention1(x1)
            out2, _ = self.attention2(x2)
            out3, _ = self.attention3(x3)
            loss_d = self.criterion(out1, labels) + self.criterion(out2, labels) + self.criterion(out3, labels)
            loss_d.backward()  # No retain_graph
            print(f"gD - Loss: {loss_d.item():.4f}")
            print(f"gD - ‖x1‖: {x1.grad.norm().item():.4f}, ‖x2‖: {x2.grad.norm().item():.4f}, "
                  f"‖x3‖: {x3.grad.norm().item():.4f}, ‖fb‖: {fb.grad.norm().item() if fb.grad is not None else 0:.4f}")
        
        # Test gFP32
        if self.fp32_clone is None:
            self._create_fp32_clone()
        
        for module in [self.fp32_clone['input'], self.fp32_clone['features'], self.fp32_clone['classifier']]:
            for m in module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
            module.zero_grad()
        
        x0_fp = self.fp32_clone['input'](inputs)
        fb_fp, x1_fp, x2_fp, x3_fp = self.fp32_clone['features'](x0_fp)
        for t in (x1_fp, x2_fp, x3_fp, fb_fp):
            t.retain_grad()
        logits_fp = self.fp32_clone['classifier'](fb_fp)
        loss_fp = self.criterion(logits_fp, labels)
        loss_fp.backward()
        print(f"gFP32 - Loss: {loss_fp.item():.4f}")
        print(f"gFP32 - ‖x1‖: {x1_fp.grad.norm().item():.4f}, ‖x2‖: {x2_fp.grad.norm().item():.4f}, "
              f"‖x3‖: {x3_fp.grad.norm().item():.4f}, ‖fb‖: {fb_fp.grad.norm().item():.4f}")
        
        print("=== Sanity Check Complete ===\n")
    
    def on_fit_end(self):
        """Lightning hook called when training ends."""
        if hasattr(self, 'csv_file') and self.csv_file:
            self.csv_file.close()
            self.csv_file = None
    
    def __del__(self):
        """Cleanup CSV file on deletion."""
        if hasattr(self, 'csv_file') and self.csv_file:
            self.csv_file.close()