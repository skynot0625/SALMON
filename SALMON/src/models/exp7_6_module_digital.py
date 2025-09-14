from typing import Any, Dict, Tuple, List, Optional
import functools
import os
import csv
import math
import glob
import numpy as np
import pandas as pd
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
        
        # Store integrated_resnet for device type detection
        self.integrated_resnet = integrated_resnet

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

        # Gradient alignment probe configuration (extended)
        self.probe_config = probe or {}
        self.probe_enabled = self.probe_config.get('enabled', False)
        self.probe_epochs = self.probe_config.get('epochs', [])
        self.probe_batches_per_epoch = self.probe_config.get('batches_per_epoch', 8)
        self.probe_alpha_eval = self.probe_config.get('alpha_eval', [0.0, 0.5, 1.0])
        self.probe_seed = self.probe_config.get('seed', 1234)
        
        # NEW: Extended probe configuration
        self.probe_core_epochs = self.probe_config.get('core_epochs', self.probe_epochs)
        self.probe_aux_epochs = self.probe_config.get('aux_epochs', [])
        self.probe_batches_per_epoch_core = self.probe_config.get('batches_per_epoch_core', self.probe_batches_per_epoch)
        self.probe_batches_per_epoch_aux = self.probe_config.get('batches_per_epoch_aux', self.probe_batches_per_epoch)
        self.probe_batch_size = self.probe_config.get('probe_batch_size', 32)
        self.probe_seeds = self.probe_config.get('seeds', [self.probe_seed])
        
        # NEW: Analysis control options
        self.include_fb = self.probe_config.get('include_fb', False)  # Include FB in reports/plots (default False)
        self.use_fp32_metrics = self.probe_config.get('use_fp32_metrics', True)  # Use FP32 metrics/plots (default True)
        self.alpha_crit_mode = self.probe_config.get('alpha_crit_mode', 'S_perp_D')  # 'S_perp_D' or 'S_perp_A'
        
        # Probe data structures
        self.probe_loader = None
        self.probe_iterator = None
        self.csv_file = None
        self.csv_writer = None
        
        # Layer handles for gradient probing
        self.layer_handles = {}
        
        # Detect device type for probe directory
        self.device_type = self._detect_device_type()

    def _detect_device_type(self):
        """Detect the device type from the integrated_resnet module."""
        # TEMPORARY HARDCODE FOR IDEALIZED RUN - REMOVE AFTER TEST
        return 'Idealized'  # TODO: REVERT THIS AFTER IDEALIZED RUN
        
        # Check the integrated_resnet directly (not from hparams since it's excluded)
        if hasattr(self, 'integrated_resnet'):
            # Try to get the RPU config from the model
            if hasattr(self.integrated_resnet, 'rpu_config'):
                rpu_config = self.integrated_resnet.rpu_config
                config_name = rpu_config.__class__.__name__
                print(f"DEBUG: RPU config class name: {config_name}")
                if 'EcRam' in config_name:
                    return 'EcRam'
                elif 'Idealized' in config_name:
                    return 'Idealized'
                elif 'ReRam' in config_name:
                    return 'ReRam'
                elif 'PCM' in config_name:
                    return 'PCM'
            else:
                print(f"DEBUG: integrated_resnet has no rpu_config attribute")
        else:
            print(f"DEBUG: No integrated_resnet attribute found")
        return 'Default'
    
    def setup(self, stage: str = None):
        """Setup probe data loader and layer handles."""
        if stage == "fit" and self.probe_enabled:
            self._setup_probe_loader()
            self._setup_layer_handles()
            self._setup_csv_logging()
            self._create_fp32_clone()

    def _setup_probe_loader(self, seed=None, batch_size=None):
        """Create deterministic probe data loader without augmentation."""
        if seed is None:
            seed = self.probe_seed
        if batch_size is None:
            batch_size = self.probe_batch_size
            
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = datasets.CIFAR10(
            root=self.dataset_path, 
            train=True, 
            download=False, 
            transform=transform
        )
        
        # Fixed indices for deterministic probing with specific seed
        torch.manual_seed(seed)
        # Get enough samples for maximum possible batches
        max_batches = max(self.probe_batches_per_epoch_core, self.probe_batches_per_epoch_aux, self.probe_batches_per_epoch)
        indices = torch.randperm(len(train_dataset))[:max_batches * batch_size]
        
        self.probe_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(indices),
            num_workers=0,  # Deterministic
            pin_memory=False  # Avoid pinned memory to reduce memory usage
        )
        self.probe_iterator = iter(self.probe_loader)

    def _create_fp32_clone(self):
        """Create FP32 digital backbone (features + classifier only) as reference."""
        from src.models.components.exp7_1_model import (
            ResNetInput, ResNetFeatures, ResNetClassifier, 
            BasicBlock, Bottleneck
        )
        import torch.nn as nn
        
        try:
            # IMPORTANT: Create a pure digital FP32 model matching our analog architecture
            # Use the same components as exp7_1_model but without analog conversion
            
            if self.model == 'resnet18':
                # Create FP32 model components matching the analog model exactly
                
                # Input module with 3x3 conv (matching ResNetInput from exp7_1_model)
                fp32_input = ResNetInput(in_channels=3, base_channels=64)
                
                # Features module (matching ResNetFeatures from exp7_1_model)
                block = BasicBlock if self.block == 'BasicBlock' else Bottleneck
                fp32_features = ResNetFeatures(
                    block=block,
                    layers=[2, 2, 2, 2],  # ResNet18 layers
                    base_channels=64
                )
                
                # Classifier module (matching ResNetClassifier from exp7_1_model)
                # ResNet18 with BasicBlock has 512 final channels
                in_features = 512 * (1 if self.block == 'BasicBlock' else 4)
                fp32_classifier = ResNetClassifier(
                    in_features=in_features,
                    num_classes=self.N_CLASSES
                )
                
                self.fp32_clone = {
                    'input': fp32_input,
                    'features': fp32_features,
                    'classifier': fp32_classifier
                }
            else:
                raise NotImplementedError(f"FP32 clone not implemented for {self.model}")
            
            # Move to same device as main model
            device = next(self.parameters()).device
            for name, module in self.fp32_clone.items():
                self.fp32_clone[name] = module.to(device)
                # [C1] Ensure FP32 is always in eval mode for BN consistency
                self.fp32_clone[name].eval()
            
            # Copy weights from analog model to FP32 model
            self._copy_analog_weights_to_fp32()
            
            print("Created FP32 backbone clone for reference gradient comparison")
            
        except Exception as e:
            print(f"Warning: Could not create FP32 clone: {e}")
            import traceback
            traceback.print_exc()
            print("Using analog gradients as reference instead")
            self.fp32_clone = None
    
    def _copy_analog_weights_to_fp32(self):
        """Copy weights from analog modules to FP32 modules."""
        import torch
        
        def copy_analog_layer(src_analog, dst_conv):
            """Copy weights from an analog layer to a conv/linear layer."""
            if hasattr(src_analog, 'get_weights'):
                weights = src_analog.get_weights()
                if weights is None:
                    return False
                    
                # Handle tuple return (weight, bias)
                if isinstance(weights, tuple):
                    weight = weights[0] if len(weights) > 0 and weights[0] is not None else None
                    bias = weights[1] if len(weights) > 1 and weights[1] is not None else None
                else:
                    weight = weights
                    bias = None
                
                if weight is not None and hasattr(dst_conv, 'weight'):
                    # Reshape if needed for Conv2d
                    if len(dst_conv.weight.shape) == 4 and len(weight.shape) == 2:
                        out_ch_dst, in_ch_dst, kh_dst, kw_dst = dst_conv.weight.shape
                        out_ch_src = weight.shape[0]
                        
                        # Try to infer kernel size from weight shape
                        if out_ch_src == out_ch_dst:
                            # Calculate expected kernel size from analog weight
                            expected_in_size = weight.shape[1]
                            kernel_size = int((expected_in_size / in_ch_dst) ** 0.5)
                            
                            if kernel_size * kernel_size * in_ch_dst == expected_in_size:
                                # Reshape analog weight to match its actual kernel size
                                weight = weight.view(out_ch_src, in_ch_dst, kernel_size, kernel_size)
                                
                                # Skip if kernel sizes don't match (e.g., 3x3 vs 7x7)
                                if kernel_size != kh_dst or kernel_size != kw_dst:
                                    print(f"    Warning: Kernel size mismatch - analog {kernel_size}x{kernel_size} vs FP32 {kh_dst}x{kw_dst}, skipping weight copy")
                                    return False
                    
                    # Convert to proper device/dtype
                    w = torch.as_tensor(weight, device=dst_conv.weight.device, dtype=dst_conv.weight.dtype)
                    
                    # Final shape check before copying
                    if w.shape != dst_conv.weight.shape:
                        print(f"    Warning: Shape mismatch - analog {w.shape} vs FP32 {dst_conv.weight.shape}, skipping weight copy")
                        return False
                        
                    dst_conv.weight.data.copy_(w)
                    
                    if bias is not None and hasattr(dst_conv, 'bias') and dst_conv.bias is not None:
                        b = torch.as_tensor(bias, device=dst_conv.bias.device, dtype=dst_conv.bias.dtype)
                        dst_conv.bias.data.copy_(b)
                    return True
            return False
        
        # Copy input module (conv1, bn1)
        if hasattr(self.input, 'conv1') and hasattr(self.input.conv1, 'get_weights'):
            copy_analog_layer(self.input.conv1, self.fp32_clone['input'].conv1)
        if hasattr(self.input, 'bn1') and hasattr(self.fp32_clone['input'], 'bn1'):
            src_bn = self.input.bn1
            dst_bn = self.fp32_clone['input'].bn1
            if hasattr(src_bn, 'running_mean'):
                dst_bn.running_mean.data.copy_(src_bn.running_mean.data)
                dst_bn.running_var.data.copy_(src_bn.running_var.data)
                if src_bn.weight is not None:
                    dst_bn.weight.data.copy_(src_bn.weight.data)
                if src_bn.bias is not None:
                    dst_bn.bias.data.copy_(src_bn.bias.data)
        
        # Copy feature layers (layer1-4)
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            if hasattr(self.features, layer_name) and hasattr(self.fp32_clone['features'], layer_name):
                src_layer = getattr(self.features, layer_name)
                dst_layer = getattr(self.fp32_clone['features'], layer_name)
                
                # Copy each block in the layer
                for block_idx, (src_block, dst_block) in enumerate(zip(src_layer, dst_layer)):
                    # Copy conv layers in residual function
                    if hasattr(src_block, 'residual_function') and hasattr(dst_block, 'residual_function'):
                        # Both have residual_function as Sequential
                        for src_module, dst_module in zip(src_block.residual_function, dst_block.residual_function):
                            if hasattr(src_module, 'get_weights'):  # AnalogConv2d
                                if isinstance(dst_module, torch.nn.Conv2d):
                                    copy_analog_layer(src_module, dst_module)
                            elif isinstance(src_module, torch.nn.BatchNorm2d) and isinstance(dst_module, torch.nn.BatchNorm2d):
                                if hasattr(src_module, 'running_mean'):
                                    dst_module.running_mean.data.copy_(src_module.running_mean.data)
                                    dst_module.running_var.data.copy_(src_module.running_var.data)
                                    if src_module.weight is not None:
                                        dst_module.weight.data.copy_(src_module.weight.data)
                                    if src_module.bias is not None:
                                        dst_module.bias.data.copy_(src_module.bias.data)
                    
                    # Copy shortcut if exists
                    if hasattr(src_block, 'shortcut') and hasattr(dst_block, 'shortcut'):
                        if len(src_block.shortcut) > 0 and len(dst_block.shortcut) > 0:
                            for src_m, dst_m in zip(src_block.shortcut, dst_block.shortcut):
                                if hasattr(src_m, 'get_weights'):
                                    if isinstance(dst_m, torch.nn.Conv2d):
                                        copy_analog_layer(src_m, dst_m)
                                elif isinstance(src_m, torch.nn.BatchNorm2d) and isinstance(dst_m, torch.nn.BatchNorm2d):
                                    dst_m.running_mean.data.copy_(src_m.running_mean.data)
                                    dst_m.running_var.data.copy_(src_m.running_var.data)
                                    if src_m.weight is not None:
                                        dst_m.weight.data.copy_(src_m.weight.data)
                                    if src_m.bias is not None:
                                        dst_m.bias.data.copy_(src_m.bias.data)
        
        # Copy classifier (fc layer)
        if hasattr(self.classifier, 'fc') and hasattr(self.classifier.fc, 'get_weights'):
            copy_analog_layer(self.classifier.fc, self.fp32_clone['classifier'].fc)
    
    def _setup_layer_handles(self):
        """Setup handles to activation tensors instead of layer modules."""
        # We'll capture gradients at activation tensors, not layer modules
        # Define depth mapping for gradient flow
        self.LOCUS_DEPTH = {
            'L1_fea': 1,  # Layer 1 output (x1)
            'L2_fea': 2,  # Layer 2 output (x2)  
            'L3_fea': 3,  # Layer 3 output (x3)
            'FB': 4       # Feature backbone (classifier input)
        }
        
        self.BRANCH_DEPTH = {
            'D1': 1,      # Attention-1 reaches up to L1
            'D2': 2,      # Attention-2 reaches up to L2
            'D3': 3,      # Attention-3 reaches up to L3
            'Dsum': 3     # Sum reaches up to L3
        }
        
        # Define reachability based on gradient flow
        self.REACHABLE = {
            'D1': {'L1_fea'},
            'D2': {'L1_fea', 'L2_fea'},
            'D3': {'L1_fea', 'L2_fea', 'L3_fea'},
            'Dsum': {'L1_fea', 'L2_fea', 'L3_fea'}
        }
        
    def _find_first_conv(self, module):
        """Find the first conv layer in a module."""
        # First check if this module itself is a conv
        if hasattr(module, 'analog_module'):
            return module
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            return module
            
        # Then check children
        for child in module.children():
            if hasattr(child, 'analog_module') or isinstance(child, (nn.Conv2d, nn.Linear)):
                return child
            result = self._find_first_conv(child)
            if result is not None:
                return result
        return None
    
    def _find_linear(self, module):
        """Find the linear layer in a module."""
        for child in module.children():
            if hasattr(child, 'analog_module') or isinstance(child, nn.Linear):
                return child
            result = self._find_linear(child)
            if result is not None:
                return result
        return None

    def _setup_csv_logging(self):
        """Setup CSV logging for probe results."""
        os.makedirs('./probes', exist_ok=True)
        # CSV files will be created per epoch in run_gradient_probes
        self.csv_files = {}  # Dictionary to store file handles per epoch
        self.csv_writers = {}  # Dictionary to store CSV writers per epoch

    @contextmanager
    def bn_eval(self):
        """Context manager to freeze BN layers during probing."""
        bn_states = {}
        
        # Save current training states and set to eval
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                bn_states[name] = module.training
                module.eval()
        
        try:
            yield
        finally:
            # Restore original states
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.train(bn_states[name])

    def _get_taps(self, inputs, use_fp32=False):
        """Get activation tensors at branch points for gradient capture."""
        # [C3] Proper existence check for fp32_clone
        if use_fp32 and getattr(self, 'fp32_clone', None) is not None:
            input_features = self.fp32_clone['input'](inputs)
            fb, x1, x2, x3 = self.fp32_clone['features'](input_features)
            # Apply classifier avgpool and flatten for FB
            fb = self.fp32_clone['classifier'].avgpool(fb) if hasattr(self.fp32_clone['classifier'], 'avgpool') else fb
            fb = torch.flatten(fb, 1)
        else:
            input_features = self.input(inputs)
            fb, x1, x2, x3 = self.features(input_features)
        return {
            'L1_fea': x1,
            'L2_fea': x2,
            'L3_fea': x3,
            'FB': fb
        }
    
    def _backward_and_capture(self, taps: Dict[str, torch.Tensor], loss: torch.Tensor):
        """Backward and capture activation gradients using tensor hooks."""
        grads = {}
        handles = []
        
        # Register hooks to capture gradients
        for name, t in taps.items():
            def _make_hook(name=name):
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
                handles.append(t.register_hook(_make_hook()))
        
        # Backward pass
        loss.backward()
        
        # [C4] Remove hooks to prevent memory accumulation
        for h in handles:
            h.remove()
        
        # Fill missing keys with zeros
        for k in taps.keys():
            if k not in grads:
                grads[k] = torch.zeros(1, device=self.device)
        
        return grads

    def get_probe_batch(self):
        """Get next probe batch, resetting iterator if needed."""
        try:
            return next(self.probe_iterator)
        except StopIteration:
            self.probe_iterator = iter(self.probe_loader)
            return next(self.probe_iterator)

    def run_gradient_probes(self, epoch: int):
        """Run gradient alignment probes comparing with FP32 reference."""
        if not self.probe_enabled:
            print(f"Probes disabled")
            return
        if epoch not in self.probe_epochs:
            print(f"Epoch {epoch} not in probe epochs {self.probe_epochs}")
            return
            
        print(f"\n=== Running Gradient Alignment Probes at Epoch {epoch} ===")
        
        # Determine number of batches for this epoch
        if epoch in self.probe_core_epochs:
            n_batches = self.probe_batches_per_epoch_core
            print(f"  Core epoch: using {n_batches} batches")
        elif epoch in self.probe_aux_epochs:
            n_batches = self.probe_batches_per_epoch_aux
            print(f"  Auxiliary epoch: using {n_batches} batches")
        else:
            n_batches = self.probe_batches_per_epoch
            print(f"  Standard epoch: using {n_batches} batches")
        
        # [C7] Synchronize FP32 weights with current analog weights before probing
        has_fp32 = (getattr(self, 'fp32_clone', None) is not None) and self.use_fp32_metrics
        if has_fp32:
            print(f"  Synchronizing FP32 weights with current analog weights...")
            self._copy_analog_weights_to_fp32()
        
        # Loop over seeds
        for seed in self.probe_seeds:
            print(f"\n  Running with seed {seed}")
            
            # Setup probe loader for this seed
            self._setup_probe_loader(seed=seed, batch_size=self.probe_batch_size)
            
            # Create CSV files for this epoch and seed
            self._create_epoch_csv_files(epoch, seed, has_fp32)
            
            with self.bn_eval():
                for batch_idx in range(n_batches):
                    inputs, labels = self.get_probe_batch()
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    # Get gradient vectors
                    g_analog = self._get_gTask(inputs, labels)  # Analog backbone gradient
                    gD_split = self._get_gD_split(inputs, labels)  # Digital attention gradients
                    g_fp32 = self._get_gFP32(inputs, labels) if has_fp32 else {}  # FP32 backbone reference
                    
                    if g_fp32:  # If FP32 model is available
                        # Compute both FP32 alignment and task-centric metrics
                        rows_fp32 = self._compute_alignment_metrics_with_fp32(g_analog, gD_split, g_fp32, self.probe_alpha_eval)
                        rows_task = self._compute_task_centric_metrics(g_analog, gD_split, self.probe_alpha_eval)
                        
                        # Write to FP32 alignment CSV (contains both FP32 and task metrics)
                        writer_key_fp32 = f'epoch_{epoch}_seed_{seed}_fp32'
                        for row in rows_fp32:
                            row['epoch'] = epoch
                            row['batch'] = batch_idx
                            row['seed'] = seed
                            self.csv_writers[writer_key_fp32].writerow(row)
                        self.csv_files[writer_key_fp32].flush()
                        
                        # Also write to task-centric CSV (task metrics only)
                        writer_key_task = f'epoch_{epoch}_seed_{seed}_task'
                        for row in rows_task:
                            row['epoch'] = epoch
                            row['batch'] = batch_idx
                            row['seed'] = seed
                            self.csv_writers[writer_key_task].writerow(row)
                        self.csv_files[writer_key_task].flush()
                    else:
                        # Only task-centric metrics available
                        rows = self._compute_task_centric_metrics(g_analog, gD_split, self.probe_alpha_eval)
                        # Write to task-centric CSV
                        writer_key = f'epoch_{epoch}_seed_{seed}_task'
                        for row in rows:
                            row['epoch'] = epoch
                            row['batch'] = batch_idx
                            row['seed'] = seed
                            self.csv_writers[writer_key].writerow(row)
                        self.csv_files[writer_key].flush()
                    
                    if batch_idx == 0:  # Log only for first batch to reduce noise
                        if g_fp32:
                            print(f"    Batch {batch_idx}: Generated {len(rows_fp32)} rows (FP32) and {len(rows_task)} rows (task)")
                        else:
                            print(f"    Batch {batch_idx}: Generated {len(rows)} rows")
                
            # Close CSV files for this seed
            self._close_epoch_csv_files(epoch)
        
        # After all seeds, aggregate and plot
        print(f"\n  Aggregating results for epoch {epoch}")
        self._aggregate_epoch_and_plot(epoch, has_fp32)

    def _get_gTask(self, inputs, labels):
        """Get task gradient using activation gradients from analog model."""
        self.zero_grad(set_to_none=True)
        taps = self._get_taps(inputs, use_fp32=False)
        logits = self.classifier(taps['FB'])
        loss = self.criterion(logits, labels)
        return self._backward_and_capture(taps, loss)
    
    def _get_gFP32(self, inputs, labels):
        """Get FP32 backbone reference gradient (backbone only, no attention)."""
        if not hasattr(self, 'fp32_clone') or self.fp32_clone is None:
            return {}
        
        # Ensure FP32 modules are on same device as inputs
        device = inputs.device
        for name, module in self.fp32_clone.items():
            if module is not None:
                # Check if module needs to be moved to device
                try:
                    module_device = next(module.parameters()).device
                    if module_device != device:
                        self.fp32_clone[name] = module.to(device)
                except StopIteration:
                    # Module has no parameters
                    pass
                module.zero_grad()
        
        # Forward pass through FP32 backbone ONLY (no attention branches)
        taps = self._get_taps(inputs, use_fp32=True)
        # FP32 classifier expects flattened features
        logits = self.fp32_clone['classifier'].fc(taps['FB'])
        loss = self.criterion(logits, labels)
        
        # This gives us the reference gradient for ideal digital backbone
        return self._backward_and_capture(taps, loss)

    def _get_gD_split(self, inputs, labels):
        """Get branch-wise attention gradients using activation gradients."""
        gradients = {}
        
        for branch in ['D1', 'D2', 'D3', 'Dsum']:
            self.zero_grad(set_to_none=True)
            taps = self._get_taps(inputs)
            
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
            
            gradients[branch] = self._backward_and_capture(taps, loss)
        
        return gradients

    def _compute_task_centric_metrics(self, g_task, gD_split, alpha_eval_list):
        """Compute task-centric metrics (analog gradients only, no FP32)."""
        rows = []
        
        print(f"    Computing task-centric metrics for {len(gD_split)} branches")
        
        for branch_name, gD_dict in gD_split.items():
            # [C6] Include FB in comparison locus
            for locus in ['L1_fea', 'L2_fea', 'L3_fea', 'FB']:
                # Check depth-based reachability
                if self.LOCUS_DEPTH[locus] > self.BRANCH_DEPTH[branch_name]:
                    continue
                
                if locus not in g_task or locus not in gD_dict:
                    continue
                
                g_T = g_task[locus]    # Task gradient
                g_D = gD_dict[locus]   # Attention gradient
                
                # Skip if gradients are too small
                norm_T = torch.norm(g_T).item()
                norm_D = torch.norm(g_D).item()
                
                if norm_T < 1e-8:
                    continue
                
                # Task-centric metrics
                cos_D_task = self._safe_cosine(g_D, g_T) if norm_D > 1e-8 else 0
                r_norm_ratio_task = norm_D / norm_T if norm_T > 0 else 0
                
                # Dot product for critical alpha and beta_task
                dot_TD = torch.dot(g_T, g_D).item() if norm_D > 1e-8 else 0
                
                # Critical alpha (mode-dependent)
                if norm_D > 1e-8 and norm_T > 1e-8:
                    if self.alpha_crit_mode == 'S_perp_A':
                        # <A + αD, A> = 0  => α = -||A||^2 / <D,A>
                        alpha_crit_task = -(norm_T**2) / dot_TD if abs(dot_TD) > 1e-8 else float('inf')
                    else:
                        # Default: <A + αD, D> = 0  => α = -<A,D> / ||D||^2
                        alpha_crit_task = -dot_TD / (norm_D ** 2)
                else:
                    alpha_crit_task = float('inf')
                
                # NEW: beta_task = <A,D>/||A||^2 = (||D||/||A||) * cos(D,A)
                beta_task = (dot_TD / (norm_T**2)) if norm_T > 1e-8 else 0.0
                
                # [C5] Only fill task-centric fields, leave FP32 fields empty
                row = {
                    'branch': branch_name,
                    'locus': locus,
                    # Task-centric metrics
                    'cos_D_task': cos_D_task,
                    'norm_gTask': norm_T,
                    'norm_gD': norm_D,
                    'r_norm_ratio_task': r_norm_ratio_task,
                    'alpha_crit_task': alpha_crit_task,
                    'beta_task': beta_task,  # NEW
                    # FP32 alignment metrics - leave empty/None
                    'cos_A_FP': None,
                    'cos_D_FP': None,
                    'cos_D_A': None,
                    'angle_A_FP_deg': None,
                    'angle_D_FP_deg': None,
                    'cos_S_FP': None,
                    'angle_improve_deg': None,
                    'norm_gA': None,
                    'norm_gFP32': None,
                    'r_norm_ratio': None,
                    'alpha_crit_fp32': None,
                }
                
                # NEW: Calculate descent gain for each alpha
                for alpha in alpha_eval_list:
                    row[f'cos_S_FP@{alpha:.2f}'] = None
                    row[f'descent_gain_task@{alpha:.2f}'] = alpha * beta_task
                
                rows.append(row)
        
        # Add descent sweep metrics
        descent_rows = self._compute_task_descent_sweep(g_task, gD_split, alpha_eval_list)
        
        # Add directional metrics
        directional_rows = self._compute_task_directional_metrics(g_task, gD_split, alpha_eval_list)
        
        # Merge metrics by branch and locus
        for descent_row in descent_rows:
            # Find matching row
            for row in rows:
                if row['branch'] == descent_row['branch'] and row['locus'] == descent_row['locus']:
                    row.update(descent_row)
                    break
        
        for dir_row in directional_rows:
            # Find matching row
            for row in rows:
                if row['branch'] == dir_row['branch'] and row['locus'] == dir_row['locus']:
                    row.update(dir_row)
                    break
        
        return rows
    
    def _safe_cosine(self, u, v, eps=1e-12):
        """Compute cosine similarity with numerical safety."""
        u_norm = torch.norm(u) + eps
        v_norm = torch.norm(v) + eps
        cos_val = torch.dot(u, v) / (u_norm * v_norm)
        return torch.clamp(cos_val, -1 + 1e-7, 1 - 1e-7).item()
    
    def _compute_task_descent_sweep(self, g_task, gD_split, alpha_eval_list, L_smooth=1.0):
        """
        Compute worst-case 1-step descent ratio based on Descent Lemma.
        base: α=0 descent = ||g_A||^2 / (2L)
        sweep: α>0 descent = <g_A, g_A + α g_D>^2 / (2L ||g_A + α g_D||^2)
        Returns: ratio(α) = sweep/base - 1 (positive means improvement)
        """
        rows = []
        for branch_name, gD_dict in gD_split.items():
            for locus in ['L1_fea', 'L2_fea', 'L3_fea', 'FB']:
                if self.LOCUS_DEPTH[locus] > self.BRANCH_DEPTH[branch_name]:
                    continue
                if locus not in g_task or locus not in gD_dict:
                    continue
                    
                a = g_task[locus]  # Task gradient
                d = gD_dict[locus]  # Digital gradient
                na = torch.norm(a).item()
                nd = torch.norm(d).item()
                
                if na < 1e-8 or nd < 1e-8:
                    continue
                
                base = (na * na) / (2.0 * L_smooth)  # α=0 descent
                dot_ad = torch.dot(a, d).item()
                
                # First-order improvement condition at α=0: dot_ad > 0
                for alpha in alpha_eval_list:
                    s = a + alpha * d
                    ns = torch.norm(s).item()
                    if ns < 1e-8:
                        continue
                    
                    # Descent with α according to Descent Lemma
                    gain = ((torch.dot(a, s).item()) ** 2) / (2.0 * L_smooth * (ns * ns))
                    ratio = (gain / base) - 1.0
                    
                    rows.append({
                        'branch': branch_name,
                        'locus': locus,
                        'descent_ratio': ratio,
                        f'descent_ratio@{alpha:.2f}': ratio,
                        'dot_ad': dot_ad,
                        'first_order_positive': dot_ad > 0
                    })
        return rows
    
    def _compute_task_directional_metrics(self, g_task, gD_split, alpha_eval_list):
        """
        Compute pure task-centric directional metrics without FP32.
        - cos_S_task@α = cos(g_A + α g_D, g_A) 
        - dir_deriv@α = <g_A, g_A + α g_D> / ||g_A|| (directional derivative)
        """
        rows = []
        for branch, gD_dict in gD_split.items():
            for locus in ['L1_fea', 'L2_fea', 'L3_fea', 'FB']:
                if self.LOCUS_DEPTH[locus] > self.BRANCH_DEPTH[branch]:
                    continue
                if locus not in g_task or locus not in gD_dict:
                    continue
                    
                a = g_task[locus]  # Task gradient
                d = gD_dict[locus]  # Digital gradient
                na = torch.norm(a).item()
                nd = torch.norm(d).item()
                
                if na < 1e-8 or nd < 1e-8:
                    continue
                
                for alpha in alpha_eval_list:
                    s = a + alpha * d
                    cos_S_task = self._safe_cosine(s, a)
                    dir_deriv = torch.dot(a, s).item() / (na + 1e-12)
                    
                    rows.append({
                        'branch': branch,
                        'locus': locus,
                        f'cos_S_task@{alpha:.2f}': cos_S_task,
                        f'dir_deriv_task@{alpha:.2f}': dir_deriv
                    })
        return rows
    
    # NEW: Statistical utility functions
    def _mean_ci_95(self, x: np.ndarray):
        """Return mean, ci_half (95% CI half-width), n"""
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        n = len(x)
        if n == 0:
            return np.nan, np.nan, 0
        m = float(np.mean(x))
        s = float(np.std(x, ddof=1)) if n > 1 else 0.0
        try:
            from scipy.stats import t
            crit = float(t.ppf(0.975, df=n-1)) if n > 1 else 0.0
        except Exception:
            crit = 1.96
        ci_half = crit * (s / math.sqrt(n)) if n > 1 else 0.0
        return m, ci_half, n
    
    def _paired_t_one_sided(self, delta: np.ndarray):
        """Paired t-test (one-sided: improvement>0). Returns (t_stat, p_value)."""
        delta = np.asarray(delta, dtype=float)
        delta = delta[np.isfinite(delta)]
        n = len(delta)
        if n < 2:
            return np.nan, np.nan
        m = float(np.mean(delta))
        s = float(np.std(delta, ddof=1))
        if s == 0:
            return np.inf if m > 0 else -np.inf, 0.0 if m > 0 else 1.0
        t_stat = m / (s / math.sqrt(n))
        # one-sided p-value
        try:
            from scipy.stats import t
            p = 1.0 - float(t.cdf(t_stat, df=n-1))
        except Exception:
            # normal approximation
            z = t_stat
            p = 0.5 * (1.0 - math.erf(z / math.sqrt(2)))
        return t_stat, p
    
    def _wilson_ci(self, p_hat: float, n: int, z: float = 1.96):
        """Wilson 95% CI for proportion."""
        if n == 0:
            return (np.nan, np.nan)
        denom = 1.0 + z**2 / n
        center = (p_hat + z*z/(2*n)) / denom
        half = z * math.sqrt((p_hat*(1-p_hat) + z*z/(4*n)) / n) / denom
        return (center - half, center + half)
    
    def _compute_alignment_metrics_with_fp32(self, g_analog, gD_split, g_fp32, alpha_eval_list):
        """Compute alignment metrics comparing analog and attention gradients with FP32 reference."""
        rows = []
        
        print(f"    Computing alignment metrics with FP32 reference")
        
        for branch_name, gD_dict in gD_split.items():
            # [C6] Include FB in comparison locus
            for locus in ['L1_fea', 'L2_fea', 'L3_fea', 'FB']:
                # Check depth-based reachability
                if self.LOCUS_DEPTH[locus] > self.BRANCH_DEPTH[branch_name]:
                    continue
                
                if locus not in g_analog or locus not in g_fp32 or locus not in gD_dict:
                    continue
                
                g_A = g_analog[locus]  # Analog backbone gradient
                g_FP = g_fp32[locus]   # FP32 backbone reference
                g_D = gD_dict[locus]   # Digital attention gradient
                
                # Skip if gradients are too small
                norm_A = torch.norm(g_A).item()
                norm_FP = torch.norm(g_FP).item()
                norm_D = torch.norm(g_D).item()
                
                if norm_FP < 1e-8:  # FP32 reference must be valid
                    continue
                
                # Key comparisons:
                # 1. Analog backbone vs FP32 backbone (how much analog deviates)
                cos_A_FP = self._safe_cosine(g_A, g_FP) if norm_A > 1e-8 else 0
                
                # 2. Digital attention vs FP32 backbone (does attention help?)
                cos_D_FP = self._safe_cosine(g_D, g_FP) if norm_D > 1e-8 else 0
                
                # 3. Digital attention vs Analog backbone (correlation)
                cos_D_A = self._safe_cosine(g_D, g_A) if norm_D > 1e-8 and norm_A > 1e-8 else 0
                
                # Angles in degrees
                angle_A_FP_deg = np.arccos(np.clip(cos_A_FP, -1, 1)) * 180 / np.pi
                angle_D_FP_deg = np.arccos(np.clip(cos_D_FP, -1, 1)) * 180 / np.pi if norm_D > 1e-8 else 90
                
                # Combined gradient: S = A + α*D
                best_improvement = 0
                best_alpha = 0
                cos_S_FP_values = {}
                
                for alpha in alpha_eval_list:
                    if norm_A > 1e-8:  # Only if analog gradient exists
                        g_S = g_A + alpha * g_D  # Combined gradient
                        norm_S = torch.norm(g_S).item()
                        
                        # Norm-preserving scaling to compare fairly
                        if norm_S > 1e-8:
                            g_S_tilde = (norm_A / norm_S) * g_S
                            cos_S_FP = self._safe_cosine(g_S_tilde, g_FP)
                        else:
                            cos_S_FP = cos_A_FP
                    else:
                        # If no analog gradient, use attention only
                        cos_S_FP = cos_D_FP
                    
                    cos_S_FP_values[f'cos_S_FP@{alpha:.2f}'] = cos_S_FP
                    
                    # Track best improvement
                    improvement = cos_S_FP - cos_A_FP
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_alpha = alpha
                
                # Angle improvement (negative means better alignment)
                if best_alpha > 0 and norm_A > 1e-8:
                    angle_improve_deg = angle_A_FP_deg - np.arccos(np.clip(cos_S_FP_values[f'cos_S_FP@{best_alpha:.2f}'], -1, 1)) * 180 / np.pi
                else:
                    angle_improve_deg = 0
                
                # Critical alpha where gradient becomes orthogonal
                if norm_A > 1e-8 and norm_D > 1e-8:
                    dot_AD = torch.dot(g_A, g_D).item()
                    alpha_crit_task = -dot_AD / (norm_D ** 2)
                    
                    dot_FP_D = torch.dot(g_FP, g_D).item()
                    dot_FP_A = torch.dot(g_FP, g_A).item()
                    alpha_crit_fp32 = -dot_FP_A / dot_FP_D if abs(dot_FP_D) > 1e-8 else float('inf')
                else:
                    alpha_crit_task = float('inf')
                    alpha_crit_fp32 = float('inf')
                
                # Create row with both task-centric and FP32 alignment metrics
                row = {
                    'branch': branch_name,
                    'locus': locus,
                    # Task-centric metrics (also available with FP32)
                    'cos_D_task': cos_D_A,  # Digital vs Analog (task gradient)
                    'norm_gTask': norm_A,   # Analog is our task gradient
                    'norm_gD': norm_D,
                    'r_norm_ratio_task': norm_D / norm_A if norm_A > 0 else float('inf'),
                    'alpha_crit_task': alpha_crit_task,
                    # FP32 alignment metrics
                    'cos_A_FP': cos_A_FP,  # Analog vs FP32 alignment
                    'cos_D_FP': cos_D_FP,  # Attention vs FP32 alignment
                    'cos_D_A': cos_D_A,    # Attention vs Analog correlation
                    'angle_A_FP_deg': angle_A_FP_deg,
                    'angle_D_FP_deg': angle_D_FP_deg,
                    'cos_S_FP': cos_S_FP_values.get(f'cos_S_FP@1.0', cos_A_FP),
                    'angle_improve_deg': angle_improve_deg,
                    'norm_gA': norm_A,
                    'norm_gFP32': norm_FP,
                    'r_norm_ratio': norm_D / norm_A if norm_A > 0 else float('inf'),
                    'alpha_crit_fp32': alpha_crit_fp32,
                    **cos_S_FP_values
                }
                rows.append(row)
        
        return rows
    
    def _compute_alignment_metrics(self, g_analog, g_fp32, gD_split, alpha_eval_list):
        """Compute analog-FP32 alignment metrics with attention gradient improvements."""
        rows = []
        
        print(f"    Computing alignment metrics for {len(gD_split)} branches")
        
        for branch_name, gD_dict in gD_split.items():
            # [C6] Include FB in comparison locus
            for locus in ['L1_fea', 'L2_fea', 'L3_fea', 'FB']:
                # Check depth-based reachability
                if self.LOCUS_DEPTH[locus] > self.BRANCH_DEPTH[branch_name]:
                    continue
                
                if locus not in g_analog or locus not in g_fp32 or locus not in gD_dict:
                    continue
                
                g_A = g_analog[locus]  # Analog gradient
                g_FP = g_fp32[locus]   # FP32 reference
                g_D = gD_dict[locus]   # Attention gradient
                
                # Skip if gradients are too small
                norm_A = torch.norm(g_A).item()
                norm_FP = torch.norm(g_FP).item()
                norm_D = torch.norm(g_D).item()
                
                if norm_A < 1e-8 or norm_FP < 1e-8:
                    continue
                
                # Compute cosine similarities
                cos_A_FP = self._safe_cosine(g_A, g_FP)  # Analog-FP32 alignment
                cos_D_FP = self._safe_cosine(g_D, g_FP) if norm_D > 1e-8 else 0  # Attention-FP32
                cos_D_A = self._safe_cosine(g_D, g_A) if norm_D > 1e-8 else 0    # Attention-Analog
                
                # Compute angles in degrees
                angle_A_FP_deg = np.arccos(cos_A_FP) * 180 / np.pi
                angle_D_FP_deg = np.arccos(cos_D_FP) * 180 / np.pi if norm_D > 1e-8 else 90
                
                # Critical alpha values
                dot_AD = torch.dot(g_A, g_D).item() if norm_D > 1e-8 else 0
                alpha_crit_task = -dot_AD / (norm_D ** 2) if norm_D > 1e-8 else float('inf')
                
                dot_FP_D = torch.dot(g_FP, g_D).item() if norm_D > 1e-8 else 0
                alpha_crit_fp32 = -torch.dot(g_FP, g_A).item() / dot_FP_D if abs(dot_FP_D) > 1e-8 else float('inf')
                
                # Compute improvement for different alpha values
                cos_S_FP_values = {}
                best_improvement = 0
                best_alpha = 0
                
                for alpha in alpha_eval_list:
                    g_S = g_A + alpha * g_D  # Combined gradient
                    norm_S = torch.norm(g_S).item()
                    
                    # Norm-preserving scaling
                    g_S_tilde = (norm_A / (norm_S + 1e-12)) * g_S
                    cos_S_FP = self._safe_cosine(g_S_tilde, g_FP)
                    cos_S_FP_values[f'cos_S_FP@{alpha:.2f}'] = cos_S_FP
                    
                    # Track best improvement
                    improvement = cos_S_FP - cos_A_FP
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_alpha = alpha
                
                # Angle improvement for best alpha
                angle_improve_deg = np.arccos(cos_A_FP) - np.arccos(cos_S_FP_values[f'cos_S_FP@{best_alpha:.2f}'])
                angle_improve_deg = angle_improve_deg * 180 / np.pi
                
                # Create row
                row = {
                    'branch': branch_name,
                    'locus': locus,
                    'cos_A_FP': cos_A_FP,
                    'cos_D_FP': cos_D_FP,
                    'cos_D_A': cos_D_A,
                    'angle_A_FP_deg': angle_A_FP_deg,
                    'angle_D_FP_deg': angle_D_FP_deg,
                    'cos_S_FP': cos_S_FP_values.get(f'cos_S_FP@1.0', cos_A_FP),  # Default α=1
                    'angle_improve_deg': angle_improve_deg,
                    'norm_gA': norm_A,
                    'norm_gD': norm_D,
                    'norm_gFP32': norm_FP,
                    'r_norm_ratio': norm_D / norm_A if norm_A > 0 else 0,
                    'alpha_crit_task': alpha_crit_task,
                    'alpha_crit_fp32': alpha_crit_fp32,
                    **cos_S_FP_values
                }
                rows.append(row)
        
        return rows

    def _aggregate_epoch_and_plot(self, epoch: int, has_fp32: bool):
        """Aggregate statistics and create plots for an epoch."""
        loss_coef_str = f"loss_coef_{self.loss_coefficient:.2f}".replace(".", "_").replace("-", "neg")
        probe_dir = f'./probes/{self.device_type}/{loss_coef_str}'
        os.makedirs(f'{probe_dir}/plots', exist_ok=True)
        
        # 1) Read all seed CSV files for this epoch
        if has_fp32:
            pattern = f'{probe_dir}/epoch_{epoch}_seed_*_fp32_alignment.csv'
        else:
            pattern = f'{probe_dir}/epoch_{epoch}_seed_*_task_centric.csv'
        
        files = sorted(glob.glob(pattern))
        if len(files) == 0:
            print(f"  [aggregate] No files found for epoch {epoch}")
            return
        
        df_list = []
        for path in files:
            try:
                df = pd.read_csv(path)
                # Convert numeric columns
                for col in df.columns:
                    if col not in ['epoch', 'batch', 'seed', 'branch', 'locus']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                df_list.append(df)
            except Exception as e:
                print(f"  [aggregate] Failed to read {path}: {e}")
        
        if not df_list:
            return
        
        df = pd.concat(df_list, ignore_index=True)
        
        # 2) Helper function for summarizing series
        def summarize_series(x):
            x = pd.to_numeric(x, errors='coerce')
            x = x[np.isfinite(x)]
            # Handle both pandas Series and numpy arrays
            if hasattr(x, 'values'):
                x_array = x.values
            else:
                x_array = np.asarray(x)
            m, ci, n = self._mean_ci_95(x_array)
            return pd.Series({'mean': m, 'ci_half': ci, 'n': n})
        
        # 3) Aggregate statistics
        rows = []
        # Loci configuration - exclude FB by default
        loci_task = ['L1_fea', 'L2_fea', 'L3_fea']
        loci_all = loci_task + (['FB'] if self.include_fb else [])
        branches = ['D1', 'D2', 'D3', 'Dsum']
        
        # Task-centric metrics (always available)
        if 'cos_D_task' in df.columns:
            for L in loci_task:  # Use loci_task to exclude FB
                for B in branches:
                    sub = df[(df['locus'] == L) & (df['branch'] == B)]
                    if len(sub) == 0:
                        continue
                    # cos(D,A)
                    if 'cos_D_task' in sub.columns:
                        stats = summarize_series(sub['cos_D_task'])
                        rows.append(dict(
                            level='branch_locus', epoch=epoch, branch=B, locus=L,
                            metric='cos_D_task', alpha='-', **stats
                        ))
                    # r = ||D||/||A|| (for scatter plot size)
                    if 'r_norm_ratio_task' in sub.columns:
                        stats = summarize_series(sub['r_norm_ratio_task'])
                        rows.append(dict(
                            level='branch_locus', epoch=epoch, branch=B, locus=L,
                            metric='r_norm_ratio_task', alpha='-', **stats
                        ))
                    elif 'r_norm_ratio' in sub.columns:  # Fallback to r_norm_ratio if exists
                        stats = summarize_series(sub['r_norm_ratio'])
                        rows.append(dict(
                            level='branch_locus', epoch=epoch, branch=B, locus=L,
                            metric='r_norm_ratio', alpha='-', **stats
                        ))
                    # alpha_crit_task
                    if 'alpha_crit_task' in sub.columns:
                        stats = summarize_series(sub['alpha_crit_task'])
                        rows.append(dict(
                            level='branch_locus', epoch=epoch, branch=B, locus=L,
                            metric='alpha_crit_task', alpha='-', **stats
                        ))
                    
                    # NEW: beta_task
                    if 'beta_task' in sub.columns:
                        stats = summarize_series(sub['beta_task'])
                        rows.append(dict(
                            level='branch_locus', epoch=epoch, branch=B, locus=L,
                            metric='beta_task', alpha='-', **stats
                        ))
                    
                    # NEW: descent_gain_task for each alpha with statistical tests
                    for col in sub.columns:
                        if col.startswith('descent_gain_task@'):
                            vals = pd.to_numeric(sub[col], errors='coerce').dropna().values
                            if len(vals) == 0:
                                continue
                            m, ci, n = self._mean_ci_95(vals)
                            tstat, p = self._paired_t_one_sided(vals)  # H1: gain > 0
                            d_eff = float(m/np.std(vals, ddof=1)) if n > 1 and np.std(vals, ddof=1) > 0 else np.nan
                            pos_ratio = float((vals > 0).mean())
                            lo, hi = self._wilson_ci(pos_ratio, n)
                            rows.append(dict(
                                level='branch_locus', epoch=epoch, branch=B, locus=L,
                                metric='descent_gain_task', alpha=col.split('@')[1],
                                mean=m, ci_half=ci, n=n,
                                t_stat=tstat, p_value=p, cohens_d=d_eff,
                                pos_ratio=pos_ratio, pos_ratio_low=lo, pos_ratio_high=hi
                            ))
                        
                        # NEW: descent_ratio metrics
                        elif col.startswith('descent_ratio@'):
                            vals = pd.to_numeric(sub[col], errors='coerce').dropna().values
                            if len(vals) == 0:
                                continue
                            m, ci, n = self._mean_ci_95(vals)
                            tstat, p = self._paired_t_one_sided(vals)  # H1: ratio > 0
                            rows.append(dict(
                                level='branch_locus', epoch=epoch, branch=B, locus=L,
                                metric='descent_ratio', alpha=col.split('@')[1],
                                mean=m, ci_half=ci, n=n,
                                t_stat=tstat, p_value=p
                            ))
                        
                        # NEW: cos_S_task metrics
                        elif col.startswith('cos_S_task@'):
                            vals = pd.to_numeric(sub[col], errors='coerce').dropna().values
                            if len(vals) == 0:
                                continue
                            stats = summarize_series(vals)
                            rows.append(dict(
                                level='branch_locus', epoch=epoch, branch=B, locus=L,
                                metric='cos_S_task', alpha=col.split('@')[1], **stats
                            ))
                        
                        # NEW: dir_deriv_task metrics
                        elif col.startswith('dir_deriv_task@'):
                            vals = pd.to_numeric(sub[col], errors='coerce').dropna().values
                            if len(vals) == 0:
                                continue
                            stats = summarize_series(vals)
                            rows.append(dict(
                                level='branch_locus', epoch=epoch, branch=B, locus=L,
                                metric='dir_deriv_task', alpha=col.split('@')[1], **stats
                            ))
        
        # FP32 alignment metrics
        if has_fp32 and 'cos_A_FP' in df.columns:
            # Locus-only cos_A_FP (deduplicated)
            for L in loci_all:  # Use loci_all for FP32 metrics
                sub = df[df['locus'] == L][['epoch', 'seed', 'batch', 'locus', 'cos_A_FP']].drop_duplicates()
                if len(sub) == 0:
                    continue
                stats = summarize_series(sub['cos_A_FP'])
                rows.append(dict(
                    level='locus_only', epoch=epoch, branch='-', locus=L,
                    metric='cos_A_FP', alpha='-', **stats
                ))
            
            # Branch × locus metrics (cos_D_FP, cos_D_A)
            for L in loci_all:  # Use loci_all for FP32 metrics
                for B in branches:
                    sub = df[(df['locus'] == L) & (df['branch'] == B)]
                    if len(sub) == 0:
                        continue
                    
                    # cos_D_FP
                    if 'cos_D_FP' in sub.columns:
                        stats = summarize_series(sub['cos_D_FP'])
                        rows.append(dict(
                            level='branch_locus', epoch=epoch, branch=B, locus=L,
                            metric='cos_D_FP', alpha='-', **stats
                        ))
                    
                    # cos_D_A (for triple-alignment plot)
                    if 'cos_D_A' in sub.columns:
                        stats = summarize_series(sub['cos_D_A'])
                        rows.append(dict(
                            level='branch_locus', epoch=epoch, branch=B, locus=L,
                            metric='cos_D_A', alpha='-', **stats
                        ))
                    
                    # r_norm_ratio for FP32 comparison
                    if 'r_norm_ratio' in sub.columns:
                        stats = summarize_series(sub['r_norm_ratio'])
                        rows.append(dict(
                            level='branch_locus', epoch=epoch, branch=B, locus=L,
                            metric='r_norm_ratio', alpha='-', **stats
                        ))
                    
                    # alpha_crit_fp32
                    if 'alpha_crit_fp32' in sub.columns:
                        stats = summarize_series(sub['alpha_crit_fp32'])
                        rows.append(dict(
                            level='branch_locus', epoch=epoch, branch=B, locus=L,
                            metric='alpha_crit_fp32', alpha='-', **stats
                        ))
                    
                    # Alpha sweep - compute delta metrics
                    alphas = []
                    for col in sub.columns:
                        if col.startswith('cos_S_FP@'):
                            try:
                                alphas.append(float(col.split('@')[1]))
                            except:
                                pass
                    alphas = sorted(set(alphas))
                    
                    best_alpha = None
                    best_mean = -np.inf
                    
                    for a in alphas:
                        col = f'cos_S_FP@{a:.1f}'
                        if col not in sub.columns or 'cos_A_FP' not in sub.columns:
                            continue
                        
                        # Compute delta = cos_S - cos_A
                        delta = (sub[col] - sub['cos_A_FP']).dropna().values
                        if len(delta) == 0:
                            continue
                        
                        m, ci, n = self._mean_ci_95(delta)
                        tstat, p = self._paired_t_one_sided(delta)
                        d_eff = float(m / np.std(delta, ddof=1)) if len(delta) > 1 and np.std(delta, ddof=1) > 0 else np.nan
                        pos_ratio = float((delta > 0).mean())
                        lo, hi = self._wilson_ci(pos_ratio, len(delta))
                        
                        rows.append(dict(
                            level='branch_locus', epoch=epoch, branch=B, locus=L,
                            metric='delta', alpha=f'{a:.2f}', mean=m, ci_half=ci, n=len(delta),
                            t_stat=tstat, p_value=p, cohens_d=d_eff,
                            pos_ratio=pos_ratio, pos_ratio_low=lo, pos_ratio_high=hi
                        ))
                        
                        if m > best_mean:
                            best_mean = m
                            best_alpha = a
                    
                    # Record best alpha
                    if best_alpha is not None:
                        rows.append(dict(
                            level='branch_locus', epoch=epoch, branch=B, locus=L,
                            metric='best_alpha', alpha=f'{best_alpha:.2f}', mean=best_alpha,
                            ci_half=np.nan, n=1
                        ))
        
        # 4) Save summary CSV
        if rows:
            summary_df = pd.DataFrame(rows)
            loss_coef_str = f"loss_coef_{self.loss_coefficient:.2f}".replace(".", "_").replace("-", "neg")
            probe_dir = f'./probes/{loss_coef_str}'
            os.makedirs(probe_dir, exist_ok=True)
            summary_path = f'{probe_dir}/epoch_{epoch}_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            print(f"  Saved summary to {summary_path}")
            
            # 5) Log metrics to wandb
            self._log_metrics_to_wandb(epoch, summary_df)
            
            # 6) Create plots
            self._create_epoch_plots(epoch, summary_df, has_fp32)
    
    def _log_metrics_to_wandb(self, epoch: int, df: pd.DataFrame):
        """Log aggregated probe metrics to wandb using Lightning's self.log()."""
        try:
            # Log aggregated metrics by branch and locus
            branch_locus_df = df[df['level'] == 'branch_locus']
            
            # Track key metrics for logging
            key_metrics_to_log = ['cos_D_task', 'alpha_crit_task', 'descent_ratio', 'beta_task', 
                                  'cos_S_task', 'dir_deriv_task', 'cos_D_FP', 'alpha_crit_fp32', 'delta']
            
            for _, row in branch_locus_df.iterrows():
                branch = row['branch']
                locus = row['locus']
                metric = row['metric']
                
                # Only log key metrics to avoid cluttering wandb
                if metric not in key_metrics_to_log:
                    continue
                
                # Create a meaningful metric name
                base_name = f"probe/{branch}_{locus}/{metric}"
                
                # For metrics with alpha values, include alpha in the name
                alpha_str = ""
                if 'alpha' in row and row['alpha'] != '-' and pd.notna(row['alpha']):
                    alpha_str = f"_a{row['alpha']}"
                
                # Log mean value
                if 'mean' in row and pd.notna(row['mean']):
                    self.log(f"{base_name}{alpha_str}", row['mean'], 
                            on_step=False, on_epoch=True, sync_dist=True)
                    
                    # Also log confidence interval for important metrics
                    if metric in ['descent_ratio', 'delta'] and 'ci_half' in row and pd.notna(row['ci_half']):
                        self.log(f"{base_name}{alpha_str}_ci_upper", row['mean'] + row['ci_half'],
                                on_step=False, on_epoch=True, sync_dist=True)
                        self.log(f"{base_name}{alpha_str}_ci_lower", row['mean'] - row['ci_half'],
                                on_step=False, on_epoch=True, sync_dist=True)
                    
                    # Log p-value for descent_ratio to track significance
                    if metric == 'descent_ratio' and 'p_value' in row and pd.notna(row['p_value']):
                        self.log(f"{base_name}{alpha_str}_pval", row['p_value'],
                                on_step=False, on_epoch=True, sync_dist=True)
            
            # Log global summary metrics
            global_df = df[df['level'] == 'global']
            for _, row in global_df.iterrows():
                metric = row['metric']
                if metric in key_metrics_to_log and 'mean' in row and pd.notna(row['mean']):
                    alpha_str = ""
                    if 'alpha' in row and row['alpha'] != '-' and pd.notna(row['alpha']):
                        alpha_str = f"_a{row['alpha']}"
                    self.log(f"probe/global/{metric}{alpha_str}", row['mean'],
                            on_step=False, on_epoch=True, sync_dist=True)
            
            # Log special aggregate metrics for descent analysis
            # Average positive descent ratio (improvement percentage)
            descent_df = df[(df['metric'] == 'descent_ratio') & (df['level'] == 'branch_locus')]
            if len(descent_df) > 0:
                for alpha in ['1.00', '0.50']:  # Focus on key alpha values
                    alpha_df = descent_df[descent_df['alpha'] == alpha]
                    if len(alpha_df) > 0:
                        # Percentage of positive descent ratios
                        pos_ratios = alpha_df['pos_ratio'].dropna()
                        if len(pos_ratios) > 0:
                            avg_pos_ratio = pos_ratios.mean()
                            self.log(f"probe/descent_improvement_rate_a{alpha}", avg_pos_ratio,
                                    on_step=False, on_epoch=True, sync_dist=True)
                        
                        # Average descent ratio magnitude
                        mean_values = alpha_df['mean'].dropna()
                        if len(mean_values) > 0:
                            avg_descent = mean_values.mean()
                            self.log(f"probe/avg_descent_ratio_a{alpha}", avg_descent,
                                    on_step=False, on_epoch=True, sync_dist=True)
            
            print(f"  Logged probe metrics to wandb for epoch {epoch}")
            
        except Exception as e:
            print(f"  Warning: Failed to log probe metrics: {e}")
    
    def _create_epoch_plots(self, epoch: int, df: pd.DataFrame, has_fp32: bool):
        """Create enhanced visualization plots for an epoch with triple-alignment panels."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Use loci_task to exclude FB by default
        loci_task = ['L1_fea', 'L2_fea', 'L3_fea']
        loci = loci_task + (['FB'] if self.include_fb else [])
        branches = ['D1', 'D2', 'D3', 'Dsum']
        
        # Create main figure with 3x3 grid
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(f'Epoch {epoch} Gradient Alignment Analysis', fontsize=16)
        
        # --- Row 1: Triple Alignment Panel ---
        
        # Plot 1-1: cos(D, A) - Digital vs Analog alignment
        ax = axes[0, 0]
        subset = df[(df['metric'] == 'cos_D_task') & (df['level'] == 'branch_locus')]
        if len(subset) > 0:
            for branch in branches:
                branch_data = subset[subset['branch'] == branch]
                if len(branch_data) == 0:
                    continue
                locus_idx = [loci.index(l) for l in branch_data['locus']]
                ax.errorbar(locus_idx, branch_data['mean'].values, 
                           yerr=branch_data['ci_half'].values,
                           label=branch, marker='o', capsize=5)
            ax.set_xticks(range(len(loci)))
            ax.set_xticklabels(loci, rotation=45)
            ax.set_xlabel('Locus')
            ax.set_ylabel('cos(D, A)')
            ax.set_title('Digital vs Analog Alignment')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([-1, 1])
        
        # Plot 1-2: cos(A, FP32) - Analog vs FP32 alignment
        ax = axes[0, 1]
        if has_fp32:
            subset = df[(df['metric'] == 'cos_A_FP') & (df['level'] == 'locus_only')]
            if len(subset) > 0:
                # cos_A_FP is locus-only (same for all branches)
                locus_idx = [loci.index(l) for l in subset['locus'] if l in loci]
                locus_vals = [subset[subset['locus'] == l]['mean'].values[0] for l in subset['locus'] if l in loci]
                locus_errs = [subset[subset['locus'] == l]['ci_half'].values[0] for l in subset['locus'] if l in loci]
                
                ax.errorbar(locus_idx, locus_vals,
                           yerr=locus_errs,
                           label='Analog-FP32', marker='s', capsize=5, linewidth=2)
                ax.set_xticks(range(len(loci)))
                ax.set_xticklabels(loci, rotation=45)
                ax.set_xlabel('Locus')
                ax.set_ylabel('cos(A, FP32)')
                ax.set_title('Analog vs FP32 Alignment')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_ylim([-1, 1])
        else:
            ax.text(0.5, 0.5, 'FP32 not available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Analog vs FP32 Alignment')
        
        # Plot 1-3: cos(D, FP32) - Digital vs FP32 alignment
        ax = axes[0, 2]
        if has_fp32:
            subset = df[(df['metric'] == 'cos_D_FP') & (df['level'] == 'branch_locus')]
            if len(subset) > 0:
                for branch in branches:
                    branch_data = subset[subset['branch'] == branch]
                    if len(branch_data) == 0:
                        continue
                    locus_idx = [loci.index(l) for l in branch_data['locus']]
                    ax.errorbar(locus_idx, branch_data['mean'].values,
                               yerr=branch_data['ci_half'].values,
                               label=branch, marker='^', capsize=5)
                ax.set_xticks(range(len(loci)))
                ax.set_xticklabels(loci, rotation=45)
                ax.set_xlabel('Locus')
                ax.set_ylabel('cos(D, FP32)')
                ax.set_title('Digital vs FP32 Alignment')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_ylim([-1, 1])
        else:
            ax.text(0.5, 0.5, 'FP32 not available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Digital vs FP32 Alignment')
        
        # --- Row 2: Scatter Plots with Norm Ratio ---
        
        # Plot 2-1: cos(D,A) vs norm ratio (size = norm ratio)
        ax = axes[1, 0]
        cos_subset = df[(df['metric'] == 'cos_D_task') & (df['level'] == 'branch_locus')]
        norm_subset = df[(df['metric'] == 'r_norm_ratio_task') & (df['level'] == 'branch_locus')]
        if len(cos_subset) > 0 and len(norm_subset) > 0:
            for branch in branches:
                cos_data = cos_subset[cos_subset['branch'] == branch]
                norm_data = norm_subset[norm_subset['branch'] == branch]
                if len(cos_data) == 0 or len(norm_data) == 0:
                    continue
                # Match by locus
                for locus in cos_data['locus'].unique():
                    cos_val = cos_data[cos_data['locus'] == locus]['mean'].values
                    norm_val = norm_data[norm_data['locus'] == locus]['mean'].values
                    if len(cos_val) > 0 and len(norm_val) > 0:
                        size = 100 * np.clip(norm_val[0], 0.1, 10)
                        ax.scatter(loci.index(locus), cos_val[0], s=size, alpha=0.6, label=branch if locus == cos_data['locus'].iloc[0] else "")
            ax.set_xlabel('Locus Index')
            ax.set_ylabel('cos(D, A)')
            ax.set_title('Task Alignment vs Norm Ratio (size ∝ norm ratio)')
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([-1, 1])
        
        # Plot 2-2: cos(A,FP32) vs norm ratio
        ax = axes[1, 1]
        if has_fp32:
            cos_subset = df[(df['metric'] == 'cos_A_FP') & (df['level'] == 'locus_only')]
            norm_subset = df[(df['metric'] == 'r_norm_ratio') & (df['level'] == 'branch_locus')]
            if len(cos_subset) > 0 and len(norm_subset) > 0:
                for locus in loci:
                    cos_data = cos_subset[cos_subset['locus'] == locus]
                    if len(cos_data) == 0:
                        continue
                    cos_val = cos_data['mean'].values[0]  # Same for all branches
                    
                    for branch in branches:
                        norm_data = norm_subset[(norm_subset['branch'] == branch) & (norm_subset['locus'] == locus)]
                        if len(norm_data) == 0:
                            continue
                        norm_val = norm_data['mean'].values[0]
                        size = 100 * np.clip(norm_val, 0.1, 10)
                        ax.scatter(loci.index(locus), cos_val, s=size, alpha=0.6, 
                                  label=branch if locus == loci[0] else "")
                ax.set_xlabel('Locus Index')
                ax.set_ylabel('cos(A, FP32)')
                ax.set_title('FP32 Alignment vs Norm Ratio (size ∝ norm ratio)')
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_ylim([-1, 1])
        else:
            ax.text(0.5, 0.5, 'FP32 metrics not available', ha='center', va='center', transform=ax.transAxes)
        
        # Plot 2-3: Norm ratio distribution
        ax = axes[1, 2]
        norm_subset = df[(df['metric'] == 'r_norm_ratio_task') & (df['level'] == 'branch_locus')]
        if len(norm_subset) > 0:
            for branch in branches:
                branch_data = norm_subset[norm_subset['branch'] == branch]
                if len(branch_data) == 0:
                    continue
                locus_idx = [loci.index(l) for l in branch_data['locus']]
                ax.errorbar(locus_idx, branch_data['mean'].values,
                           yerr=branch_data['ci_half'].values,
                           label=branch, marker='D', capsize=5)
            ax.set_xticks(range(len(loci)))
            ax.set_xticklabels(loci, rotation=45)
            ax.set_xlabel('Locus')
            ax.set_ylabel('Norm Ratio (||g_D||/||g_A||)')
            ax.set_title('Gradient Norm Ratios')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
        
        # --- Row 3: Task-centric Alpha Critical ---
        
        # Plot 3-1: Alpha critical by branch (task-centric)
        ax = axes[2, 0]
        alpha_subset = df[(df['metric'] == 'alpha_crit_task') & (df['level'] == 'branch_locus')]
        if len(alpha_subset) > 0:
            width = 0.2
            x_base = np.arange(len(loci))
            for i, branch in enumerate(branches):
                branch_data = alpha_subset[alpha_subset['branch'] == branch]
                if len(branch_data) == 0:
                    continue
                # Ensure correct order
                alpha_means = []
                alpha_errs = []
                for locus in loci:
                    locus_data = branch_data[branch_data['locus'] == locus]
                    if len(locus_data) > 0:
                        alpha_means.append(locus_data['mean'].values[0])
                        alpha_errs.append(locus_data['ci_half'].values[0])
                    else:
                        alpha_means.append(0)
                        alpha_errs.append(0)
                
                ax.bar(x_base + i * width, alpha_means, width, 
                      yerr=alpha_errs, label=branch, capsize=3)
            
            ax.set_xlabel('Locus')
            ax.set_ylabel('Critical Alpha (Task-centric)')
            ax.set_title('Critical Alpha by Branch and Locus')
            ax.set_xticks(x_base + width * 1.5)
            ax.set_xticklabels(loci, rotation=45)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Plot 3-2: Beta_task (or FP32 alpha_crit if enabled)
        ax = axes[2, 1]
        if self.use_fp32_metrics and has_fp32:
            # Show FP32 alpha_crit if FP32 metrics are enabled
            alpha_subset = df[(df['metric'] == 'alpha_crit_fp32') & (df['level'] == 'branch_locus')]
            if len(alpha_subset) > 0:
                width = 0.2
                x_base = np.arange(len(loci))
                for i, branch in enumerate(branches):
                    branch_data = alpha_subset[alpha_subset['branch'] == branch]
                    if len(branch_data) == 0:
                        continue
                    alpha_means = []
                    alpha_errs = []
                    for locus in loci:
                        locus_data = branch_data[branch_data['locus'] == locus]
                        if len(locus_data) > 0:
                            alpha_means.append(locus_data['mean'].values[0])
                            alpha_errs.append(locus_data['ci_half'].values[0])
                        else:
                            alpha_means.append(0)
                            alpha_errs.append(0)
                    
                    ax.bar(x_base + i * width, alpha_means, width,
                          yerr=alpha_errs, label=branch, capsize=3)
                
                ax.set_xlabel('Locus')
                ax.set_ylabel('Critical Alpha (FP32)')
                ax.set_title('FP32 Critical Alpha by Branch and Locus')
                ax.set_xticks(x_base + width * 1.5)
                ax.set_xticklabels(loci, rotation=45)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3, axis='y')
                ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        else:
            # Show beta_task when FP32 is not used (default)
            beta_subset = df[(df['metric'] == 'beta_task') & (df['level'] == 'branch_locus')]
            if len(beta_subset) > 0:
                width = 0.2
                x_base = np.arange(len(loci))
                for i, branch in enumerate(branches):
                    branch_data = beta_subset[beta_subset['branch'] == branch]
                    if len(branch_data) == 0:
                        continue
                    beta_means = []
                    beta_errs = []
                    for locus in loci:
                        locus_data = branch_data[branch_data['locus'] == locus]
                        if len(locus_data) > 0:
                            beta_means.append(locus_data['mean'].values[0])
                            beta_errs.append(locus_data['ci_half'].values[0])
                        else:
                            beta_means.append(0)
                            beta_errs.append(0)
                    
                    ax.bar(x_base + i * width, beta_means, width,
                          yerr=beta_errs, label=branch, capsize=3)
                
                ax.set_xlabel('Locus')
                ax.set_ylabel(r'$\beta_{task} = \langle A,D\rangle/\|A\|^2$')
                ax.set_title('Task-centric β by Branch and Locus')
                ax.set_xticks(x_base + width * 1.5)
                ax.set_xticklabels(loci, rotation=45)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3, axis='y')
                ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            else:
                ax.text(0.5, 0.5, 'Beta_task not available', ha='center', va='center', transform=ax.transAxes)
        
        # Plot 3-3: Task-centric Descent Score (descent_ratio or descent_gain)
        ax = axes[2, 2]
        # Show descent ratio at optimal alpha (e.g., α=0.5 or 1.0)
        descent_subset = df[(df['metric'] == 'descent_ratio') & (df['level'] == 'branch_locus')]
        if len(descent_subset) > 0:
            # Get descent ratio for a specific alpha (e.g., 1.0)
            target_alpha = '1.00'
            descent_alpha = descent_subset[descent_subset['alpha'] == target_alpha]
            
            if len(descent_alpha) > 0:
                # Create heatmap of descent ratios
                descent_matrix = np.zeros((len(branches), len(loci)))
                has_data = np.zeros((len(branches), len(loci)), dtype=bool)
                
                for i, branch in enumerate(branches):
                    for j, locus in enumerate(loci):
                        mask = (descent_alpha['branch'] == branch) & (descent_alpha['locus'] == locus)
                        if mask.any():
                            descent_matrix[i, j] = descent_alpha.loc[mask, 'mean'].values[0]
                            has_data[i, j] = True
                
                # Use diverging colormap centered at 0 (negative=bad, positive=good)
                vmax = np.abs(descent_matrix[has_data]).max() if has_data.any() else 1
                im = ax.imshow(descent_matrix, cmap='RdBu', aspect='auto', vmin=-vmax, vmax=vmax)
                ax.set_xticks(range(len(loci)))
                ax.set_xticklabels(loci, rotation=45)
                ax.set_yticks(range(len(branches)))
                ax.set_yticklabels(branches)
                ax.set_title(f'Descent Ratio @ α={float(target_alpha):.1f}\n(>0 improves optimization)')
                
                # Add text annotations with significance
                for i in range(len(branches)):
                    for j in range(len(loci)):
                        if has_data[i, j]:
                            val = descent_matrix[i, j]
                            # Check if we have p-value
                            mask = (descent_alpha['branch'] == branches[i]) & (descent_alpha['locus'] == loci[j])
                            if mask.any() and 'p_value' in descent_alpha.columns:
                                p_val = descent_alpha.loc[mask, 'p_value'].values[0]
                                sig = ''
                                if p_val < 0.01:
                                    sig = '***'
                                elif p_val < 0.05:
                                    sig = '**'
                                elif p_val < 0.1:
                                    sig = '*'
                                text = ax.text(j, i, f'{val:.3f}\n{sig}',
                                             ha="center", va="center", 
                                             color="white" if abs(val) > vmax*0.6 else "black",
                                             fontsize=8)
                            else:
                                text = ax.text(j, i, f'{val:.3f}',
                                             ha="center", va="center",
                                             color="white" if abs(val) > vmax*0.6 else "black")
                
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Descent Ratio', rotation=270, labelpad=15)
            else:
                # Alternative: Show descent gain if descent ratio not available
                gain_subset = df[(df['metric'] == 'descent_gain_task') & (df['level'] == 'branch_locus')]
                if len(gain_subset) > 0:
                    target_alpha = '1.0'
                    gain_alpha = gain_subset[gain_subset['alpha'] == target_alpha]
                    if len(gain_alpha) > 0:
                        width = 0.2
                        x_base = np.arange(len(loci))
                        for i, branch in enumerate(branches):
                            branch_data = gain_alpha[gain_alpha['branch'] == branch]
                            if len(branch_data) == 0:
                                continue
                            gain_means = []
                            gain_errs = []
                            for locus in loci:
                                locus_data = branch_data[branch_data['locus'] == locus]
                                if len(locus_data) > 0:
                                    gain_means.append(locus_data['mean'].values[0])
                                    gain_errs.append(locus_data['ci_half'].values[0])
                                else:
                                    gain_means.append(0)
                                    gain_errs.append(0)
                            
                            ax.bar(x_base + i * width, gain_means, width,
                                  yerr=gain_errs, label=branch, capsize=3)
                        
                        ax.set_xlabel('Locus')
                        ax.set_ylabel('Descent Gain')
                        ax.set_title(f'Task Descent Gain @ α={target_alpha}')
                        ax.set_xticks(x_base + width * 1.5)
                        ax.set_xticklabels(loci, rotation=45)
                        ax.legend(fontsize=8)
                        ax.grid(True, alpha=0.3, axis='y')
                        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                else:
                    ax.text(0.5, 0.5, 'No descent metrics available', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No descent metrics available', ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(f'./probes/{self.device_type}/plots', exist_ok=True)
        plot_path = f'./probes/{self.device_type}/plots/epoch_{epoch}_triple_alignment.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved triple-alignment visualization to {plot_path}")
        
        # Create individual Gain_task(α) curves for each branch×locus
        for L in loci_task:
            for B in branches:
                sub = df[(df['metric'] == 'descent_gain_task') & (df['branch'] == B) & (df['locus'] == L)]
                if len(sub) == 0:
                    continue
                
                # Sort by alpha value
                s = sub.copy()
                s['alpha_val'] = pd.to_numeric(s['alpha'], errors='coerce')
                s = s.dropna(subset=['alpha_val']).sort_values('alpha_val')
                
                if len(s) > 0:
                    a_vals = s['alpha_val'].values
                    m_vals = s['mean'].values
                    e_vals = s['ci_half'].values
                    p_vals = s['p_value'].values if 'p_value' in s.columns else None
                    
                    plt.figure(figsize=(8, 6))
                    plt.plot(a_vals, m_vals, marker='o', linewidth=2, markersize=8)
                    plt.fill_between(a_vals, m_vals - e_vals, m_vals + e_vals, alpha=0.3)
                    plt.axhline(y=0, linestyle='--', color='red', alpha=0.5)
                    
                    # Add significance markers
                    if p_vals is not None:
                        for i, (a, m, p) in enumerate(zip(a_vals, m_vals, p_vals)):
                            if p < 0.01:
                                plt.text(a, m + 0.01, '***', ha='center', fontsize=10)
                            elif p < 0.05:
                                plt.text(a, m + 0.01, '**', ha='center', fontsize=10)
                            elif p < 0.1:
                                plt.text(a, m + 0.01, '*', ha='center', fontsize=10)
                    
                    plt.xlabel('α', fontsize=12)
                    plt.ylabel('Task-centric Gain (↑ better)', fontsize=12)
                    plt.title(f'Epoch {epoch}: Gain_task(α) @ {B}/{L}', fontsize=14)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    gain_path = f'./probes/{self.device_type}/plots/epoch_{epoch}_gain_task_{B}_{L}.png'
                    plt.savefig(gain_path, dpi=150)
                    plt.close()
                    print(f"  Saved gain curve to {gain_path}")
        
        # Create descent ratio curves based on Descent Lemma
        for L in loci_task:
            for B in branches:
                sub = df[(df['metric'] == 'descent_ratio') & (df['branch'] == B) & (df['locus'] == L)]
                if len(sub) == 0:
                    continue
                
                # Sort by alpha value
                s = sub.copy()
                s['alpha_val'] = pd.to_numeric(s['alpha'], errors='coerce')
                s = s.dropna(subset=['alpha_val']).sort_values('alpha_val')
                
                if len(s) > 0:
                    a_vals = s['alpha_val'].values
                    m_vals = s['mean'].values
                    e_vals = s['ci_half'].values
                    p_vals = s['p_value'].values if 'p_value' in s.columns else None
                    
                    plt.figure(figsize=(8, 6))
                    plt.plot(a_vals, m_vals, marker='s', linewidth=2, markersize=8, color='darkblue')
                    plt.fill_between(a_vals, m_vals - e_vals, m_vals + e_vals, alpha=0.3, color='blue')
                    plt.axhline(y=0, linestyle='--', color='red', alpha=0.5)
                    
                    # Add significance markers
                    if p_vals is not None:
                        for i, (a, m, p) in enumerate(zip(a_vals, m_vals, p_vals)):
                            if p < 0.01:
                                plt.text(a, m + 0.02, '***', ha='center', fontsize=10)
                            elif p < 0.05:
                                plt.text(a, m + 0.02, '**', ha='center', fontsize=10)
                            elif p < 0.1:
                                plt.text(a, m + 0.02, '*', ha='center', fontsize=10)
                    
                    plt.xlabel('α', fontsize=12)
                    plt.ylabel('Descent Ratio (vs α=0)', fontsize=12)
                    plt.title(f'Epoch {epoch}: Descent Lemma Ratio @ {B}/{L}', fontsize=14)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    descent_path = f'./probes/{self.device_type}/plots/epoch_{epoch}_descent_ratio_{B}_{L}.png'
                    plt.savefig(descent_path, dpi=150)
                    plt.close()
                    print(f"  Saved descent ratio curve to {descent_path}")
        
        # Create directional derivative plots
        for L in loci_task:
            for B in branches:
                sub = df[(df['metric'] == 'dir_deriv_task') & (df['branch'] == B) & (df['locus'] == L)]
                if len(sub) == 0:
                    continue
                
                # Sort by alpha value
                s = sub.copy()
                s['alpha_val'] = pd.to_numeric(s['alpha'], errors='coerce')
                s = s.dropna(subset=['alpha_val']).sort_values('alpha_val')
                
                if len(s) > 0:
                    a_vals = s['alpha_val'].values
                    m_vals = s['mean'].values
                    e_vals = s['ci_half'].values
                    
                    # Also get cos_S_task for comparison
                    cos_sub = df[(df['metric'] == 'cos_S_task') & (df['branch'] == B) & (df['locus'] == L)]
                    if len(cos_sub) > 0:
                        cos_s = cos_sub.copy()
                        cos_s['alpha_val'] = pd.to_numeric(cos_s['alpha'], errors='coerce')
                        cos_s = cos_s.dropna(subset=['alpha_val']).sort_values('alpha_val')
                        cos_vals = cos_s['mean'].values
                        cos_errs = cos_s['ci_half'].values
                    else:
                        cos_vals = None
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                    
                    # Plot directional derivative
                    ax1.plot(a_vals, m_vals, marker='o', linewidth=2, markersize=8, color='green')
                    ax1.fill_between(a_vals, m_vals - e_vals, m_vals + e_vals, alpha=0.3, color='lightgreen')
                    ax1.axhline(y=m_vals[0] if len(m_vals) > 0 else 0, linestyle=':', color='gray', alpha=0.5)
                    ax1.set_xlabel('α', fontsize=12)
                    ax1.set_ylabel('Directional Derivative', fontsize=12)
                    ax1.set_title(f'⟨A, A+αD⟩/||A||', fontsize=12)
                    ax1.grid(True, alpha=0.3)
                    
                    # Plot cos_S_task if available
                    if cos_vals is not None:
                        ax2.plot(a_vals, cos_vals, marker='^', linewidth=2, markersize=8, color='purple')
                        ax2.fill_between(a_vals, cos_vals - cos_errs, cos_vals + cos_errs, alpha=0.3, color='lavender')
                        ax2.axhline(y=1.0, linestyle=':', color='gray', alpha=0.5)
                        ax2.set_xlabel('α', fontsize=12)
                        ax2.set_ylabel('cos(S, A)', fontsize=12)
                        ax2.set_title(f'cos(A+αD, A)', fontsize=12)
                        ax2.grid(True, alpha=0.3)
                        ax2.set_ylim([-1, 1])
                    
                    fig.suptitle(f'Epoch {epoch}: Task-centric Directional Metrics @ {B}/{L}', fontsize=14)
                    plt.tight_layout()
                    
                    dir_path = f'./probes/{self.device_type}/plots/epoch_{epoch}_directional_{B}_{L}.png'
                    plt.savefig(dir_path, dpi=150)
                    plt.close()
                    print(f"  Saved directional metrics to {dir_path}")
    
    def on_train_epoch_start(self):
        """Run probes at start of training epoch."""
        if self.probe_enabled:
            self.run_gradient_probes(self.current_epoch + 1)

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

    def _create_epoch_csv_files(self, epoch: int, seed: int, has_fp32: bool):
        """Create CSV files for a specific epoch and seed."""
        # Include loss_coefficient in the directory name
        loss_coef_str = f"loss_coef_{self.loss_coefficient:.2f}".replace(".", "_").replace("-", "neg")
        probe_dir = f'./probes/{self.device_type}/{loss_coef_str}'
        os.makedirs(probe_dir, exist_ok=True)
        
        # Always create task-centric CSV
        csv_path_task = f'{probe_dir}/epoch_{epoch}_seed_{seed}_task_centric.csv'
        csv_file_task = open(csv_path_task, 'w', newline='')
        fieldnames_task = [
            'epoch', 'batch', 'seed', 'branch', 'locus',
            # Task-centric metrics only
            'cos_D_task', 'norm_gTask', 'norm_gD', 'r_norm_ratio_task', 'alpha_crit_task',
            'beta_task',  # NEW
            'dot_ad', 'first_order_positive'  # NEW: Descent Lemma metrics
        ]
        # Add descent gain columns for each alpha
        for alpha in self.probe_alpha_eval:
            fieldnames_task.append(f'descent_gain_task@{alpha:.2f}')
            fieldnames_task.append(f'descent_ratio@{alpha:.2f}')  # NEW
            fieldnames_task.append(f'cos_S_task@{alpha:.2f}')  # NEW
            fieldnames_task.append(f'dir_deriv_task@{alpha:.2f}')  # NEW
        writer_task = csv.DictWriter(csv_file_task, fieldnames=fieldnames_task, extrasaction='ignore')
        writer_task.writeheader()
        self.csv_files[f'epoch_{epoch}_seed_{seed}_task'] = csv_file_task
        self.csv_writers[f'epoch_{epoch}_seed_{seed}_task'] = writer_task
        print(f"  Created task-centric CSV: {csv_path_task}")
        
        # Also create FP32 alignment CSV if FP32 is available
        if has_fp32:
            csv_path_fp32 = f'{probe_dir}/epoch_{epoch}_seed_{seed}_fp32_alignment.csv'
            csv_file_fp32 = open(csv_path_fp32, 'w', newline='')
            fieldnames_fp32 = [
                'epoch', 'batch', 'seed', 'branch', 'locus',
                # Task-centric metrics (also available with FP32)
                'cos_D_task', 'norm_gTask', 'norm_gD', 'r_norm_ratio_task', 'alpha_crit_task',
                # FP32 alignment metrics
                'cos_A_FP', 'cos_D_FP', 'cos_D_A',
                'angle_A_FP_deg', 'angle_D_FP_deg',
                'cos_S_FP', 'angle_improve_deg',
                'norm_gA', 'norm_gFP32', 'r_norm_ratio', 'alpha_crit_fp32'
            ]
            # Add alpha sweep metrics
            for alpha in self.probe_alpha_eval:
                fieldnames_fp32.append(f'cos_S_FP@{alpha:.2f}')
            
            writer_fp32 = csv.DictWriter(csv_file_fp32, fieldnames=fieldnames_fp32)
            writer_fp32.writeheader()
            self.csv_files[f'epoch_{epoch}_seed_{seed}_fp32'] = csv_file_fp32
            self.csv_writers[f'epoch_{epoch}_seed_{seed}_fp32'] = writer_fp32
            print(f"  Created FP32 alignment CSV: {csv_path_fp32}")
    
    def _close_epoch_csv_files(self, epoch: int):
        """Close CSV files for a specific epoch."""
        keys_to_close = [k for k in self.csv_files.keys() if f'epoch_{epoch}' in k]
        for key in keys_to_close:
            if key in self.csv_files:
                self.csv_files[key].close()
                del self.csv_files[key]
                del self.csv_writers[key]
                print(f"  Closed CSV file for {key}")
    
    def _build_longitudinal_plots(self):
        """Build longitudinal trend plots across epochs."""
        import matplotlib.pyplot as plt
        
        loss_coef_str = f"loss_coef_{self.loss_coefficient:.2f}".replace(".", "_").replace("-", "neg")
        probe_dir = f'./probes/{self.device_type}/{loss_coef_str}'
        paths = sorted(glob.glob(f'{probe_dir}/epoch_*_summary.csv'))
        if not paths:
            print("No summary files found for longitudinal plots")
            return
        
        dfs = []
        for p in paths:
            try:
                df = pd.read_csv(p)
                df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
                dfs.append(df)
            except Exception as e:
                print(f"Failed to read {p}: {e}")
        
        if not dfs:
            return
        
        df = pd.concat(dfs, ignore_index=True)
        
        loci = ['L1_fea', 'L2_fea', 'L3_fea', 'FB']
        branches = ['D1', 'D2', 'D3', 'Dsum']
        
        # Trend: cos(A,FP32) by locus
        for L in loci:
            sub = df[(df['metric'] == 'cos_A_FP') & (df['locus'] == L) & (df['level'] == 'locus_only')]
            if len(sub) == 0:
                continue
            sub = sub.sort_values('epoch')
            
            plt.figure(figsize=(8, 6))
            plt.errorbar(sub['epoch'], sub['mean'], yerr=sub['ci_half'], marker='o', linestyle='-')
            plt.title(f'Trend: cos(A,FP32) @ {L}')
            plt.xlabel('Epoch')
            plt.ylabel('cos(A, FP32)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{probe_dir}/plots/trend_cosA_{L}.png', dpi=150)
            plt.close()
        
        # Trend: cos(D,task) by branch×locus
        for L in loci:
            for B in branches:
                sub = df[(df['metric'] == 'cos_D_task') & (df['locus'] == L) & (df['branch'] == B)]
                if len(sub) == 0:
                    continue
                sub = sub.sort_values('epoch')
                
                plt.figure(figsize=(8, 6))
                plt.errorbar(sub['epoch'], sub['mean'], yerr=sub['ci_half'], marker='o', linestyle='-')
                plt.title(f'Trend: cos(D,task) @ {B}/{L}')
                plt.xlabel('Epoch')
                plt.ylabel('cos(D, task)')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f'{probe_dir}/plots/trend_cosD_task_{B}_{L}.png', dpi=150)
                plt.close()
        
        # Trend: Best delta by branch×locus
        for L in loci:
            for B in branches:
                # Find the best alpha for each epoch
                best_deltas = []
                epochs = sorted(df['epoch'].unique())
                
                for ep in epochs:
                    delta_rows = df[(df['epoch'] == ep) & (df['branch'] == B) & 
                                   (df['locus'] == L) & (df['metric'] == 'delta')]
                    if len(delta_rows) == 0:
                        continue
                    
                    # Find best delta
                    best_row = delta_rows.loc[delta_rows['mean'].idxmax()]
                    best_deltas.append({
                        'epoch': ep,
                        'mean': best_row['mean'],
                        'ci_half': best_row['ci_half'],
                        'pos_ratio': best_row.get('pos_ratio', np.nan)
                    })
                
                if best_deltas:
                    bd_df = pd.DataFrame(best_deltas)
                    
                    fig, ax1 = plt.subplots(figsize=(8, 6))
                    
                    # Delta on primary axis
                    ax1.errorbar(bd_df['epoch'], bd_df['mean'], yerr=bd_df['ci_half'], 
                                marker='o', linestyle='-', color='tab:blue', label='Δ@best')
                    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Δ cosine (best α)', color='tab:blue')
                    ax1.tick_params(axis='y', labelcolor='tab:blue')
                    ax1.grid(True, alpha=0.3)
                    
                    # Positive ratio on secondary axis if available
                    if not bd_df['pos_ratio'].isna().all():
                        ax2 = ax1.twinx()
                        ax2.plot(bd_df['epoch'], bd_df['pos_ratio'] * 100, 
                                marker='s', linestyle='--', color='tab:orange', label='Pos. ratio')
                        ax2.set_ylabel('Positive ratio (%)', color='tab:orange')
                        ax2.tick_params(axis='y', labelcolor='tab:orange')
                    
                    plt.title(f'Trend: Δ@best @ {B}/{L}')
                    plt.tight_layout()
                    plt.savefig(f'{probe_dir}/plots/trend_delta_best_{B}_{L}.png', dpi=150)
                    plt.close()
        
        print("Created longitudinal trend plots")
    
    def on_fit_end(self):
        """Called when training ends."""
        if self.probe_enabled:
            print("\n=== Building Longitudinal Plots ===")
            self._build_longitudinal_plots()
    
    def __del__(self):
        """Cleanup all CSV files on deletion."""
        if hasattr(self, 'csv_files'):
            for csv_file in self.csv_files.values():
                if csv_file and not csv_file.closed:
                    csv_file.close()