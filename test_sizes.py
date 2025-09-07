import sys
sys.path.append('/root/SALMON')

from src.models.exp7_6_module import SalmonLitModule
from omegaconf import OmegaConf
import torch

# Create minimal config
config = OmegaConf.create({
    'integrated_resnet': {
        '_target_': 'src.models.components.exp7_1_model.IntegratedResNet',
        'architecture': 'resnet18',
        'num_classes': 10,
        'rpu_config': {'_target_': 'aihwkit.simulator.presets.IdealizedPreset'}
    },
    'probe': {'enabled': False},
    'optimizer': {'lr': 0.01, 'momentum': 0.9},
    'scheduler': {'T_max': 300, 'eta_min': 0.0001}
})

# Create model
model = SalmonLitModule(config)
model = model.cuda()

# Setup layer handles
model._setup_layer_handles()

# Print layer info
for locus, layer in model.layer_handles.items():
    print(f"\n{locus}: {type(layer).__name__}")
    if hasattr(layer, 'weight'):
        print(f"  Weight shape: {layer.weight.shape}")
        print(f"  Weight size: {layer.weight.numel()}")
    
    # For analog layers
    if hasattr(layer, 'analog_module'):
        analog_module = layer.analog_module
        if hasattr(analog_module, 'tile'):
            weights = analog_module.tile.get_weights()
            if weights:
                w = weights[0]
                print(f"  Analog weight shape: {w.shape}")
                print(f"  Analog weight size: {w.numel()}")

# Test forward pass
x = torch.randn(32, 3, 32, 32).cuda()
x.requires_grad = True

# Forward
out = model(x)
loss = out.sum()
loss.backward()

# Check gradient sizes
print("\n\nGradient sizes after backward:")
for locus, layer in model.layer_handles.items():
    print(f"\n{locus}:")
    
    # Check analog context
    if hasattr(layer, 'analog_module'):
        analog_module = layer.analog_module
        if hasattr(analog_module, 'analog_ctx'):
            ctx = analog_module.analog_ctx
            if hasattr(ctx, 'analog_grad_output') and ctx.analog_grad_output:
                grad_out = ctx.analog_grad_output[-1]
                print(f"  Analog grad output shape: {grad_out.shape}")
                print(f"  Analog grad output size: {grad_out.numel()}")
    
    # Check standard gradients
    if hasattr(layer, 'weight') and layer.weight.grad is not None:
        print(f"  Weight grad shape: {layer.weight.grad.shape}")
        print(f"  Weight grad size: {layer.weight.grad.numel()}")
