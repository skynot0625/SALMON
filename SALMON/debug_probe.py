import torch
from aihwkit.nn import AnalogConv2d, AnalogLinear
from aihwkit.simulator.presets import IdealizedPreset

# Create a sample analog conv layer
rpu_config = IdealizedPreset()
layer = AnalogConv2d(256, 256, kernel_size=3, stride=1, padding=1, rpu_config=rpu_config).cuda()

# Forward pass
x = torch.randn(32, 256, 8, 8).cuda()
x.requires_grad = True
out = layer(x)

# Backward pass
loss = out.sum()
loss.backward()

# Check analog context
ctx = layer.analog_module.analog_ctx
if ctx.analog_grad_output:
    grad_out = ctx.analog_grad_output[-1]
    print(f"Grad output shape: {grad_out.shape}")
    print(f"Flattened size: {grad_out.numel()}")
    
if ctx.analog_input:
    inp = ctx.analog_input[-1]
    print(f"Input shape: {inp.shape}")

# Check weight shape
weights = layer.get_weights()
if weights:
    w = weights[0]
    print(f"Weight shape: {w.shape}")
    print(f"Weight flattened size: {w.numel()}")
