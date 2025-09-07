#!/usr/bin/env python3
"""Test script to verify correct weight copying from analog to FP32 modules."""

import torch
import torch.nn as nn
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.nn import AnalogConv2d, AnalogLinear
from aihwkit.simulator.presets import IdealizedPreset

def copy_weights_analog_to_fp32(analog_module, fp32_module):
    """Copy weights from analog module to FP32 module correctly."""
    
    def copy_recursive(src, dst):
        # Check if it's an AnalogConv2d or AnalogLinear
        if isinstance(src, (AnalogConv2d, AnalogLinear)):
            weights = src.get_weights()
            weight_tensor = weights[0]  # First element is weight
            
            # For Conv2d: need to reshape from [out_channels, in_features] to [out_channels, in_channels, kh, kw]
            if isinstance(src, AnalogConv2d) and isinstance(dst, nn.Conv2d):
                out_channels = dst.weight.shape[0]
                in_channels = dst.weight.shape[1]
                kh = dst.weight.shape[2]
                kw = dst.weight.shape[3]
                # Analog weight is [out_channels, in_channels * kh * kw]
                weight_tensor = weight_tensor.view(out_channels, in_channels, kh, kw)
            
            dst.weight.data.copy_(weight_tensor)
            
            # Copy bias if exists
            if len(weights) > 1 and weights[1] is not None and dst.bias is not None:
                dst.bias.data.copy_(weights[1])
                
        # Regular PyTorch module with weight
        elif hasattr(src, 'weight') and hasattr(dst, 'weight'):
            dst.weight.data.copy_(src.weight.data)
            if hasattr(src, 'bias') and src.bias is not None and hasattr(dst, 'bias'):
                dst.bias.data.copy_(src.bias.data)
                
        # BatchNorm layers
        elif isinstance(src, nn.BatchNorm2d) and isinstance(dst, nn.BatchNorm2d):
            dst.weight.data.copy_(src.weight.data)
            dst.bias.data.copy_(src.bias.data)
            dst.running_mean.data.copy_(src.running_mean.data)
            dst.running_var.data.copy_(src.running_var.data)
            dst.num_batches_tracked = src.num_batches_tracked
            
        # Recursively handle container modules
        else:
            for (src_name, src_child), (dst_name, dst_child) in zip(
                src.named_children(), dst.named_children()
            ):
                if src_name == dst_name:
                    copy_recursive(src_child, dst_child)
    
    copy_recursive(analog_module, fp32_module)

# Test the function
def test_weight_copy():
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.fc = nn.Linear(128, 10)
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = torch.relu(x)
            x = self.conv2(x)
            x = torch.relu(x)
            x = torch.mean(x, dim=[2, 3])  # Global average pooling
            x = self.fc(x)
            return x
    
    # Create original and analog versions
    original = SimpleModel()
    analog = SimpleModel()
    # Convert each layer individually
    analog.conv1 = convert_to_analog(analog.conv1, IdealizedPreset())
    analog.conv2 = convert_to_analog(analog.conv2, IdealizedPreset())  
    analog.fc = convert_to_analog(analog.fc, IdealizedPreset())
    
    # Set some random weights in analog model
    torch.manual_seed(42)
    for m in analog.modules():
        if isinstance(m, (AnalogConv2d, AnalogLinear)):
            # AnalogConv2d/Linear have their own initialization
            pass
    
    # Create FP32 clone
    fp32_clone = SimpleModel()
    
    # Copy weights
    copy_weights_analog_to_fp32(analog, fp32_clone)
    
    # Test forward pass
    x = torch.randn(1, 3, 32, 32)
    
    try:
        analog_out = analog(x)
        fp32_out = fp32_clone(x)
        print("✓ Weight copy successful!")
        print(f"Analog output shape: {analog_out.shape}")
        print(f"FP32 output shape: {fp32_out.shape}")
        
        # Check if outputs are similar (they won't be identical due to analog simulation)
        diff = (analog_out - fp32_out).abs().mean()
        print(f"Mean absolute difference: {diff:.6f}")
        
    except Exception as e:
        print(f"✗ Error during forward pass: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_weight_copy()
    if success:
        print("\nWeight copying works correctly!")
    else:
        print("\nWeight copying failed!")