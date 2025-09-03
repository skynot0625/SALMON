import torch
import torch.nn as nn
import sys
import os
from typing import Tuple, Dict
from pow_model import IntegratedResNet as PowModel
from pow_model1 import IntegratedResNet as PowModel1

def count_flops_conv2d(m: nn.Conv2d, x: torch.Tensor, y: torch.Tensor) -> int:
    """Conv2d layer의 FLOPs 계산"""
    x = x[0]  # Remove batch dimension
    out_h = y.size(2)
    out_w = y.size(3)
    
    kernel_ops = m.kernel_size[0] * m.kernel_size[1]
    bias_ops = 1 if m.bias is not None else 0
    
    # Depthwise convolution (groups == in_channels)
    if m.groups == m.in_channels:
        # Depthwise: kernel_ops * channels * H * W
        ops_per_element = kernel_ops * (2 if bias_ops else 1)
        total_ops = ops_per_element * m.in_channels * out_h * out_w
    # 1x1 convolution
    elif m.kernel_size[0] == 1:
        # 1x1 conv: in_channels * out_channels * H * W
        ops_per_element = m.in_channels * (2 if bias_ops else 1) / m.groups
        total_ops = ops_per_element * m.out_channels * out_h * out_w
    else:
        # Regular convolution
        ops_per_element = kernel_ops * m.in_channels * (2 if bias_ops else 1) / m.groups
        total_ops = ops_per_element * m.out_channels * out_h * out_w
    
    return int(total_ops)

def count_flops_linear(m: nn.Linear, x: torch.Tensor, y: torch.Tensor) -> int:
    """Linear layer의 FLOPs 계산"""
    total_ops = m.in_features * m.out_features * 2  # multiply-add
    if m.bias is not None:
        total_ops += m.out_features
    
    batch_size = x[0].size(0)
    total_ops *= batch_size
    
    return int(total_ops)

def count_flops_bn(m: nn.BatchNorm2d, x: torch.Tensor, y: torch.Tensor) -> int:
    """BatchNorm2d layer의 FLOPs 계산"""
    nelements = x[0].numel()
    total_ops = 2 * nelements  # multiply and add
    
    return int(total_ops)

def calculate_model_flops(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 32, 32), 
                         is_analog: bool = False) -> Dict[str, float]:
    """모델의 FLOPs 계산"""
    
    # Register hooks for each layer
    flops_dict = {}
    def hook_fn(name):
        def fn(module, x, y):
            if isinstance(module, nn.Conv2d):
                flops = count_flops_conv2d(module, x, y)
            elif isinstance(module, nn.Linear):
                flops = count_flops_linear(module, x, y)
            elif isinstance(module, nn.BatchNorm2d):
                flops = count_flops_bn(module, x, y)
            else:
                flops = 0
            flops_dict[name] = flops
        return fn
    
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Forward pass with dummy input
    device = next(model.parameters()).device
    x = torch.randn(input_size).to(device)
    model(x)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Calculate total FLOPs
    total_flops = 0
    features_flops = 0
    classifier_flops = 0
    
    print("\nLayer-wise FLOPs breakdown:")
    print("-" * 40)
    for name, flops in flops_dict.items():
        print(f"{name}: {flops/1e9:.4f} GFLOPs")
        # features 부분의 FLOPs 따로 계산
        if ('features' in name or 'input_module' in name) and 'attention' not in name:
            features_flops += flops
            print(f"Features: {name}")
        else:
            classifier_flops += flops
            print(f"Classifier: {name}")
    
    print(f"\nFeatures total: {features_flops/1e9:.4f} GFLOPs")
    print(f"Classifier total: {classifier_flops/1e9:.4f} GFLOPs")
    
    if is_analog:
        # features 부분만 1/100로 계산
        total_flops = (features_flops / 100) + classifier_flops
        print(f"Features after 1/100 reduction: {(features_flops/100)/1e9:.4f} GFLOPs")
    else:
        total_flops = features_flops + classifier_flops
    
    # Convert to GFLOPs
    training_gflops = total_flops * 3 / 1e9  # *3 for backward pass
    inference_gflops = total_flops / 1e9
    
    return {
        'training_gflops': training_gflops,
        'inference_gflops': inference_gflops
    }

def print_model_flops(model_name: str = "pow_model", architecture: str = "resnet10", 
                     input_size: Tuple[int, int, int, int] = (1, 3, 32, 32)):
    """모델의 FLOPs 출력"""
    # Create model
    if model_name == "pow_model":
        model = PowModel(architecture=architecture)
    else:
        model = PowModel1(architecture=architecture)
    
    # Calculate FLOPs
    digital_flops = calculate_model_flops(model, input_size, is_analog=False)
    analog_flops = calculate_model_flops(model, input_size, is_analog=True)
    
    print(f"\n{model_name.upper()} ({architecture.upper()}) FLOPs Analysis")
    print("-" * 50)
    print(f"Input size: {input_size}")
    print("\nDigital Model:")
    print(f"- Training GFLOPs: {digital_flops['training_gflops']:.2f}")
    print(f"- Inference GFLOPs: {digital_flops['inference_gflops']:.2f}")
    print("\nAnalog Model (1/100 FLOPS for features):")
    print(f"- Training GFLOPs: {analog_flops['training_gflops']:.2f}")
    print(f"- Inference GFLOPs: {analog_flops['inference_gflops']:.2f}")
    print("-" * 50)

if __name__ == "__main__":
    print("\nFLOPs Analysis for pow_model and pow_model1")
    print("=" * 50)
    
    architectures = ["resnet10", "resnet18", "resnet34", "resnet50"]
    model_types = ["pow_model", "pow_model1"]
    
    for model_type in model_types:
        print(f"\n{model_type.upper()} Analysis")
        print("-" * 30)
        for arch in architectures:
            print_model_flops(model_type, arch, (1, 3, 32, 32))
            
    print("\n주의: 이 계산에는 ReLU, MaxPool 등의 연산은 포함되지 않았습니다.")
    print("실제 연산량은 이보다 약간 더 높을 수 있습니다.")
