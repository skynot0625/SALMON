import torch
import sys
sys.path.append('/root/SALMON')

from src.models.components.exp7_1_model import IntegratedResNet
from src.models.exp7_6_module_digital import SalmonLitModule
from aihwkit.simulator.presets import IdealizedPreset

# Create a test model
integrated_resnet = IntegratedResNet(
    architecture='resnet18',
    num_classes=10,
    rpu_config=IdealizedPreset()
)

# Create the module
model = SalmonLitModule(
    model='resnet18',
    integrated_resnet=integrated_resnet,
    compile=False,
    optimizer={'lr': 0.01, 'weight_decay': 0.0, 'momentum': 0.9, 'dampening': 0, 'nesterov': False},
    dataset='cifar10',
    epoch=300,
    loss_coefficient=0.0,
    feature_loss_coefficient=0.0,
    dataset_path='data',
    autoaugment=False,
    temperature=3,
    batchsize=128,
    init_lr=0.1,
    N_CLASSES=10,
    block='BasicBlock',
    alpha=0.3,
    p_max=10000,
    opt_config='AnalogSGD',
    sd_config='true',
    FC_Digit='true',
    sch_config='off-schedule',
    scheduler={'T_max': 300, 'eta_min': 0.0001},
    probe={
        'enabled': True,
        'epochs': [1],
        'batches_per_epoch': 1,
        'batch_size': 32,
        'alpha_eval': [0.0, 0.5, 1.0],
        'seed': 42
    }
)

# Setup the model
model.setup(stage='fit')

# Create dummy input
inputs = torch.randn(32, 3, 32, 32)
labels = torch.randint(0, 10, (32,))

print("Testing gradient extraction from analog modules...")
print("=" * 60)

# Check layer handles
print("\nLayer handles found:")
for name, layer in model.layer_handles.items():
    print(f"  {name}: {layer.__class__.__name__}")
    if hasattr(layer, 'analog_module'):
        print(f"    - Has analog_module: True")
        if hasattr(layer, 'get_weights'):
            print(f"    - Has get_weights method: True")
            weight, bias = layer.get_weights()
            print(f"    - Weight shape: {weight.shape}")
    else:
        print(f"    - Standard PyTorch layer")

print("\n" + "=" * 60)
print("Testing gradient extraction...")

# Test gradient extraction
model.zero_grad(set_to_none=True)

# Forward pass
input_features = model.input(inputs)
feature_backbone, x1, x2, x3 = model.features(input_features)
logits = model.classifier(feature_backbone)
loss = model.criterion(logits, labels)

# Backward pass
loss.backward()

# Check gradients
print("\nGradient extraction results:")
gradients = model.flatten_grads(model.layer_handles)

for locus, grad in gradients.items():
    print(f"  {locus}:")
    print(f"    - Gradient shape: {grad.shape}")
    print(f"    - Gradient norm: {grad.norm().item():.6f}")
    print(f"    - Non-zero elements: {(grad != 0).sum().item()}/{grad.numel()}")
    print(f"    - Min/Max values: {grad.min().item():.6f} / {grad.max().item():.6f}")

print("\n" + "=" * 60)
print("Testing branch gradient extraction...")

# Test branch gradients
gD_split = model._get_gD_split(inputs, labels)

for branch, branch_grads in gD_split.items():
    print(f"\nBranch {branch}:")
    for locus, grad in branch_grads.items():
        print(f"  {locus}:")
        print(f"    - Gradient norm: {grad.norm().item():.6f}")
        print(f"    - Non-zero ratio: {(grad != 0).sum().item() / grad.numel():.2%}")

print("\n" + "=" * 60)
print("Test completed successfully!")
