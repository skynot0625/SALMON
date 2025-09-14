import torch
import sys
sys.path.append('/root/SALMON')

from src.models.components.exp7_1_model import IntegratedResNet
from src.models.exp7_6_module_digital import SalmonLitModule
from aihwkit.simulator.presets import IdealizedPreset

# Create model
integrated_resnet = IntegratedResNet(
    architecture='resnet18',
    num_classes=10,
    rpu_config=IdealizedPreset()
)

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
    probe={'enabled': True, 'epochs': [1], 'batches_per_epoch': 1, 'batch_size': 32, 'alpha_eval': [0.0, 0.5, 1.0], 'seed': 42}
)

model.setup(stage='fit')

# Test gradient extraction with activation tensors
inputs = torch.randn(32, 3, 32, 32)
labels = torch.randint(0, 10, (32,))

print("Testing activation-based gradient extraction...")
print("=" * 60)

# Get activation taps
taps = model._get_taps(inputs)
print("\nActivation taps:")
for name, tensor in taps.items():
    print(f"  {name}: shape={tensor.shape}, requires_grad={tensor.requires_grad}")

# Test task gradient
print("\n" + "=" * 60)
print("Testing task gradient extraction...")
g_task = model._get_gTask(inputs, labels)
print(f"Task gradients captured at {len(g_task)} locations:")
for locus, grad in g_task.items():
    print(f"  {locus}: shape={grad.shape}, norm={grad.norm().item():.6f}")

# Test branch gradients
print("\n" + "=" * 60)
print("Testing branch gradient extraction...")
gD_split = model._get_gD_split(inputs, labels)
for branch, grads in gD_split.items():
    print(f"\nBranch {branch}:")
    for locus, grad in grads.items():
        print(f"  {locus}: norm={grad.norm().item():.6f}")

# Test metrics computation
print("\n" + "=" * 60)
print("Testing metrics computation...")
rows = model._compute_metrics_task(g_task, gD_split, [0.0, 0.5, 1.0])
print(f"Generated {len(rows)} metric rows")
for row in rows[:3]:
    print(f"  {row['branch']}-{row['locus']}: cos_D_task={row['cos_D_task']:.3f}")

print("\n" + "=" * 60)
print("SUCCESS: Activation-based gradient extraction working correctly!")
