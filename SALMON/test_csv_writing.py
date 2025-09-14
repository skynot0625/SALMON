import sys
import torch
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
    optimizer={'lr': 0.01, 'weight_decay': 0.0, 'momentum': 0.9},
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

# Test gradient extraction and CSV writing
inputs = torch.randn(32, 3, 32, 32)
labels = torch.randint(0, 10, (32,))

print("Testing CSV writing...")
print("CSV fieldnames:", model.csv_writer.fieldnames)
print()

# Get gradients
g_task = model._get_gTask(inputs, labels)
gD_split = model._get_gD_split(inputs, labels)

# Compute metrics
rows = model._compute_task_centric_metrics(g_task, gD_split, [0.0, 0.5, 1.0])
print(f"Generated {len(rows)} rows")

if rows:
    print("\nFirst row keys:", list(rows[0].keys()))
    print("\nSample row:")
    for k, v in rows[0].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    
    # Try to write the row
    try:
        rows[0]['epoch'] = 1
        rows[0]['batch'] = 0
        model.csv_writer.writerow(rows[0])
        print("\nSuccessfully wrote row to CSV!")
    except Exception as e:
        print(f"\nError writing row: {e}")
        print("\nMissing fields:", set(model.csv_writer.fieldnames) - set(rows[0].keys()))
        print("Extra fields:", set(rows[0].keys()) - set(model.csv_writer.fieldnames))
