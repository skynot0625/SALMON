import torch
import sys
sys.path.append('/root/SALMON')

from src.models.components.exp7_1_model import IntegratedResNet
from src.models.exp7_6_module_digital import SalmonLitModule
from aihwkit.simulator.presets import IdealizedPreset

# Create model with analog tiles
integrated_resnet = IntegratedResNet(
    architecture='resnet18',
    num_classes=10,
    rpu_config=IdealizedPreset()  # This creates analog tiles
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

# Setup the model (this should create FP32 clone)
model.setup(stage='fit')

# Check if FP32 clone was created
if hasattr(model, 'fp32_clone') and model.fp32_clone is not None:
    print("✓ FP32 clone created successfully")
    print(f"  - Has input: {'input' in model.fp32_clone}")
    print(f"  - Has features: {'features' in model.fp32_clone}")
    print(f"  - Has classifier: {'classifier' in model.fp32_clone}")
    
    # Test gradient extraction
    inputs = torch.randn(32, 3, 32, 32)
    labels = torch.randint(0, 10, (32,))
    
    print("\nTesting gradient extraction:")
    try:
        g_analog = model._get_gTask(inputs, labels)
        print(f"  ✓ Analog gradients: {len(g_analog)} locations")
        
        g_fp32 = model._get_gFP32(inputs, labels)
        print(f"  ✓ FP32 gradients: {len(g_fp32)} locations")
        
        if g_analog and g_fp32:
            # Compare a gradient
            locus = list(g_analog.keys())[0]
            if locus in g_fp32:
                cos_sim = model._safe_cosine(g_analog[locus], g_fp32[locus])
                print(f"  ✓ Cosine similarity at {locus}: {cos_sim:.4f}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("✗ FP32 clone not created")
