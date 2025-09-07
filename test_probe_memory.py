#!/usr/bin/env python3
"""
Test script to verify gradient probe memory management is working correctly.
"""

import torch
import sys
sys.path.append('/root/SALMON')

from src.models.exp7_6_module import SalmonLitModule
from src.models.components.exp7_1_model import IntegratedResNet
from aihwkit.simulator.presets import IdealizedPreset

def test_probe_memory():
    """Test that gradient probes don't cause memory issues."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    integrated_resnet = IntegratedResNet(
        architecture="resnet18",
        num_classes=10,
        rpu_config=IdealizedPreset()
    )
    
    model = SalmonLitModule(
        model="resnet18",
        integrated_resnet=integrated_resnet,
        compile=False,
        optimizer={"lr": 0.01, "momentum": 0.9, "weight_decay": 0.0},
        dataset="cifar10",
        epoch=100,
        loss_coefficient=0.0,
        feature_loss_coefficient=0.0,
        dataset_path="data",
        autoaugment=False,
        temperature=3,
        batchsize=128,
        init_lr=0.1,
        N_CLASSES=10,
        block="BasicBlock",
        alpha=0.3,
        p_max=10000,
        opt_config="AnalogSGD",
        sd_config="true",
        FC_Digit="true",
        sch_config="off-schedule",
        scheduler={"T_max": 300, "eta_min": 0.0001},
        probe={
            "enabled": True,
            "epochs": [1],
            "batches_per_epoch": 2,  # Test with small number first
            "alpha_eval": [0.0, 0.5, 1.0],
            "seed": 1234
        }
    )
    
    model = model.to(device)
    model.setup(stage="fit")
    
    print("\n=== Testing gradient probe memory management ===")
    
    # Get initial memory usage
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(device) / 1024**2
        print(f"Initial GPU memory: {initial_memory:.2f} MB")
    
    # Run probes (simulating epoch 1)
    try:
        model.run_gradient_probes(epoch=1)
        print("✓ Gradient probes completed without memory errors")
        
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated(device) / 1024**2
            print(f"Final GPU memory: {final_memory:.2f} MB")
            print(f"Memory increase: {final_memory - initial_memory:.2f} MB")
            
    except RuntimeError as e:
        if "CUDA" in str(e) or "memory" in str(e).lower():
            print(f"✗ Memory error occurred: {e}")
            return False
        else:
            raise
    
    print("\n=== Memory test passed! ===")
    return True

if __name__ == "__main__":
    success = test_probe_memory()
    sys.exit(0 if success else 1)