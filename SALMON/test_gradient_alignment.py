#!/usr/bin/env python3
"""Test gradient alignment between analog and FP32 models."""

import sys
import torch
sys.path.append('/root/SALMON')

from src.models.components.exp7_1_model import IntegratedResNet
from src.models.exp7_6_module_digital import SalmonLitModule
from aihwkit.simulator.presets import IdealizedPreset

def test_gradient_alignment():
    """Test the complete gradient alignment pipeline."""
    
    print("=" * 60)
    print("Testing Gradient Alignment Pipeline")
    print("=" * 60)
    
    # Create model with analog tiles
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
    
    # Setup the model
    print("\n1. Setting up model...")
    model.setup(stage='fit')
    
    # Check FP32 clone
    if hasattr(model, 'fp32_clone') and model.fp32_clone is not None:
        print("   ✓ FP32 clone created successfully")
    else:
        print("   ✗ FP32 clone not created")
        return
    
    # Test data
    inputs = torch.randn(32, 3, 32, 32)
    labels = torch.randint(0, 10, (32,))
    
    print("\n2. Testing gradient extraction...")
    
    # Get analog backbone gradients
    try:
        g_analog = model._get_gTask(inputs, labels)
        print(f"   ✓ Analog gradients: {len(g_analog)} locations")
        for locus, grad in g_analog.items():
            if grad is not None:
                print(f"      - {locus}: shape {grad.shape}, norm {grad.norm().item():.6f}")
    except Exception as e:
        print(f"   ✗ Error getting analog gradients: {e}")
        return
    
    # Get FP32 gradients
    try:
        g_fp32 = model._get_gFP32(inputs, labels)
        print(f"   ✓ FP32 gradients: {len(g_fp32)} locations")
        for locus, grad in g_fp32.items():
            if grad is not None:
                print(f"      - {locus}: shape {grad.shape}, norm {grad.norm().item():.6f}")
    except Exception as e:
        print(f"   ✗ Error getting FP32 gradients: {e}")
        return
    
    # Get attention gradients
    try:
        gD_split = model._get_gD_split(inputs, labels)
        print(f"   ✓ Attention gradients: {len(gD_split)} branches")
        for branch, grads in gD_split.items():
            if grads:
                print(f"      - {branch}: {len(grads)} locations")
    except Exception as e:
        print(f"   ✗ Error getting attention gradients: {e}")
        return
    
    print("\n3. Computing alignment metrics...")
    
    # Compute task-centric metrics
    try:
        rows = model._compute_task_centric_metrics(g_analog, gD_split, [0.0, 0.5, 1.0])
        print(f"   ✓ Generated {len(rows)} metric rows")
        
        if rows:
            # Show sample metrics
            sample = rows[0]
            print("\n   Sample metrics (alpha=0.0):")
            
            # Analog vs FP32 alignment
            if 'cos_A_FP' in sample:
                print(f"      - cos(analog, FP32): {sample['cos_A_FP']:.6f}")
                print(f"      - angle(analog, FP32): {sample['angle_A_FP_deg']:.2f}°")
            
            # Digital attention vs FP32 alignment
            if 'cos_D_FP' in sample:
                print(f"      - cos(attention, FP32): {sample['cos_D_FP']:.6f}")
                print(f"      - angle(attention, FP32): {sample['angle_D_FP_deg']:.2f}°")
            
            # Digital vs Analog alignment
            if 'cos_D_A' in sample:
                print(f"      - cos(attention, analog): {sample['cos_D_A']:.6f}")
                print(f"      - angle(attention, analog): {sample['angle_D_A_deg']:.2f}°")
            
            # Gradient norms
            print(f"      - norm(analog): {sample['norm_gA']:.6f}")
            print(f"      - norm(attention): {sample['norm_gD']:.6f}")
            print(f"      - norm(FP32): {sample['norm_gFP32']:.6f}")
            
    except Exception as e:
        print(f"   ✗ Error computing metrics: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n4. Testing CSV writing...")
    
    # Test CSV writing
    try:
        if rows:
            rows[0]['epoch'] = 1
            rows[0]['batch'] = 0
            model.csv_writer.writerow(rows[0])
            print("   ✓ Successfully wrote to CSV")
            print(f"   ✓ CSV file: {model.csv_file}")
    except Exception as e:
        print(f"   ✗ Error writing CSV: {e}")
        return
    
    print("\n" + "=" * 60)
    print("✓ All tests passed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    test_gradient_alignment()