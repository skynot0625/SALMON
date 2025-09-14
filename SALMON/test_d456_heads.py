#!/usr/bin/env python3
"""
Focused test for D4, D5, D6 attention heads only.
Tests their individual operation and integration.
"""

import torch
import torch.nn as nn
from aihwkit.simulator.configs import SingleRPUConfig, IdealDevice
from aihwkit.simulator.presets import IdealizedPreset

# Add path for imports
import sys
sys.path.append('/root/SALMON')

from src.models.components.exp7_6_digital_model import IntegratedResNet, ResNetFeatures
from src.models.exp7_6_digital_module import SalmonLitModule


def test_d456_taps():
    """Test that D4, D5, D6 get correct intermediate taps from ResNet."""
    print("=" * 70)
    print("Testing D4, D5, D6 Intermediate Tap Points")
    print("=" * 70)
    
    # Create ResNetFeatures to test tap points
    from src.models.components.exp7_6_digital_model import BasicBlock
    features = ResNetFeatures(
        block=BasicBlock,
        layers=[2, 2, 2, 2],  # ResNet18 configuration
    )
    
    # Test input
    x = torch.randn(2, 3, 32, 32)
    
    print("\n1. Testing forward_with_taps for intermediate outputs...")
    out4_feature, x1, x2, x3, x2_mid, x3_mid, x4_mid = features.forward_with_taps(x)
    
    # Check shapes for each tap point
    print("\n2. Verifying tap point dimensions:")
    print(f"   x1 (for D1):     {x1.shape} - After layer1")
    print(f"   x2_mid (for D4): {x2_mid.shape} - After layer2[0]")
    print(f"   x2 (for D2):     {x2.shape} - After layer2 complete")
    print(f"   x3_mid (for D5): {x3_mid.shape} - After layer3[0]")
    print(f"   x3 (for D3):     {x3.shape} - After layer3 complete")
    print(f"   x4_mid (for D6): {x4_mid.shape} - After layer4[0]")
    
    # Verify channel dimensions
    assert x1.shape[1] == 64, f"x1 should have 64 channels, got {x1.shape[1]}"
    assert x2_mid.shape[1] == 128, f"x2_mid should have 128 channels, got {x2_mid.shape[1]}"
    assert x2.shape[1] == 128, f"x2 should have 128 channels, got {x2.shape[1]}"
    assert x3_mid.shape[1] == 256, f"x3_mid should have 256 channels, got {x3_mid.shape[1]}"
    assert x3.shape[1] == 256, f"x3 should have 256 channels, got {x3.shape[1]}"
    assert x4_mid.shape[1] == 512, f"x4_mid should have 512 channels, got {x4_mid.shape[1]}"
    
    print("\n✅ All tap points have correct dimensions!")
    return features


def test_d456_attention_modules():
    """Test D4, D5, D6 attention modules specifically."""
    print("\n" + "=" * 70)
    print("Testing D4, D5, D6 Attention Modules")
    print("=" * 70)
    
    # Create model
    rpu_config = SingleRPUConfig(device=IdealDevice())
    model = IntegratedResNet(
        architecture="resnet18",
        num_classes=10,
        rpu_config=rpu_config
    )
    
    print("\n1. Checking attention module types:")
    print(f"   attention4 (D4): {type(model.attention4).__name__}")
    print(f"   attention5 (D5): {type(model.attention5).__name__}")
    print(f"   attention6 (D6): {type(model.attention6).__name__}")
    
    # Verify D4 uses ResNetAttention2 (for 128-channel input)
    assert type(model.attention4).__name__ == "ResNetAttention2", "D4 should use ResNetAttention2"
    # Verify D5 uses ResNetAttention3 (for 256-channel input)
    assert type(model.attention5).__name__ == "ResNetAttention3", "D5 should use ResNetAttention3"
    # Verify D6 uses ResNetAttention4 (for 512-channel input)
    assert type(model.attention6).__name__ == "ResNetAttention4", "D6 should use ResNetAttention4"
    
    print("\n2. Testing individual attention modules with correct input sizes:")
    
    # Test D4 (expects 128-channel input)
    x_128 = torch.randn(2, 128, 16, 16)  # Simulated x2_mid
    d4_out, d4_feat = model.attention4(x_128)
    print(f"   D4: input {x_128.shape} -> output {d4_out.shape}, features {d4_feat.shape}")
    assert d4_out.shape == (2, 10), f"D4 output shape wrong: {d4_out.shape}"
    assert d4_feat.shape == (2, 512), f"D4 feature shape wrong: {d4_feat.shape}"
    
    # Test D5 (expects 256-channel input)
    x_256 = torch.randn(2, 256, 8, 8)  # Simulated x3_mid
    d5_out, d5_feat = model.attention5(x_256)
    print(f"   D5: input {x_256.shape} -> output {d5_out.shape}, features {d5_feat.shape}")
    assert d5_out.shape == (2, 10), f"D5 output shape wrong: {d5_out.shape}"
    assert d5_feat.shape == (2, 512), f"D5 feature shape wrong: {d5_feat.shape}"
    
    # Test D6 (expects 512-channel input)
    x_512 = torch.randn(2, 512, 4, 4)  # Simulated x4_mid
    d6_out, d6_feat = model.attention6(x_512)
    print(f"   D6: input {x_512.shape} -> output {d6_out.shape}, features {d6_feat.shape}")
    assert d6_out.shape == (2, 10), f"D6 output shape wrong: {d6_out.shape}"
    assert d6_feat.shape == (2, 512), f"D6 feature shape wrong: {d6_feat.shape}"
    
    print("\n✅ D4, D5, D6 attention modules work correctly!")
    return model


def test_d456_only_configuration():
    """Test configuration with ONLY D4, D5, D6 (no D1, D2, D3)."""
    print("\n" + "=" * 70)
    print("Testing Configuration with ONLY D4, D5, D6")
    print("=" * 70)
    
    # Create model and module with only D4, D5, D6
    rpu_config = SingleRPUConfig(device=IdealDevice())
    integrated_resnet = IntegratedResNet(
        architecture="resnet18",
        num_classes=10,
        rpu_config=rpu_config
    )
    
    module = SalmonLitModule(
        model='resnet18',
        integrated_resnet=integrated_resnet,
        compile=False,
        optimizer={'lr': 0.01, 'weight_decay': 0.0001},
        dataset='cifar10',
        epoch=300,
        loss_coefficient=0.0,
        feature_loss_coefficient=0.0,
        dataset_path='data',
        autoaugment=False,
        temperature=3,
        batchsize=128,
        init_lr=0.01,
        N_CLASSES=10,
        block='BasicBlock',
        alpha=0.3,
        p_max=10000,
        opt_config='AnalogSGD',
        sd_config='true',
        FC_Digit='true',
        sch_config='cosine',
        scheduler={'T_max': 300, 'eta_min': 0.0001},
        use_all_heads=True,
        active_heads=['D4', 'D5', 'D6']  # Only D4, D5, D6
    )
    
    print("\n1. Module configuration:")
    print(f"   Active heads: {module.active_heads}")
    print(f"   Head names: {module.head_names}")
    print(f"   Number of heads: {module.num_heads}")
    
    # Verify configuration
    assert module.active_heads == ['D4', 'D5', 'D6']
    assert module.head_names == ['backbone', 'D4', 'D5', 'D6']
    assert module.num_heads == 4
    
    print("\n2. Testing forward pass with D4, D5, D6 only...")
    batch = (torch.randn(2, 3, 32, 32), torch.randint(0, 10, (2,)))
    outputs, features, labels = module.model_step_all_heads(batch)
    
    print(f"   Number of outputs: {len(outputs)}")
    print(f"   Number of features: {len(features)}")
    
    assert len(outputs) == 4, f"Expected 4 outputs (backbone + D4,D5,D6), got {len(outputs)}"
    assert len(features) == 4, f"Expected 4 features, got {len(features)}"
    
    print("\n3. Testing loss calculation with only D4, D5, D6...")
    loss = module.training_step(batch, 0)
    print(f"   Loss computed: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    
    print("\n✅ D4, D5, D6 only configuration works correctly!")
    return module


def test_d456_in_full_forward():
    """Test D4, D5, D6 outputs in full forward pass."""
    print("\n" + "=" * 70)
    print("Testing D4, D5, D6 in Full Forward Pass")
    print("=" * 70)
    
    # Create model
    rpu_config = SingleRPUConfig(device=IdealDevice())
    model = IntegratedResNet(
        architecture="resnet18",
        num_classes=10,
        rpu_config=rpu_config
    )
    
    # Test input
    x = torch.randn(2, 3, 32, 32)
    
    print("\n1. Running forward_all_heads...")
    outputs = model.forward_all_heads(x)
    
    # Extract D4, D5, D6 outputs
    D4_out, D4_feat = outputs[4], outputs[5]
    D5_out, D5_feat = outputs[8], outputs[9]
    D6_out, D6_feat = outputs[12], outputs[13]
    
    print("\n2. D4, D5, D6 outputs from full forward:")
    print(f"   D4: logits {D4_out.shape}, features {D4_feat.shape}")
    print(f"   D5: logits {D5_out.shape}, features {D5_feat.shape}")
    print(f"   D6: logits {D6_out.shape}, features {D6_feat.shape}")
    
    # Verify outputs are different (not identical)
    print("\n3. Verifying D4, D5, D6 produce different outputs:")
    d4_d5_diff = torch.abs(D4_out - D5_out).mean().item()
    d5_d6_diff = torch.abs(D5_out - D6_out).mean().item()
    d4_d6_diff = torch.abs(D4_out - D6_out).mean().item()
    
    print(f"   Mean absolute difference D4-D5: {d4_d5_diff:.4f}")
    print(f"   Mean absolute difference D5-D6: {d5_d6_diff:.4f}")
    print(f"   Mean absolute difference D4-D6: {d4_d6_diff:.4f}")
    
    assert d4_d5_diff > 0.01, "D4 and D5 outputs should be different"
    assert d5_d6_diff > 0.01, "D5 and D6 outputs should be different"
    assert d4_d6_diff > 0.01, "D4 and D6 outputs should be different"
    
    print("\n✅ D4, D5, D6 produce distinct outputs as expected!")
    
    # Compare with D1, D2, D3
    D1_out = outputs[2]
    D2_out = outputs[6]
    D3_out = outputs[10]
    
    print("\n4. Comparing D4,D5,D6 with D1,D2,D3:")
    print(f"   D1 vs D4 difference: {torch.abs(D1_out - D4_out).mean().item():.4f}")
    print(f"   D2 vs D5 difference: {torch.abs(D2_out - D5_out).mean().item():.4f}")
    print(f"   D3 vs D6 difference: {torch.abs(D3_out - D6_out).mean().item():.4f}")
    
    return model


def test_d456_gradients():
    """Test that D4, D5, D6 produce gradients during backprop."""
    print("\n" + "=" * 70)
    print("Testing D4, D5, D6 Gradient Flow")
    print("=" * 70)
    
    # Create model
    rpu_config = SingleRPUConfig(device=IdealDevice())
    model = IntegratedResNet(
        architecture="resnet18",
        num_classes=10,
        rpu_config=rpu_config
    )
    
    # Test input and target
    target = torch.randint(0, 10, (2,))
    criterion = nn.CrossEntropyLoss()
    
    # Test gradient flow for D4
    print("\n1. Testing D4 gradient flow:")
    model.zero_grad()
    x = torch.randn(2, 3, 32, 32)
    outputs = model.forward_all_heads(x)
    D4_out = outputs[4]
    loss_d4 = criterion(D4_out, target)
    loss_d4.backward()
    
    # Check if attention4 has gradients
    has_grad_d4 = any(p.grad is not None and p.grad.abs().sum() > 0 
                       for p in model.attention4.parameters())
    print(f"   D4 (attention4) has gradients: {has_grad_d4}")
    assert has_grad_d4, "D4 should have gradients"
    
    # Test gradient flow for D5
    print("\n2. Testing D5 gradient flow:")
    model.zero_grad()
    x = torch.randn(2, 3, 32, 32)
    outputs = model.forward_all_heads(x)
    D5_out = outputs[8]
    loss_d5 = criterion(D5_out, target)
    loss_d5.backward()
    
    has_grad_d5 = any(p.grad is not None and p.grad.abs().sum() > 0 
                       for p in model.attention5.parameters())
    print(f"   D5 (attention5) has gradients: {has_grad_d5}")
    assert has_grad_d5, "D5 should have gradients"
    
    # Test gradient flow for D6
    print("\n3. Testing D6 gradient flow:")
    model.zero_grad()
    x = torch.randn(2, 3, 32, 32)
    outputs = model.forward_all_heads(x)
    D6_out = outputs[12]
    loss_d6 = criterion(D6_out, target)
    loss_d6.backward()
    
    has_grad_d6 = any(p.grad is not None and p.grad.abs().sum() > 0 
                       for p in model.attention6.parameters())
    print(f"   D6 (attention6) has gradients: {has_grad_d6}")
    assert has_grad_d6, "D6 should have gradients"
    
    print("\n✅ D4, D5, D6 all have proper gradient flow!")


def main():
    """Run all D4, D5, D6 focused tests."""
    print("\n" + "=" * 70)
    print("FOCUSED TESTING FOR D4, D5, D6 ATTENTION HEADS")
    print("=" * 70)
    
    try:
        # Test 1: Intermediate tap points
        test_d456_taps()
        
        # Test 2: Attention module types and functionality
        test_d456_attention_modules()
        
        # Test 3: Configuration with only D4, D5, D6
        test_d456_only_configuration()
        
        # Test 4: D4, D5, D6 in full forward pass
        test_d456_in_full_forward()
        
        # Test 5: Gradient flow
        test_d456_gradients()
        
        print("\n" + "=" * 70)
        print("✅ ALL D4, D5, D6 TESTS PASSED SUCCESSFULLY!")
        print("=" * 70)
        
        print("\nVerified:")
        print("✓ D4 correctly uses layer2[0] intermediate tap (128 channels)")
        print("✓ D5 correctly uses layer3[0] intermediate tap (256 channels)")
        print("✓ D6 correctly uses layer4[0] intermediate tap (512 channels)")
        print("✓ Each head produces distinct outputs")
        print("✓ Configurations with only D4,D5,D6 work correctly")
        print("✓ Gradient flow works for all three heads")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())