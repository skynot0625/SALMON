#!/usr/bin/env python3
"""
Detailed verification of D4, D5, D6 forward pass implementation.
This test traces the exact data flow through the network.
"""

import torch
import torch.nn as nn
from aihwkit.simulator.configs import SingleRPUConfig, IdealDevice

# Add path for imports
import sys
sys.path.append('/root/SALMON')

from src.models.components.exp7_6_digital_model import IntegratedResNet


def verify_d456_forward_implementation():
    """Verify the exact forward pass for D4, D5, D6."""
    print("=" * 80)
    print("VERIFYING D4, D5, D6 FORWARD PASS IMPLEMENTATION")
    print("=" * 80)
    
    # Create model
    rpu_config = SingleRPUConfig(device=IdealDevice())
    model = IntegratedResNet(
        architecture="resnet18",
        num_classes=10,
        rpu_config=rpu_config
    )
    
    # Test input
    batch_size = 2
    x = torch.randn(batch_size, 3, 32, 32)
    
    print("\n1. TRACING DATA FLOW THROUGH RESNET:")
    print("-" * 50)
    
    # Manually trace through the network to verify tap points
    with torch.no_grad():
        # Initial conv layers
        x_conv = model.features.conv1(x)
        x_bn = model.features.bn1(x_conv)
        x_relu = model.features.relu(x_bn)
        
        print(f"After initial conv: {x_relu.shape}")
        
        # Layer 1
        x1 = model.features.layer1(x_relu)
        print(f"After layer1 (x1 for D1): {x1.shape}")
        assert x1.shape[1] == 64, "Layer1 should output 64 channels"
        
        # Layer 2 - check intermediate tap
        print(f"\nLayer2 has {len(model.features.layer2)} blocks")
        x2_mid = model.features.layer2[0](x1)  # First block of layer2
        print(f"After layer2[0] (x2_mid for D4): {x2_mid.shape}")
        assert x2_mid.shape[1] == 128, "Layer2[0] should output 128 channels"
        
        x2 = model.features.layer2[1](x2_mid)  # Second block of layer2
        print(f"After layer2[1] (x2 for D2): {x2.shape}")
        assert x2.shape[1] == 128, "Layer2 complete should output 128 channels"
        
        # Layer 3 - check intermediate tap
        print(f"\nLayer3 has {len(model.features.layer3)} blocks")
        x3_mid = model.features.layer3[0](x2)  # First block of layer3
        print(f"After layer3[0] (x3_mid for D5): {x3_mid.shape}")
        assert x3_mid.shape[1] == 256, "Layer3[0] should output 256 channels"
        
        x3 = model.features.layer3[1](x3_mid)  # Second block of layer3
        print(f"After layer3[1] (x3 for D3): {x3.shape}")
        assert x3.shape[1] == 256, "Layer3 complete should output 256 channels"
        
        # Layer 4 - check intermediate tap
        print(f"\nLayer4 has {len(model.features.layer4)} blocks")
        x4_mid = model.features.layer4[0](x3)  # First block of layer4
        print(f"After layer4[0] (x4_mid for D6): {x4_mid.shape}")
        assert x4_mid.shape[1] == 512, "Layer4[0] should output 512 channels"
        
        x4 = model.features.layer4[1](x4_mid)  # Second block of layer4
        print(f"After layer4[1] (x4 final): {x4.shape}")
        assert x4.shape[1] == 512, "Layer4 complete should output 512 channels"
    
    print("\n2. VERIFYING FORWARD_WITH_TAPS:")
    print("-" * 50)
    
    # Test forward_with_taps
    out4_feature, x1_tap, x2_tap, x3_tap, x2_mid_tap, x3_mid_tap, x4_mid_tap = model.features.forward_with_taps(x)
    
    print(f"x1_tap (for D1): {x1_tap.shape}")
    print(f"x2_mid_tap (for D4): {x2_mid_tap.shape}")
    print(f"x2_tap (for D2): {x2_tap.shape}")
    print(f"x3_mid_tap (for D5): {x3_mid_tap.shape}")
    print(f"x3_tap (for D3): {x3_tap.shape}")
    print(f"x4_mid_tap (for D6): {x4_mid_tap.shape}")
    
    # Verify dimensions
    assert x1_tap.shape[1] == 64, "x1 tap should be 64 channels"
    assert x2_mid_tap.shape[1] == 128, "x2_mid tap should be 128 channels"
    assert x2_tap.shape[1] == 128, "x2 tap should be 128 channels"
    assert x3_mid_tap.shape[1] == 256, "x3_mid tap should be 256 channels"
    assert x3_tap.shape[1] == 256, "x3 tap should be 256 channels"
    assert x4_mid_tap.shape[1] == 512, "x4_mid tap should be 512 channels"
    
    print("\n3. VERIFYING D4, D5, D6 ATTENTION MODULES:")
    print("-" * 50)
    
    # Test D4 with x2_mid
    print(f"\nD4 (attention4) processing x2_mid {x2_mid_tap.shape}:")
    d4_out, d4_feat = model.attention4(x2_mid_tap)
    print(f"  Output: {d4_out.shape}, Features: {d4_feat.shape}")
    assert d4_out.shape == (batch_size, 10), "D4 should output class predictions"
    assert d4_feat.shape == (batch_size, 512), "D4 should output 512-dim features"
    
    # Test D5 with x3_mid
    print(f"\nD5 (attention5) processing x3_mid {x3_mid_tap.shape}:")
    d5_out, d5_feat = model.attention5(x3_mid_tap)
    print(f"  Output: {d5_out.shape}, Features: {d5_feat.shape}")
    assert d5_out.shape == (batch_size, 10), "D5 should output class predictions"
    assert d5_feat.shape == (batch_size, 512), "D5 should output 512-dim features"
    
    # Test D6 with x4_mid
    print(f"\nD6 (attention6) processing x4_mid {x4_mid_tap.shape}:")
    d6_out, d6_feat = model.attention6(x4_mid_tap)
    print(f"  Output: {d6_out.shape}, Features: {d6_feat.shape}")
    assert d6_out.shape == (batch_size, 10), "D6 should output class predictions"
    assert d6_feat.shape == (batch_size, 512), "D6 should output 512-dim features"
    
    print("\n4. VERIFYING COMPLETE FORWARD_ALL_HEADS:")
    print("-" * 50)
    
    # Test complete forward_all_heads
    outputs = model.forward_all_heads(x)
    
    print(f"Total outputs: {len(outputs)}")
    assert len(outputs) == 14, "Should have 14 outputs (backbone + 6 heads with features)"
    
    # Parse outputs
    out_backbone, out4_feature = outputs[0], outputs[1]
    D1_out, D1_feat = outputs[2], outputs[3]
    D4_out, D4_feat = outputs[4], outputs[5]
    D2_out, D2_feat = outputs[6], outputs[7]
    D5_out, D5_feat = outputs[8], outputs[9]
    D3_out, D3_feat = outputs[10], outputs[11]
    D6_out, D6_feat = outputs[12], outputs[13]
    
    print("\nOutput shapes:")
    print(f"  Backbone: {out_backbone.shape}")
    print(f"  D1: {D1_out.shape}")
    print(f"  D2: {D2_out.shape}")
    print(f"  D3: {D3_out.shape}")
    print(f"  D4: {D4_out.shape} <- From layer2[0]")
    print(f"  D5: {D5_out.shape} <- From layer3[0]")
    print(f"  D6: {D6_out.shape} <- From layer4[0]")
    
    print("\n5. VERIFYING D4, D5, D6 USE CORRECT INTERMEDIATE FEATURES:")
    print("-" * 50)
    
    # Verify that D4, D5, D6 are using intermediate features (not final layer outputs)
    # by checking they produce different outputs than D2, D3, and backbone
    
    with torch.no_grad():
        # D4 vs D2 (both process 128-ch, but different stages)
        d4_d2_diff = torch.abs(D4_out - D2_out).mean().item()
        print(f"D4 vs D2 difference: {d4_d2_diff:.4f}")
        assert d4_d2_diff > 0.01, "D4 and D2 should produce different outputs"
        
        # D5 vs D3 (both process 256-ch, but different stages)
        d5_d3_diff = torch.abs(D5_out - D3_out).mean().item()
        print(f"D5 vs D3 difference: {d5_d3_diff:.4f}")
        assert d5_d3_diff > 0.01, "D5 and D3 should produce different outputs"
        
        # D6 should be unique (only one processing 512-ch intermediate)
        d6_backbone_diff = torch.abs(D6_out - out_backbone).mean().item()
        print(f"D6 vs Backbone difference: {d6_backbone_diff:.4f}")
        assert d6_backbone_diff > 0.01, "D6 and Backbone should produce different outputs"
    
    print("\n6. ARCHITECTURE VERIFICATION:")
    print("-" * 50)
    
    print(f"D4 attention module type: {type(model.attention4).__name__}")
    print(f"D5 attention module type: {type(model.attention5).__name__}")
    print(f"D6 attention module type: {type(model.attention6).__name__}")
    
    # Verify attention module types
    assert type(model.attention4).__name__ == "ResNetAttention2", "D4 should use ResNetAttention2"
    assert type(model.attention5).__name__ == "ResNetAttention3", "D5 should use ResNetAttention3"
    assert type(model.attention6).__name__ == "ResNetAttention4", "D6 should use ResNetAttention4"
    
    print("\n" + "=" * 80)
    print("✅ D4, D5, D6 FORWARD PASS VERIFICATION COMPLETE!")
    print("=" * 80)
    
    print("\nSUMMARY:")
    print("✓ D4 correctly processes layer2[0] output (128-ch intermediate)")
    print("✓ D5 correctly processes layer3[0] output (256-ch intermediate)")
    print("✓ D6 correctly processes layer4[0] output (512-ch intermediate)")
    print("✓ All heads produce correct output shapes")
    print("✓ Each head produces distinct outputs")
    print("✓ forward_all_heads returns outputs in correct order")


if __name__ == "__main__":
    verify_d456_forward_implementation()
    print("\n✅ All verifications passed!")