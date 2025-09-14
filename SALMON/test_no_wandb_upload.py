#!/usr/bin/env python3
"""
Test script to verify model runs without uploading to WandB
"""

import os
import sys
import hydra
from omegaconf import DictConfig
from pathlib import Path

# Add project root to path
sys.path.append('/root/SALMON')

@hydra.main(version_base="1.3", config_path="/root/SALMON/configs", config_name="train")
def test_config(cfg: DictConfig):
    """Test configuration without actually training."""
    
    print("=" * 70)
    print("TESTING CONFIGURATION WITHOUT WANDB UPLOADS")
    print("=" * 70)
    
    # Check logger configuration
    if cfg.get('logger'):
        if isinstance(cfg.logger, dict):
            for logger_name, logger_cfg in cfg.logger.items():
                print(f"\nLogger: {logger_name}")
                if 'wandb' in logger_name.lower():
                    # Check WandB settings
                    log_model = logger_cfg.get('log_model', None)
                    save_code = logger_cfg.get('save_code', None)
                    offline = logger_cfg.get('offline', None)
                    
                    print(f"  log_model: {log_model}")
                    print(f"  save_code: {save_code}")
                    print(f"  offline: {offline}")
                    
                    if log_model:
                        print("  ⚠️  WARNING: log_model is True - models will be uploaded!")
                    else:
                        print("  ✓ log_model is False - models won't be uploaded")
                        
                    if save_code:
                        print("  ⚠️  WARNING: save_code is True - code will be uploaded!")
                    else:
                        print("  ✓ save_code is False/None - code won't be uploaded")
                else:
                    print(f"  Type: {logger_cfg.get('_target_', 'Unknown')}")
    else:
        print("\n✓ No logger configured - nothing will be uploaded")
    
    # Check model configuration
    print(f"\nModel Configuration:")
    print(f"  Architecture: {cfg.model.get('model', 'Unknown')}")
    print(f"  Heads: {cfg.model.get('active_heads', 'Not specified')}")
    print(f"  Loss coefficient: {cfg.model.get('loss_coefficient', 'Not specified')}")
    print(f"  Learning rate: {cfg.model.optimizer.get('lr', 'Not specified')}")
    
    # Check callbacks
    if cfg.get('callbacks'):
        print(f"\nCallbacks:")
        if cfg.callbacks.get('model_checkpoint'):
            checkpoint_cfg = cfg.callbacks.model_checkpoint
            dirpath = checkpoint_cfg.get('dirpath', 'Not specified')
            print(f"  Model checkpoint directory: {dirpath}")
            print(f"  ✓ Checkpoints saved locally only")
    
    print("\n" + "=" * 70)
    print("CONFIGURATION CHECK COMPLETE")
    print("=" * 70)
    
    return cfg


if __name__ == "__main__":
    # Test different configurations
    configs_to_test = [
        "experiment=exp7_6_digital_idealdevice_4heads_local",  # Local only
        "experiment=exp7_6_digital_idealdevice_4heads",  # Original (might have WandB)
    ]
    
    for config in configs_to_test:
        print(f"\n\nTesting: {config}")
        print("-" * 50)
        try:
            sys.argv = ["test_no_wandb_upload.py", config]
            test_config()
        except Exception as e:
            print(f"Error: {e}")