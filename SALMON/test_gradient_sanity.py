#!/usr/bin/env python3
"""
Test script to verify gradient computation sanity check.
This script initializes the model and runs the sanity check without full training.
"""

import torch
import hydra
from omegaconf import DictConfig
from src.models.exp7_6_module import SalmonLitModule
from src.models.components.exp7_1_model import IntegratedResNet
from aihwkit.simulator.presets import IdealizedPreset

@hydra.main(version_base="1.3", config_path="../configs", config_name="test.yaml")
def test_sanity_check(cfg: DictConfig):
    """Test the gradient sanity check function."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create integrated ResNet model
    integrated_resnet = IntegratedResNet(
        architecture="resnet18",
        num_classes=10,
        rpu_config=IdealizedPreset()
    )
    
    # Create the Lightning module with probe enabled
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
            "epochs": [1, 10, 20],
            "batches_per_epoch": 8,
            "alpha_eval": [0.0, 0.5, 1.0],
            "seed": 1234
        }
    )
    
    # Move model to device
    model = model.to(device)
    
    # Setup the model (initializes probe loader and layer handles)
    model.setup(stage="fit")
    
    # Run the sanity check
    model.sanity_check_gradients()
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    # Simple test without hydra
    import sys
    sys.path.append('/root/SALMON')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create integrated ResNet model
    integrated_resnet = IntegratedResNet(
        architecture="resnet18",
        num_classes=10,
        rpu_config=IdealizedPreset()
    )
    
    # Create the Lightning module with probe enabled
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
            "epochs": [1, 10, 20],
            "batches_per_epoch": 8,
            "alpha_eval": [0.0, 0.5, 1.0],
            "seed": 1234
        }
    )
    
    # Move model to device
    model = model.to(device)
    
    # Setup the model (initializes probe loader and layer handles)
    model.setup(stage="fit")
    
    # Run the sanity check
    model.sanity_check_gradients()
    
    print("\nTest completed successfully!")