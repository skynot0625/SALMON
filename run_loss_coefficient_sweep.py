#!/usr/bin/env python
"""
Run sweep for loss_coefficient parameter in exp7_6_digital experiment
Testing values: 0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0
The actual loss multiplier is (1 - loss_coefficient)
"""

import os
import subprocess
import sys

# Set wandb API key
os.environ['WANDB_API_KEY'] = '2601d5461d965e275d39beb448abba392abb5e0b'

# Loss coefficient values to test
loss_coefficients = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0]

print("Starting loss_coefficient sweep for exp7_6_digital experiment")
print(f"Testing values: {loss_coefficients}")
print("=" * 60)

# Run Hydra multirun with all values
cmd = [
    "python", "src/train.py",
    "-m",  # Enable multirun mode
    "experiment=exp7_6_digital",
    "trainer=gpu",
    f"model.loss_coefficient={','.join(map(str, loss_coefficients))}",
    "logger.wandb.group=loss_coefficient_sweep",
    "trainer.max_epochs=50",  # Reduced for faster sweep
    "hydra.sweep.dir=logs/sweeps/loss_coefficient",
    "hydra.sweep.subdir=${model.loss_coefficient}",
]

print("Running command:")
print(" ".join(cmd))
print("-" * 60)

# Execute the sweep
result = subprocess.run(cmd, capture_output=False, text=True)

if result.returncode == 0:
    print("\nSweep completed successfully!")
    print("Results are logged to wandb under group: loss_coefficient_sweep")
else:
    print(f"\nSweep failed with return code: {result.returncode}")
    sys.exit(1)