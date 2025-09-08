#!/bin/bash

# Run Hydra-Optuna sweep for loss_coefficient parameter
# alpha = (1 - loss_coefficient) is the actual loss multiplier
# Testing 7 different values with full 300 epochs

echo "Starting Alpha Sweep for exp7_6_digital"
echo "========================================"
echo "Testing loss_coefficient values: [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0]"
echo "Corresponding alpha values: [1.0, 0.99, 0.9, 0.5, 0.0, -1.0, -9.0]"
echo ""
echo "Each experiment will run for 300 epochs on GPU"
echo "Results will be tracked in wandb group: alpha_sweep_300epochs"
echo "========================================"

# Set wandb API key
export WANDB_API_KEY=2601d5461d965e275d39beb448abba392abb5e0b

# Run the sweep using Hydra multirun with Optuna
WANDB_API_KEY=2601d5461d965e275d39beb448abba392abb5e0b python src/train.py \
    --multirun \
    experiment=exp7_6_digital \
    hparams_search=exp7_6_alpha_sweep \
    trainer=gpu \
    trainer.max_epochs=300 \
    trainer.min_epochs=100 \
    logger.wandb.group="alpha_sweep_300epochs" \
    logger.wandb.project="salmon" \
    logger.wandb.entity="spk" \
    'logger.wandb.tags=["alpha_sweep","exp7_6_digital","300_epochs"]' \
    hydra.sweep.dir="logs/sweeps/alpha_sweep_${now:%Y-%m-%d_%H-%M-%S}"

echo ""
echo "Sweep completed! Check wandb for results."
echo "Group: alpha_sweep_300epochs"