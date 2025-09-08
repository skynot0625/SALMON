#!/bin/bash

# Direct Hydra multirun sweep for loss_coefficient parameter
# alpha = (1 - loss_coefficient) is the actual loss multiplier

echo "Starting Direct Alpha Sweep for exp7_6_digital"
echo "=============================================="
echo "Testing loss_coefficient values: [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0]"
echo "Corresponding alpha values: [1.0, 0.99, 0.9, 0.5, 0.0, -1.0, -9.0]"
echo ""

# Set wandb API key
export WANDB_API_KEY=2601d5461d965e275d39beb448abba392abb5e0b

# Run direct multirun without Optuna
WANDB_API_KEY=2601d5461d965e275d39beb448abba392abb5e0b python src/train.py \
    --multirun \
    experiment=exp7_6_digital \
    trainer=gpu \
    trainer.max_epochs=300 \
    trainer.min_epochs=100 \
    model.loss_coefficient=0.0,0.01,0.1,0.5,1.0,2.0,10.0 \
    logger.wandb.group="alpha_sweep_direct" \
    logger.wandb.project="salmon" \
    logger.wandb.entity="spk" \
    'logger.wandb.name="alpha_${model.loss_coefficient}"' \
    'logger.wandb.tags=["alpha_sweep","exp7_6_digital","loss_coef_${model.loss_coefficient}"]' \
    hydra.sweep.dir="logs/sweeps/alpha_direct" \
    hydra.sweep.subdir="loss_coef_${model.loss_coefficient}"

echo ""
echo "Sweep completed! Check wandb for results."
echo "Group: alpha_sweep_direct"