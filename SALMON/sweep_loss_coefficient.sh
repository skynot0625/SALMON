#!/bin/bash

# Sweep over different loss_coefficient values for exp7_6_digital experiment
# The loss is computed as: loss * (1 - loss_coefficient)
# Testing values: 0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0

echo "Starting loss_coefficient sweep for exp7_6_digital experiment"
echo "Testing values: 0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0"
echo "================================================"

# Set wandb API key
export WANDB_API_KEY=2601d5461d965e275d39beb448abba392abb5e0b

# Run experiments with different loss_coefficient values
for loss_coef in 0.0 0.01 0.1 0.5 1.0 2.0 10.0
do
    echo ""
    echo "Running experiment with loss_coefficient=$loss_coef"
    echo "Effective loss multiplier: $(echo "1 - $loss_coef" | bc -l)"
    echo "----------------------------------------"
    
    # Run the experiment with the current loss_coefficient
    python src/train.py \
        experiment=exp7_6_digital \
        trainer=gpu \
        model.loss_coefficient=$loss_coef \
        logger.wandb.group="loss_coefficient_sweep" \
        logger.wandb.tags="[loss_coef_${loss_coef}]" \
        logger.wandb.name="exp7_6_digital_loss_${loss_coef}" \
        trainer.max_epochs=50  # Reduced for faster sweep, adjust as needed
        
    echo "Completed experiment with loss_coefficient=$loss_coef"
    echo "================================================"
done

echo "Sweep completed successfully!"