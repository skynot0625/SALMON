#!/bin/bash

# Test sweep with 1 epoch to verify configuration
# This will run quickly to ensure everything is set up correctly

echo "=========================================="
echo "TEST SWEEP - 1 EPOCH ONLY"
echo "Testing loss_coefficient sweep configuration"
echo "=========================================="
echo ""

# Set wandb API key
export WANDB_API_KEY=2601d5461d965e275d39beb448abba392abb5e0b

# Test with just 2 values for quick verification
echo "Testing with loss_coefficient values: [0.0, 1.0]"
echo "Corresponding alpha values: [1.0, 0.0]"
echo ""

# Create test screen session
screen -dmS test_sweep bash -c "
    cd /root/SALMON
    export WANDB_API_KEY=2601d5461d965e275d39beb448abba392abb5e0b
    echo 'Starting TEST sweep with 1 epoch'
    python src/train.py \
        --multirun \
        experiment=exp7_6_digital \
        trainer=gpu \
        trainer.max_epochs=1 \
        trainer.min_epochs=1 \
        model.loss_coefficient=0.0,1.0 \
        logger.wandb.group='test_sweep_1epoch' \
        logger.wandb.project='salmon' \
        logger.wandb.entity='spk' \
        'logger.wandb.name=test_alpha_\${model.loss_coefficient}' \
        'logger.wandb.tags=[test_sweep,1epoch,loss_coef_\${model.loss_coefficient}]' \
        hydra.sweep.dir='logs/test_sweep' \
        hydra.sweep.subdir='loss_\${model.loss_coefficient}' \
        2>&1 | tee logs/test_sweep_$(date +%Y%m%d_%H%M%S).log
    echo 'TEST sweep completed!'
"

echo "Test sweep started in screen session 'test_sweep'"
echo ""
echo "Commands:"
echo "  - Watch progress:     screen -r test_sweep"
echo "  - Detach:            Ctrl+A then D"
echo "  - Check status:      screen -ls"
echo ""
echo "This test should complete in a few minutes."
echo "Check wandb group: test_sweep_1epoch"