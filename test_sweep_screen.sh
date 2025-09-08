#!/bin/bash

# Test script with 1 epoch to verify screen setup works correctly

echo "=============================================="
echo "TEST SWEEP - 1 EPOCH"
echo "Testing screen setup with 2 loss_coefficient values"
echo "=============================================="
echo ""

# Create logs directory
mkdir -p /root/SALMON/logs/test

# Test with 2 values
echo "Testing loss_coefficient = 0.0 (multiplier = 1.0) and 1.0 (multiplier = 0.0)"
echo ""

# Start test in screen
screen -dmS test_sweep bash -c "
    cd /root/SALMON
    export WANDB_API_KEY=2601d5461d965e275d39beb448abba392abb5e0b
    
    echo 'TEST SWEEP - 1 EPOCH'
    echo 'Testing 2 values: loss_coefficient = 0.0 and 1.0'
    echo 'Start time: \$(date)'
    echo '========================================'
    
    # Test first value
    echo 'Test 1: loss_coefficient=0.0 (multiplier=1.0)'
    python src/train.py \
        experiment=exp7_6_digital \
        trainer=gpu \
        trainer.max_epochs=1 \
        trainer.min_epochs=1 \
        model.loss_coefficient=0.0 \
        logger.wandb.group='test_sweep_1ep' \
        logger.wandb.project='salmon' \
        logger.wandb.entity='spk' \
        2>&1 | tee logs/test/test_loss_0.0.log
    
    echo 'Test 1 completed'
    echo '----------------------------------------'
    
    # Test second value
    echo 'Test 2: loss_coefficient=1.0 (multiplier=0.0)'
    python src/train.py \
        experiment=exp7_6_digital \
        trainer=gpu \
        trainer.max_epochs=1 \
        trainer.min_epochs=1 \
        model.loss_coefficient=1.0 \
        logger.wandb.group='test_sweep_1ep' \
        logger.wandb.project='salmon' \
        logger.wandb.entity='spk' \
        2>&1 | tee logs/test/test_loss_1.0.log
    
    echo 'Test 2 completed'
    echo '========================================'
    echo 'TEST SWEEP COMPLETED!'
    echo 'End time: \$(date)'
    
    # Keep session alive for a moment to check results
    sleep 10
"

echo "✅ Test sweep started in screen session 'test_sweep'"
echo ""
echo "Commands:"
echo "  - Watch progress:  screen -r test_sweep"
echo "  - Check status:    screen -ls"
echo "  - Detach:         Ctrl+A then D"
echo ""
echo "This test should complete in 2-3 minutes."
echo "Check logs in: logs/test/"