#!/bin/bash

# Direct test run without screen first to verify the command works

cd /root/SALMON

echo "=========================================="
echo "DIRECT TEST - 1 EPOCH"
echo "Testing single loss_coefficient value"
echo "=========================================="
echo ""

# Set wandb API key
export WANDB_API_KEY=2601d5461d965e275d39beb448abba392abb5e0b

echo "Running test with loss_coefficient=0.5 (alpha=0.5)"
echo ""

# Run single test
WANDB_API_KEY=2601d5461d965e275d39beb448abba392abb5e0b python src/train.py \
    experiment=exp7_6_digital \
    trainer=gpu \
    trainer.max_epochs=1 \
    trainer.min_epochs=1 \
    model.loss_coefficient=0.5 \
    logger.wandb.group="test_direct" \
    logger.wandb.project="salmon" \
    logger.wandb.entity="spk" \
    logger.wandb.name="test_loss_0.5_1epoch" \
    'logger.wandb.tags=["test","1epoch","loss_0.5"]'

echo ""
echo "Test completed!"