#!/bin/bash

# Run sweep with screen for persistent execution
# alpha = (1 - loss_coefficient) is the actual loss multiplier

echo "Starting Alpha Sweep in Screen Session"
echo "======================================="
echo "Testing loss_coefficient values: [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0]"
echo "Corresponding alpha values: [1.0, 0.99, 0.9, 0.5, 0.0, -1.0, -9.0]"
echo ""

# Set wandb API key
export WANDB_API_KEY=2601d5461d965e275d39beb448abba392abb5e0b

# Create a screen session and run the sweep
screen -dmS alpha_sweep bash -c "
    cd /root/SALMON
    export WANDB_API_KEY=2601d5461d965e275d39beb448abba392abb5e0b
    python src/train.py \
        --multirun \
        experiment=exp7_6_digital \
        trainer=gpu \
        trainer.max_epochs=300 \
        trainer.min_epochs=100 \
        model.loss_coefficient=0.0,0.01,0.1,0.5,1.0,2.0,10.0 \
        logger.wandb.group='alpha_sweep_screen' \
        logger.wandb.project='salmon' \
        logger.wandb.entity='spk' \
        'logger.wandb.name=alpha_\${model.loss_coefficient}' \
        'logger.wandb.tags=[alpha_sweep,exp7_6_digital,loss_coef_\${model.loss_coefficient}]' \
        hydra.sweep.dir='logs/sweeps/alpha_screen' \
        hydra.sweep.subdir='loss_coef_\${model.loss_coefficient}' \
        2>&1 | tee logs/sweep_alpha_$(date +%Y%m%d_%H%M%S).log
"

echo "Sweep started in screen session 'alpha_sweep'"
echo ""
echo "Commands to manage the screen session:"
echo "  - View the running session:  screen -r alpha_sweep"
echo "  - Detach from session:       Ctrl+A then D"
echo "  - List all sessions:         screen -ls"
echo "  - Kill the session:          screen -X -S alpha_sweep quit"
echo ""
echo "Results will be logged to:"
echo "  - Wandb group: alpha_sweep_screen"
echo "  - Log file: logs/sweep_alpha_*.log"