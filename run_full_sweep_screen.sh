#!/bin/bash

# Full sweep with screen for 300 epochs
# alpha = (1 - loss_coefficient)

echo "=========================================="
echo "FULL ALPHA SWEEP - 300 EPOCHS"
echo "=========================================="
echo "Testing loss_coefficient values: [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0]"
echo "Corresponding alpha values: [1.0, 0.99, 0.9, 0.5, 0.0, -1.0, -9.0]"
echo ""

# Create logs directory if it doesn't exist
mkdir -p /root/SALMON/logs/sweeps

# Start screen session with multirun sweep
screen -dmS alpha_sweep_full bash -c "
    cd /root/SALMON
    export WANDB_API_KEY=2601d5461d965e275d39beb448abba392abb5e0b
    echo 'Starting FULL alpha sweep with 300 epochs'
    echo 'Start time: $(date)'
    
    python src/train.py \
        --multirun \
        experiment=exp7_6_digital \
        trainer=gpu \
        trainer.max_epochs=300 \
        trainer.min_epochs=100 \
        model.loss_coefficient=0.0,0.01,0.1,0.5,1.0,2.0,10.0 \
        logger.wandb.group='alpha_sweep_300ep' \
        logger.wandb.project='salmon' \
        logger.wandb.entity='spk' \
        hydra.sweep.dir='logs/sweeps/alpha_300ep' \
        hydra.sweep.subdir='loss_\${model.loss_coefficient}' \
        2>&1 | tee logs/sweeps/alpha_sweep_300ep_$(date +%Y%m%d_%H%M%S).log
    
    echo 'End time: $(date)'
    echo 'Alpha sweep completed!'
"

echo "✅ Full sweep started in screen session 'alpha_sweep_full'"
echo ""
echo "📊 Monitoring Commands:"
echo "  - Attach to session:  screen -r alpha_sweep_full"
echo "  - Detach:            Ctrl+A then D"
echo "  - Check status:      screen -ls"
echo "  - View log:          tail -f logs/sweeps/alpha_sweep_300ep_*.log"
echo ""
echo "🔗 Wandb tracking:"
echo "  - Group: alpha_sweep_300ep"
echo "  - Project: salmon"
echo "  - Entity: spk"
echo ""
echo "⏱️  Estimated time: ~7 × (time for 300 epochs)"
echo "    This will run all 7 experiments sequentially"