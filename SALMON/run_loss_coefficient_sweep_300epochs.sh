#!/bin/bash

# Production-ready sweep script for loss_coefficient parameter
# Runs 300 epochs for each value to properly evaluate performance

echo "=================================================="
echo "LOSS COEFFICIENT SWEEP - 300 EPOCHS"
echo "exp7_6_digital with SALMON Digital Module"
echo "=================================================="
echo ""
echo "Understanding the loss computation:"
echo "  - Teacher network: always gets full loss"
echo "  - Student networks: loss * (1 - loss_coefficient)"
echo ""
echo "Testing 7 loss_coefficient values:"
echo "  0.00 -> multiplier = 1.00 (full student loss)"
echo "  0.01 -> multiplier = 0.99"
echo "  0.10 -> multiplier = 0.90"
echo "  0.50 -> multiplier = 0.50"  
echo "  1.00 -> multiplier = 0.00 (no student loss)"
echo "  2.00 -> multiplier = -1.00 (negative loss)"
echo "  10.0 -> multiplier = -9.00 (strong negative)"
echo ""

# Create necessary directories
mkdir -p /root/SALMON/logs/sweeps/loss_coefficient_300ep

# Check if sequential or parallel mode requested
MODE=${1:-sequential}

if [ "$MODE" == "sequential" ]; then
    echo "Mode: SEQUENTIAL (experiments run one after another)"
    echo ""
    
    # Start sequential sweep in screen
    screen -dmS loss_sweep_300ep bash -c "
        cd /root/SALMON
        export WANDB_API_KEY=2601d5461d965e275d39beb448abba392abb5e0b
        
        echo '=================================================='
        echo 'SEQUENTIAL LOSS COEFFICIENT SWEEP - 300 EPOCHS'
        echo 'Start time: \$(date)'
        echo '=================================================='
        
        # Array of loss coefficient values
        loss_values=(0.0 0.01 0.1 0.5 1.0 2.0 10.0)
        
        for loss_coef in \${loss_values[@]}; do
            multiplier=\$(echo \"scale=2; 1 - \$loss_coef\" | bc)
            
            echo ''
            echo '=================================================='
            echo \"Starting: loss_coefficient=\$loss_coef (multiplier=\$multiplier)\"
            echo \"Time: \$(date)\"
            echo '=================================================='
            
            python src/train.py \
                experiment=exp7_6_digital \
                trainer=gpu \
                trainer.max_epochs=300 \
                trainer.min_epochs=100 \
                model.loss_coefficient=\$loss_coef \
                logger.wandb.group='loss_sweep_300ep' \
                logger.wandb.project='salmon' \
                logger.wandb.entity='spk' \
                logger.wandb.tags=\"['loss_sweep','300_epochs','loss_coef_\${loss_coef}']\" \
                2>&1 | tee logs/sweeps/loss_coefficient_300ep/loss_\${loss_coef}_\$(date +%Y%m%d_%H%M%S).log
            
            echo \"Completed: loss_coefficient=\$loss_coef\"
            echo '=================================================='
        done
        
        echo ''
        echo '=================================================='
        echo 'SWEEP COMPLETED SUCCESSFULLY!'
        echo 'End time: \$(date)'
        echo '=================================================='
    "
    
    echo "✅ Sequential sweep started in screen session: 'loss_sweep_300ep'"
    
elif [ "$MODE" == "parallel" ]; then
    echo "Mode: PARALLEL (all experiments run simultaneously)"
    echo "⚠️  Warning: This requires significant GPU memory!"
    echo ""
    
    # Start each experiment in its own screen session
    for loss_coef in 0.0 0.01 0.1 0.5 1.0 2.0 10.0; do
        session_name="loss_${loss_coef//./_}_300ep"
        multiplier=$(echo "scale=2; 1 - $loss_coef" | bc)
        
        echo "Starting session: $session_name (loss=$loss_coef, multiplier=$multiplier)"
        
        screen -dmS "$session_name" bash -c "
            cd /root/SALMON
            export WANDB_API_KEY=2601d5461d965e275d39beb448abba392abb5e0b
            
            echo 'loss_coefficient=$loss_coef (multiplier=$multiplier)'
            echo 'Start: \$(date)'
            
            python src/train.py \
                experiment=exp7_6_digital \
                trainer=gpu \
                trainer.max_epochs=300 \
                trainer.min_epochs=100 \
                model.loss_coefficient=$loss_coef \
                logger.wandb.group='loss_sweep_300ep_parallel' \
                logger.wandb.project='salmon' \
                logger.wandb.entity='spk' \
                2>&1 | tee logs/sweeps/loss_coefficient_300ep/parallel_loss_${loss_coef}_\$(date +%Y%m%d_%H%M%S).log
            
            echo 'End: \$(date)'
        "
        
        sleep 5  # Delay between starting sessions
    done
    
    echo "✅ All parallel sessions started!"
    
else
    echo "Usage: $0 [sequential|parallel]"
    echo ""
    echo "  sequential - Run experiments one after another (default)"
    echo "  parallel   - Run all experiments simultaneously"
    exit 1
fi

echo ""
echo "📊 Screen Management:"
echo "  - List sessions:     screen -ls"
echo "  - Attach to session: screen -r <session_name>"
echo "  - Detach:           Ctrl+A then D"
echo "  - Kill session:     screen -X -S <session_name> quit"
echo ""
echo "📁 Log files: /root/SALMON/logs/sweeps/loss_coefficient_300ep/"
echo ""
echo "🔗 Wandb tracking:"
echo "  - Project: salmon"
echo "  - Entity: spk"
if [ "$MODE" == "sequential" ]; then
    echo "  - Group: loss_sweep_300ep"
    echo "  - Session: loss_sweep_300ep"
else
    echo "  - Group: loss_sweep_300ep_parallel"
    echo "  - Sessions: loss_0_0_300ep, loss_0_01_300ep, etc."
fi
echo ""
echo "⏱️  Estimated time:"
if [ "$MODE" == "sequential" ]; then
    echo "  ~7 × (time for 300 epochs) - running sequentially"
else
    echo "  ~1 × (time for 300 epochs) - running in parallel"
fi