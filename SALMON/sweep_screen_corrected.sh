#!/bin/bash

# Corrected sweep script for loss_coefficient in exp7_6_digital
# Understanding: loss = teacher_loss + sum(student_losses * (1 - loss_coefficient))
# loss_coefficient = 0.0 -> students get full loss (multiplier = 1.0)
# loss_coefficient = 1.0 -> students get no loss (multiplier = 0.0)
# loss_coefficient = 2.0 -> students get negative loss (multiplier = -1.0)

echo "=============================================="
echo "LOSS COEFFICIENT SWEEP - exp7_6_digital"
echo "=============================================="
echo ""
echo "Loss computation:"
echo "  Teacher: always full loss"
echo "  Students: loss * (1 - loss_coefficient)"
echo ""
echo "Testing values:"
echo "  loss_coefficient = 0.0  -> student multiplier = 1.0  (full loss)"
echo "  loss_coefficient = 0.01 -> student multiplier = 0.99"
echo "  loss_coefficient = 0.1  -> student multiplier = 0.9"
echo "  loss_coefficient = 0.5  -> student multiplier = 0.5"
echo "  loss_coefficient = 1.0  -> student multiplier = 0.0  (no loss)"
echo "  loss_coefficient = 2.0  -> student multiplier = -1.0 (negative loss)"
echo "  loss_coefficient = 10.0 -> student multiplier = -9.0 (strong negative)"
echo ""

# Create logs directory
mkdir -p /root/SALMON/logs/sweeps

# Function to run a single experiment in screen
run_single_experiment() {
    local loss_coef=$1
    local session_name="loss_${loss_coef//./_}"  # Replace dots with underscores
    local multiplier=$(echo "scale=2; 1 - $loss_coef" | bc)
    
    echo "Starting experiment: loss_coefficient=$loss_coef (multiplier=$multiplier)"
    
    screen -dmS "$session_name" bash -c "
        cd /root/SALMON
        export WANDB_API_KEY=2601d5461d965e275d39beb448abba392abb5e0b
        
        echo '========================================'
        echo 'Experiment: loss_coefficient=$loss_coef'
        echo 'Student loss multiplier: $multiplier'
        echo 'Start time: \$(date)'
        echo '========================================'
        
        python src/train.py \
            experiment=exp7_6_digital \
            trainer=gpu \
            trainer.max_epochs=300 \
            trainer.min_epochs=100 \
            model.loss_coefficient=$loss_coef \
            logger.wandb.group='loss_coef_sweep' \
            logger.wandb.project='salmon' \
            logger.wandb.entity='spk' \
            logger.wandb.tags='[\"loss_coef_sweep\",\"exp7_6_digital\",\"loss_$loss_coef\",\"multiplier_$multiplier\"]' \
            2>&1 | tee logs/sweeps/loss_coef_${loss_coef}_\$(date +%Y%m%d_%H%M%S).log
        
        echo 'Experiment completed for loss_coefficient=$loss_coef'
        echo 'End time: \$(date)'
    "
    
    sleep 3  # Small delay between starting sessions
}

# Option 1: Run all experiments in parallel (each in its own screen)
run_parallel() {
    echo "Starting parallel execution (each experiment in separate screen)..."
    for loss_coef in 0.0 0.01 0.1 0.5 1.0 2.0 10.0; do
        run_single_experiment $loss_coef
    done
    echo ""
    echo "✅ All experiments started in parallel!"
}

# Option 2: Run all experiments sequentially in one screen
run_sequential() {
    echo "Starting sequential execution (all experiments in one screen)..."
    
    screen -dmS loss_sweep_seq bash -c "
        cd /root/SALMON
        export WANDB_API_KEY=2601d5461d965e275d39beb448abba392abb5e0b
        
        echo 'Starting sequential sweep of all loss_coefficient values'
        echo 'Start time: \$(date)'
        echo '========================================'
        
        for loss_coef in 0.0 0.01 0.1 0.5 1.0 2.0 10.0; do
            multiplier=\$(echo \"scale=2; 1 - \$loss_coef\" | bc)
            echo ''
            echo \"Running: loss_coefficient=\$loss_coef (multiplier=\$multiplier)\"
            
            python src/train.py \
                experiment=exp7_6_digital \
                trainer=gpu \
                trainer.max_epochs=300 \
                trainer.min_epochs=100 \
                model.loss_coefficient=\$loss_coef \
                logger.wandb.group='loss_coef_sweep_seq' \
                logger.wandb.project='salmon' \
                logger.wandb.entity='spk' \
                2>&1 | tee -a logs/sweeps/loss_sweep_seq_\$(date +%Y%m%d_%H%M%S).log
            
            echo \"Completed: loss_coefficient=\$loss_coef\"
            echo '----------------------------------------'
        done
        
        echo 'Sequential sweep completed!'
        echo 'End time: \$(date)'
    "
    
    echo "✅ Sequential sweep started in screen 'loss_sweep_seq'"
}

# Ask user which mode to run
if [ "$1" == "parallel" ]; then
    run_parallel
elif [ "$1" == "sequential" ]; then
    run_sequential
else
    echo "Usage: $0 [parallel|sequential]"
    echo ""
    echo "  parallel   - Run all experiments simultaneously (faster, more GPU memory)"
    echo "  sequential - Run experiments one after another (slower, less memory)"
    echo ""
    echo "Example: $0 parallel"
    exit 1
fi

echo ""
echo "📊 Screen Management Commands:"
echo "  - List sessions:    screen -ls"
echo "  - Attach to session: screen -r <session_name>"
echo "  - Detach:           Ctrl+A then D"
echo "  - Kill session:     screen -X -S <session_name> quit"
echo ""
echo "📝 Log files location: logs/sweeps/"
echo ""
echo "🔗 Wandb tracking:"
if [ "$1" == "parallel" ]; then
    echo "  - Group: loss_coef_sweep"
    echo "  - Sessions: loss_0_0, loss_0_01, loss_0_1, loss_0_5, loss_1_0, loss_2_0, loss_10_0"
else
    echo "  - Group: loss_coef_sweep_seq"
    echo "  - Session: loss_sweep_seq"
fi