#!/bin/bash

# Run individual experiments in separate screen sessions
# This allows parallel execution and better monitoring

echo "Starting Individual Screen Sessions for Each Alpha Value"
echo "========================================================="

# Set wandb API key
export WANDB_API_KEY=2601d5461d965e275d39beb448abba392abb5e0b

# Array of loss_coefficient values
loss_coeffs=(0.0 0.01 0.1 0.5 1.0 2.0 10.0)

# Start each experiment in its own screen session
for loss_coef in "${loss_coeffs[@]}"
do
    # Calculate alpha for display
    alpha=$(echo "scale=2; 1 - $loss_coef" | bc)
    
    # Create screen session name (replace dots with underscore for valid name)
    session_name="exp_loss_${loss_coef//./_}"
    
    echo "Starting session: $session_name (loss_coef=$loss_coef, alpha=$alpha)"
    
    # Create screen session and run experiment
    screen -dmS "$session_name" bash -c "
        cd /root/SALMON
        export WANDB_API_KEY=2601d5461d965e275d39beb448abba392abb5e0b
        echo 'Starting experiment with loss_coefficient=$loss_coef (alpha=$alpha)'
        python src/train.py \
            experiment=exp7_6_digital \
            trainer=gpu \
            trainer.max_epochs=300 \
            trainer.min_epochs=100 \
            model.loss_coefficient=$loss_coef \
            logger.wandb.group='alpha_sweep_parallel' \
            logger.wandb.project='salmon' \
            logger.wandb.entity='spk' \
            logger.wandb.name='exp7_6_loss_${loss_coef}' \
            'logger.wandb.tags=[alpha_sweep,exp7_6_digital,loss_coef_${loss_coef},alpha_${alpha}]' \
            2>&1 | tee logs/exp_loss_${loss_coef}_$(date +%Y%m%d_%H%M%S).log
        echo 'Experiment completed for loss_coefficient=$loss_coef'
    "
    
    # Small delay between starting sessions
    sleep 2
done

echo ""
echo "All experiments started in separate screen sessions!"
echo ""
echo "Useful screen commands:"
echo "  - List all sessions:     screen -ls"
echo "  - Attach to a session:   screen -r <session_name>"
echo "  - Detach from session:   Ctrl+A then D"
echo "  - Kill a session:        screen -X -S <session_name> quit"
echo "  - Kill all exp sessions: screen -ls | grep exp_loss | cut -d. -f1 | awk '{print \$1}' | xargs -I {} screen -X -S {} quit"
echo ""
echo "Session names:"
for loss_coef in "${loss_coeffs[@]}"
do
    session_name="exp_loss_${loss_coef//./_}"
    echo "  - $session_name (loss_coef=$loss_coef)"
done
echo ""
echo "Monitor progress on wandb group: alpha_sweep_parallel"