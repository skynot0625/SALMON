#!/bin/bash

# Enhanced sweep script with proper wandb tagging and hyperparameter logging
# Tags will include the loss_coefficient value for easy filtering in wandb

echo "=================================================="
echo "LOSS COEFFICIENT SWEEP WITH ENHANCED TAGGING"
echo "exp7_6_digital - 300 epochs per experiment"
echo "=================================================="
echo ""
echo "Loss coefficient values and their effects:"
echo "  0.00 -> multiplier = 1.00 (full student loss)"
echo "  0.01 -> multiplier = 0.99"
echo "  0.10 -> multiplier = 0.90"
echo "  0.50 -> multiplier = 0.50"
echo "  1.00 -> multiplier = 0.00 (no student loss)"
echo "  2.00 -> multiplier = -1.00 (negative loss)"
echo "  10.0 -> multiplier = -9.00 (strong negative)"
echo ""

# Create logs directory
mkdir -p /root/SALMON/logs/sweeps/loss_coefficient_tagged

# Function to format loss coefficient for tags (replace . with _)
format_loss_for_tag() {
    echo "$1" | sed 's/\./_/g'
}

# Choose mode
MODE=${1:-sequential}

if [ "$MODE" == "sequential" ]; then
    echo "Running SEQUENTIAL sweep with enhanced tagging..."
    echo ""
    
    screen -dmS loss_sweep_tagged bash -c "
        cd /root/SALMON
        export WANDB_API_KEY=2601d5461d965e275d39beb448abba392abb5e0b
        
        echo '=================================================='
        echo 'SEQUENTIAL SWEEP WITH ENHANCED TAGGING'
        echo 'Start time: \$(date)'
        echo '=================================================='
        
        # Loop through loss coefficient values
        for loss_coef in 0.0 0.01 0.1 0.5 1.0 2.0 10.0; do
            # Calculate multiplier for display
            multiplier=\$(echo \"scale=2; 1 - \$loss_coef\" | bc)
            
            # Format loss coefficient for tag (replace . with _)
            loss_tag=\$(echo \"\$loss_coef\" | sed 's/\./_/g')
            
            echo ''
            echo '=================================================='
            echo \"Experiment: loss_coefficient=\$loss_coef\"
            echo \"Student loss multiplier: \$multiplier\"
            echo \"Tag: loss_coef_\$loss_tag\"
            echo \"Start: \$(date)\"
            echo '=================================================='
            
            # Run experiment with proper tagging
            python src/train.py \
                experiment=exp7_6_digital \
                trainer=gpu \
                trainer.max_epochs=300 \
                trainer.min_epochs=100 \
                model.loss_coefficient=\$loss_coef \
                logger.wandb.group='loss_coef_sweep_tagged' \
                logger.wandb.project='salmon' \
                logger.wandb.entity='spk' \
                logger.wandb.tags='[\"cifar10\",\"Resnet18\",\"gradient_alignment\",\"IdealizedPreset\",\"Device2\",\"probe_analysis\",\"digital_module\",\"loss_sweep\",\"loss_coef_'\"\$loss_tag\"'\",\"multiplier_'\"\$multiplier\"'\"]' \
                hydra.run.dir='logs/sweeps/loss_coefficient_tagged/loss_\${loss_coef}' \
                2>&1 | tee logs/sweeps/loss_coefficient_tagged/loss_\${loss_coef}_\$(date +%Y%m%d_%H%M%S).log
            
            echo \"Completed: loss_coefficient=\$loss_coef\"
            echo '=================================================='
        done
        
        echo ''
        echo '=================================================='
        echo 'SEQUENTIAL SWEEP COMPLETED!'
        echo 'End time: \$(date)'
        echo '=================================================='
    "
    
    echo "✅ Sequential sweep started in screen: 'loss_sweep_tagged'"
    
elif [ "$MODE" == "parallel" ]; then
    echo "Running PARALLEL sweep with enhanced tagging..."
    echo "⚠️  Warning: This requires significant GPU memory!"
    echo ""
    
    # Start each experiment in its own screen
    for loss_coef in 0.0 0.01 0.1 0.5 1.0 2.0 10.0; do
        # Calculate multiplier
        multiplier=$(echo "scale=2; 1 - $loss_coef" | bc)
        
        # Format for session name and tag
        loss_formatted=$(format_loss_for_tag $loss_coef)
        session_name="loss_${loss_formatted}_tagged"
        
        echo "Starting: $session_name (loss=$loss_coef, mult=$multiplier)"
        
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
                logger.wandb.group='loss_coef_sweep_parallel_tagged' \
                logger.wandb.project='salmon' \
                logger.wandb.entity='spk' \
                logger.wandb.tags='[\"cifar10\",\"Resnet18\",\"gradient_alignment\",\"IdealizedPreset\",\"Device2\",\"probe_analysis\",\"digital_module\",\"loss_sweep\",\"loss_coef_${loss_formatted}\",\"multiplier_${multiplier}\"]' \
                hydra.run.dir='logs/sweeps/loss_coefficient_tagged/parallel_loss_${loss_coef}' \
                2>&1 | tee logs/sweeps/loss_coefficient_tagged/parallel_loss_${loss_coef}_\$(date +%Y%m%d_%H%M%S).log
            
            echo 'End: \$(date)'
        "
        
        sleep 5
    done
    
    echo "✅ All parallel sessions started!"
    
elif [ "$MODE" == "test" ]; then
    echo "Running TEST mode (1 epoch, 2 values)..."
    echo ""
    
    # Test with just 2 values and 1 epoch
    screen -dmS test_tags bash -c "
        cd /root/SALMON
        export WANDB_API_KEY=2601d5461d965e275d39beb448abba392abb5e0b
        
        echo 'TEST MODE - 1 epoch, 2 values'
        
        for loss_coef in 0.0 1.0; do
            multiplier=\$(echo \"scale=2; 1 - \$loss_coef\" | bc)
            loss_tag=\$(echo \"\$loss_coef\" | sed 's/\./_/g')
            
            echo \"Testing: loss_coef=\$loss_coef, tag=loss_coef_\$loss_tag\"
            
            python src/train.py \
                experiment=exp7_6_digital \
                trainer=gpu \
                trainer.max_epochs=1 \
                trainer.min_epochs=1 \
                model.loss_coefficient=\$loss_coef \
                logger.wandb.group='test_tags' \
                logger.wandb.project='salmon' \
                logger.wandb.entity='spk' \
                logger.wandb.tags='[\"test\",\"loss_coef_'\"\$loss_tag\"'\",\"multiplier_'\"\$multiplier\"'\"]' \
                2>&1 | tee logs/sweeps/loss_coefficient_tagged/test_\${loss_coef}.log
        done
        
        echo 'Test completed!'
    "
    
    echo "✅ Test started in screen: 'test_tags'"
    
else
    echo "Usage: $0 [sequential|parallel|test]"
    echo ""
    echo "  sequential - Run experiments one after another"
    echo "  parallel   - Run all experiments simultaneously"
    echo "  test       - Quick test with 1 epoch, 2 values"
    exit 1
fi

echo ""
echo "📊 Wandb Features:"
echo "  - Each run is tagged with its loss_coefficient value"
echo "  - loss_coefficient and multiplier are logged as config params"
echo "  - Easy filtering by tags: loss_coef_0_0, loss_coef_0_5, etc."
echo ""
echo "📊 Screen Management:"
echo "  - List sessions:    screen -ls"
echo "  - Attach:          screen -r <session_name>"
echo "  - Detach:          Ctrl+A then D"
echo "  - Kill:            screen -X -S <session_name> quit"
echo ""
echo "📁 Logs: /root/SALMON/logs/sweeps/loss_coefficient_tagged/"
echo ""
echo "🔗 Wandb:"
echo "  - Project: salmon"
echo "  - Entity: spk"
if [ "$MODE" == "sequential" ]; then
    echo "  - Group: loss_coef_sweep_tagged"
elif [ "$MODE" == "parallel" ]; then
    echo "  - Group: loss_coef_sweep_parallel_tagged"
else
    echo "  - Group: test_tags"
fi