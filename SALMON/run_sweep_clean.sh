#!/bin/bash

# Clean sweep script for loss_coefficient parameter with proper tagging
# Runs 300 epochs for each value with wandb tracking

echo "=================================================="
echo "LOSS COEFFICIENT SWEEP - CLEAN VERSION"
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
mkdir -p /root/SALMON/logs/sweeps/loss_coefficient_clean

# Choose mode
MODE=${1:-sequential}

if [ "$MODE" == "sequential" ]; then
    echo "Running SEQUENTIAL sweep..."
    echo ""
    
    screen -dmS loss_sweep_clean bash -c "
        cd /root/SALMON
        export WANDB_API_KEY=2601d5461d965e275d39beb448abba392abb5e0b
        
        echo '=================================================='
        echo 'SEQUENTIAL SWEEP - CLEAN VERSION'
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
            echo \"Start: \$(date)\"
            echo '=================================================='
            
            # Run experiment with proper tagging via tags override
            python src/train.py \
                experiment=exp7_6_digital \
                trainer=gpu \
                trainer.max_epochs=300 \
                trainer.min_epochs=100 \
                model.loss_coefficient=\$loss_coef \
                logger.wandb.group='loss_coef_sweep' \
                logger.wandb.project='salmon' \
                logger.wandb.entity='spk' \
                'tags=[\"cifar10\",\"Resnet18\",\"gradient_alignment\",\"IdealizedPreset\",\"Device2\",\"probe_analysis\",\"digital_module\",\"loss_sweep\",\"loss_coef_'\"\$loss_tag\"'\"]' \
                hydra.run.dir=\"logs/sweeps/loss_coefficient_clean/loss_\$loss_coef\" \
                2>&1 | tee logs/sweeps/loss_coefficient_clean/loss_\${loss_coef}_\$(date +%Y%m%d_%H%M%S).log
            
            echo \"Completed: loss_coefficient=\$loss_coef\"
            echo '=================================================='
        done
        
        echo ''
        echo '=================================================='
        echo 'SEQUENTIAL SWEEP COMPLETED!'
        echo 'End time: \$(date)'
        echo '=================================================='
    "
    
    echo "✅ Sequential sweep started in screen: 'loss_sweep_clean'"
    
elif [ "$MODE" == "test" ]; then
    echo "Running TEST mode (1 epoch, 2 values)..."
    echo ""
    
    # Test with just 2 values and 1 epoch
    screen -dmS test_clean bash -c "
        cd /root/SALMON
        export WANDB_API_KEY=2601d5461d965e275d39beb448abba392abb5e0b
        
        echo 'TEST MODE - 1 epoch, 2 values'
        
        for loss_coef in 0.0 1.0; do
            multiplier=\$(echo \"scale=2; 1 - \$loss_coef\" | bc)
            loss_tag=\$(echo \"\$loss_coef\" | sed 's/\./_/g')
            
            echo \"Testing: loss_coef=\$loss_coef\"
            
            python src/train.py \
                experiment=exp7_6_digital \
                trainer=gpu \
                trainer.max_epochs=1 \
                trainer.min_epochs=1 \
                model.loss_coefficient=\$loss_coef \
                logger.wandb.group='test_clean' \
                logger.wandb.project='salmon' \
                logger.wandb.entity='spk' \
                'tags=[\"test\",\"loss_coef_'\"\$loss_tag\"'\"]' \
                2>&1 | tee logs/sweeps/loss_coefficient_clean/test_\${loss_coef}.log
        done
        
        echo 'Test completed!'
    "
    
    echo "✅ Test started in screen: 'test_clean'"
    
else
    echo "Usage: $0 [sequential|test]"
    echo ""
    echo "  sequential - Run experiments one after another"
    echo "  test       - Quick test with 1 epoch, 2 values"
    exit 1
fi

echo ""
echo "📊 Screen Management:"
echo "  - List sessions:    screen -ls"
echo "  - Attach:          screen -r <session_name>"
echo "  - Detach:          Ctrl+A then D"
echo "  - Kill:            screen -X -S <session_name> quit"
echo ""
echo "📁 Logs: /root/SALMON/logs/sweeps/loss_coefficient_clean/"
echo ""
echo "🔗 Wandb:"
echo "  - Project: salmon"
echo "  - Entity: spk"
if [ "$MODE" == "sequential" ]; then
    echo "  - Group: loss_coef_sweep"
    echo "  - Names: loss_coef_0_0, loss_coef_0_5, etc."
else
    echo "  - Group: test_clean"
fi