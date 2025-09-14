# WandB Configuration Summary

## Configuration Changes Made

### 1. Created `wandb_metrics_only.yaml` Logger Configuration
- Located at: `/root/SALMON/configs/logger/wandb_metrics_only.yaml`
- **Key setting**: `log_model: False` - Prevents model checkpoint uploads to WandB
- **What it does**: Only sends metrics (loss, accuracy, etc.) to WandB
- **What it doesn't do**: Does NOT upload model checkpoints, artifacts, or code

### 2. Updated All Experiment Configurations
All exp7_6_digital experiment files now use `wandb_metrics_only` logger:
- exp7_6_digital_idealdevice_4heads.yaml
- exp7_6_digital_idealdevice_5heads.yaml
- exp7_6_digital_idealdevice_6heads.yaml
- exp7_6_digital_idealized_4heads.yaml
- exp7_6_digital_idealized_5heads.yaml
- exp7_6_digital_idealized_6heads.yaml
- exp7_6_digital_ecram_4heads.yaml
- exp7_6_digital_ecram_5heads.yaml
- exp7_6_digital_ecram_6heads.yaml

### 3. Local Model Checkpoints
- Models are still saved locally in: `${paths.output_dir}/checkpoints`
- Controlled by: `/root/SALMON/configs/callbacks/model_checkpoint.yaml`
- These local checkpoints are NOT uploaded to WandB

## What Gets Uploaded to WandB
✅ **Only these are uploaded:**
- Training metrics (loss, accuracy)
- Validation metrics
- Test metrics
- Learning rate schedules
- System metrics (GPU usage, etc.)
- Hyperparameters

❌ **These are NOT uploaded:**
- Model checkpoint files (.ckpt)
- Model artifacts
- Code files
- Dataset files

## How to Run

### Normal training with metrics-only WandB:
```bash
python src/train.py experiment=exp7_6_digital_idealdevice_4heads
```

### If WandB connection times out, run offline:
```bash
WANDB_MODE=offline python src/train.py experiment=exp7_6_digital_idealdevice_4heads
```

### To completely disable WandB and use local logging only:
```bash
python src/train.py experiment=exp7_6_digital_idealdevice_4heads_local
```

## Verification
To verify no models are being uploaded:
1. Check WandB project page - there should be no "Artifacts" tab
2. Check WandB run page - no model files should appear
3. Only metrics graphs and hyperparameters should be visible

## Rollback
If you want to enable model uploads again:
- Change `log_model: False` to `log_model: True` in `wandb_metrics_only.yaml`
- Or use the original `wandb.yaml` logger configuration