#!/usr/bin/env python3
"""
Update all experiment configurations to use wandb_metrics_only logger
"""

import os
from pathlib import Path

configs_dir = Path("/root/SALMON/configs/experiment")

# Files to update
files_to_update = [
    "exp7_6_digital_ecram_4heads.yaml",
    "exp7_6_digital_ecram_5heads.yaml",
    "exp7_6_digital_ecram_6heads.yaml",
    "exp7_6_digital_idealdevice_5heads.yaml",
    "exp7_6_digital_idealdevice_6heads.yaml",
    "exp7_6_digital_idealized_4heads.yaml",
    "exp7_6_digital_idealized_5heads.yaml",
    "exp7_6_digital_idealized_6heads.yaml",
]

for filename in files_to_update:
    filepath = configs_dir / filename
    if filepath.exists():
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Check if it already has a logger override
        if "- override /logger:" not in content:
            # Add logger override to defaults
            content = content.replace(
                "  - override /trainer: default",
                "  - override /trainer: default\n  - override /logger: wandb_metrics_only  # Only metrics, no model uploads"
            )
        else:
            # Update existing logger override
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if "- override /logger:" in line:
                    lines[i] = "  - override /logger: wandb_metrics_only  # Only metrics, no model uploads"
            content = '\n'.join(lines)
        
        # Remove any existing logger configuration section
        if "logger:" in content:
            lines = content.split('\n')
            new_lines = []
            skip_logger_section = False
            
            for line in lines:
                if line.strip().startswith("logger:") and not line.strip().startswith("logger:"):
                    skip_logger_section = True
                    new_lines.append("# Logger configuration handled by wandb_metrics_only.yaml")
                    continue
                
                if skip_logger_section:
                    # Check if we've reached a new top-level section
                    if line and not line.startswith(' ') and not line.startswith('\t'):
                        skip_logger_section = False
                        new_lines.append(line)
                    # Skip logger section lines
                    continue
                else:
                    new_lines.append(line)
            
            content = '\n'.join(new_lines)
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"✓ Updated {filename}")
    else:
        print(f"✗ File not found: {filename}")

print("\nAll configurations updated to use wandb_metrics_only logger!")
print("This ensures only metrics are sent to WandB, no model artifacts.")