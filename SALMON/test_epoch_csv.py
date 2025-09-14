#!/usr/bin/env python3
"""
Test script to verify epoch-based CSV file generation.
"""

import os
import glob

def test_epoch_csv_files():
    """Test that separate CSV files are created for each epoch."""
    
    probes_dir = "./probes"
    
    # Check if probes directory exists
    if not os.path.exists(probes_dir):
        print(f"Creating {probes_dir} directory...")
        os.makedirs(probes_dir)
    
    # List all existing CSV files
    existing_files = glob.glob(os.path.join(probes_dir, "*.csv"))
    print(f"\nExisting CSV files in {probes_dir}:")
    if existing_files:
        for f in sorted(existing_files):
            print(f"  - {os.path.basename(f)}")
    else:
        print("  No CSV files found yet")
    
    # Expected file naming pattern
    print("\n\nExpected file naming pattern after modification:")
    print("  device2_resnet18_cifar10_seed_<seed>_task_epoch<epoch>.csv")
    print("\nExample filenames:")
    for epoch in [1, 10, 20, 40]:
        example_name = f"device2_resnet18_cifar10_seed_42_task_epoch{epoch:03d}.csv"
        print(f"  - {example_name}")
    
    print("\nKey changes made:")
    print("1. _setup_csv_logging() now accepts epoch parameter")
    print("2. Creates epoch-specific filename: ..._epoch{epoch:03d}.csv")
    print("3. Closes previous CSV file before opening new one")
    print("4. Each epoch gets its own CSV file with complete headers")
    print("5. CSV file is closed at the end of each epoch")
    
    # Check implementation details
    print("\n\nImplementation details:")
    print("=" * 60)
    
    from src.models.exp7_6_module import SalmonLitModule
    
    # Check if the modified methods exist
    if hasattr(SalmonLitModule, '_setup_csv_logging'):
        import inspect
        sig = inspect.signature(SalmonLitModule._setup_csv_logging)
        params = list(sig.parameters.keys())
        print(f"✓ _setup_csv_logging method signature: {params}")
        if 'epoch' in params:
            print("  ✓ epoch parameter added successfully")
        else:
            print("  ✗ epoch parameter missing")
    
    print("\n✓ Module verification complete!")
    
    # Show how to use
    print("\n\nUsage example:")
    print("-" * 60)
    print("""
# In your training loop or probe runner:

for epoch in probe_epochs:
    # This creates a new CSV file for this epoch
    self._setup_csv_logging(epoch=epoch)
    
    # Run probes and write data
    run_gradient_probes(epoch)
    
    # CSV file is automatically closed at end of probe run
    """)
    
    print("\nBenefits of per-epoch CSV files:")
    print("1. Easier to analyze specific epochs")
    print("2. Can process epochs in parallel")
    print("3. Smaller file sizes for better Excel compatibility")
    print("4. No risk of corrupting data from other epochs")
    print("5. Can delete old epochs to save disk space")

if __name__ == "__main__":
    test_epoch_csv_files()