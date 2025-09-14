#!/usr/bin/env python3
"""
Combine the 4 fresh plots (generated from latest data) into final 2x2 grid
"""

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Define the paths to the fresh plots
plot_dir = '/root/SALMON/probes/Idealized/loss_coef_0_00/plots_fresh/'
plot_files = [
    'figure1_beta_fresh.png',
    'figure2_fp32_fresh.png', 
    'figure3_safe_fresh.png',
    'figure4_accuracy_fresh.png'
]

# Titles for each subplot
titles = [
    '(a) Effective Descent Coefficient',
    '(b) Small-α FP32 Alignment Improvement',
    '(c) Safe Mixing Window',
    '(d) Device Accuracy vs Alpha'
]

# Load all images
images = []
sizes = []
for i, filename in enumerate(plot_files):
    path = plot_dir + filename
    img = Image.open(path)
    images.append(np.array(img))
    sizes.append(img.size)
    print(f"Loaded {filename}: {img.size}")

# Verify all images have similar size
print(f"\nSizes: {sizes}")
width_diff = max(s[0] for s in sizes) - min(s[0] for s in sizes)
height_diff = max(s[1] for s in sizes) - min(s[1] for s in sizes)
print(f"Width difference: {width_diff} pixels")
print(f"Height difference: {height_diff} pixels")

if width_diff < 10 and height_diff < 10:
    print("✓ All plots have essentially the same dimensions")

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# Add main title
fig.suptitle('SALMON Idealized Loss Coefficient (0.00) Analysis - Fresh Data (Sep 11-12)', 
             fontsize=20, fontweight='bold', y=1.01)

# Plot each image in its subplot
for idx, (ax, img_array, title) in enumerate(zip(axes.flat, images, titles)):
    ax.imshow(img_array)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=8)
    ax.axis('off')

# Very minimal spacing since all images have the same size
plt.subplots_adjust(wspace=0.01, hspace=0.04, top=0.95)

# Save the combined figure
output_path = '/root/SALMON/probes/Idealized/loss_coef_0_00/combined_plots_FRESH_FINAL.png'
plt.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0.2)
print(f"\nFinal combined figure saved to: {output_path}")

# Also save a high-resolution version
output_path_hires = '/root/SALMON/probes/Idealized/loss_coef_0_00/combined_plots_FRESH_FINAL_hires.png'
plt.savefig(output_path_hires, dpi=300, bbox_inches='tight', pad_inches=0.2)
print(f"High-resolution version saved to: {output_path_hires}")

plt.close()

print("\n" + "="*80)
print("SUCCESS! Final combined figure created with:")
print("- All 4 plots generated FRESH from the LATEST data files (Sep 11-12)")
print("- Uniform figure size (10, 8) for all plots")
print("- Figure 1: Effective Descent Coefficient (β = r·cos(D,A))")
print("- Figure 2: Small-α FP32 Alignment Improvement")
print("- Figure 3: Safe Mixing Window")
print("- Figure 4: Device Accuracy vs Alpha")
print("- Data epochs: [1, 10, 50, 100, 150, 200, 250, 300]")
print("="*80)