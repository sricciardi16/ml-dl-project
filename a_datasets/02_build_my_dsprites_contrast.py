"""
Contrast-Modified dSprites Dataset Generator

This script generates a contrast-modified version of the dSprites dataset by applying
different contrast transformations to the original binary images. The resulting dataset
contains 5 contrast classes, each representing a different visual appearance of the
same shapes.

Integer to Float Scale Mapping:
    The dataset uses a 0-20 integer scale which maps to a 0.0 to 1.0 float scale.
    Each integer unit equals 0.05 on the float scale (e.g., integer 10 = float 0.5).

Contrast Classes:
    The dataset contains 5 contrast classes with the following specifications:

    Class 0 - Standard Anchor:
        - Background (integer): 0
        - Foreground (integer): 20
        - Background (float): 0.0
        - Foreground (float): 1.0
        This is the standard high-contrast representation matching the original
        binary images (black background, white foreground).

    Class 1 - Faded Shift:
        - Background (integer): 8
        - Foreground (integer): 14
        - Background (float): 0.4
        - Foreground (float): 0.7
        A low-contrast representation with gray tones.

    Class 2 - Dark Inversion:
        - Background (integer): 16
        - Foreground (integer): 0
        - Background (float): 0.8
        - Foreground (float): 0.0
        An inverted representation with a dark gray background and black foreground.

    Class 3 - Misty Inversion:
        - Background (integer): 12
        - Foreground (integer): 4
        - Background (float): 0.6
        - Foreground (float): 0.2
        A misty inverted representation with medium gray tones.

    Class 4 - Extreme Edge Case:
        - Background (integer): 10
        - Foreground (integer): 11
        - Background (float): 0.5
        - Foreground (float): 0.55
        An extremely low-contrast edge case where foreground and background
        are nearly indistinguishable.

Output:
    The script produces a compressed numpy archive containing:
    - expanded_imgs: The contrast-modified images (5x the original count)
    - contrast_labels: Labels indicating the contrast class (0-4)
    - All original label arrays expanded to match the new image count
"""

import gc
import os
import random
import sys
import numpy as np

# Set seed to 29 (User's birth date) for strict determinism
os.environ['PYTHONHASHSEED'] = '29'
random.seed(29)
np.random.seed(29)

# Add project root to sys.path if not already present
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from shared.paths import MY_DSPRITES, MY_DSPRITES_CONTRAST

# Idempotency check
if os.path.exists(MY_DSPRITES_CONTRAST):
    print(f"File already exists at {MY_DSPRITES_CONTRAST}. Exiting.")
    sys.exit(0)

print("Starting data loading...")

# Load the original dataset
data = np.load(MY_DSPRITES)
imgs = data['imgs']
shape_and_scale_labels = data['shape_and_scale_labels']
shape_labels = data['shape_labels']
scale_labels = data['scale_labels']
orientation_labels = data['orientation_labels']
pos_x_labels = data['pos_x_labels']
pos_y_labels = data['pos_y_labels']

# Define contrast mappings: (background, foreground) integer values for each class
contrast_mappings = [
    (0, 20),   # Class 0: Standard Anchor
    (8, 14),   # Class 1: Faded Shift
    (16, 0),   # Class 2: Dark Inversion
    (12, 4),   # Class 3: Misty Inversion
    (10, 11),  # Class 4: Extreme Edge Case
]

print("Starting processing...")

# Pre-allocate arrays for memory efficiency
num_imgs = len(imgs)
num_classes = len(contrast_mappings)
expanded_imgs = np.empty((num_imgs * num_classes, 64, 64), dtype=np.uint8)
contrast_labels = np.empty(num_imgs * num_classes, dtype=np.int8)

# Process each contrast class
for class_idx, (bg, fg) in enumerate(contrast_mappings):
    start_idx = class_idx * num_imgs
    end_idx = start_idx + num_imgs
    # Replace 1s with fg value, 0s with bg value, explicitly cast to uint8
    expanded_imgs[start_idx:end_idx] = np.where(imgs == 1, np.uint8(fg), np.uint8(bg))
    contrast_labels[start_idx:end_idx] = class_idx

# Free memory by deleting original data
del imgs
del data
gc.collect()

# Duplicate existing labels 5 times to align with expanded images
expanded_shape_and_scale_labels = np.tile(shape_and_scale_labels, 5)
expanded_shape_labels = np.tile(shape_labels, 5)
expanded_scale_labels = np.tile(scale_labels, 5)
expanded_orientation_labels = np.tile(orientation_labels, 5)
expanded_pos_x_labels = np.tile(pos_x_labels, 5)
expanded_pos_y_labels = np.tile(pos_y_labels, 5)

print("Starting saving...")

# Save the new dataset
np.savez_compressed(
    MY_DSPRITES_CONTRAST,
    imgs=expanded_imgs,
    shape_and_scale_labels=expanded_shape_and_scale_labels,
    shape_labels=expanded_shape_labels,
    scale_labels=expanded_scale_labels,
    contrast_labels=contrast_labels,
    orientation_labels=expanded_orientation_labels,
    pos_x_labels=expanded_pos_x_labels,
    pos_y_labels=expanded_pos_y_labels
)

print("Successfully created contrast-modified dataset!")