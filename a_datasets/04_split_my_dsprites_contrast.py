"""
Script to split the contrast dataset into train/validation/test sets.

Creates an 80/10/10 stratified split of the my-dSprites-contrast dataset.
This split aligns perfectly with the standard dataset's split to prevent cross-dataset 
evaluation leakage during adversarial testing.

Split Structure:
    - Train: 80% of the dataset
    - Validation: 10% of the dataset
    - Test: 10% of the dataset

The split is stratified using a base-first splitting strategy on shape_and_scale_labels,
then expanded to include all contrast levels.

Label Roles:
    - shape_and_scale_labels: Composite label used for stratification
    - shape_labels: Shape category of each sample
    - scale_labels: Size category of each sample
    - contrast_labels: Contrast level of each sample
    - orientation_labels: Rotation angle category of each sample
    - pos_x_labels: X-position category of each sample
    - pos_y_labels: Y-position category of each sample

Generative Factors:
    - shape: 3 classes
    - scale: 6 classes
    - orientation: 40 classes
    - posX: 32 classes
    - posY: 32 classes
    - contrast: 5 classes
"""

import os
import sys
import random
import numpy as np
from sklearn.model_selection import train_test_split

# Set seed to 29 (User's birth date) for strict determinism
os.environ['PYTHONHASHSEED'] = '29'
random.seed(29)
np.random.seed(29)

# Add project root to sys.path if not already present
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from shared.paths import MY_DSPRITES_CONTRAST, MY_DSPRITES_CONTRAST_SPLIT

# Idempotency check: exit if split already exists
if os.path.exists(MY_DSPRITES_CONTRAST_SPLIT):
    print(f"Data split is already done. '{MY_DSPRITES_CONTRAST_SPLIT}' exists.")
    sys.exit(0)

# Load the dataset
data = np.load(MY_DSPRITES_CONTRAST)

# Extract the 8 arrays from the dataset
imgs = data['imgs']
shape_and_scale_labels = data['shape_and_scale_labels']
shape_labels = data['shape_labels']
scale_labels = data['scale_labels']
contrast_labels = data['contrast_labels']
orientation_labels = data['orientation_labels']
pos_x_labels = data['pos_x_labels']
pos_y_labels = data['pos_y_labels']

# Isolating the base indices and using a fixed seed (29) forces the contrast dataset
# to partition its underlying base images exactly as the standard dataset does.
# This ensures that the train/val/test splits contain the exact same base images
# across both datasets, preventing any cross-dataset evaluation leakage.
# Implicit assumption: the contrast dataset is structured as 5 contiguous blocks of N images.
# Block 0 (indices 0 to N-1) contains all Contrast 0 images, Block 1 (indices N to 2N-1) contains
# all Contrast 1 images, and so on, and the base images in each block are in the exact same order.
# Calculate the number of base images, denoted as N, by dividing the total length of the loaded images array by 5
N = len(imgs) // 5

# Create an array of base indices from 0 to N-1
# This creates an array of integers representing only the first block (Contrast 0),
# and slicing the labels up to N extracts the labels for just this first block to use as the stratification target.
base_indices = np.arange(N)

# Extract the first N elements of the shape_and_scale_labels array to serve as the base stratification key
base_stratify_key = shape_and_scale_labels[:N]

# First split: Train (80%) / Temp (20%) on base indices
train_base_idx, temp_base_idx = train_test_split(
    base_indices,
    test_size=0.2,
    random_state=29,
    stratify=base_stratify_key
)

# Second split: Validation (50% of temp) / Test (50% of temp) on base indices
val_base_idx, test_base_idx = train_test_split(
    temp_base_idx,
    test_size=0.5,
    random_state=29,
    stratify=base_stratify_key[temp_base_idx]
)

# Helper function to expand a given set of base indices.
# Expanding the base indices to include all 5 contrast variants ensures the standard split
# (which corresponds to Contrast 0) remains a perfect, mathematically aligned subset of
# the contrast split. This ensures downstream adversarial evaluation on the standard test set
# is valid and leak-free, as no base images from the test set will have their contrast
# variants present in the training or validation sets.
def expand_indices(base_idx, N):
    # i represents a specific base image index.
    # k represents the contrast multiplier (0 through 4).
    # k * N calculates the exact memory offset to jump to the next contrast block.
    # i + k * N grabs the exact same base image across all 5 contrast levels.
    expanded = [i + k * N for i in base_idx for k in range(5)]
    return np.array(expanded)

# Apply expansion to the train, val, and test base indices
train_idx = expand_indices(train_base_idx, N)
val_idx = expand_indices(val_base_idx, N)
test_idx = expand_indices(test_base_idx, N)

# Apply numpy.random.shuffle to each expanded index array to mix the contrast classes
# Because expand_indices appends the variants in a structured, repeating order, the resulting
# arrays are highly ordered, so the in-place shuffle is mathematically required to mix the
# contrast levels randomly before they are fed into a neural network.
np.random.shuffle(train_idx)
np.random.shuffle(val_idx)
np.random.shuffle(test_idx)

# Save the results (24 arrays, ignoring stratification arrays)
np.savez_compressed(
    MY_DSPRITES_CONTRAST_SPLIT,
    X_train=imgs[train_idx],
    X_val=imgs[val_idx],
    X_test=imgs[test_idx],
    y_shape_and_scale_train=shape_and_scale_labels[train_idx],
    y_shape_and_scale_val=shape_and_scale_labels[val_idx],
    y_shape_and_scale_test=shape_and_scale_labels[test_idx],
    y_shape_train=shape_labels[train_idx],
    y_shape_val=shape_labels[val_idx],
    y_shape_test=shape_labels[test_idx],
    y_scale_train=scale_labels[train_idx],
    y_scale_val=scale_labels[val_idx],
    y_scale_test=scale_labels[test_idx],
    y_contrast_train=contrast_labels[train_idx],
    y_contrast_val=contrast_labels[val_idx],
    y_contrast_test=contrast_labels[test_idx],
    y_orientation_train=orientation_labels[train_idx],
    y_orientation_val=orientation_labels[val_idx],
    y_orientation_test=orientation_labels[test_idx],
    y_pos_x_train=pos_x_labels[train_idx],
    y_pos_x_val=pos_x_labels[val_idx],
    y_pos_x_test=pos_x_labels[test_idx],
    y_pos_y_train=pos_y_labels[train_idx],
    y_pos_y_val=pos_y_labels[val_idx],
    y_pos_y_test=pos_y_labels[test_idx]
)

print("Successfully created 80/10/10 split and saved to MY_DSPRITES_CONTRAST_SPLIT")