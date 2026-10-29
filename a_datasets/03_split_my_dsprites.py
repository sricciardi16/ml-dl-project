"""
This script splits the custom `my_dsprites` dataset into training, validation,
and test sets using a stratified 80/10/10 split.

Label Roles:
    Primary Targets (Classification):
        - shape: The shape category (3 classes).
        - scale: The size category (3 classes).
        - shape_and_scale_labels: An engineered combined label (9 classes) that
          serves as the main classification target.

    Generative Factors (Factors of Variation):
        - orientation: The rotation angle of the shape.
        - pos_x: The x-coordinate position.
        - pos_y: The y-coordinate position.
        These factors determine where and how the images are generated and
        rendered spatially, but are not the primary targets for classification.
"""

import os
import random
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Insert project root into sys.path if not already present, allowing the script
# to run from any directory while still finding the `shared` module.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from sklearn.model_selection import train_test_split

from shared.paths import MY_DSPRITES, MY_DSPRITES_SPLIT

# Set seed to 29 (User's birth date) for strict determinism
os.environ['PYTHONHASHSEED'] = '29'
random.seed(29)
np.random.seed(29)

# Idempotency check: prevents redundant processing and accidental overwrites
if os.path.exists(MY_DSPRITES_SPLIT):
    print(f"Data split is already done. '{MY_DSPRITES_SPLIT}' exists.")
    sys.exit(0)
    
# Load custom dataset
data = np.load(MY_DSPRITES)

# Extract arrays - these 7 arrays will all be split in parallel
imgs = data['imgs']
shape_and_scale_labels = data['shape_and_scale_labels']
shape_labels = data['shape_labels']
scale_labels = data['scale_labels']
orientation_labels = data['orientation_labels']
pos_x_labels = data['pos_x_labels']
pos_y_labels = data['pos_y_labels']

# Create index array for memory-efficient splitting
indices = np.arange(len(imgs))

"""
First split: Train (80%) and Temp (20%)

We stratify using `shape_and_scale_labels` because it is our primary 9-class
classification target. Stratification guarantees that the exact class
distribution is preserved in both the training and temporary sets. This
prevents class imbalance and ensures the model learns from a representative
sample of each class.
"""
train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=29,  # 29 is the user's birth date
    stratify=shape_and_scale_labels
)

"""
Second split: Validation (10%) and Test (10%) from Temp (20%)

Splitting the 20% temporary set with test_size=0.5 yields two 10% sets
(Validation and Test). This split is also stratified on the combined labels
to ensure the validation and test sets perfectly reflect the training distribution.
"""
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    random_state=29,  # 29 is the user's birth date
    stratify=shape_and_scale_labels[temp_idx]
)

# Save all 21 resulting arrays (7 features/labels × 3 splits) into a single
# compressed archive for efficient loading during model training.
np.savez_compressed(
    MY_DSPRITES_SPLIT,
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
print(f"Data successfully split and saved to '{MY_DSPRITES_SPLIT}'.")