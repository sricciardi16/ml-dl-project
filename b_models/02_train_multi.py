"""
Multi-task Baseline Model Training Script

This script trains a multi-output Convolutional Neural Network (CNN) on the dSprites dataset 
using the Keras Functional API. The multi-task architecture is designed to predict three 
related properties of an image simultaneously:
1. A combined 'shape and scale' class (9 classes).
2. The 'shape' class (3 classes).
3. The 'scale' class (3 classes).

Imports:
- argparse: For parsing command-line arguments (pool_type, dataset).
- numpy: For data manipulation and loading/saving arrays.
- os, sys: For path manipulation and system operations.
- tensorflow.keras: For building, compiling, and training the deep learning model.
- shared.paths: For retrieving standardized file paths for datasets and output artifacts.

Multi-Task Architecture:
- Shared Trunk: The model begins with a shared convolutional trunk consisting of three blocks of 
  Conv2D and MaxPooling2D layers. This shared representation learns common features useful for all tasks.
- Branching Logic: After the shared trunk, the network splits into three independent branches.
  Each branch has its own Dense layer (128 units, ReLU) and a final Dense layer outputting raw logits
  corresponding to the number of classes for that specific task.
- Conditional Dropout: Dropout (0.5) is conditionally applied to the trunk output for each branch 
  only when 'flatten' pooling is used. This helps prevent overfitting in the high-dimensional flattened space, 
  whereas Global Average Pooling (GAP) inherently reduces dimensionality and acts as a regularizer, 
  making Dropout less critical.
"""

import argparse
import numpy as np
import os
import random
import sys
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    GlobalAveragePooling2D,
    BatchNormalization,
    Dropout,
    Activation,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import model_to_dot

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from shared.paths import (
    MY_DSPRITES_SPLIT,
    MY_DSPRITES_CONTRAST_SPLIT,
    get_model_paths,
)

# Set global random seed to 29 (User's birth date) for reproducibility
os.environ['PYTHONHASHSEED'] = '29'
random.seed(29)
np.random.seed(29)
tf.random.set_seed(29)
tf.keras.utils.set_random_seed(29)

parser = argparse.ArgumentParser(description="Train a multi-output CNN on the dSprites dataset.")
parser.add_argument('--pool_type', type=str, choices=['flatten', 'gap'], default='flatten',
                    help="Type of pooling to use after the shared trunk.")
parser.add_argument('--dataset', type=str, choices=['standard', 'contrast'], default='standard',
                    help="Dataset to use for training.")
parser.add_argument('--loss_weight', type=float, choices=[0.5, 1.0, 2.0], default=1.0,
                    help="Controls the loss weight of the shape_and_scale branch.")
args = parser.parse_args()

# Startup print statements
print(f"--- Starting Training Script: {os.path.basename(__file__)}")
print(f"Configuration: Pool Type = {args.pool_type}, Dataset = {args.dataset}")

# Resolve paths based on dataset and pool_type
model_filepath, history_filepath, dot_filepath = get_model_paths('multi', args.dataset, args.pool_type, args.loss_weight)

# Ensure the output directory exists
os.makedirs(os.path.dirname(model_filepath), exist_ok=True)

# Single stage idempotency check immediately after path resolution
if os.path.exists(model_filepath) and os.path.exists(history_filepath) and os.path.exists(dot_filepath):
    print("All output artifacts (model, history, architecture plot) already exist. Exiting gracefully.")
    sys.exit(0)

# Loading the dataset
dataset_path = MY_DSPRITES_SPLIT if args.dataset == 'standard' else MY_DSPRITES_CONTRAST_SPLIT
print(f"Loading dataset from {dataset_path}")

with np.load(dataset_path) as data:
    # Extract the training and validation images as uint8
    X_train = data['X_train']
    X_val = data['X_val']

    # Extract the three separate label arrays for multi-task learning:
    # 1. y_shape_and_scale: Combined target representing both shape and scale (9 classes)
    # 2. y_shape: Target representing shape only (3 classes)
    # 3. y_scale: Target representing scale only (3 classes)
    y_shape_and_scale_train = data['y_shape_and_scale_train']
    y_shape_train = data['y_shape_train']
    y_scale_train = data['y_scale_train']

    y_shape_and_scale_val = data['y_shape_and_scale_val']
    y_shape_val = data['y_shape_val']
    y_scale_val = data['y_scale_val']

# Preprocessing data (adding channel dimension)...
print("Preprocessing data (adding channel dimension)...")

# Add a channel dimension (from [N, H, W] to [N, H, W, 1]) as required by Conv2D layers in Keras
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)

def preprocess_multi(x, y):
    x = tf.cast(x, tf.float32)
    if args.dataset == 'contrast':
        x = x / 20.0
    return x, y

# Defining the model architecture using the Keras Functional API.
# The Functional API is ideal here because it allows us to define a model with multiple outputs
# branching from a shared feature extractor (trunk).
inputs = Input(shape=(64, 64, 1))

# Following the original formulation of Ioffe & Szegedy (2015), Batch Normalization should be applied before the non-linearity (ReLU). Applying BN after ReLU creates the 'ReLU paradox': ReLU zeroes out negative values to ensure sparse activation, but the subsequent BN shifts and rescales the distribution to bring the mean back to 0 and the variance to 1, artificially recreating negative values and destroying the sparsity effect. The Conv -> BN -> ReLU order guarantees mathematical consistency and better gradient stability.
# Shared Trunk:
# A series of Conv2D, BatchNormalization, and MaxPooling2D layers.
# This trunk extracts general visual features (edges, textures, shapes) that are shared across all tasks.
# MaxPooling2D is used to progressively reduce the spatial dimensions of the feature maps,
# which decreases computational load, reduces the number of parameters, and helps achieve spatial invariance.
x = Conv2D(32, kernel_size=(3, 3), padding='same')(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(128, kernel_size=(3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Pooling layer:
# We either flatten the 3D feature maps to a 1D vector or use Global Average Pooling (GAP).
# GAP averages each feature map, drastically reducing the parameter count and acting as a regularizer.
if args.pool_type == 'gap':
    trunk_output = GlobalAveragePooling2D()(x)
else:
    trunk_output = Flatten()(x)

# Branching Logic:
# From the trunk output, we split into three separate branches, one for each task.
# Each branch has its own Dense layer (128 units, ReLU) and a final Dense layer with softmax activation.
#
# Conditional Dropout:
# Dropout (0.5) is applied to the trunk output for each branch only when 'flatten' pooling is used.
# Flattening results in a very high-dimensional vector, making the subsequent Dense layers highly prone to overfitting.
# Dropout randomly deactivates units during training to improve generalization.
# When using GAP, the feature vector is already small, so Dropout is not applied.

# Branch 1 (Main Task: Shape and Scale)
if args.pool_type == 'flatten':
    branch1_input = Dropout(0.5)(trunk_output)
else:
    branch1_input = trunk_output
branch1 = Dense(128, activation='relu')(branch1_input)
out_main = Dense(9, name='shape_and_scale')(branch1)

# Branch 2 (Auxiliary Task: Shape)
if args.pool_type == 'flatten':
    branch2_input = Dropout(0.5)(trunk_output)
else:
    branch2_input = trunk_output
branch2 = Dense(128, activation='relu')(branch2_input)
out_shape = Dense(3, name='shape')(branch2)

# Branch 3 (Auxiliary Task: Scale)
if args.pool_type == 'flatten':
    branch3_input = Dropout(0.5)(trunk_output)
else:
    branch3_input = trunk_output
branch3 = Dense(128, activation='relu')(branch3_input)
out_scale = Dense(3, name='scale')(branch3)

# Instantiate the model with one input and three outputs
model = Model(inputs=inputs, outputs=[out_main, out_shape, out_scale])

# Print model summary
print("\n--- Model Architecture ---")
model.summary()

# Generate architecture dot file if missing
if not os.path.exists(dot_filepath):
    print(f"Saving model architecture visualization to {dot_filepath}")
    dot = model_to_dot(model, show_shapes=True, show_layer_names=True, show_layer_activations=True, show_dtype=True)
    dot.write(dot_filepath)

# Define explicit variables for compilation and training
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
# The model is configured to return raw linear logits (from_logits=True) rather than post-softmax probabilities. This is TensorFlow's official recommendation to prevent numerical instability and overflow/underflow issues during cross-entropy calculation. Moreover, in Adversarial Machine Learning, using logits avoids the 'softmax saturation' phenomenon (where probabilities close to 1.0 cause gradient vanishing, making FGSM/PGD attacks blind or ineffective). Working with logits ensures consistently strong gradients and mathematically rigorous adversarial perturbation computation.
loss_functions = {
    'shape_and_scale': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    'shape': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    'scale': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
}
loss_weights = {
    'shape_and_scale': args.loss_weight,
    'shape': 1.0,
    'scale': 1.0
}
metrics = {
    'shape_and_scale': ['accuracy'],
    'shape': ['accuracy'],
    'scale': ['accuracy']
}
patience = 10
epochs = 50
# "Friends don't let friends use minibatches larger than 32" - Yann LeCun
batch_size = 32

# Compiling the model:
# We use a dictionary to specify the loss function for each output branch.
# Since we have multiple outputs, we can also specify loss weights.
# Here, all tasks are weighted equally with a weight of 1.0, meaning the total loss
# is the simple sum of the individual losses.
model.compile(
    optimizer=optimizer,
    loss=loss_functions,
    loss_weights=loss_weights,
    metrics=metrics
)

# Configuring callbacks
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, min_delta=1e-8),
    ModelCheckpoint(filepath=model_filepath, monitor='val_loss', save_best_only=True),
    reduce_lr
]

# Detailed print block listing the training configuration
print("\n--- Training Configuration ---")
print(f"Optimizer: {optimizer}")
print(f"Loss Functions: {loss_functions}")
print(f"Loss Weights: {loss_weights}")
print(f"Metrics: {metrics}")
print(f"Epochs: {epochs}")
print(f"Batch Size: {batch_size}")
print(f"Early Stopping Patience: {patience}")
print(f"Model Checkpoint: Saving best model to {model_filepath}")
print("------------------------------\n")

# Group the training labels into a dictionary
y_train_dict = {
    'shape_and_scale': y_shape_and_scale_train,
    'shape': y_shape_train,
    'scale': y_scale_train
}

# Create a similar dictionary for the validation labels
y_val_dict = {
    'shape_and_scale': y_shape_and_scale_val,
    'shape': y_shape_val,
    'scale': y_scale_val
}

# Create training and validation datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_dict)) \
    .shuffle(buffer_size=len(X_train), seed=29) \
    .map(preprocess_multi, num_parallel_calls=tf.data.AUTOTUNE) \
    .batch(batch_size) \
    .prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val_dict)) \
    .map(preprocess_multi, num_parallel_calls=tf.data.AUTOTUNE) \
    .batch(batch_size) \
    .prefetch(tf.data.AUTOTUNE)

print("\nStarting model training...")

# Training the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=callbacks
)

# Saving the history
print(f"Saving training history to {history_filepath}")
np.save(history_filepath, history.history)
print(f"Model and history successfully saved to {os.path.dirname(model_filepath)}")

print("Training script completed successfully.")