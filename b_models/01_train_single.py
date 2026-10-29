"""
Single-task Baseline Model Training Script

This script trains a baseline Convolutional Neural Network (CNN) on the dSprites dataset 
to classify 9 combinations of shape and scale.

Imports explanation:
- os: Standard library module for interacting with the operating system, used here for path manipulation and directory creation.
- sys: Standard library module providing access to system-specific parameters and functions, used here to modify the import path and exit the script.
- argparse: Standard library module for parsing command-line arguments, allowing the user to configure pool type and dataset options.
- numpy (np): Fundamental package for scientific computing in Python, used for loading datasets, manipulating arrays, and saving training history.
- tensorflow.keras: High-level neural networks API, used for defining the CNN architecture (Sequential, Conv2D, MaxPooling2D, etc.), compiling, and training the model.
- shared.paths: Custom module containing predefined file paths for datasets and output artifacts to ensure consistency across the project.
"""

import os
import sys
import argparse
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
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

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a single-output CNN on the dSprites dataset.")
parser.add_argument('--pool_type', type=str, choices=['flatten', 'gap'], default='flatten', help="Type of pooling layer")
parser.add_argument('--dataset', type=str, choices=['standard', 'contrast'], default='standard', help="Path to the dataset")
args = parser.parse_args()

# Log startup and configuration
print(f"--- Starting Training Script: {os.path.basename(__file__)}")
print(f"Configuration: Pool Type = {args.pool_type}, Dataset = {args.dataset}")

# Resolve output file paths based on configuration
model_filepath, history_filepath, dot_filepath = get_model_paths('single', args.dataset, args.pool_type)

# Consolidated idempotency check
if os.path.exists(model_filepath) and os.path.exists(history_filepath) and os.path.exists(dot_filepath):
    print("All output artifacts (model, history, architecture plot) already exist. Exiting gracefully.")
    sys.exit(0)

os.makedirs(os.path.dirname(model_filepath), exist_ok=True)

# Define the model architecture
# We construct a Sequential CNN model to extract hierarchical features from the input images.
layers = [
    # Input layer specifying the shape of the input images (64x64 pixels, 1 channel).
    Input(shape=(64, 64, 1)),
    
    # Following the original formulation of Ioffe & Szegedy (2015), Batch Normalization should be applied before the non-linearity (ReLU). Applying BN after ReLU creates the 'ReLU paradox': ReLU zeroes out negative values to ensure sparse activation, but the subsequent BN shifts and rescales the distribution to bring the mean back to 0 and the variance to 1, artificially recreating negative values and destroying the sparsity effect. The Conv -> BN -> ReLU order guarantees mathematical consistency and better gradient stability.
    # First Convolutional block:
    # Conv2D extracts spatial features using 32 filters of size 3x3. ReLU activation introduces non-linearity.
    Conv2D(32, (3, 3), padding='same'),
    # BatchNormalization normalizes the activations of the previous layer, which stabilizes and accelerates the learning process.
    BatchNormalization(),
    Activation('relu'),
    # MaxPooling2D downsamples the feature maps by taking the maximum value in 2x2 windows, reducing spatial dimensions and computational complexity.
    MaxPooling2D((2, 2)),
    
    # Second Convolutional block:
    # Conv2D extracts higher-level spatial features using 64 filters.
    Conv2D(64, (3, 3), padding='same'),
    # BatchNormalization stabilizes learning by reducing internal covariate shift.
    BatchNormalization(),
    Activation('relu'),
    # MaxPooling2D reduces spatial dimensions further.
    MaxPooling2D((2, 2)),
    
    # Third Convolutional block:
    # Conv2D extracts complex spatial features using 128 filters.
    Conv2D(128, (3, 3), padding='same'),
    # BatchNormalization stabilizes learning.
    BatchNormalization(),
    Activation('relu'),
    # MaxPooling2D reduces spatial dimensions to prepare for pooling/flattening.
    MaxPooling2D((2, 2))
]

if args.pool_type == 'gap':
    layers.append(GlobalAveragePooling2D())
else:
    # Flatten converts the 2D feature maps into a 1D vector
    layers.append(Flatten())
    # Dropout is only applied to the flattened representation
    layers.append(Dropout(0.5))

layers.extend([
    Dense(128, activation='relu'),
    # The final Dense(9) outputs raw linear logits across the 9 target classes.
    Dense(9)
])

model = Sequential(layers)

# Generate and save the model architecture plot if it does not exist
if not os.path.exists(dot_filepath):
    print(f"Saving model architecture visualization to {dot_filepath}")
    from tensorflow.keras.utils import model_to_dot
    dot_obj = model_to_dot(model, show_shapes=True, show_layer_activations=True, show_dtype=True, show_layer_names=True)
    dot_obj.write(dot_filepath)

# Load the dataset
if args.dataset == 'standard':
    dataset_path = MY_DSPRITES_SPLIT
else:
    dataset_path = MY_DSPRITES_CONTRAST_SPLIT

print(f"Loading dataset from {dataset_path}")
with np.load(dataset_path) as data:
    X_train = data['X_train']
    X_val = data['X_val']
    y_train = data['y_shape_and_scale_train']
    y_val = data['y_shape_and_scale_val']

# Preprocess the image data
print("Preprocessing data (casting to float32, normalizing, adding channel dimension)...")

# np.expand_dims is used to add a channel dimension at the end of the shape (axis=-1).
# Keras Conv2D layers expect input tensors to have a shape of (batch_size, height, width, channels).
# Since our raw images are grayscale and only have height and width, we must explicitly add the channel dimension.
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = ['accuracy']
patience = 10
epochs = 50
# "Friends don't let friends use minibatches larger than 32" - Yann LeCun
batch_size = 32

def preprocess_data(x, y):
    x = tf.cast(x, tf.float32)
    if args.dataset == 'contrast':
        x = x / 20.0
    return x, y

# Create training and validation datasets using tf.data API
train_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(buffer_size=len(X_train), seed=29)
    .map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

# The model is configured to return raw linear logits (from_logits=True) rather than post-softmax probabilities. This is TensorFlow's official recommendation to prevent numerical instability and overflow/underflow issues during cross-entropy calculation. Moreover, in Adversarial Machine Learning, using logits avoids the 'softmax saturation' phenomenon (where probabilities close to 1.0 cause gradient vanishing, making FGSM/PGD attacks blind or ineffective). Working with logits ensures consistently strong gradients and mathematically rigorous adversarial perturbation computation.
# Compile the model
# sparse_categorical_crossentropy is used because labels are integers, not one-hot encoded vectors.
# adam optimizer is an adaptive learning rate optimization algorithm.
model.compile(
    optimizer=optimizer,
    loss=loss_function,
    metrics=metrics
)

print("\n--- Model Architecture ---")
model.summary()

# Define callbacks
# EarlyStopping is a regularization technique used to prevent overfitting.
# It monitors the validation loss and stops the training process once the validation loss stops improving
# for a specified number of epochs (patience). It also restores the weights of the best performing epoch.
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=patience,
    restore_best_weights=True,
    min_delta=1e-8
)

# ModelCheckpoint is used to save the model at various intervals during training.
# Here, it is configured to save the model to disk only when a new minimum validation loss is achieved,
# ensuring that we do not lose the best-performing model weights if training continues past the optimal point.
model_checkpoint = ModelCheckpoint(
    filepath=model_filepath,
    monitor='val_loss',
    save_best_only=True
)

# ReduceLROnPlateau callback reduces the learning rate when a metric has stopped improving.
# This helps the model converge better by taking smaller steps when nearing a minimum.
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# Detailed print block listing the training configuration
print("\n--- Training Configuration ---")
print(f"Optimizer: {optimizer}")
print(f"Loss Function: {loss_function}")
print(f"Metrics: {metrics}")
print(f"Epochs: {epochs}")
print(f"Batch Size: {batch_size}")
print(f"Early Stopping Patience: {patience}")
print("------------------------------\n")

print("Starting model training...")

# Train the model
# The fit method trains the model for a fixed number of epochs (iterations on a dataset).
# epochs defines the maximum number of times the learning algorithm will work through the entire training dataset.
history = model.fit(
    train_dataset, 
    validation_data=val_dataset,
    epochs=epochs,  # Set to a high number, relying on EarlyStopping to halt training
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

# Save the training history
# history.history is a dictionary of metrics saved as a .npy file for later analysis.
print(f"Saving training history to {history_filepath}")
np.save(history_filepath, history.history)
print(f"Model and history successfully saved to {os.path.dirname(model_filepath)}")

print("Training script completed successfully.")