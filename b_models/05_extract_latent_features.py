"""
Script to extract latent features from trained models on the dSprites dataset.
The script processes both single-task and multi-task model configurations,
extracts features from the layer before the final dense layer (trunk),
and saves them as compressed numpy arrays for later analysis.
"""

import os
import sys
import gc
import numpy as np
import tensorflow as tf

# Add project root to sys.path to allow imports from shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.paths import MODELS_LATENT_DIR, MY_DSPRITES_SPLIT, get_model_paths, get_latent_features_path

# Ensure the directory for saving latent features exists
os.makedirs(MODELS_LATENT_DIR, exist_ok=True)

# Load the standard dataset
with np.load(MY_DSPRITES_SPLIT) as dataset:
    # X_train = dataset['X_train']
    # X_val = dataset['X_val']
    X_test = dataset['X_test']

# Cast to float32 and add channel dimension for compatibility with CNN models
# X_train = np.expand_dims(X_train.astype(np.float32), axis=-1)
# X_val = np.expand_dims(X_val.astype(np.float32), axis=-1)
X_test = np.expand_dims(X_test.astype(np.float32), axis=-1)

# Note: No division by 20.0 is performed, even for contrast models,
# because the standard dataset values are already in the expected float interval [0, 1].

# Define helper function for batch prediction
def predict_in_batches(model, X, batch_size=4096):
    """Predict features in batches to handle large datasets efficiently."""
    features = []
    num_samples = X.shape[0]
    
    for i in range(0, num_samples, batch_size):
        batch = X[i : i + batch_size]
        batch_preds = model.predict_on_batch(batch)
        
        # Convert TensorFlow tensor to NumPy array if needed
        if hasattr(batch_preds, 'numpy'):
            batch_preds = batch_preds.numpy()
        
        features.append(batch_preds)
    
    return np.concatenate(features, axis=0)

# Define all model configurations to process
configurations = []

# Single-task models: dataset x pool_type combinations
for dataset_type in ['standard', 'contrast']:
    for pool_type in ['flatten', 'gap']:
        configurations.append({
            'task': 'single',
            'dataset': dataset_type,
            'pool_type': pool_type,
            'loss_weight': None
        })

# Multi-task models: dataset x pool_type x loss_weight combinations
for dataset_type in ['standard', 'contrast']:
    for pool_type in ['flatten', 'gap']:
        for loss_weight in [0.5, 1.0, 2.0]:
            configurations.append({
                'task': 'multi',
                'dataset': dataset_type,
                'pool_type': pool_type,
                'loss_weight': loss_weight
            })

print(f"Processing {len(configurations)} model configurations...")

# Process each configuration
for config in configurations:
    task = config['task']
    dataset_type = config['dataset']
    pool_type = config['pool_type']
    loss_weight = config['loss_weight']
    
    # Get target save path for latent features
    save_path = get_latent_features_path(task, dataset_type, pool_type, loss_weight)
    
    # Skip if features already exist
    if os.path.exists(save_path):
        print(f"Skipping {task}/{dataset_type}/{pool_type}/loss_weight={loss_weight}: features already exist at {save_path}")
        continue
    
    print(f"Processing {task}/{dataset_type}/{pool_type}/loss_weight={loss_weight}...")
    
    # Get model path and load the trained model
    model_path, _, _ = get_model_paths(task, dataset_type, pool_type, loss_weight)
    model = tf.keras.models.load_model(model_path)
    
    # Find the trunk layer (layer before final dense layer)
    trunk_layer = None
    for layer in model.layers:
        layer_name = layer.name.lower()
        if 'flatten' in layer_name or 'global_average_pooling2d' in layer_name:
            trunk_layer = layer
            break
    
    if trunk_layer is None:
        print(f"Warning: Could not find trunk layer in model {model_path}. Skipping.")
        continue
    
    # Create trunk model that outputs features from the trunk layer
    trunk_model = tf.keras.Model(inputs=model.inputs, outputs=trunk_layer.output)
    
    # Extract latent features for all splits using batch prediction
    # print(f"  Extracting features for training set...")
    # latent_train = predict_in_batches(trunk_model, X_train, batch_size=4096)
    
    # print(f"  Extracting features for validation set...")
    # latent_val = predict_in_batches(trunk_model, X_val, batch_size=4096)
    
    print(f"  Extracting features for test set...")
    latent_test = predict_in_batches(trunk_model, X_test, batch_size=4096)
    
    # Save compressed numpy arrays
    np.savez_compressed(
        save_path,
        # latent_train=latent_train,
        # latent_val=latent_val,
        latent_test=latent_test
    )
    
    print(f"  Saved latent features to {save_path}")
    
    # Clean up to free memory
    del trunk_model
    del model
    tf.keras.backend.clear_session()
    gc.collect()

print("All configurations processed.")