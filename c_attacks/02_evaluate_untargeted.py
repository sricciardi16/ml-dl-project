"""
Script to evaluate adversarial attacks on a multi-output model.

This script loads a pre-trained multi-output model and evaluates its performance
on adversarial examples generated using different attack methods (FGSM, PGD)
under different norm constraints (L-inf, L2).

The model has three outputs:
1. Main output: Combined shape and scale prediction (shape * 3 + scale)
2. Shape output: Shape classification
3. Scale output: Scale classification

This script only saves raw per-sample predictions to a CSV file.
"""

import os
import sys

os.environ['PYTHONHASHSEED'] = '29'

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import random
import numpy as np
import tensorflow as tf

# Set global random seed to 29 (User's birth date) for reproducibility
random.seed(29)
np.random.seed(29)
tf.random.set_seed(29)
tf.keras.utils.set_random_seed(29)

import argparse
import pandas as pd
from tensorflow.keras.models import load_model
from shared.paths import (
    get_untargeted_attack_path,
    get_attack_eval_paths,
    get_model_paths
)

# Set up argument parser to accept command line arguments
parser = argparse.ArgumentParser(description='Evaluate adversarial attacks on multi-output model')
parser.add_argument('--pool_type', type=str, choices=['flatten', 'gap'],
                    required=True, help='Pooling type used in the model')
parser.add_argument('--dataset', type=str, choices=['standard', 'contrast'],
                    default='standard', help='Dataset type to evaluate')
parser.add_argument('--loss_weight', type=float, choices=[0.5, 1.0, 2.0],
                    default=1.0, help='Loss weight used in the model')

args = parser.parse_args()

# Determine the model path based on dataset and pool_type
model_path, _, _ = get_model_paths('multi', args.dataset, args.pool_type, args.loss_weight)

print(f"Loading model from: {model_path}")
model = load_model(model_path)
print(f"Model loaded successfully. Model has {len(model.outputs)} outputs.")

# Define the attack configurations
configs = [
    ('fgsm', 'linf', np.inf, [0.01, 0.05, 0.1, 0.15, 0.2]),
    ('fgsm', 'l2', 2, [0.1, 0.5, 1.0, 2.0, 3.0]),
    ('pgd', 'linf', np.inf, [0.01, 0.05, 0.1, 0.15, 0.2]),
    ('pgd', 'l2', 2, [0.1, 0.5, 1.0, 2.0, 3.0])
]

# Iterate over each attack configuration
for attack_name, norm_name, art_norm, epsilons in configs:
    # Generate output file path at the very top of the loop
    csv_path = get_attack_eval_paths(args.dataset, args.pool_type, attack_name, norm_name, args.loss_weight)
    
    # Idempotency check: skip if file already exists
    if os.path.exists(csv_path):
        print(f"Evaluation for {attack_name} ({norm_name}) is already complete. Skipping...")
        continue

    # Initialize results for this specific attack/norm combination
    csv_dfs = []
    
    for eps in epsilons:
        print(f"\n{'='*60}")
        print(f"Evaluating {attack_name} ({norm_name}) with epsilon: {eps}")
        print(f"{'='*60}")
        
        # Construct the data path for adversarial examples
        data_path = get_untargeted_attack_path(args.dataset, args.pool_type, attack_name, norm_name, eps, args.loss_weight)
        print(f"Loading adversarial data from: {data_path}")
        
        # Load the .npz file containing adversarial examples and labels using context manager
        with np.load(data_path) as data:
            X_test = data['X_test']
            y_shape_and_scale_test = data['y_shape_and_scale_test'].flatten()
            y_shape_test = data['y_shape_test'].flatten()
            y_scale_test = data['y_scale_test'].flatten()
        
        print(f"Loaded {len(X_test)} adversarial samples.")
        
        # Preprocess X_test by expanding dimensions to include channel axis
        # Original shape: (N, 64, 64) -> New shape: (N, 64, 64, 1)
        X_test = np.expand_dims(X_test, axis=-1).astype(np.float32)
        print(f"Preprocessed X_test shape: {X_test.shape}")
        
        # Get model predictions. Note that model.predict returns logits, which are unpacked
        # and explicitly converted to probability arrays via tf.nn.softmax before being passed
        # to np.argmax for class label extraction.
        print("Running model predictions...")
        probs_main, probs_shape, probs_scale = model.predict(X_test, batch_size=4096, verbose=1)
        probs_main = tf.nn.softmax(probs_main, axis=-1).numpy()
        probs_shape = tf.nn.softmax(probs_shape, axis=-1).numpy()
        probs_scale = tf.nn.softmax(probs_scale, axis=-1).numpy()
        
        # Convert probability arrays to predicted class labels using np.argmax.
        # np.argmax extracts the class predictions by identifying the index of the maximum probability value along the class axis (axis=1).
        pred_main = np.argmax(probs_main, axis=1)
        pred_shape = np.argmax(probs_shape, axis=1)
        pred_scale = np.argmax(probs_scale, axis=1)
        
        print("Calculating uncertainty metrics...")
        
        # Calculate confidence and entropy for main task
        confidence = np.max(probs_main, axis=1)
        entropy = -np.sum(probs_main * np.log(probs_main + 1e-9), axis=1)
        
        # Calculate confidence and entropy for shape branch
        confidence_shape = np.max(probs_shape, axis=1)
        entropy_shape = -np.sum(probs_shape * np.log(probs_shape + 1e-9), axis=1)
        
        # Calculate confidence and entropy for scale branch
        confidence_scale = np.max(probs_scale, axis=1)
        entropy_scale = -np.sum(probs_scale * np.log(probs_scale + 1e-9), axis=1)
        
        # Create a Pandas DataFrame for the current epsilon containing per-sample data
        df_eps = pd.DataFrame({
            'sample_index': np.arange(len(X_test)),
            'epsilon': eps,
            'y_shape_and_scale': y_shape_and_scale_test,
            'y_shape': y_shape_test,
            'y_scale': y_scale_test,
            'pred_main': pred_main,
            'pred_shape': pred_shape,
            'pred_scale': pred_scale,
            'confidence': confidence,
            'entropy': entropy,
            'confidence_shape': confidence_shape,
            'entropy_shape': entropy_shape,
            'confidence_scale': confidence_scale,
            'entropy_scale': entropy_scale
        })
        csv_dfs.append(df_eps)

    # End of epsilon loop, still inside attack/norm loop
    
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Concatenate and save CSV dataframes
    print(f"Saving CSV report to: {csv_path}")
    if csv_dfs:
        df_all = pd.concat(csv_dfs, ignore_index=True)
        df_all.to_csv(csv_path, index=False, compression='gzip')
    print("CSV report saved successfully!")
    print(f"{'='*60}\n")