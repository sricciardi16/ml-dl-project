"""
Script to evaluate the multi-output baseline model.
Loads the test dataset and the trained multi-output model, evaluates it on 
three specific output heads (shape_and_scale, shape, and scale), and computes metrics 
including loss, accuracy, confusion matrix, and classification report.
The results are saved to a JSON file.

This script handles both standard and contrast datasets. For standard models,
it evaluates on the standard test split. For contrast models, it performs a
two-step evaluation: first on the contrast test split (primary evaluation),
and second on the standard test split (secondary evaluation). The standard test set
is a perfect subset of the contrast test set, and this secondary evaluation establishes
a common baseline for evaluating adversarial attacks across all models.
"""

import os
import sys
import json
import argparse
import random
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

# Set global random seed to 29 (User's birth date) for reproducibility
# Setting PYTHONHASHSEED to a fixed value ensures deterministic behavior for certain hash-based operations in Python.
os.environ['PYTHONHASHSEED'] = '29'
# Seed the built-in random number generator for reproducibility.
random.seed(29)
# Seed the NumPy random number generator to ensure consistent data shuffling and other NumPy operations.
np.random.seed(29)
# Seed the TensorFlow random number generator to ensure deterministic weight initialization and training operations.
tf.random.set_seed(29)
tf.keras.utils.set_random_seed(29)

# Resolve the project root directory and insert it into sys.path.
# This allows importing modules from the shared package (e.g., shared.paths) regardless of where the script is executed.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from shared.paths import (
    MY_DSPRITES_SPLIT,
    MY_DSPRITES_CONTRAST_SPLIT,
    get_model_paths,
    get_eval_paths
)

def get_metric(metrics, prefix, suffix_list):
    """
    Helper function to extract a metric value from the evaluation metrics dictionary
    using a list of potential suffix names.

    Keras metric names can vary depending on the TensorFlow version, model compilation,
    and output layer naming (e.g., accuracy, acc, shape_accuracy, sparse_categorical_accuracy).
    To safely extract the correct metric value for a specific task head, this helper
    iterates through a list of potential suffixes, combines them with the head's prefix,
    and returns the first matching value found in the metrics dictionary.
    """
    for suffix in suffix_list:
        key = f"{prefix}_{suffix}"
        if key in metrics:
            return float(metrics[key])
    return 0.0

def load_and_preprocess_multi(dataset_path, is_contrast):
    """
    Loads the dataset and preprocesses the images and labels.
    """
    # Load the test dataset from the specified path.
    # We extract three distinct label arrays:
    # - y_shape_and_scale_test: the combined shape and scale labels (main task)
    # - y_shape_test: the shape-only labels
    # - y_scale_test: the scale-only labels
    with np.load(dataset_path) as data:
        X_test = data['X_test']
        y_shape_and_scale_test = data['y_shape_and_scale_test']
        y_shape_test = data['y_shape_test']
        y_scale_test = data['y_scale_test']
        
    X_test = X_test.astype(np.float32)
    # Normalize the images. If the dataset is a contrast dataset, we divide by 20.0
    # to scale the pixel values back to the [0, 1] range.
    if is_contrast:
        X_test = X_test / 20.0
    # Expand the dimensions of the image array to add a channel dimension (axis=-1)
    # so that the shape becomes (num_samples, height, width, 1), matching Keras input requirements.
    X_test = np.expand_dims(X_test, axis=-1)
    
    # Create the ground truth dictionary.
    # Note that the dictionary keys ('shape_and_scale', 'shape', 'scale') must perfectly match
    # the names of the output layers defined in the multi-output Keras model.
    y_true_dict = {
        'shape_and_scale': y_shape_and_scale_test,
        'shape': y_shape_test,
        'scale': y_scale_test
    }
    return X_test, y_true_dict

def evaluate_and_save_multi(model, X_test, y_true_dict, output_path, summary_str):
    """
    Evaluates the model, computes metrics, and saves them to a JSON file.
    """
    # Run model.evaluate with return_dict=True.
    # This returns a dictionary of metrics mapped to respective output heads (e.g., 'loss', 'shape_and_scale_loss', 'shape_accuracy', etc.)
    # instead of a flat list, making it easier and safer to extract specific values.
    eval_metrics = model.evaluate(X_test, y_true_dict, return_dict=True, batch_size=4096, verbose=1)
    
    # Run model.predict to obtain the raw logits for all test samples.
    predictions = model.predict(X_test, batch_size=4096, verbose=1)
    # Unpacking predictions (probs_main, probs_shape, probs_scale) handles the list of three logit arrays
    # corresponding to the three task heads returned by the model. Note that these arrays only become
    # probabilities after the tf.nn.softmax lines are executed below.
    probs_main, probs_shape, probs_scale = predictions
    probs_main = tf.nn.softmax(probs_main, axis=-1).numpy()
    probs_shape = tf.nn.softmax(probs_shape, axis=-1).numpy()
    probs_scale = tf.nn.softmax(probs_scale, axis=-1).numpy()
    
    # The np.argmax block converts probabilities to discrete class predictions for all three tasks
    # by selecting the index of the maximum probability along the class axis (axis=1).
    y_pred_shape_and_scale = np.argmax(probs_main, axis=1)
    y_pred_shape = np.argmax(probs_shape, axis=1)
    y_pred_scale = np.argmax(probs_scale, axis=1)
    
    # Flatten() operations ensure true label arrays are 1D to match prediction shapes for accurate comparison.
    # This avoids any potential shape mismatch issues (e.g., (N, 1) vs (N,)).
    y_true_shape_and_scale_flat = y_true_dict['shape_and_scale'].flatten()
    y_true_shape_flat = y_true_dict['shape'].flatten()
    y_true_scale_flat = y_true_dict['scale'].flatten()
    
    # Identify misclassifications for the main task.
    # The misclassification blocks (using np.where) find exact indices where predictions fail,
    # and the subsequent list comprehension builds a detailed record of true vs. predicted labels for debugging.
    misclass_main_idx = np.where(y_true_shape_and_scale_flat != y_pred_shape_and_scale)[0]
    misclassifications_main = [
        {
            "index": int(idx),
            "true_label": int(y_true_shape_and_scale_flat[idx]),
            "predicted_label": int(y_pred_shape_and_scale[idx])
        }
        for idx in misclass_main_idx
    ]
    
    # Identify misclassifications for the shape task.
    # The misclassification blocks (using np.where) find exact indices where predictions fail,
    # and the subsequent list comprehension builds a detailed record of true vs. predicted labels for debugging.
    misclass_shape_idx = np.where(y_true_shape_flat != y_pred_shape)[0]
    misclassifications_shape = [
        {
            "index": int(idx),
            "true_label": int(y_true_shape_flat[idx]),
            "predicted_label": int(y_pred_shape[idx])
        }
        for idx in misclass_shape_idx
    ]
    
    # Identify misclassifications for the scale task.
    # The misclassification blocks (using np.where) find exact indices where predictions fail,
    # and the subsequent list comprehension builds a detailed record of true vs. predicted labels for debugging.
    misclass_scale_idx = np.where(y_true_scale_flat != y_pred_scale)[0]
    misclassifications_scale = [
        {
            "index": int(idx),
            "true_label": int(y_true_scale_flat[idx]),
            "predicted_label": int(y_pred_scale[idx])
        }
        for idx in misclass_scale_idx
    ]
    
    # Compute confusion matrix and classification report for all three tasks
    cm_main = confusion_matrix(y_true_shape_and_scale_flat, y_pred_shape_and_scale)
    cr_main = classification_report(y_true_shape_and_scale_flat, y_pred_shape_and_scale, output_dict=True)
    
    cm_shape = confusion_matrix(y_true_shape_flat, y_pred_shape)
    cr_shape = classification_report(y_true_shape_flat, y_pred_shape, output_dict=True)
    
    cm_scale = confusion_matrix(y_true_scale_flat, y_pred_scale)
    cr_scale = classification_report(y_true_scale_flat, y_pred_scale, output_dict=True)
    
    # Extract loss and accuracy metrics.
    # The metric extraction block (using get_metric) isolates loss and accuracy for the main, shape, and scale tasks
    # by trying multiple potential metric suffixes to handle naming variations across Keras versions.
    shape_and_scale_loss = get_metric(eval_metrics, 'shape_and_scale', ['loss'])
    shape_and_scale_acc = get_metric(eval_metrics, 'shape_and_scale', ['accuracy', 'acc', 'categorical_accuracy', 'sparse_categorical_accuracy'])
    
    shape_loss = get_metric(eval_metrics, 'shape', ['loss'])
    shape_acc = get_metric(eval_metrics, 'shape', ['accuracy', 'acc', 'categorical_accuracy', 'sparse_categorical_accuracy'])
    
    scale_loss = get_metric(eval_metrics, 'scale', ['loss'])
    scale_acc = get_metric(eval_metrics, 'scale', ['accuracy', 'acc', 'categorical_accuracy', 'sparse_categorical_accuracy'])
    
    # Build nested dictionary.
    # The metrics_dict construction builds the nested JSON schema containing total loss, task-specific metrics,
    # classification reports, confusion matrices, misclassification details, and the model summary.
    metrics_dict = {
        "test_loss_total": float(eval_metrics.get('loss', 0.0)),
        "shape_and_scale_task": {
            "loss": shape_and_scale_loss,
            "accuracy": shape_and_scale_acc,
            "classification_report": cr_main,
            "confusion_matrix": cm_main.tolist(),
            "misclassifications": misclassifications_main
        },
        "shape_task": {
            "loss": shape_loss,
            "accuracy": shape_acc,
            "classification_report": cr_shape,
            "confusion_matrix": cm_shape.tolist(),
            "misclassifications": misclassifications_shape
        },
        "scale_task": {
            "loss": scale_loss,
            "accuracy": scale_acc,
            "classification_report": cr_scale,
            "confusion_matrix": cm_scale.tolist(),
            "misclassifications": misclassifications_scale
        },
        "model_summary": summary_str
    }
    
    # Ensure output directory exists and save JSON
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
        
    print(f"Saved evaluation metrics to {output_path}")
    print(f"Total Loss: {metrics_dict['test_loss_total']:.4f}")
    print(f"Main Task Accuracy: {metrics_dict['shape_and_scale_task']['accuracy']:.4f}")
    print(f"Shape Task Accuracy: {metrics_dict['shape_task']['accuracy']:.4f}")
    print(f"Scale Task Accuracy: {metrics_dict['scale_task']['accuracy']:.4f}")

def main():
    # Argparse setup: Define command-line arguments to specify the pooling type ('flatten' or 'gap')
    # and the dataset type ('standard' or 'contrast') for the model evaluation.
    parser = argparse.ArgumentParser(description="Evaluate multi-output models.")
    parser.add_argument('--pool_type', type=str, choices=['flatten', 'gap'], default='flatten', help='Pooling type')
    parser.add_argument('--dataset', type=str, choices=['standard', 'contrast'], default='standard', help='Dataset type')
    parser.add_argument('--loss_weight', type=float, choices=[0.5, 1.0, 2.0], default=1.0, help='Loss weight')
    args = parser.parse_args()
    
    # Path Resolution block: Select the appropriate model path and evaluation output paths based on the arguments.
    # For standard models, we only have a primary evaluation path.
    # For contrast models, we set a secondary path to evaluate the model's performance on the standard dataset
    # as well, establishing a common baseline for evaluating adversarial attacks across all models.
    model_path = get_model_paths('multi', args.dataset, args.pool_type, args.loss_weight)[0]
    primary_eval_path, secondary_eval_path = get_eval_paths('multi', args.dataset, args.pool_type, args.loss_weight)

    # Idempotency Check block: Check if the evaluation output files already exist.
    # If they do, we exit early to avoid redundant computations and ensure the script is idempotent.
    if args.dataset == 'standard':
        if os.path.exists(primary_eval_path):
            print(f"Evaluation file already exists: {primary_eval_path}. Skipping evaluation.")
            sys.exit(0)
    elif args.dataset == 'contrast':
        if os.path.exists(primary_eval_path) and os.path.exists(secondary_eval_path):
            print(f"Both evaluation files already exist: {primary_eval_path} and {secondary_eval_path}. Skipping evaluation.")
            sys.exit(0)

    # Model Loading
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    # Model summary capture block: Capture the model's architecture summary as a string using io.StringIO().
    # This allows us to save the model summary directly inside the JSON evaluation file for documentation.
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_str = stream.getvalue()
    stream.close()
    
    # Execution Logic block: Perform the evaluation based on the dataset type.
    # For standard models, we evaluate once on the standard test split.
    # For contrast models, we perform a two-step evaluation process:
    # 1. Evaluate on the contrast split first (primary evaluation) to measure in-distribution performance.
    # 2. Evaluate on the standard split second (secondary evaluation) to establish a common baseline.
    if args.dataset == 'standard':
        X_test, y_true_dict = load_and_preprocess_multi(MY_DSPRITES_SPLIT, is_contrast=False)
        evaluate_and_save_multi(model, X_test, y_true_dict, primary_eval_path, summary_str)
    elif args.dataset == 'contrast':
        # First, load MY_DSPRITES_CONTRAST_SPLIT with is_contrast=True, then call evaluate_and_save_multi to the primary path.
        X_test_contrast, y_true_dict_contrast = load_and_preprocess_multi(MY_DSPRITES_CONTRAST_SPLIT, is_contrast=True)
        evaluate_and_save_multi(model, X_test_contrast, y_true_dict_contrast, primary_eval_path, summary_str)
        
        # -------------------------------------------------------------------------
        # Standard dataset values are exactly 0 (background) and 1 (object).
        # Contrast dataset values, after being divided by 20.0 during preprocessing,
        # fall into the 0.0 to 1.0 range. Therefore, 1 == 20/20, meaning the standard
        # dataset perfectly matches the float interval expected by the contrast-trained model.
        # WARNING: Do not divide the standard dataset by 20, as it would ruin the data
        # (turning 1 into 0.05).
        #
        # Because the standard dataset is a perfect subset of the contrast dataset by construction,
        # this evaluation does not measure out-of-distribution performance. Instead, it simply
        # provides a uniform baseline test set for the adversarial attack phase, ensuring all 8
        # models can be compared fairly on the exact same data.
        # -------------------------------------------------------------------------
        # Second, load MY_DSPRITES_SPLIT with is_contrast=False, then call evaluate_and_save_multi to the secondary path.
        X_test_standard, y_true_dict_standard = load_and_preprocess_multi(MY_DSPRITES_SPLIT, is_contrast=False)
        evaluate_and_save_multi(model, X_test_standard, y_true_dict_standard, secondary_eval_path, summary_str)

if __name__ == "__main__":
    main()