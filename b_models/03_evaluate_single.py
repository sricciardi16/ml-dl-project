"""
This script evaluates a single-output Convolutional Neural Network on the dSprites dataset (standard or contrast).
It computes loss, accuracy, confusion matrices, and classification reports, and serializes the results to JSON.

The two-step evaluation for contrast models is performed because the standard test set is a perfect subset of the
contrast test set by construction. Evaluating on the standard test set provides a common baseline for comparing
adversarial attacks across all 8 models later in the pipeline.
"""

import os
# Setting the PYTHONHASHSEED environment variable to '29' ensures deterministic hash-based operations.
os.environ['PYTHONHASHSEED'] = '29'
import sys
import json
import argparse
import io
import random
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

# Set global random seed to 29 (User's birth date) for reproducibility
random.seed(29)
np.random.seed(29)
tf.random.set_seed(29)
tf.keras.utils.set_random_seed(29)

# This allows importing from the shared module located in the project root.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from shared.paths import (
    MY_DSPRITES_SPLIT,
    MY_DSPRITES_CONTRAST_SPLIT,
    get_model_paths,
    get_eval_paths
)

def load_and_preprocess(dataset_path, is_contrast):
    """
    Loads the dSprites test dataset and preprocesses the images and labels.

    Parameters:
    -----------
    dataset_path : str
        The file path to the .npz dataset.
    is_contrast : bool
        Flag indicating whether the dataset is contrast-adjusted.

    Returns:
    --------
    X_test : numpy.ndarray
        Preprocessed image array with shape (N, 64, 64, 1).
    y_test : numpy.ndarray
        Label array containing combined shape-and-scale labels.
    """
    print(f"Loading dataset from {dataset_path}...")
    # The .npz loading block extracts test images and combined shape-and-scale labels.
    with np.load(dataset_path) as data:
        X_test = data['X_test']
        y_test = data['y_shape_and_scale_test']
    
    # astype(np.float32) ensures compatibility with TensorFlow.
    X_test = X_test.astype(np.float32)
    
    # The if is_contrast block dividing by 20.0 normalizes contrast-adjusted pixel values.
    if is_contrast:
        X_test = X_test / 20.0
        
    # np.expand_dims with axis=-1 adds the channel dimension required by the CNN (e.g., turning (N, 64, 64) into (N, 64, 64, 1)).
    X_test = np.expand_dims(X_test, axis=-1)
    return X_test, y_test

def evaluate_and_save(model, X_test, y_test, output_path, summary_str):
    """
    Runs inference on the test dataset, calculates evaluation metrics, and saves the results to a JSON report.

    Parameters:
    -----------
    model : tensorflow.keras.Model
        The trained CNN model to evaluate.
    X_test : numpy.ndarray
        The preprocessed test images.
    y_test : numpy.ndarray
        The true labels.
    output_path : str
        The file path where the JSON report will be saved.
    summary_str : str
        The string representation of the model's summary.
    """
    print(f"Evaluating model on dataset...")
    # model.evaluate with verbose=1 shows progress bar.
    loss, accuracy = model.evaluate(X_test, y_test, batch_size=4096, verbose=1)
    
    print("Generating predictions...")
    # model.predict outputs raw logits, and the subsequent tf.nn.softmax call dynamically converts these logits into a probability distribution before np.argmax extracts the class index.
    y_pred_probs = model.predict(X_test, batch_size=4096, verbose=1)
    y_pred_probs = tf.nn.softmax(y_pred_probs, axis=-1).numpy()
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Identify misclassifications by finding indices where the true label does not match the predicted label.
    # We use np.where to get the indices and then build a list of dictionaries with standard Python ints for JSON serialization.
    misclassified_indices = np.where(y_test != y_pred)[0]
    misclassifications = [
        {
            "index": int(idx),
            "true_label": int(y_test[idx]),
            "predicted_label": int(y_pred[idx])
        }
        for idx in misclassified_indices
    ]
    
    print("Computing metrics...")
    # confusion_matrix and classification_report compare true against predicted labels, with output_dict=True formatting the report for JSON serialization.
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    
    metrics = {
        "test_loss_total": float(loss),
        "shape_and_scale_task": {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "classification_report": cr,
            "confusion_matrix": cm.tolist(),
            "misclassifications": misclassifications
        },
        "model_summary": summary_str
    }
    
    # The os.makedirs and json.dump block ensures the target directory exists before writing the formatted metrics dictionary to disk.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved evaluation metrics to {output_path}")

def main():
    # The argparse setup defines available command-line arguments (--pool_type and --dataset).
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool_type', choices=['flatten', 'gap'], default='flatten')
    parser.add_argument('--dataset', choices=['standard', 'contrast'], default='standard')
    args = parser.parse_args()

    # Path Resolution block selects the correct pre-trained model and output JSON paths based on arguments,
    # noting that contrast models require a secondary_eval_path for evaluation on both contrast and standard test sets.
    model_path = get_model_paths('single', args.dataset, args.pool_type)[0]
    primary_eval_path, secondary_eval_path = get_eval_paths('single', args.dataset, args.pool_type)

    # Idempotency Check block prevents redundant computation by exiting early if target JSON files exist.
    if args.dataset == 'standard':
        if os.path.exists(primary_eval_path):
            print(f"Evaluation file already exists: {primary_eval_path}. Exiting.")
            sys.exit(0)
    else:
        if os.path.exists(primary_eval_path) and os.path.exists(secondary_eval_path):
            print(f"Evaluation files already exist: {primary_eval_path} and {secondary_eval_path}. Exiting.")
            sys.exit(0)

    # Model Loading block uses io.StringIO() to capture the standard output of model.summary() into a string variable for the JSON report.
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # Capture model summary
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_str = stream.getvalue()

    # Execution Logic block branches so standard models are evaluated once,
    # while contrast models are evaluated twice (first on contrast, then on standard to establish a common baseline).
    if args.dataset == 'standard':
        X_test, y_test = load_and_preprocess(MY_DSPRITES_SPLIT, is_contrast=False)
        evaluate_and_save(model, X_test, y_test, primary_eval_path, summary_str)
    else:
        # First, evaluate on contrast test set
        X_test_contrast, y_test_contrast = load_and_preprocess(MY_DSPRITES_CONTRAST_SPLIT, is_contrast=True)
        evaluate_and_save(model, X_test_contrast, y_test_contrast, primary_eval_path, summary_str)
        
        # We evaluate the contrast-trained model on the standard test set. Note that there is no out-of-distribution (OOD)
        # evaluation here; the standard test set is a perfect subset of the contrast test set by construction.
        # Evaluating on the standard test set guarantees a unified, common baseline for the upcoming adversarial attack
        # evaluations across all 8 models later in the pipeline.
        # WARNING: Do not divide the standard dataset by 20, as it would ruin the data (turning 1 into 0.05).
        # Second, evaluate on standard test set
        X_test_standard, y_test_standard = load_and_preprocess(MY_DSPRITES_SPLIT, is_contrast=False)
        evaluate_and_save(model, X_test_standard, y_test_standard, secondary_eval_path, summary_str)

if __name__ == '__main__':
    main()