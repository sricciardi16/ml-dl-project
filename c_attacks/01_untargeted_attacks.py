import argparse
import os
import random
import sys
import numpy as np
import tensorflow as tf
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from shared.paths import (
    MY_DSPRITES_SPLIT,
    UNTARGETED_ATTACKS_DIR,
    get_untargeted_attack_path,
    get_model_paths
)

# Set global random seed to 29 (User's birth date) for reproducibility
os.environ['PYTHONHASHSEED'] = '29'
random.seed(29)
np.random.seed(29)
tf.random.set_seed(29)
tf.keras.utils.set_random_seed(29)

# Set up an argument parser to handle command-line options for the attack configuration
parser = argparse.ArgumentParser(description="Generate untargeted adversarial attacks on a specific sub-model.")
parser.add_argument('--pool_type', type=str, choices=['flatten', 'gap'], default='flatten',
                    help="Type of pooling used in the model ('flatten' or 'gap').")
parser.add_argument('--dataset', type=str, choices=['standard', 'contrast'], default='standard',
                    help="Dataset type ('standard' or 'contrast').")
parser.add_argument('--batch_size', type=int, default=4096,
                    help="Batch size for generating adversarial examples")
parser.add_argument('--loss_weight', type=float, choices=[0.5, 1.0, 2.0], default=1.0,
                    help="Loss weight for the shape_and_scale branch")
args = parser.parse_args()

# Load the dataset containing the test images and labels
data_path = MY_DSPRITES_SPLIT

# Using a context manager ensures the file handle is safely closed after loading
with np.load(data_path) as data:
    # Extract the test images and all corresponding label arrays.
    # The standard test set is used universally across all 4 multi-output models (Standard/Contrast combined with Flatten/GAP).
    # Because the standard test set is a perfect subset of the contrast set (values 0 and 1 map perfectly to the contrast float interval),
    # it provides a strict, common baseline to fairly compare how different training paradigms react to the exact same adversarial perturbations.
    X_test = data['X_test']
    y_shape_and_scale_test = data['y_shape_and_scale_test']
    y_shape_test = data['y_shape_test']
    y_scale_test = data['y_scale_test']
    y_orientation_test = data['y_orientation_test']
    y_pos_x_test = data['y_pos_x_test']
    y_pos_y_test = data['y_pos_y_test']

# Regardless of whether the target model is standard or contrast, we always use the standard test set (which contains values of exactly 0 or 1). This is mathematically correct because it perfectly matches the [0.0, 1.0] expected input range, so no / 20.0 scaling is needed.
X_test = X_test.astype(np.float32)

# Expand dimensions on the last axis to change the shape from (N, 64, 64) to (N, 64, 64, 1).
# This dimension expansion is required to match the model's expected input shape, which expects a channel dimension.
X_test = np.expand_dims(X_test, axis=-1)

# Load the trained Keras model based on the specified dataset and pooling type
model_path, _, _ = get_model_paths('multi', args.dataset, args.pool_type, args.loss_weight)

model = tf.keras.models.load_model(model_path)

# Create a new Keras Model to isolate the target branch.
# We set its inputs to the loaded model's input and its outputs to the output of the layer named 'shape_and_scale'.
# This sub-model ensures the attack algorithm only computes gradients with respect to the 'shape_and_scale' loss,
# remaining completely blind to the auxiliary 'shape' and 'scale' branches. This isolation is crucial for
# generating attacks that specifically target the joint shape and scale classification task without interference.
sub_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('shape_and_scale').output)

# Instantiate a loss object for the classification task
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Wrap the isolated sub-model in ART's TensorFlowV2Classifier.
# This wrapper provides a unified interface for ART's attack algorithms to interact with the TensorFlow model.
# We hardcode nb_classes=9 because this sub-model isolates the shape_and_scale branch, which has exactly 9 classes.
# This ensures the attack algorithm computes gradients solely for this primary task without interference from the auxiliary shape and scale branches.
classifier = TensorFlowV2Classifier(
    model=sub_model,
    nb_classes=9,
    input_shape=(64, 64, 1),
    loss_object=loss_object,
    clip_values=(0.0, 1.0)
)

# Ensure the output directory exists before generating attacks
os.makedirs(UNTARGETED_ATTACKS_DIR, exist_ok=True)

# Define the configurations for the attacks
configs = [
    ('fgsm', 'linf', np.inf, [0.01, 0.05, 0.1, 0.15, 0.2]),
    ('fgsm', 'l2', 2, [0.1, 0.5, 1.0, 2.0, 3.0]),
    ('pgd', 'linf', np.inf, [0.01, 0.05, 0.1, 0.15, 0.2]),
    ('pgd', 'l2', 2, [0.1, 0.5, 1.0, 2.0, 3.0])
]

# Iterate over the configurations and epsilon values to generate attacks
for attack_name, norm_name, art_norm, epsilons in configs:
    for eps in epsilons:
        # Instantiate the chosen attack method
        if attack_name == 'fgsm':
            attack = FastGradientMethod(estimator=classifier, norm=art_norm, eps=eps, targeted=False, batch_size=args.batch_size)
        elif attack_name == 'pgd':
            # This step size uses Madry's heuristic (alpha = 2.5 * epsilon / N_iter) to ensure optimal traversal of the perturbation space for both L2 and L-infinity norms.
            # Setting num_random_init=1 enables the random initialization required for standard PGD behavior, preventing the attack from defaulting to a deterministic Basic Iterative Method (BIM).
            attack = ProjectedGradientDescent(
                estimator=classifier, 
                norm=art_norm, 
                eps=eps, 
                eps_step=(2.5 * eps) / 40.0, 
                max_iter=40,
                targeted=False,
                num_random_init=1,
                batch_size=args.batch_size
            )

        # Generate the adversarial examples with manual batching to prevent OOM errors.
        # Passing the true labels (y=y_shape_and_scale_test) explicitly instructs the Adversarial Robustness Toolbox (ART)
        # to maximize the loss with respect to the ground-truth class, which effectively pushes the prediction
        # away from the correct label to create an untargeted attack.
        # Note: ART's internal batching is bypassed because the script manually feeds 
        # pre-sized batches to prevent OOM errors.
        n_samples = len(X_test)
        num_batches = int(np.ceil(n_samples / args.batch_size))
        adv_x_batches = []
        print(f"Generating attacks for {attack_name} {norm_name} epsilon {eps}...")
        for i in range(num_batches):
            start_idx = i * args.batch_size
            end_idx = min((i + 1) * args.batch_size, n_samples)
            x_batch = X_test[start_idx:end_idx]
            print(f"  Processing batch {i+1}/{num_batches} (size: {len(x_batch)})...")
            batch_adv_x = attack.generate(x=x_batch, y=y_shape_and_scale_test[start_idx:end_idx])
            adv_x_batches.append(batch_adv_x)
        adv_x = np.concatenate(adv_x_batches, axis=0)

        # Squeeze the channel dimension from the generated adversarial images.
        # This reverts the shape back to (N, 64, 64) to match the original dataset format.
        adv_x_squeezed = np.squeeze(adv_x, axis=-1)

        # Construct the output path for the generated adversarial dataset
        output_path = get_untargeted_attack_path(args.dataset, args.pool_type, attack_name, norm_name, eps, args.loss_weight)

        # Save the adversarial images and all original test label arrays to a compressed .npz file
        np.savez_compressed(
            output_path,
            X_test=adv_x_squeezed,
            y_shape_and_scale_test=y_shape_and_scale_test,
            y_shape_test=y_shape_test,
            y_scale_test=y_scale_test,
            y_orientation_test=y_orientation_test,
            y_pos_x_test=y_pos_x_test,
            y_pos_y_test=y_pos_y_test
        )
        print(f"Saved adversarial examples to {output_path}")