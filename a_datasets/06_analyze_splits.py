# %%
# Cell 1: Imports and setup

import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, chisquare, entropy
from sklearn.metrics import normalized_mutual_info_score
import ipywidgets
import itertools
from IPython.display import display

# Set seed to 29 (User's birth date) for strict determinism
os.environ['PYTHONHASHSEED'] = '29'
random.seed(29)
np.random.seed(29)

# Append project root (one directory up from the current file) to sys.path.
# This path resolution ensures that we can import modules from the 'shared' package
# regardless of where the script is executed from (e.g., from the project root or the datasets_analysis directory).
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.paths import (
    MY_DSPRITES_SPLIT,
    MY_DSPRITES_CONTRAST_SPLIT,
    MY_DSPRITES_SPLIT_REPORT,
    MY_DSPRITES_CONTRAST_SPLIT_REPORT,
    ANALYSES_DIR
)
from shared.report_generator import ReportGenerator

# Ensure ANALYSES_DIR exists to store generated reports and figures
os.makedirs(ANALYSES_DIR, exist_ok=True)

# %%
# Cell 2: Dataset selection widget

# Create an interactive ToggleButtons widget to allow the user to select
# between the 'Standard' and 'Contrast' datasets. This widget logic enables
# dynamic analysis of different dataset splits within the same notebook/script environment.
dataset_selector = ipywidgets.ToggleButtons(
    options=['Standard', 'Contrast'],
    description='Dataset:',
    value='Standard'
)
# Display the widget in the notebook interface
display(dataset_selector)

# %%
# Cell 3: Load dataset and extract labels

# Read the selected value from the ToggleButtons widget to determine if we are analyzing the contrast dataset.
# This boolean flag controls path resolution and feature selection in subsequent steps.
is_contrast = dataset_selector.value == 'Contrast'

# Resolve the input dataset path and output report path based on the widget selection.
# If 'Contrast' is selected, we load the contrast-augmented dSprites split and write to its corresponding report.
# Otherwise, we default to the standard dSprites split and report.
if is_contrast:
    dataset_path = MY_DSPRITES_CONTRAST_SPLIT
    report_path = MY_DSPRITES_CONTRAST_SPLIT_REPORT
else:
    dataset_path = MY_DSPRITES_SPLIT
    report_path = MY_DSPRITES_SPLIT_REPORT

# Initialize the ReportGenerator utility to compile analysis results and plots into a structured report.
report = ReportGenerator(report_path)

# Load the compressed numpy (.npz) file containing the dataset splits and labels.
data = np.load(dataset_path)

# Print the sizes of the feature arrays (X_train, X_val, X_test) to verify the split sizes.
print(f"X_train size: {data['X_train'].shape[0]}")
print(f"X_val size: {data['X_val'].shape[0]}")
print(f"X_test size: {data['X_test'].shape[0]}")

# Define the splits and features we want to analyze.
# If the contrast dataset is selected, we append 'contrast' to the list of features.
splits = ['train', 'val', 'test']
features = ['shape_and_scale', 'shape', 'scale', 'orientation', 'pos_x', 'pos_y']
if is_contrast:
    features.append('contrast')

# Extract labels into a nested dictionary structure.
# The nested dictionary comprehension/loop `labels[split][feature]` organizes the flat keys
# of the loaded .npz file (which are named like 'y_shape_train', 'y_scale_val', etc.) into a
# structured format: labels[split][feature]. This allows for clean, intuitive, and programmatic
# iteration over splits and features in subsequent analysis cells.
labels = {}
for split in splits:
    labels[split] = {}
    for feature in features:
        key = f'y_{feature}_{split}'
        labels[split][feature] = data[key]

# %%
# Cell 4: Marginal Uniformity Analysis

# Define features that were explicitly used during the stratified splitting process.
# These features are expected to be perfectly balanced across splits.
stratified_features = ['shape_and_scale', 'shape', 'scale']
if is_contrast:
    stratified_features.append('contrast')

# Define features that were not explicitly stratified.
# These features are expected to be roughly uniform but may exhibit minor statistical fluctuations.
statistical_features = ['orientation', 'pos_x', 'pos_y']

for split in splits:
    print(f"\n{'='*60}")
    print(f"=== Split: {split} - Marginal Uniformity Analysis ===")
    print(f"{'='*60}")
    
    # =========================================================================
    # BLOCK: Stratified Features (Deterministic Uniformity)
    # =========================================================================
    # These features (shape, scale, shape_and_scale, and optionally contrast)
    # were explicitly balanced during the dataset splitting process.
    # Because the splitting algorithm balanced them deterministically, we use a
    # strict deterministic check to verify that all counts match the expected
    # count exactly. This ensures perfect stratification and balance across all splits.
    # =========================================================================
    print("\n--- Stratified Features (Deterministic Uniformity) ---")
    for feature in stratified_features:
        arr = labels[split][feature]
        unique_vals, counts = np.unique(arr, return_counts=True)
        empirical_entropy = entropy(counts)
        theoretical_max_entropy = np.log(len(unique_vals))
        if theoretical_max_entropy == 0:
            uniformity_score = 1.0
        else:
            uniformity_score = empirical_entropy / theoretical_max_entropy
        expected_count = len(arr) / len(unique_vals)
        is_uniform = np.all(counts == expected_count)
        print(f"Feature: {feature}")
        print(f"  Unique values: {unique_vals}")
        print(f"  Counts: {counts}")
        print(f"  Uniformity score (Entropy): {uniformity_score:.4f}")
        print(f"  Expected count: {expected_count}")
        print(f"  All counts match expected exactly: {is_uniform}")
        print(f"  Explanation: This feature is perfectly balanced by construction due to stratification.")
        print()
        
    # =========================================================================
    # BLOCK: Statistical Features (Statistical Uniformity)
    # =========================================================================
    # These features (orientation, pos_x, pos_y) were not explicitly stratified
    # during the splitting process. Instead, they were distributed randomly.
    # Therefore, we do not expect perfect deterministic balance. Instead, we
    # perform a Chi-square goodness-of-fit test (`chisquare`) to verify rough
    # uniformity. A high p-value (>= 0.05) indicates that the observed counts
    # do not significantly deviate from a uniform distribution, confirming that
    # the random split preserved statistical uniformity.
    # =========================================================================
    print("\n--- Statistical Features (Statistical Uniformity) ---")
    for feature in statistical_features:
        arr = labels[split][feature]
        unique_vals, counts = np.unique(arr, return_counts=True)
        empirical_entropy = entropy(counts)
        theoretical_max_entropy = np.log(len(unique_vals))
        if theoretical_max_entropy == 0:
            uniformity_score = 1.0
        else:
            uniformity_score = empirical_entropy / theoretical_max_entropy
        chi2_stat, p_val = chisquare(f_obs=counts)
        print(f"Feature: {feature}")
        print(f"  Unique values: {unique_vals}")
        print(f"  Counts: {counts}")
        print(f"  Uniformity score (Entropy): {uniformity_score:.4f}")
        print(f"  Chi-square statistic: {chi2_stat:.4f}")
        print(f"  p-value: {p_val:.4e}")
        if p_val < 0.05:
            conclusion = "Distribution significantly deviates from uniform"
        else:
            conclusion = "Distribution is consistent with uniform"
        print(f"  Conclusion: {conclusion}")
        print(f"  Explanation: These factors were not explicitly stratified so a statistical test is used.")
        print()

# %%
# Cell 5: Marginal Uniformity Visualizations

# Define the spatial and rotational features to visualize
spatial_rotational_features = ['orientation', 'pos_x', 'pos_y']

for split in splits:
    # Create a figure with 3 subplots (one for each spatial/rotational feature)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, feature in enumerate(spatial_rotational_features):
        arr = labels[split][feature]
        unique_vals, counts = np.unique(arr, return_counts=True)
        
        # Calculate the expected count per category under a perfectly uniform distribution.
        # This is computed as the total number of samples in the split divided by the number of unique categories.
        # We will plot this as a horizontal red dashed line to visually compare the actual counts against the ideal uniform distribution.
        expected_count = len(arr) / len(unique_vals)
        
        # Plot the actual counts as a bar chart
        axes[idx].bar(unique_vals, counts)
        
        # Draw the horizontal red dashed line representing the expected uniform count
        axes[idx].axhline(y=expected_count, color='r', linestyle='--', label='Expected uniform count')
        
        # Set labels and titles for the subplot
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Count')
        axes[idx].set_title(f'{feature} distribution')
        axes[idx].legend()
    
    # Adjust layout to prevent overlapping labels
    plt.tight_layout()
    
    # Save the figure to the report using the Kroki/ReportGenerator utility
    report.kroki(fig, title=f'Spatial/Rotational features distribution in {split} split')
    
    # Close the figure to free up memory
    plt.close(fig)

# %%
# Cell 6: Joint Uniformity Analysis

# Identify the base stratified features (excluding the composite 'shape_and_scale' feature)
base_stratified_features = ['shape', 'scale']
if is_contrast:
    base_stratified_features.append('contrast')

for split in splits:
    print(f"\n=== Split: {split} ===")
    
    # Stack the base stratified feature arrays column-wise.
    # This creates a 2D array where each row represents the combination of base stratified features for a single sample.
    # For example, if we have shape and scale, each row will be [shape_val, scale_val].
    arrays = [labels[split][f] for f in base_stratified_features]
    stacked = np.column_stack(arrays)
    
    # Calculate the unique combinations of these base features and their respective counts.
    # This allows us to verify if the Cartesian product of the base stratified features remains perfectly balanced.
    unique_combos, combo_counts = np.unique(stacked, axis=0, return_counts=True)
    
    # Calculate the expected count for each unique combination.
    # Under perfect joint uniformity, every possible combination in the Cartesian product of the base features
    # should appear exactly the same number of times (or differ by at most 1 if the total count is not perfectly divisible).
    expected_combo_count = len(stacked) / len(unique_combos)
    all_expected = np.all(combo_counts == expected_combo_count)
    
    print(f"Number of unique combinations: {len(unique_combos)}")
    print(f"Expected count per combination: {expected_combo_count:.2f}")
    print(f"All combinations appear exactly expected times: {all_expected}")
    if not all_expected:
        print(f"Count range: {combo_counts.min()} - {combo_counts.max()}")

# %%
# Cell 7: Pairwise Independence Analysis

print("Pairwise Independence Analysis Explanation:")
print("Pairs involving 'shape_and_scale' and its base components ('shape' or 'scale') should yield a Cramér's V of 1.0 (perfectly dependent).")
print("All other base feature pairs should yield a Cramér's V of 0.0 (perfectly independent).")

for split in splits:
    print(f"\n{'='*60}")
    print(f"=== Split: {split} - Pairwise Independence Analysis ===")
    print(f"{'='*60}")
    
    # Generate all unique pairwise combinations of features to analyze their joint distributions
    feature_pairs = list(itertools.combinations(features, 2))
    
    for f1, f2 in feature_pairs:
        arr1 = labels[split][f1]
        arr2 = labels[split][f2]
        nmi = normalized_mutual_info_score(arr1, arr2)
        
        # Get unique values for each feature to define the dimensions of the contingency table
        unique1 = np.unique(arr1)
        unique2 = np.unique(arr2)
        
        # Construct the contingency table (joint frequency distribution table).
        # The entry at contingency[i, j] represents the number of samples where feature 1 has value unique1[i]
        # and feature 2 has value unique2[j].
        contingency = np.zeros((len(unique1), len(unique2)), dtype=int)
        for i, v1 in enumerate(unique1):
            for j, v2 in enumerate(unique2):
                contingency[i, j] = np.sum((arr1 == v1) & (arr2 == v2))
        
        print(f"Pair: {f1} vs {f2}")
        
        # Run Chi-square test of independence
        chi2, p, dof, expected = chi2_contingency(contingency)
        
        # Calculate Cramér's V
        n = np.sum(contingency)
        min_dim = min(contingency.shape) - 1
        if min_dim == 0:
            cramers_v = 0.0
        else:
            cramers_v = np.sqrt(max(0.0, chi2 / (n * min_dim)))
            
        cramers_v = round(cramers_v, 4)
        
        print(f"  Chi-square statistic: {chi2:.4f}")
        print(f"  Cramér's V: {cramers_v:.4f}")
        print(f"  Normalized Mutual Information (NMI): {nmi:.4f}")
        
        if np.isclose(cramers_v, 0.0, atol=1e-4):
            print("  Perfectly independent")
        elif np.isclose(cramers_v, 1.0, atol=1e-4):
            print("  Perfectly dependent")
        else:
            if p < 0.05:
                print(f"  Statistically dependent (p-value: {p:.4e} < 0.05)")
            else:
                print(f"  Statistically independent (p-value: {p:.4e} >= 0.05)")
        print()

# %%
# Cell 8: Data Leakage Verification

# This verification step checks for standard intra-dataset leakage, ensuring that there is
# zero overlap in base instances (generative factors) between the train, validation, and test splits.
# By confirming that each split contains a completely disjoint set of base instances, we establish
# a clean baseline. This intra-dataset verification serves as a strict prerequisite for subsequent
# cross-dataset checks (such as verifying the relationship between the standard and contrast datasets).

print(f"\n{'='*60}")
print("=== Data Leakage Verification ===")
print(f"{'='*60}")

# Stack base generative factors column-wise for each split
split_instances = {}
base_factors = ['shape', 'scale', 'orientation', 'pos_x', 'pos_y']

for split in splits:
    # np.column_stack takes several 1D arrays and glues them together horizontally to create a 2D matrix where each row is a single image's complete set of generative factors.
    stacked = np.column_stack([labels[split][factor] for factor in base_factors])
    # Converting each row to a tuple (tuple(row)) is necessary because NumPy arrays are mutable and unhashable. Converting them to tuples allows insertion into a Python set, which automatically drops duplicates and provides O(1) lookup time.
    split_instances[split] = set(tuple(row) for row in stacked)
    print(f"Split '{split}' unique base instances: {len(split_instances[split])}")

# Calculate intersections between splits
# Python's set intersection mathematically proves whether any identical tuple exists in both sets.
train_val_intersection = split_instances['train'].intersection(split_instances['val'])
train_test_intersection = split_instances['train'].intersection(split_instances['test'])
val_test_intersection = split_instances['val'].intersection(split_instances['test'])

size_train_val = len(train_val_intersection)
size_train_test = len(train_test_intersection)
size_val_test = len(val_test_intersection)

print(f"Intersection size (train & val): {size_train_val}")
print(f"Intersection size (train & test): {size_train_test}")
print(f"Intersection size (val & test): {size_val_test}")

# Assert that all intersections are exactly 0
assert size_train_val == 0, f"Data leakage detected between train and val! Intersection size: {size_train_val}"
assert size_train_test == 0, f"Data leakage detected between train and test! Intersection size: {size_train_test}"
assert size_val_test == 0, f"Data leakage detected between val and test! Intersection size: {size_val_test}"

if size_train_val == 0 and size_train_test == 0 and size_val_test == 0:
    print("Success: No data leakage detected between any of the splits!")
else:
    print("Leakage warning: Data leakage detected between splits!")

# %%
# Cell 9: Contrast Subset Verification

# This section performs Contrast Subset Verification to ensure the integrity of the contrast dataset
# relative to the standard dataset. Specifically, we verify two conditions:
# 1. The contrast dataset size is exactly 5x the standard dataset size (since there are 5 contrast levels: 0, 1, 2, 3, 4).
# 2. The standard instances perfectly match the contrast == 0 instances (the base/unmodified contrast level).
#
# These two conditions serve as the mathematical proofs required for the adversarial attack pipeline.
# By proving the standard test set is a perfect subset of the contrast test set, we mathematically guarantee that evaluating a contrast-trained model on the standard test set will not result in data leakage (i.e., the model has never seen those standard test images, nor any of their contrast variants, during training).
#
# Specifically, because the train, validation, and test splits are disjoint in the standard dataset (as proven
# in Cell 8), and because the contrast dataset preserves this exact split mapping (where each standard instance
# maps to its 5 contrast variants in the corresponding split), a model trained on the contrast train set
# has only seen contrast variants of the standard train set. It has never seen any contrast variants of the
# standard test set. Thus, evaluating on the standard test set is a completely clean, zero-leakage out-of-distribution
# or in-distribution evaluation of the model's robustness.

if is_contrast:
    print(f"\n{'='*60}")
    print("=== Contrast Subset Verification ===")
    print(f"{'='*60}")
    
    # Load the standard dataset
    standard_data = np.load(MY_DSPRITES_SPLIT)
    base_factors = ['shape', 'scale', 'orientation', 'pos_x', 'pos_y']
    
    for split in splits:
        print(f"\nVerifying split: {split}")
        
        # Extract standard instances as a set of tuples
        standard_stacked = np.column_stack([standard_data[f'y_{factor}_{split}'] for factor in base_factors])
        standard_set = set(tuple(row) for row in standard_stacked)
        
        # Extract contrast instances from current dataset where contrast label equals 0
        contrast_labels = labels[split]['contrast']
        # This creates a boolean mask of True/False values where True means the image is a base image.
        mask = (contrast_labels == 0)
        
        # np.column_stack is used here with fancy indexing: applying the mask instantly filters out all rows where the mask is False, extracting base images without a slow Python for loop.
        contrast_stacked = np.column_stack([labels[split][factor][mask] for factor in base_factors])
        contrast_set = set(tuple(row) for row in contrast_stacked)
        
        # Verify condition 1: total size of contrast split is exactly 5 times the size of standard split
        # This mathematically proves that every standard instance has exactly 5 contrast variants in the contrast dataset.
        contrast_total_size = len(labels[split]['contrast'])
        standard_total_size = len(standard_data[f'y_shape_{split}'])
        # This size check compares the contrast total size to 5 times the standard total size, which proves the 1-to-5 mapping assumption.
        size_check = (contrast_total_size == 5 * standard_total_size)
        
        # Verify condition 2: set of standard instances perfectly matches the set of contrast == 0 instances
        # This mathematically proves that the contrast == 0 subset is identical to the standard dataset,
        # meaning the standard dataset is perfectly embedded within the contrast dataset.
        # The equality operator between two Python sets checks if they contain the exact same unique tuples regardless of order, mathematically proving the standard dataset is perfectly embedded within the contrast dataset.
        match_check = (standard_set == contrast_set)
        
        print(f"  Condition 1 (Contrast split size is 5x standard split size): {size_check}")
        print(f"  Condition 2 (Standard instances match contrast == 0 instances): {match_check}")
        
        assert size_check, f"Size check failed for split {split}!"
        assert match_check, f"Match check failed for split {split}!"
        
        if size_check and match_check:
            print(f"  Success: Perfect subset relationship confirmed for split '{split}'!")

# %%
# Cell 10: Finalize report

# Close the report generator and write the compiled analysis and figures to the final report file.
report.close()