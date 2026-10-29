# %%
# Cell 1: Setup
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import ipywidgets as widgets
from IPython.display import display

# Set seed to 29 (User's birth date) for strict determinism
os.environ['PYTHONHASHSEED'] = '29'
random.seed(29)
np.random.seed(29)

# Calculate the absolute path of the parent directory (project root)
# This allows us to perform absolute imports from the 'shared' module
# regardless of where the script is executed from.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Insert the project root directory at the beginning of sys.path to ensure
# that the 'shared' package can be imported successfully.
sys.path.insert(0, project_root)

from shared.paths import MY_DSPRITES, MY_DSPRITES_REPORT, MY_DSPRITES_CONTRAST, MY_DSPRITES_CONTRAST_REPORT, ANALYSES_DIR
from shared.report_generator import ReportGenerator

# Create the analysis directory if it does not already exist.
# This directory will store the generated reports and analysis outputs.
os.makedirs(ANALYSES_DIR, exist_ok=True)

# %%
# Cell 2: Dataset Selection
# Create an interactive ToggleButtons widget using ipywidgets to allow the user
# to select between the 'Standard' and 'Contrast' datasets.
# This provides a user-friendly interface in Jupyter notebooks to toggle analysis modes.
dataset_widget = widgets.ToggleButtons(options=['Standard', 'Contrast'], description='Dataset:')
display(dataset_widget)

# %%
# Cell 3: Data Loading
# Determine whether to analyze the contrast dataset based on the user's selection in the widget.
# The boolean flag 'analyze_contrast' is set to True if 'Contrast' is selected, and False otherwise.
analyze_contrast = (dataset_widget.value == 'Contrast')

# Based on the 'analyze_contrast' flag, select the appropriate dataset file path (.npz)
# and the corresponding report output path. This ensures that the correct data is loaded
# and the results are written to the correct report file.
if analyze_contrast:
    dataset_path = MY_DSPRITES_CONTRAST
    report_path = MY_DSPRITES_CONTRAST_REPORT
else:
    dataset_path = MY_DSPRITES
    report_path = MY_DSPRITES_REPORT

# Load the selected dataset using numpy's np.load function.
data = np.load(dataset_path)
imgs = data['imgs']
shape_and_scale_labels = data['shape_and_scale_labels']
shape_labels = data['shape_labels']
scale_labels = data['scale_labels']
orientation_labels = data['orientation_labels']
pos_x_labels = data['pos_x_labels']
pos_y_labels = data['pos_y_labels']

# Create a dictionary mapping human-readable feature names to their corresponding numpy label arrays.
# This mapping simplifies iterating over features during analysis and validation checks.
labels_dict = {
    'shape_and_scale': shape_and_scale_labels,
    'shape': shape_labels,
    'scale': scale_labels,
    'orientation': orientation_labels,
    'pos_x': pos_x_labels,
    'pos_y': pos_y_labels
}

# If the contrast dataset is selected, extract the contrast labels and add them to the labels dictionary.
if analyze_contrast:
    contrast_labels = data['contrast_labels']
    labels_dict['contrast'] = contrast_labels

# Instantiate the ReportGenerator with the selected report path to document the analysis.
report = ReportGenerator(report_path)

print("\nDataset Overview")
print(f"Total images in dataset: {len(imgs)}")

# %%
# Cell 4: Marginal Uniformity Check
# Define descriptions for each feature to provide context during the uniformity check.
feature_descriptions = {
    "shape_and_scale": "Combined 9-class target. 0-2: Square (Small, Med, Large), 3-5: Ellipse (Small, Med, Large), 6-8: Heart (Small, Med, Large).",
    "shape": "Base shape. 0: Square, 1: Ellipse, 2: Heart.",
    "scale": "Base scale. 0: Small, 1: Medium, 2: Large.",
    "orientation": "Rotation angle. 40 distinct values evenly spaced from 0 to 2π.",
    "pos_x": "Horizontal position. 32 distinct values.",
    "pos_y": "Vertical position. 32 distinct values.",
    "contrast": "Contrast level. 0: Standard Anchor, 1: Faded Shift, 2: Dark Inversion, 3: Misty Inversion, 4: Extreme Edge Case."
}

# Iterate through each feature in the labels dictionary to verify marginal uniformity.
# Marginal uniformity means that each distinct value of a feature occurs with the exact same frequency.
# The mathematical logic is: expected_count = total_images / num_distinct.
# We count the unique values using np.unique and verify if all counts match this expected count,
# indicating a perfectly flat distribution across all categories/values of that feature.
for feature_name, labels in labels_dict.items():
    # Find the unique values and their respective counts for the current feature
    distinct_values, counts = np.unique(labels, return_counts=True)
    print(f"Feature: {feature_name}")
    print(f"Description: {feature_descriptions[feature_name]}")
    print(f"Distinct values: {distinct_values}")
    
    # Calculate the expected count for a perfectly uniform distribution
    total_images = len(labels)
    num_distinct = len(distinct_values)
    expected_count = total_images / num_distinct
    
    # Check if all actual counts are equal to the expected count
    marginal_uniform = np.all(counts == expected_count)
    
    if marginal_uniform:
        print(f"Result: Marginal uniformity holds. For each distinct value, there are exactly {int(expected_count)} images, as expected ({total_images} total / {num_distinct} distinct values).")
    else:
        print(f"Result: Marginal uniformity does not hold. Counts vary across distinct values.")
    print()

# %%
# Cell 5: Joint Uniformity Check
print("This step checks if the dataset contains all possible combinations of the base generative latent factors.")

# Combine the base generative factors into a single 2D array using np.column_stack.
# This operation stacks the 1D label arrays as columns, representing each image as a vector of its latent factors.
# This allows us to analyze the Cartesian product of all distinct latent factor values.
if analyze_contrast:
    stacked = np.column_stack([shape_labels, scale_labels, orientation_labels, pos_x_labels, pos_y_labels, contrast_labels])
else:
    stacked = np.column_stack([shape_labels, scale_labels, orientation_labels, pos_x_labels, pos_y_labels])

# Find the unique combinations (rows) and their counts to verify joint uniformity.
unique_combinations, counts = np.unique(stacked, axis=0, return_counts=True)

total_images = len(stacked)
num_combinations = len(unique_combinations)
# Calculate the expected count for each combination if the joint distribution is perfectly uniform.
expected_joint_count = total_images / num_combinations

# Check if every single combination appears the exact same number of times in the dataset.
# If joint_uniform is True, it confirms that the dataset is a complete and balanced Cartesian product of the factors.
joint_uniform = np.all(counts == expected_joint_count)

print(f"There are {num_combinations} unique combinations, corresponding to the Cartesian product of all distinct latent factor values.")

if joint_uniform:
    print(f"Result: Joint uniformity holds. Every combination appears exactly {int(expected_joint_count)} times ({total_images} total / {num_combinations} combinations).")
else:
    print(f"Result: Joint uniformity does not hold. Some combinations appear more than others.")

# %%
# Cell 6: Statistical Independence Check
print("\nStatistical Independence Check")
print("Cramér's V will be used to measure effect size, expecting exactly 0.0 for perfect independence (base factors) or 1.0 for perfect dependence (engineered features).")
print("Expected outcomes:")
print("- The engineered 'shape_and_scale' feature is expected to be highly dependent on 'shape' and 'scale' (by construction).")
print("- All other base generative factors (shape, scale, orientation, pos_x, pos_y) should be mutually independent because they were generated via a Cartesian product.")
print()

# Get the list of feature names to perform pairwise independence tests.
feature_names = list(labels_dict.keys())

# Use nested loops to compare every unique pair of features (i, j) where i < j.
# This avoids redundant comparisons and self-comparisons.
for i in range(len(feature_names)):
    for j in range(i + 1, len(feature_names)):
        name1, name2 = feature_names[i], feature_names[j]
        labels1, labels2 = labels_dict[name1], labels_dict[name2]
        
        # Construct a 2D contingency table (cross-tabulation) for the two features.
        # The table cell at (idx1, idx2) stores the frequency of co-occurrence of val1 from feature1 and val2 from feature2.
        unique1 = np.unique(labels1)
        unique2 = np.unique(labels2)
        contingency_table = np.zeros((len(unique1), len(unique2)), dtype=int)
        
        for idx1, val1 in enumerate(unique1):
            for idx2, val2 in enumerate(unique2):
                contingency_table[idx1, idx2] = np.sum((labels1 == val1) & (labels2 == val2))
        
        # Perform the Chi-Square test of independence on the contingency table.
        # This test evaluates whether the distribution of the two features is independent.
        # A p-value > 0.05 indicates that we fail to reject the null hypothesis of independence,
        # meaning the two features are statistically independent.
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Calculate Cramér's V to measure effect size
        n = np.sum(contingency_table)
        min_dim = min(contingency_table.shape) - 1
        if min_dim == 0:
            cramers_v = 0.0
        else:
            cramers_v = np.sqrt(chi2 / (n * min_dim))
        cramers_v = round(cramers_v, 4)
        
        print(f"Pair: {name1} and {name2}")
        print(f"Chi-square statistic: {chi2:.4f}")
        print(f"Cramér's V: {cramers_v:.4f}")
        
        if np.isclose(cramers_v, 0.0, atol=1e-4):
            print("Confirmed perfect statistical independence.")
        elif np.isclose(cramers_v, 1.0, atol=1e-4):
            print("Confirmed perfect dependence.")
        else:
            print(f"Warning: Unexpected intermediate dependence. p-value: {p_value:.4f}")
        print()

# %%
# Cell 7: Visualization
# Set up a 3x3 grid of subplots to visualize one random sample for each of the 9 shape_and_scale classes.
fig, axes = plt.subplots(3, 3, figsize=(8, 8))

shapes = ["Square", "Ellipse", "Heart"]
scales = ["Small", "Medium", "Large"]

# Define the maximum pixel value based on the dataset type.
# The contrast dataset has pixel values up to 20, while the standard dataset is binary (0 or 1).
max_pixel_value = 20 if analyze_contrast else 1

# Iterate through each of the 9 classes (0 to 8) representing combinations of shape and scale.
for class_idx in range(9):
    # Masking logic: isolate the indices in the dataset where the 'shape_and_scale' label matches the current class index.
    class_indices = np.where(shape_and_scale_labels == class_idx)[0]
    
    # Random sampling: select exactly one random index from the isolated class indices.
    random_idx = np.random.choice(class_indices)
    
    # Calculate the grid position (row and column) for the current class index.
    # Row is determined by integer division (class_idx // 3), and column by modulo (class_idx % 3).
    row = class_idx // 3
    col = class_idx % 3
    
    # Access the specific axis in the 3x3 subplot grid.
    ax = axes[row, col]
    
    # Display the randomly sampled image on the axis with appropriate colormap and intensity scaling.
    ax.imshow(imgs[random_idx], cmap='gray', vmin=0, vmax=max_pixel_value)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set titles for the columns (scales) and labels for the rows (shapes) to make the grid readable.
    if row == 0:
        ax.set_title(scales[col])
    if col == 0:
        ax.set_ylabel(shapes[row])

plt.tight_layout()
# Save the generated figure to the report using the report generator's kroki method.
report.kroki(fig, title="Random Samples per Shape & Scale Class")

# %%
# Cell 8: Contrast Variants Visualization
# If the contrast dataset is selected, visualize the exact same base image across all 5 contrast variants.
if analyze_contrast:
    # The contrast dataset is structured such that the 5 contrast variants of each base image
    # are stored at regular intervals. The base size represents the number of unique base images (total / 5).
    base_size = len(imgs) // 5
    
    # Select a random index within the range of the base images.
    random_index = np.random.randint(0, base_size)
    
    # Set up a 1x5 grid of subplots to display the 5 contrast variants side-by-side.
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    titles = ["Standard", "Faded", "Dark Inv", "Misty Inv", "Extreme"]
    
    # Iterate through the 5 contrast variants.
    for k in range(5):
        # Calculate the exact index of the k-th contrast variant for the selected base image.
        # Since the variants are stacked sequentially in blocks of size 'base_size',
        # the index is calculated as: random_index + k * base_size.
        calculated_index = random_index + k * base_size
        
        # Display the image variant with the appropriate colormap and intensity scaling (0 to 20).
        axes[k].imshow(imgs[calculated_index], cmap='gray', vmin=0, vmax=20)
        axes[k].set_title(titles[k])
        axes[k].set_xticks([])
        axes[k].set_yticks([])
        
    plt.tight_layout()
    # Save the contrast variants figure to the report.
    report.kroki(fig, title="Random Image Across All Contrast Variants")

# %%
# Cell 9: Teardown
# Close the report generator to finalize and save the report document.
report.close()