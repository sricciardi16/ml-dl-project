"""
Dataset Preparation Script for Custom dSprites Dataset
========================================================

This script downloads the dSprites dataset, filters it to include only specific
scales, and saves a processed version with engineered features.

Expected Final Dataset Structure (my_dsprites.npz):
----------------------------------------------------
The output .npz file contains the following arrays:

1. imgs: numpy.ndarray
   - Shape: (N, 64, 64)
   - Binary images of the filtered sprites
   - N = number of samples with scales 0, 3, or 5

2. shape_and_scale_labels: numpy.ndarray
   - Shape: (N,)
   - Combined 9-class labels (shape * 3 + remapped_scale)
   - Values range from 0 to 8

3. shape_labels: numpy.ndarray
   - Shape: (N,)
   - Shape category (0=square, 1=ellipse, 2=heart)
   - Values: 0, 1, or 2

4. scale_labels: numpy.ndarray
   - Shape: (N,)
   - Remapped scale values (original 0,3,5 → 0,1,2)
   - Values: 0, 1, or 2

5. orientation_labels: numpy.ndarray
   - Shape: (N,)
   - Orientation index (40 possible values)
   - Values: 0 to 39

6. pos_x_labels: numpy.ndarray
   - Shape: (N,)
   - Horizontal position index (32 possible values)
   - Values: 0 to 31

7. pos_y_labels: numpy.ndarray
   - Shape: (N,)
   - Vertical position index (32 possible values)
   - Values: 0 to 31

Note: This dataset contains only the filtered scales (0, 3, 5) mapped to a
9-class classification problem (3 shapes × 3 scales).
"""

# Imports Section:
# 'io' is used for in-memory byte stream handling.
# 'os' is used for path manipulation to construct platform-independent file paths.
# 'random' is used for generating random numbers and setting seeds.
# 'sys' is used for early script termination via exit codes.
# 'urllib.request' is used for downloading the dataset over HTTP.
# 'numpy' is used for array manipulation and saving processed data.
import io
import os
import random
import sys
import urllib.request
import numpy as np

# Set seed to 29 (User's birth date) for strict determinism
os.environ['PYTHONHASHSEED'] = '29'
random.seed(29)
np.random.seed(29)

# Path Variables:
# Calling absolute path on the current file gets the script's location, and navigating up
# the directory tree twice (via dirname) reaches the project root.
# The processed data path is the final output file.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from shared.paths import DATASETS_DIR, MY_DSPRITES

# Column Index Constants:
# These constants define the column indices in the latents_classes array,
# making the code more readable and maintainable by replacing magic numbers.
COL_COLOR = 0       # Color latent (always 1 value in dSprites)
COL_SHAPE = 1       # Shape latent (3 values: square, ellipse, heart)
COL_SCALE = 2       # Scale latent (6 values representing different sizes)
COL_ORIENTATION = 3 # Orientation latent (40 values covering 0 to 2*pi radians)
COL_POS_X = 4       # Position X latent (32 values for horizontal position)
COL_POS_Y = 5       # Position Y latent (32 values for vertical position)

# Target Scale Constants:
# These constants define the specific scale values we want to extract from the dataset.
# We filter to only include scales 0, 3, and 5 for our custom 9-class problem.
TARGET_SCALE_0 = 0  # Smallest scale, will be remapped to 0
TARGET_SCALE_3 = 3  # Medium scale, will be remapped to 1
TARGET_SCALE_5 = 5  # Largest scale, will be remapped to 2

# Idempotency Check:
# This checks if the final output ('my_dsprites.npz') already exists.
# If it does, the script exits with code 0 (success) to prevent redundant downloading
# and processing, which saves time in automated pipelines.
if os.path.exists(MY_DSPRITES):
    print("Data is already prepared.")
    sys.exit(0)

# Directory Creation:
# Using 'exist_ok=True' ensures the 'data/dataset/' directory exists without throwing an error
# if it is already present.
os.makedirs(DATASETS_DIR, exist_ok=True)

# In-Memory Download Logic:
# The script downloads the official 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz' file
# directly from DeepMind's GitHub repository into memory.
# Using urlopen and BytesIO avoids unnecessary disk I/O by keeping the raw dataset
# entirely in memory during the download and loading process.
url = 'https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
print(f"Downloading dSprites dataset from {url}...")
response = urllib.request.urlopen(url)
data_bytes = response.read()
data = np.load(io.BytesIO(data_bytes), encoding='latin1')
print("Download and load complete.")

# Data Loading:
# The 'encoding="latin1"' argument is required to load older NumPy archives (created in Python 2)
# into Python 3. The raw archive contains an 'imgs' array (the pixel data) and a 'latents_classes'
# array (the ground truth attributes for each image).
print("Processing data...")
imgs = data['imgs']
latents_classes = data['latents_classes']

# Structure of the raw dSprites latents_classes array:
# The latents_classes array is a 2D NumPy array where each row corresponds to a single image
# in the dataset, and each column represents a distinct generative factor (latent dimension).
# The dSprites dataset defines 6 generative factors in the following column order:
#   - Column 0: Color (always 1 value - white, since dSprites is grayscale)
#   - Column 1: Shape (3 possible values: square, ellipse, heart)
#   - Column 2: Scale (6 possible values representing different sizes)
#   - Column 3: Orientation (40 possible values covering 0 to 2*pi radians)
#   - Column 4: Position X (32 possible values for horizontal position)
#   - Column 5: Position Y (32 possible values for vertical position)
# This structured representation allows for precise indexing and filtering of images
# based on any combination of these generative factors.

# Filtering Logic (Masking):
# We are building a custom dSprites dataset (my_dsprites) by isolating exactly 3 specific
# scales (indices 0, 3, and 5) from the latent classes. This targeted extraction forms
# the foundation of the custom dataset used for downstream tasks.
# The boolean mask uses bitwise OR to identify rows where the scale matches our target values.
scale_col = latents_classes[:, COL_SCALE]
mask = (scale_col == TARGET_SCALE_0) | (scale_col == TARGET_SCALE_3) | (scale_col == TARGET_SCALE_5)

# Applying the Mask:
# Applying the boolean mask to both the images and latents arrays simultaneously ensures that
# the images and their corresponding labels remain perfectly aligned while discarding unwanted scales.
imgs = imgs[mask]
filtered_latents = latents_classes[mask]

# Attribute Extraction:
# Extracting attributes in sequential column index order for clarity and maintainability.
# Column 1 (COL_SHAPE) represents Shape.
shape_labels = filtered_latents[:, COL_SHAPE]
# Column 2 (COL_SCALE) represents Scale.
scale_array = filtered_latents[:, COL_SCALE]
# Column 3 (COL_ORIENTATION) represents Orientation.
orientation_labels = filtered_latents[:, COL_ORIENTATION]
# Column 4 (COL_POS_X) represents Position X.
pos_x_labels = filtered_latents[:, COL_POS_X]
# Column 5 (COL_POS_Y) represents Position Y.
pos_y_labels = filtered_latents[:, COL_POS_Y]

# Scale Remapping:
# The original scale values (0, 3, 5) are non-contiguous.
# The logic maps them to a contiguous range (0, 1, 2) respectively, which is necessary for
# standard classification tasks. The default fallback of -1 handles unexpected values.
scale_labels = np.select(
    [scale_array == TARGET_SCALE_0, scale_array == TARGET_SCALE_3, scale_array == TARGET_SCALE_5],
    [0, 1, 2],
    default=-1
)

# Label Engineering:
# The formula '(shape_labels * 3) + scale_labels' combines two categorical features.
# Since there are 3 shapes (values 0-2) and 3 remapped scales (values 0-2), multiplying the shape
# by 3 and adding the scale creates a unique, contiguous integer from 0 to 8.
# Mathematical logic: For each shape value s (0,1,2) and scale value k (0,1,2),
# the combined label = s * 3 + k produces:
#   - Shape 0 (square): labels 0, 1, 2 for scales 0, 1, 2
#   - Shape 1 (ellipse): labels 3, 4, 5 for scales 0, 1, 2
#   - Shape 2 (heart): labels 6, 7, 8 for scales 0, 1, 2
# This effectively flattens the two features into a single 9-class target variable.
shape_and_scale_labels = (shape_labels * 3) + scale_labels

# Saving Processed Data:
# This packages the filtered images and all the newly extracted/engineered label arrays into a
# single, space-efficient compressed archive. The specific keys used here are explicitly expected
# by the downstream '02_split_data.py' script.
print("Saving processed data...")
np.savez_compressed(
    MY_DSPRITES,
    imgs=imgs,
    shape_and_scale_labels=shape_and_scale_labels,
    shape_labels=shape_labels,
    scale_labels=scale_labels,
    orientation_labels=orientation_labels,
    pos_x_labels=pos_x_labels,
    pos_y_labels=pos_y_labels
)
print("Data preparation complete.")