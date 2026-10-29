# Dataset Generation, Splitting & Validation Pipeline (`a_datasets/`)

This directory contains the operational scripts for Stage 1 of the experimental pipeline: constructing two controlled variants of the DeepMind dSprites dataset—a binary base dataset (`my_dsprites`) and a $5\times$ contrast-augmented dataset (`my_dsprites_contrast`)—partitioning both into 80/10/10 train/validation/test splits without cross-dataset leakage, and validating their statistical properties.

For file path constants and archive schemas, see [`data/README.md`](../data/README.md) and [`shared/paths.py`](../shared/paths.py).

---

## 1. Pipeline Overview & Source Data

The dataset pipeline derives from the DeepMind dSprites archive (`dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz`), which contains $737,280$ binary $64 \times 64$ images generated from the Cartesian product of 5 independent ground-truth latent factors:
1. **Shape:** 3 categories (`0=Square`, `1=Ellipse`, `2=Heart`).
2. **Scale:** 6 categories (indices `0..5`).
3. **Orientation:** 40 discrete rotation angles in $[0, 2\pi]$.
4. **Position X (`pos_x`):** 32 horizontal positions (`0..31`).
5. **Position Y (`pos_y`):** 32 vertical positions (`0..31`).

All scripts in `a_datasets/` enforce deterministic execution via `PYTHONHASHSEED='29'`, `random.seed(29)`, and `np.random.seed(29)`.

---

## 2. Script-by-Script Functional Breakdown

### 2.1 `01_build_my_dsprites.py` — Base Dataset Construction

Downloads the raw DeepMind dSprites archive directly into memory (`urllib.request.urlopen` + `io.BytesIO`, loaded with `encoding='latin1'`), filters the scale dimension, engineers the 9-class composite target, and serializes the base dataset.

- **Idempotency Check:** Exits immediately (`sys.exit(0)`) if `MY_DSPRITES` (`data/a_datasets/my_dsprites.npz`) already exists.
- **Scale Filtering & Contiguous Remapping:** Retains only rows where `latents_classes[:, 2]` is in `{0, 3, 5}` (`TARGET_SCALE_0 = 0`, `TARGET_SCALE_3 = 3`, `TARGET_SCALE_5 = 5`), reducing the dataset by half to $N = 3 \times 3 \times 40 \times 32 \times 32 = 368,640$ images of shape $(64, 64)$. Selected scale indices are remapped via `np.select` to contiguous integers:
  - Original scale `0` $\to$ `0` (`Small`)
  - Original scale `3` $\to$ `1` (`Medium`)
  - Original scale `5` $\to$ `2` (`Large`)
- **Composite Target Engineering:** Constructs the 9-class main classification label ($40,960$ samples per class):
  $$\text{shape\_and\_scale\_labels} = (\text{shape\_labels} \times 3) + \text{scale\_labels} \in \{0, \dots, 8\}$$
- **Output Archive (`data/a_datasets/my_dsprites.npz`):** Compressed NumPy archive containing 7 arrays:
  - `imgs`: `(368640, 64, 64)` binary images (`{0, 1}`).
  - `shape_and_scale_labels`: `(368640,)` composite class indices (`0..8`).
  - `shape_labels`: `(368640,)` shape indices (`0..2`).
  - `scale_labels`: `(368640,)` remapped scale indices (`0..2`).
  - `orientation_labels`: `(368640,)` orientation indices (`0..39`).
  - `pos_x_labels`: `(368640,)` horizontal position indices (`0..31`).
  - `pos_y_labels`: `(368640,)` vertical position indices (`0..31`).

---

### 2.2 `02_build_my_dsprites_contrast.py` — Contrast Dataset Construction

Loads `data/a_datasets/my_dsprites.npz` and expands it by a factor of $5$ ($5N = 1,843,200$ images) across 5 discrete contrast classes mapped onto an integer intensity scale in $[0, 20]$ (`uint8`), where each integer unit corresponds to $0.05$ after division by `20.0` during model preprocessing.

- **Idempotency Check:** Exits immediately if `MY_DSPRITES_CONTRAST` (`data/a_datasets/my_dsprites_contrast.npz`) already exists.
- **Contrast Class Specification:**

| Contrast Class | Class Name | Background (`uint8` / float) | Foreground (`uint8` / float) | Polarity |
| :--- | :--- | :---: | :---: | :--- |
| **Class 0** | Standard Anchor | `0` ($0.00$) | `20` ($1.00$) | Normal (matches standard binary `{0.0, 1.0}`) |
| **Class 1** | Faded Shift | `8` ($0.40$) | `14` ($0.70$) | Normal |
| **Class 2** | Dark Inversion | `16` ($0.80$) | `0` ($0.00$) | Inverted |
| **Class 3** | Misty Inversion | `12` ($0.60$) | `4` ($0.20$) | Inverted |
| **Class 4** | Extreme Edge Case | `10` ($0.50$) | `11` ($0.55$) | Normal (near-threshold $|\Delta| = 1$) |

- **Contiguous Block Allocation & Label Tiling:** Pre-allocates `expanded_imgs` of shape $(1843200, 64, 64)$ (`np.uint8`) and `contrast_labels` of shape $(1843200,)$ (`np.int8`). For each `class_idx` $k \in \{0, 1, 2, 3, 4\}$, writes block $[kN : (k+1)N]$ via `np.where(imgs == 1, np.uint8(fg), np.uint8(bg))`. All 6 original label arrays are duplicated 5 times via `np.tile(..., 5)`.
- **Output Archive (`data/a_datasets/my_dsprites_contrast.npz`):** Compressed NumPy archive containing 8 arrays:
  - `imgs`: `(1843200, 64, 64)` `uint8` images in `[0, 20]`.
  - `shape_and_scale_labels`, `shape_labels`, `scale_labels`, `orientation_labels`, `pos_x_labels`, `pos_y_labels`: `(1843200,)` tiled label arrays.
  - `contrast_labels`: `(1843200,)` `int8` contrast indices (`0..4`).

---

### 2.3 `03_split_my_dsprites.py` — Base Dataset Stratified Splitting

Partitions `data/a_datasets/my_dsprites.npz` into 80% training ($294,912$), 10% validation ($36,864$), and 10% test ($36,864$) subsets.

- **Idempotency Check:** Exits immediately if `MY_DSPRITES_SPLIT` (`data/a_datasets/my_dsprites_split.npz`) already exists.
- **Two-Stage Stratified Split (`random_state=29`):**
  1. Splits `indices = np.arange(368640)` into `train_idx` (80%) and `temp_idx` (20%) via `train_test_split(..., test_size=0.2, random_state=29, stratify=shape_and_scale_labels)`.
  2. Splits `temp_idx` into `val_idx` (10%) and `test_idx` (10%) via `train_test_split(..., test_size=0.5, random_state=29, stratify=shape_and_scale_labels[temp_idx])`.
- **Output Archive (`data/a_datasets/my_dsprites_split.npz`):** Compressed NumPy archive containing **21 arrays** (7 features/labels $\times$ 3 splits):
  - Images: `X_train` `(294912, 64, 64)`, `X_val` `(36864, 64, 64)`, `X_test` `(36864, 64, 64)`
  - Composite target: `y_shape_and_scale_train`, `y_shape_and_scale_val`, `y_shape_and_scale_test`
  - Shape target: `y_shape_train`, `y_shape_val`, `y_shape_test`
  - Scale target: `y_scale_train`, `y_scale_val`, `y_scale_test`
  - Orientation factor: `y_orientation_train`, `y_orientation_val`, `y_orientation_test`
  - Horizontal position factor: `y_pos_x_train`, `y_pos_x_val`, `y_pos_x_test`
  - Vertical position factor: `y_pos_y_train`, `y_pos_y_val`, `y_pos_y_test`

---

### 2.4 `04_split_my_dsprites_contrast.py` — Base-First Contrast Dataset Splitting

Partitions `data/a_datasets/my_dsprites_contrast.npz` into 80% training ($1,474,560$), 10% validation ($184,320$), and 10% test ($184,320$) subsets using a **base-first splitting strategy** to prevent cross-dataset evaluation leakage.

- **Idempotency Check:** Exits immediately if `MY_DSPRITES_CONTRAST_SPLIT` (`data/a_datasets/my_dsprites_contrast_split.npz`) already exists.
- **Cross-Dataset Leakage Prevention & Base-First Algorithm:**
  Downstream evaluation and adversarial testing require evaluating contrast-trained models on the standard test set (`X_test` from `my_dsprites_split.npz`). If `my_dsprites_contrast.npz` were split independently across all $1,843,200$ images, contrast variants ($k \in \{1, 2, 3, 4\}$) of a base geometric instance `(shape, scale, orientation, pos_x, pos_y)` present in the standard test set would appear in the contrast training set.
  To guarantee zero cross-dataset leakage:
  1. **Base Index Isolation:** Computes $N = \text{len}(\text{imgs}) // 5 = 368,640$ and isolates `base_indices = np.arange(N)` (Block 0, corresponding to `Contrast 0`) and `base_stratify_key = shape_and_scale_labels[:N]`.
  2. **Identical Base Partitioning:** Executes the exact same two sequential `train_test_split` calls (`test_size=0.2` then `test_size=0.5`, `random_state=29`, stratified on `base_stratify_key`), yielding `train_base_idx`, `val_base_idx`, and `test_base_idx` identical to `03_split_my_dsprites.py`.
  3. **5-Block Index Expansion:** Expands each base index array across all 5 contiguous contrast blocks via:
     $$\text{expand\_indices}(\text{base\_idx}, N) = [i + k \cdot N \;\text{for}\; i \in \text{base\_idx} \;\text{for}\; k \in \{0, 1, 2, 3, 4\}]$$
  4. **In-Place Shuffling:** Applies `np.random.shuffle` (`seed=29`) to `train_idx`, `val_idx`, and `test_idx` to break the deterministic 5-contrast-class repeating order before saving.
  This guarantees that all 5 contrast variants of any base geometric instance reside in the same split, and that the standard test set is an exact subset (`contrast == 0`) of the contrast test set.
- **Output Archive (`data/a_datasets/my_dsprites_contrast_split.npz`):** Compressed NumPy archive containing **24 arrays** (the 21 array keys from `my_dsprites_split.npz` with $5\times$ sample counts, plus the contrast label arrays):
  - Images: `X_train` `(1474560, 64, 64)`, `X_val` `(184320, 64, 64)`, `X_test` `(184320, 64, 64)`
  - Targets & factors: `y_shape_and_scale_{train,val,test}`, `y_shape_{train,val,test}`, `y_scale_{train,val,test}`, `y_contrast_{train,val,test}`, `y_orientation_{train,val,test}`, `y_pos_x_{train,val,test}`, `y_pos_y_{train,val,test}`

---

### 2.5 `05_analyze_my_dsprites.py` — Full-Dataset Statistical Validation

Interactive `# %%` cell script using `ipywidgets.ToggleButtons(options=['Standard', 'Contrast'])` to validate `my_dsprites.npz` or `my_dsprites_contrast.npz`.

- **Validation Checks Performed:**
  1. **Marginal Uniformity:** Verifies deterministically that every distinct class of every factor (`shape_and_scale`, `shape`, `scale`, `orientation`, `pos_x`, `pos_y`, and `contrast` when selected) has count exactly equal to $\text{Total Images} / \text{Number of Distinct Values}$.
  2. **Joint Uniformity:** Stacks all base generative factors (`5` for Standard, `6` for Contrast) via `np.column_stack` and verifies via `np.unique(..., axis=0, return_counts=True)` that all Cartesian-product combinations ($368,640$ or $1,843,200$) appear exactly once (`expected_joint_count == 1`).
  3. **Pairwise Statistical Independence:** Computes contingency tables, $\chi^2$ statistics (`scipy.stats.chi2_contingency`), and Cramér's $V = \sqrt{\chi^2 / (n \cdot \min(R-1, C-1))}$ for all feature pairs, verifying $V = 1.0000$ between `shape_and_scale` and its constituent factors (`shape`, `scale`) and $V = 0.0000$ across all base generative factor pairs.
  4. **Visual Inspection Grids:** Generates and embeds a $3 \times 3$ random sample grid across the 9 `shape_and_scale` classes and (for Contrast) a $1 \times 5$ strip showing a single randomly selected base image across all 5 contrast classes (`Standard`, `Faded`, `Dark Inv`, `Misty Inv`, `Extreme`).
- **Generated Reports (`ReportGenerator`):**
  - Standard: `data/_analyses/01_my_dsprites_analysis.md` (`MY_DSPRITES_REPORT`)
  - Contrast: `data/_analyses/02_my_dsprites_contrast_analysis.md` (`MY_DSPRITES_CONTRAST_REPORT`)

---

### 2.6 `06_analyze_splits.py` — Split Integrity & Leakage Validation

Interactive `# %%` cell script using `ipywidgets.ToggleButtons(options=['Standard', 'Contrast'])` to validate `my_dsprites_split.npz` or `my_dsprites_contrast_split.npz` across `train`, `val`, and `test`.

- **Validation Checks Performed:**
  1. **Split Marginal Uniformity:**
     - *Stratified Features (`shape_and_scale`, `shape`, `scale`, and `contrast`):* Deterministic verification that every class count equals $\text{Split Size} / K$ exactly and normalized Shannon entropy $H / \ln K = 1.0000$.
     - *Unstratified Spatial/Rotational Features (`orientation`, `pos_x`, `pos_y`):* Computes normalized Shannon entropy and a one-way $\chi^2$ goodness-of-fit test (`scipy.stats.chisquare`) against a uniform distribution ($p \ge 0.05$ threshold), and embeds $1 \times 3$ distribution bar charts per split via `report.kroki`.
  2. **Split Joint Uniformity:** Verifies that all combinations of stratified base factors (`shape`, `scale`, and `contrast`) occur with exact uniform frequency within each split.
  3. **Split Pairwise Independence:** Computes $\chi^2$, Cramér's $V$, and Normalized Mutual Information (`sklearn.metrics.normalized_mutual_info_score`) for all feature pairs within each split to confirm that partitioning introduced no spurious correlations.
  4. **Intra-Dataset Zero-Leakage Verification:** Extracts sets of 5-tuples `(shape, scale, orientation, pos_x, pos_y)` for `train`, `val`, and `test` and asserts zero pairwise intersection (`len(train & val) == 0`, `len(train & test) == 0`, `len(val & test) == 0`).
  5. **Cross-Dataset Contrast Subset Verification (Contrast mode):** Loads `my_dsprites_split.npz` alongside `my_dsprites_contrast_split.npz` and asserts for every split (`train`, `val`, `test`):
     - `len(contrast_split) == 5 * len(standard_split)`
     - Exact set equality between the 5-tuple base instances in the standard split and the `contrast == 0` subset of the contrast split.
- **Generated Reports (`ReportGenerator`):**
  - Standard: `data/_analyses/03_my_dsprites_split_analysis.md` (`MY_DSPRITES_SPLIT_REPORT`)
  - Contrast: `data/_analyses/04_my_dsprites_contrast_split_analysis.md` (`MY_DSPRITES_CONTRAST_SPLIT_REPORT`)

---

## 3. Execution Guide

### 3.1 Environment Setup

The dataset construction and validation scripts run on CPU and can be executed using either a lightweight Python virtual environment (`shared/requirements.txt`) or the project Conda environment (`shared/environment.yml`).

**Option A — Python `venv` (CPU):**
```bash
python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

pip install -r shared/requirements.txt
```

**Option B — Conda Environment:**
```bash
conda env create -f shared/environment.yml
conda activate ml-dl-project-training
```

### 3.2 Sequential CLI Execution (Dataset Construction & Splitting)

Run the four build and split scripts sequentially from the **project root**:

```bash
# 1. Build base dataset (data/a_datasets/my_dsprites.npz)
python a_datasets/01_build_my_dsprites.py

# 2. Build 5x contrast dataset (data/a_datasets/my_dsprites_contrast.npz)
python a_datasets/02_build_my_dsprites_contrast.py

# 3. Split base dataset 80/10/10 (data/a_datasets/my_dsprites_split.npz)
python a_datasets/03_split_my_dsprites.py

# 4. Split contrast dataset 80/10/10 via base-first strategy (data/a_datasets/my_dsprites_contrast_split.npz)
python a_datasets/04_split_my_dsprites_contrast.py
```

### 3.3 Interactive Validation & Report Generation

The validation scripts `a_datasets/05_analyze_my_dsprites.py` and `a_datasets/06_analyze_splits.py` are structured with `# %%` interactive cells and `ipywidgets.ToggleButtons` (`Standard` vs. `Contrast`):

1. Open `a_datasets/05_analyze_my_dsprites.py` or `a_datasets/06_analyze_splits.py` in VS Code (Jupyter Interactive Window) or an IPython-enabled notebook environment.
2. Execute Cell 1 and Cell 2 to display the `Dataset:` toggle widget (`Standard` or `Contrast`).
3. Select the desired dataset variant and execute the remaining cells sequentially through the final `report.close()` cell to generate the corresponding Markdown report in `data/_analyses/`.