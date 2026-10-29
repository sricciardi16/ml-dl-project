# `data/` Artifact Directory Specification

This document specifies the structure, naming conventions, schemas, and lineage of every artifact produced by the pipeline. The entire `data/` tree is generated at runtime, is excluded from version control via `.gitignore`, and is created and resolved dynamically by `shared/paths.py`.

No file in this document is authored by hand. All paths are produced by the constants and helper functions defined in `shared/paths.py`.

---

## 1. Directory Hierarchy Overview

The directory tree rooted at `data/` is:

```
data/
├── a_datasets/                          # DATASETS_DIR
│   ├── my_dsprites.npz
│   ├── my_dsprites_contrast.npz
│   ├── my_dsprites_split.npz
│   └── my_dsprites_contrast_split.npz
├── b_models/
│   ├── trained/                         # MODELS_TRAINED_DIR
│   ├── evaluations/                     # MODELS_EVAL_DIR
│   └── latent_features/                 # MODELS_LATENT_DIR
├── c_attacks/
│   └── untargeted/
│       ├── datasets/                    # UNTARGETED_ATTACKS_DIR
│       └── evaluations/                 # UNTARGETED_EVAL_DIR
└── _analyses/                           # ANALYSES_DIR
```

| Subdirectory | `shared/paths.py` constant |
| --- | --- |
| `data/a_datasets/` | `DATASETS_DIR` |
| `data/b_models/trained/` | `MODELS_TRAINED_DIR` |
| `data/b_models/evaluations/` | `MODELS_EVAL_DIR` |
| `data/b_models/latent_features/` | `MODELS_LATENT_DIR` |
| `data/c_attacks/untargeted/datasets/` | `UNTARGETED_ATTACKS_DIR` |
| `data/c_attacks/untargeted/evaluations/` | `UNTARGETED_EVAL_DIR` |
| `data/_analyses/` | `ANALYSES_DIR` |

Auxiliary base constants also defined in `shared/paths.py`:

- `DATA_DIR = <PROJECT_ROOT>/data`
- `MODELS_BASE_DIR = DATA_DIR/b_models`
- `ATTACKS_BASE_DIR = DATA_DIR/c_attacks`

`PROJECT_ROOT` is computed as the parent directory of the `shared/` package directory (`os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`), so all paths are absolute and independent of the working directory.

---

## 2. Path Resolution Helpers (`shared/paths.py`)

### 2.1 Dataset constants

| Constant | Resolved path |
| --- | --- |
| `MY_DSPRITES` | `data/a_datasets/my_dsprites.npz` |
| `MY_DSPRITES_SPLIT` | `data/a_datasets/my_dsprites_split.npz` |
| `MY_DSPRITES_CONTRAST` | `data/a_datasets/my_dsprites_contrast.npz` |
| `MY_DSPRITES_CONTRAST_SPLIT` | `data/a_datasets/my_dsprites_contrast_split.npz` |

### 2.2 Report constants (`ANALYSES_DIR`)

| Constant | Resolved path |
| --- | --- |
| `MY_DSPRITES_REPORT` | `data/_analyses/01_my_dsprites_analysis.md` |
| `MY_DSPRITES_CONTRAST_REPORT` | `data/_analyses/02_my_dsprites_contrast_analysis.md` |
| `MY_DSPRITES_SPLIT_REPORT` | `data/_analyses/03_my_dsprites_split_analysis.md` |
| `MY_DSPRITES_CONTRAST_SPLIT_REPORT` | `data/_analyses/04_my_dsprites_contrast_split_analysis.md` |
| `MODELS_TRAINING_REPORT` | `data/_analyses/05_models_training_analysis.md` |
| `UNTARGETED_ATTACKS_REPORT` | `data/_analyses/06_untargeted_attacks_analysis.md` |

### 2.3 Helper functions

All helpers use a single naming rule: a file base name is `{task}_{dataset}_{pool_type}` and, when `loss_weight is not None`, the suffix `_w{weight}` is appended, where `{weight}` is the string form of the loss weight with every `.` replaced by `p` (e.g., `0.5 → w0p5`, `1.0 → w1p0`, `2.0 → w2p0`).

```python
def get_model_paths(task, dataset, pool_type, loss_weight=None):
    # returns (model_path, history_path, dot_path) in MODELS_TRAINED_DIR

def get_eval_paths(task, dataset, pool_type, loss_weight=None):
    # returns (primary_eval_path, secondary_eval_path) in MODELS_EVAL_DIR

def get_latent_features_path(task, dataset, pool_type, loss_weight=None):
    # returns a single .npz path in MODELS_LATENT_DIR

def get_untargeted_attack_path(dataset, pool_type, attack, norm, eps, loss_weight=None):
    # returns a single .npz path in UNTARGETED_ATTACKS_DIR

def get_attack_eval_paths(dataset, pool_type, attack, norm, loss_weight=None):
    # returns a single .csv.gz path in UNTARGETED_EVAL_DIR
```

Argument conventions across all helpers:

- `task ∈ {'single', 'multi'}`
- `dataset ∈ {'standard', 'contrast'}`
- `pool_type ∈ {'flatten', 'gap'}`
- `loss_weight ∈ {0.5, 1.0, 2.0}` (or `None` for single-output models)
- `attack ∈ {'fgsm', 'pgd'}`
- `norm ∈ {'linf', 'l2'}`
- `eps ∈ {0.01, 0.05, 0.10, 0.15, 0.20}` for `linf` or `{0.10, 0.50, 1.00, 2.00, 3.00}` for `l2`.

Numeric strings embedded in file names replace the decimal point with `p`: for epsilons `0.01 → eps0p01`, `0.05 → eps0p05`, `0.1 → eps0p1`, `0.15 → eps0p15`, `0.2 → eps0p2`, `0.5 → eps0p5`, `1.0 → eps1p0`, `2.0 → eps2p0`, `3.0 → eps3p0`.

---

## 3. `data/a_datasets/`

Four NumPy archives produced by `a_datasets/01_build_my_dsprites.py`, `02_build_my_dsprites_contrast.py`, `03_split_my_dsprites.py`, and `04_split_my_dsprites_contrast.py`. All images are stored with shape `(N, 64, 64)`. Keys are listed exactly as written by `np.savez_compressed`.

### 3.1 `my_dsprites.npz`

7 arrays. Total samples `N = 368,640`.

| Key | Shape | dtype / range | Description |
| --- | --- | --- | --- |
| `imgs` | `(368640, 64, 64)` | `{0, 1}` | Binary images retained at scales `0, 3, 5`. |
| `shape_and_scale_labels` | `(368640,)` | `{0..8}` | Composite target `shape * 3 + scale`. |
| `shape_labels` | `(368640,)` | `{0, 1, 2}` | `0=Square`, `1=Ellipse`, `2=Heart`. |
| `scale_labels` | `(368640,)` | `{0, 1, 2}` | `0=Small`, `1=Medium`, `2=Large`. |
| `orientation_labels` | `(368640,)` | `{0..39}` | Rotation category. |
| `pos_x_labels` | `(368640,)` | `{0..31}` | x-position category. |
| `pos_y_labels` | `(368640,)` | `{0..31}` | y-position category. |

### 3.2 `my_dsprites_contrast.npz`

8 arrays. Total samples `N = 1,843,200` (`= 5 × 368,640`), laid out as 5 contiguous blocks of `368,640` images.

| Key | Shape | dtype / range | Description |
| --- | --- | --- | --- |
| `imgs` | `(1843200, 64, 64)` | `uint8`, `[0, 20]` | Pixel intensities for the 5 contrast classes. |
| `shape_and_scale_labels` | `(1843200,)` | `{0..8}` | Tiled 5× from the base dataset. |
| `shape_labels` | `(1843200,)` | `{0, 1, 2}` | Tiled 5×. |
| `scale_labels` | `(1843200,)` | `{0, 1, 2}` | Tiled 5×. |
| `contrast_labels` | `(1843200,)` | `int8`, `{0..4}` | Contrast class; block `k` is `contrast == k`. |
| `orientation_labels` | `(1843200,)` | `{0..39}` | Tiled 5×. |
| `pos_x_labels` | `(1843200,)` | `{0..31}` | Tiled 5×. |
| `pos_y_labels` | `(1843200,)` | `{0..31}` | Tiled 5×. |

Contrast classes on the integer `[0, 20]` scale (background / foreground):

| Class | Background | Foreground | Polarity |
| --- | --- | --- | --- |
| 0 (Standard Anchor) | 0 | 20 | normal |
| 1 (Faded Shift) | 8 | 14 | normal |
| 2 (Dark Inversion) | 16 | 0 | inverted |
| 3 (Misty Inversion) | 12 | 4 | inverted |
| 4 (Extreme Edge Case) | 10 | 11 | near-threshold |

### 3.3 `my_dsprites_split.npz`

21 arrays: `3` image arrays + `6` label features × `3` splits. Split sizes: train `294,912`, val `36,864`, test `36,864`.

| Key | Shape |
| --- | --- |
| `X_train` | `(294912, 64, 64)` |
| `X_val` | `(36864, 64, 64)` |
| `X_test` | `(36864, 64, 64)` |
| `y_shape_and_scale_{train,val,test}` | `(294912/36864/36864,)` |
| `y_shape_{train,val,test}` | `(294912/36864/36864,)` |
| `y_scale_{train,val,test}` | `(294912/36864/36864,)` |
| `y_orientation_{train,val,test}` | `(294912/36864/36864,)` |
| `y_pos_x_{train,val,test}` | `(294912/36864/36864,)` |
| `y_pos_y_{train,val,test}` | `(294912/36864/36864,)` |

where `{train,val,test}` expands to the three suffixes (`_train`, `_val`, `_test`), yielding the full set of 18 label arrays. `X_test` is the unified standard test set of size `N = 36,864` used by all downstream evaluation and attack scripts.

### 3.4 `my_dsprites_contrast_split.npz`

24 arrays: the 21 keys of `my_dsprites_split.npz` plus `y_contrast_{train,val,test}`. Split sizes are `5×` the standard split: train `1,474,560`, val `184,320`, test `184,320`.

| Key | Shape |
| --- | --- |
| `X_train` | `(1474560, 64, 64)` |
| `X_val` | `(184320, 64, 64)` |
| `X_test` | `(184320, 64, 64)` |
| `y_shape_and_scale_{train,val,test}` | `(1474560/184320/184320,)` |
| `y_shape_{train,val,test}` | `(1474560/184320/184320,)` |
| `y_scale_{train,val,test}` | `(1474560/184320/184320,)` |
| `y_contrast_{train,val,test}` | `(1474560/184320/184320,)` |
| `y_orientation_{train,val,test}` | `(1474560/184320/184320,)` |
| `y_pos_x_{train,val,test}` | `(1474560/184320/184320,)` |
| `y_pos_y_{train,val,test}` | `(1474560/184320/184320,)` |

Lineage: `a_datasets/04_split_my_dsprites_contrast.py` partitions the first `N = len(imgs) // 5` base indices using the same stratified `train_test_split` (`random_state=29`) applied to the standard dataset, then expands each base index `i` to `i + k·N` for `k ∈ {0,1,2,3,4}` and shuffles. Consequently the standard test split is exactly the `contrast == 0` subset of the contrast test split, with zero cross-dataset leakage.

---

## 4. `data/b_models/trained/`

Produced by `b_models/01_train_single.py` and `b_models/02_train_multi.py` via `get_model_paths(task, dataset, pool_type, loss_weight)`.

**Naming pattern:** `{task}_{dataset}_{pool_type}[_w{weight}]_{model.keras | history.npy | dot.dot}`

**File counts:** 16 model configurations × 3 files = **48 files**.

| Task | Configurations | Files per config |
| --- | --- | --- |
| `single` | `standard`, `contrast` × `flatten`, `gap` (4) | `_model.keras`, `_history.npy`, `_dot.dot` |
| `multi` | `standard`, `contrast` × `flatten`, `gap` × `w0p5`, `w1p0`, `w2p0` (12) | `_model.keras`, `_history.npy`, `_dot.dot` |

Examples:

- `single_standard_flatten_model.keras`
- `multi_contrast_gap_w0p5_history.npy`
- `multi_standard_flatten_w2p0_dot.dot`

| Suffix | Content |
| --- | --- |
| `_model.keras` | Saved model (`.keras` format) with best-validation-loss weights. |
| `_history.npy` | NumPy pickled `History.history` dictionary of per-epoch loss/accuracy series. |
| `_dot.dot` | Graphviz `.dot` architecture diagram string. |

---

## 5. `data/b_models/evaluations/`

Produced by `b_models/03_evaluate_single.py` and `b_models/04_evaluate_multi.py` via `get_eval_paths(task, dataset, pool_type, loss_weight)`.

**Naming pattern:** `{base_name}_eval.json` and `{base_name}_standard_test_eval.json`, where `{base_name} = {task}_{dataset}_{pool_type}[_w{weight}]`.

**File counts:** 16 primary `_eval.json` (all 16 models) + 8 secondary `_standard_test_eval.json` (the 8 contrast-trained models only) = **24 JSON files**.

- `_eval.json`: primary evaluation. For `dataset='standard'`, evaluated on `MY_DSPRITES_SPLIT`. For `dataset='contrast'`, evaluated on `MY_DSPRITES_CONTRAST_SPLIT` with images divided by `20.0`.
- `_standard_test_eval.json`: secondary baseline. Written only for `dataset='contrast'` models; evaluated on `MY_DSPRITES_SPLIT` **without** dividing by `20.0`, to establish the unified baseline on the standard test set.

### 5.1 JSON schema — single-output model

```json
{
  "test_loss_total": <float>,
  "shape_and_scale_task": {
    "loss": <float>,
    "accuracy": <float>,
    "classification_report": { ... sklearn output_dict ... },
    "confusion_matrix": [[ ... ]],
    "misclassifications": [
      {"index": <int>, "true_label": <int>, "predicted_label": <int>}, ...
    ]
  },
  "model_summary": "<string from model.summary()>"
}
```

### 5.2 JSON schema — multi-output model

```json
{
  "test_loss_total": <float>,
  "shape_and_scale_task": { "loss": <float>, "accuracy": <float>, "classification_report": {...}, "confusion_matrix": [[...]], "misclassifications": [...] },
  "shape_task":          { "loss": <float>, "accuracy": <float>, "classification_report": {...}, "confusion_matrix": [[...]], "misclassifications": [...] },
  "scale_task":          { "loss": <float>, "accuracy": <float>, "classification_report": {...}, "confusion_matrix": [[...]], "misclassifications": [...] },
  "model_summary": "<string from model.summary()>"
}
```

Per-task nested keys: `loss`, `accuracy`, `classification_report` (`sklearn.metrics.classification_report(..., output_dict=True)`), `confusion_matrix` (row-major list), and `misclassifications` (list of `{index, true_label, predicted_label}`).

---

## 6. `data/b_models/latent_features/`

Produced by `b_models/05_extract_latent_features.py` via `get_latent_features_path(task, dataset, pool_type, loss_weight)`.

**Naming pattern:**

- `single_{dataset}_{pool_type}_latent.npz` (4 files)
- `multi_{dataset}_{pool_type}_w{weight}_latent.npz` (12 files)

**File counts:** **16 `.npz` files**.

Each archive stores a single array `latent_test` of shape `(36864, D)`, where `D` is the trunk output dimensionality:

| `pool_type` | `D` | Source layer |
| --- | --- | --- |
| `flatten` | `8192` (`8 × 8 × 128`) | `Flatten` layer output |
| `gap` | `128` | `GlobalAveragePooling2D` layer output |

Latent features are always extracted on `X_test` from `MY_DSPRITES_SPLIT` (`N = 36,864`).

---

## 7. `data/c_attacks/untargeted/datasets/`

Produced by `c_attacks/01_untargeted_attacks.py` via `get_untargeted_attack_path(dataset, pool_type, attack, norm, eps, loss_weight)`.

**Naming pattern:** `untargeted_adv_{dataset}_{pool_type}_w{weight}_{attack}_{norm}_eps{eps_str}.npz`

**File counts:** 12 multi-output models × 20 attack settings = **240 `.npz` archives**.

Example: `untargeted_adv_standard_gap_w1p0_pgd_linf_eps0p05.npz`.

Each archive stores 7 arrays:

| Key | Shape | dtype / range |
| --- | --- | --- |
| `X_test` | `(36864, 64, 64)` | `float32`, `[0.0, 1.0]` | adversarial images (channel axis squeezed) |
| `y_shape_and_scale_test` | `(36864,)` | `{0..8}` |
| `y_shape_test` | `(36864,)` | `{0, 1, 2}` |
| `y_scale_test` | `(36864,)` | `{0, 1, 2}` |
| `y_orientation_test` | `(36864,)` | `{0..39}` |
| `y_pos_x_test` | `(36864,)` | `{0..31}` |
| `y_pos_y_test` | `(36864,)` | `{0..31}` |

All 20 attack settings per model are the cross product of `attack ∈ {fgsm, pgd}`, `norm ∈ {linf, l2}`, and 5 epsilon budgets (see Section 2.3). Label arrays are identical across all 20 settings (they are the ground-truth labels of the unified standard test set).

---

## 8. `data/c_attacks/untargeted/evaluations/`

Produced by `c_attacks/02_evaluate_untargeted.py` via `get_attack_eval_paths(dataset, pool_type, attack, norm, loss_weight)`.

**Naming pattern:** `eval_{dataset}_{pool_type}_w{weight}_{attack}_{norm}_details.csv.gz`

**File counts:** 12 multi-output models × 4 `(attack, norm)` pairs = **48 gzip-compressed CSV files**.

Example: `eval_contrast_flatten_w0p5_fgsm_l2_details.csv.gz`.

Each CSV contains **14 columns** and **184,320 rows** (`= 5 epsilon steps × 36,864 samples`):

| Column | Type | Description |
| --- | --- | --- |
| `sample_index` | int | `np.arange(36864)`, repeated per epsilon block. |
| `epsilon` | float | Perturbation budget of the row's block. |
| `y_shape_and_scale` | int | Ground-truth composite label `{0..8}`. |
| `y_shape` | int | Ground-truth shape `{0,1,2}`. |
| `y_scale` | int | Ground-truth scale `{0,1,2}`. |
| `pred_main` | int | Predicted composite class `argmax p_main`. |
| `pred_shape` | int | Predicted shape class `argmax p_shape`. |
| `pred_scale` | int | Predicted scale class `argmax p_scale`. |
| `confidence` | float | `max p_main`. |
| `entropy` | float | Raw Shannon entropy (nats) of `p_main`. |
| `confidence_shape` | float | `max p_shape`. |
| `entropy_shape` | float | Raw Shannon entropy (nats) of `p_shape`. |
| `confidence_scale` | float | `max p_scale`. |
| `entropy_scale` | float | Raw Shannon entropy (nats) of `p_scale`. |

Predictions, confidences, and entropies are computed on the full 3-output network `f(x')` after `tf.nn.softmax` of the raw logits; entropy uses a `1e-9` numerical stability offset inside the logarithm.

---

## 9. `data/_analyses/`

Markdown reports generated by the analysis scripts using `shared/report_generator.py` (`ReportGenerator`), which captures stdout into fenced code blocks and embeds figures/diagrams via the `kroki` method.

| Report file | Generating script |
| --- | --- |
| `01_my_dsprites_analysis.md` | `a_datasets/05_analyze_my_dsprites.py` (Standard toggle) |
| `02_my_dsprites_contrast_analysis.md` | `a_datasets/05_analyze_my_dsprites.py` (Contrast toggle) |
| `03_my_dsprites_split_analysis.md` | `a_datasets/06_analyze_splits.py` (Standard toggle) |
| `04_my_dsprites_contrast_split_analysis.md` | `a_datasets/06_analyze_splits.py` (Contrast toggle) |
| `05_models_training_analysis.md` | `b_models/06_analyze.py` |
| `06_untargeted_attacks_analysis.md` | `c_attacks/04_analyze_untargeted.py` |

The `ReportGenerator` context manager redirects `sys.stdout` during a script run so that printed text is written into the target Markdown file as fenced code blocks, and renders embedded matplotlib figures and Graphviz objects as inline SVG/PNG through the external Kroki CLI. The constants `MY_DSPRITES_REPORT`, `MY_DSPRITES_CONTRAST_REPORT`, `MY_DSPRITES_SPLIT_REPORT`, `MY_DSPRITES_CONTRAST_SPLIT_REPORT`, `MODELS_TRAINING_REPORT`, and `UNTARGETED_ATTACKS_REPORT` (Section 2.2) resolve to these six files.

Note: `c_attacks/03_analyze_untargeted_eda.py` prints interactive summary tables and provides `ipywidgets` tools but does not persist a Markdown report to `data/_analyses/`.

---

## 10. Artifact Summary

| Directory | Producer scripts | File type | Count |
| --- | --- | --- | --- |
| `data/a_datasets/` | `a_datasets/01`–`04` | `.npz` | 4 |
| `data/b_models/trained/` | `b_models/01`, `02` | `.keras`, `.npy`, `.dot` | 48 |
| `data/b_models/evaluations/` | `b_models/03`, `04` | `.json` | 24 |
| `data/b_models/latent_features/` | `b_models/05` | `.npz` | 16 |
| `data/c_attacks/untargeted/datasets/` | `c_attacks/01` | `.npz` | 240 |
| `data/c_attacks/untargeted/evaluations/` | `c_attacks/02` | `.csv.gz` | 48 |
| `data/_analyses/` | `a_datasets/05`, `06`, `b_models/06`, `c_attacks/04` | `.md` | 6 |