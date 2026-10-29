# Model Training, Evaluation, and Analysis Pipeline

This directory trains, evaluates, and analyzes a grid of 16 Convolutional Neural Network configurations on the `my_dsprites` and `my_dsprites_contrast` dataset variants. The models serve two roles: single-output baselines that verify multi-task auxiliary heads do not degrade main-task representation or clean accuracy, and multi-output models that are the targets of the downstream adversarial robustness study.

For the artifact directory layout, file naming conventions, and schemas, see [`data/README.md`](../data/README.md).

---

## Experimental Matrix

The pipeline trains 16 models, defined by three controlled factors:

| Factor | Symbol | Values |
|---|---|---|
| Dataset variant | $d$ | `standard`, `contrast` |
| Spatial pooling | $\pi$ | `flatten`, `gap` |
| Main-task loss weight | $w$ | `0.5`, `1.0`, `2.0` (multi-output only) |

### Single-Output Baselines (4 models)

`task='single'`, predicting only `shape_and_scale` (9 classes):

- `single_standard_flatten`
- `single_standard_gap`
- `single_contrast_flatten`
- `single_contrast_gap`

### Multi-Output Models (12 models)

`task='multi'`, simultaneously predicting `shape_and_scale` (9 classes), `shape` (3 classes), and `scale` (3 classes). The total loss is:

$$\mathcal{L}_{\rm total} = w\,\mathcal{L}_{\rm main} + 1.0\,\mathcal{L}_{\rm shape} + 1.0\,\mathcal{L}_{\rm scale}$$

with `loss_weights = {'shape_and_scale': w, 'shape': 1.0, 'scale': 1.0}`.

- `multi_{standard,contrast}_{flatten,gap}_w{0p5,1p0,2p0}`

The 12 multi-output models are the targets of the untargeted adversarial attack analysis in `c_attacks/`. The 4 single-output models provide reference baselines.

---

## Script-by-Script Breakdown

### `01_train_single.py` — Single-Output Model Training

Trains a single-output Sequential CNN given `--dataset` (`standard` or `contrast`) and `--pool_type` (`flatten` or `gap`).

**Architecture:**
- Input: $(64, 64, 1)$ `float32` tensor in $[0.0, 1.0]$.
- Shared convolutional trunk (3 blocks):
  - Block 1: `Conv2D(32, (3,3), padding='same')` → `BatchNormalization()` → `Activation('relu')` → `MaxPooling2D((2,2))` (output $32 \times 32 \times 32$).
  - Block 2: `Conv2D(64, (3,3), padding='same')` → `BatchNormalization()` → `Activation('relu')` → `MaxPooling2D((2,2))` (output $16 \times 16 \times 64$).
  - Block 3: `Conv2D(128, (3,3), padding='same')` → `BatchNormalization()` → `Activation('relu')` → `MaxPooling2D((2,2))` (output $8 \times 8 \times 128$).
  - `BatchNormalization` is placed before `Activation('relu')` following Ioffe & Szegedy (2015), avoiding the "ReLU paradox" where post-ReLU normalization shifts zeroed negative activations back into negative values and destroys activation sparsity.
- Spatial pooling $\pi$:
  - `flatten`: `Flatten()` produces an 8,192-dimensional vector ($8 \times 8 \times 128$), followed by `Dropout(0.5)`.
  - `gap`: `GlobalAveragePooling2D()` averages each of the 128 spatial feature maps into a scalar, producing a 128-dimensional vector. No dropout is applied because GAP inherently acts as a structural regularizer.
- Classification head: `Dense(128, activation='relu')` → `Dense(9)` (raw linear logits).

**Preprocessing, compilation, and training:**
- Determinism: global seed `29` across `PYTHONHASHSEED`, `random`, `numpy`, `tf.random`, and `tf.keras.utils.set_random_seed(29)`.
- Inside the `tf.data` mapping function: cast `uint8` to `float32`; divide by `20.0` only when `--dataset contrast` (standard images are already in `{0, 1}`).
- `tf.data.Dataset` pipeline: training set uses `.shuffle(buffer_size=len(X_train), seed=29).map(..., num_parallel_calls=AUTOTUNE).batch(32).prefetch(AUTOTUNE)`; validation set omits `.shuffle`.
- Optimizer: `Adam(learning_rate=0.0001, clipnorm=1.0)`.
- Loss: `SparseCategoricalCrossentropy(from_logits=True)`.
- Hyperparameters: `epochs=50`, `batch_size=32`.
- Callbacks: `EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-8, restore_best_weights=True)`, `ModelCheckpoint(filepath=..., monitor='val_loss', save_best_only=True)`, `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)`.

**Outputs and idempotency:**
- Saves to `data/b_models/trained/`: `_model.keras`, `_history.npy`, `_dot.dot`.
- Idempotency check: if all three artifacts already exist, the script exits without retraining.

### `02_train_multi.py` — Multi-Output Model Training

Trains a multi-output Functional CNN given `--dataset`, `--pool_type`, and `--loss_weight` (`0.5`, `1.0`, `2.0`).

**Architecture:**
- Identical 3-block convolutional trunk $\phi(x)$ and pooling layer as the single-output model.
- After the pooling layer, the network splits into three parallel branches from `trunk_output`, each applying a conditional `Dropout(0.5)` (only when `--pool_type flatten`), `Dense(128, activation='relu')`, and a task-named linear logit `Dense` layer:
  - Branch 1 (`main`): `Dense(9, name='shape_and_scale')`.
  - Branch 2 (`shape`): `Dense(3, name='shape')`.
  - Branch 3 (`scale`): `Dense(3, name='scale')`.
- Each `flatten` branch has its own independent `Dropout(0.5)` layer applied to `trunk_output`. Under `gap`, no dropout is applied to any branch.

**Compilation and loss:**
- Optimizer: `Adam(learning_rate=0.0001, clipnorm=1.0)`.
- Loss functions: `SparseCategoricalCrossentropy(from_logits=True)` for all three heads.
- `loss_weights = {'shape_and_scale': args.loss_weight, 'shape': 1.0, 'scale': 1.0}`. The main-task loss weight $w$ is controlled by `--loss_weight` and is not fixed to `1.0`.
- Metrics: `{'shape_and_scale': ['accuracy'], 'shape': ['accuracy'], 'scale': ['accuracy']}`.
- Hyperparameters and callbacks identical to the single-output script: `epochs=50`, `batch_size=32`, `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`.

**Outputs and idempotency:**
- Saves to `data/b_models/trained/`: `_model.keras`, `_history.npy`, `_dot.dot`.
- Idempotency check: if all three artifacts already exist, the script exits without retraining.

### `03_evaluate_single.py` — Single-Output Model Evaluation

Evaluates a trained single-output model given `--dataset` and `--pool_type`.

**Inference mechanics:**
- `model.evaluate` and `model.predict` with `batch_size=4096`.
- Explicit `tf.nn.softmax(..., axis=-1)` applied to logits before `np.argmax(..., axis=1)`.
- Computes loss, accuracy, confusion matrix, `classification_report`, and misclassification records (`index`, `true_label`, `predicted_label`).
- Embeds the `model.summary()` string into the JSON output.
- JSON keys: `test_loss_total`, `shape_and_scale_task` (with `loss`, `accuracy`, `classification_report`, `confusion_matrix`, `misclassifications`), `model_summary`.

**Dual-evaluation protocol for `--dataset contrast`:**
1. Primary evaluation on `MY_DSPRITES_CONTRAST_SPLIT` with `is_contrast=True` (pixels divided by `20.0`), saved to `*_eval.json`.
2. Secondary baseline evaluation on `MY_DSPRITES_SPLIT` with `is_contrast=False` (pixels not divided by `20.0`), saved to `*_standard_test_eval.json`.

Because `MY_DSPRITES_CONTRAST` `Class 0` has foreground `20`, which becomes `20 / 20.0 = 1.0` after normalization, and `MY_DSPRITES_SPLIT` foreground pixels are already `1.0`, the standard test split is an exact in-distribution subset (`Class 0`) of the contrast test split. Dividing the standard dataset by `20.0` would corrupt foreground intensity to `0.05` and must not be done. This establishes a leak-free, unified baseline across all 16 models.

**Idempotency:**
- Standard models: exits if `*_eval.json` exists.
- Contrast models: exits if both `*_eval.json` and `*_standard_test_eval.json` exist.

### `04_evaluate_multi.py` — Multi-Output Model Evaluation

Evaluates a trained multi-output model given `--dataset`, `--pool_type`, and `--loss_weight` across all three heads.

**Inference mechanics:**
- `model.evaluate` and `model.predict` with `batch_size=4096`.
- Unpacks the three logit arrays (`probs_main`, `probs_shape`, `probs_scale`), applies `tf.nn.softmax(..., axis=-1)` to each, and extracts predicted classes via `np.argmax(..., axis=1)`.
- Computes per-head loss, accuracy, confusion matrix, `classification_report`, and misclassification records.
- Embeds the `model.summary()` string into the JSON output.
- JSON keys: `test_loss_total`, `shape_and_scale_task`, `shape_task`, `scale_task` (each with `loss`, `accuracy`, `classification_report`, `confusion_matrix`, `misclassifications`), `model_summary`.

**Dual-evaluation protocol for `--dataset contrast`:**
- Same two-step protocol as `03_evaluate_single.py`: primary on `MY_DSPRITES_CONTRAST_SPLIT` (`is_contrast=True`) saved to `*_eval.json`; secondary on `MY_DSPRITES_SPLIT` (`is_contrast=False`) saved to `*_standard_test_eval.json`.

**Idempotency:**
- Standard models: exits if `*_eval.json` exists.
- Contrast models: exits if both `*_eval.json` and `*_standard_test_eval.json` exist.

### `05_extract_latent_features.py` — Trunk Latent Feature Extraction

Extracts latent feature representations from all 16 trained models.

**Procedure:**
- Iterates over 4 single-task configurations (`dataset × pool_type`) and 12 multi-task configurations (`dataset × pool_type × loss_weight`).
- For each model, locates the shared trunk output layer by checking whether `'flatten'` or `'global_average_pooling2d'` is contained in the layer name.
- Instantiates a sub-model `tf.keras.Model(inputs=model.inputs, outputs=trunk_layer.output)` that outputs the trunk representation $\phi(x)$.
- Runs batched inference via `predict_on_batch` (batch size `4096`) on `X_test` from `MY_DSPRITES_SPLIT` ($N = 36{,}864$). No division by `20.0` is performed because standard test images are already in $[0.0, 1.0]$.
- After each model, releases memory with `del model`, `tf.keras.backend.clear_session()`, and `gc.collect()`.
- Saves a compressed `.npz` archive containing the `latent_test` array to `data/b_models/latent_features/`. The array shape is $(36864, 8192)$ for `flatten` ($8 \times 8 \times 128$) and $(36864, 128)$ for `gap`.
- Skips a configuration if its output file already exists.

### `06_analyze.py` — Training and PCA Analysis

Interactive `# %%` cell script that aggregates training diagnostics, evaluation summaries, architecture diagrams, and latent-space visualizations into `data/_analyses/05_models_training_analysis.md` via `ReportGenerator`.

**Contents:**
- **Training curves:** Single-output loss and accuracy curves in a $2 \times 2$ grid (`dataset × pool_type`); multi-output total loss and main-task accuracy curves, plus separate per-head loss and accuracy curves (`shape_and_scale`, `shape`, `scale`), in a $4 \times 3$ grid (`dataset × pool_type × loss_weight`).
- **Convergence metrics:** For each model, reports total epochs, best epoch (`argmin(val_loss) + 1`), final training/validation loss and accuracy, Generalization Gap (`val_loss[-1] - loss[-1]`), Accuracy Gap (`train_acc[-1] - val_acc[-1]`), and Validation Stability (`std(val_loss[-5:])`).
- **Accuracy verification:** Cross-checks that `accuracy == 1.0` against per-class precision, recall, and F1-score in the `classification_report`, for both primary and secondary (standard-test) evaluation JSONs of contrast models.
- **Architecture diagrams:** Renders the Graphviz `.dot` files (standard-only, since architecture does not vary by dataset or loss weight) via `pydot` and `report.kroki`.
- **2D PCA projections:** Applies `sklearn.decomposition.PCA(n_components=2)` to each `latent_test` array, subsamples 2,000 points with `np.random.seed(29)`, and scatter-plots them color-coded by the 9 `shape × scale` classes (blue hues for `Square`, green hues for `Ellipse`, red hues for `Heart`; light-to-dark shade for `Small`, `Medium`, `Large`). Four PCA figures are produced: single-task $2 \times 2$ grid, multi-task $4 \times 3$ grid (4 model families × 3 loss weights), standard vs. contrast comparison (`gap`, $w=0.5$), and a loss-weight trajectory (`con|gap`: $w=0.5 \to 1.0 \to 2.0$).

---

## Execution Instructions

### Environment Setup

Use the Conda GPU environment defined in `shared/environment.yml` (do not use `shared/requirements.txt`, which is a lightweight CPU environment for `a_datasets/`):

```bash
conda env create -f shared/environment.yml
conda activate ml-dl-project-training
```

### GPU Check

Verify TensorFlow can access a physical GPU before training:

```bash
python shared/check_gpu.py
```

### Train All 16 Models

Run from the project root. All commands are idempotent.

#### Single-Output Models

```bash
python b_models/01_train_single.py --dataset standard --pool_type flatten
python b_models/01_train_single.py --dataset standard --pool_type gap
python b_models/01_train_single.py --dataset contrast --pool_type flatten
python b_models/01_train_single.py --dataset contrast --pool_type gap
```

#### Multi-Output Models

```bash
python b_models/02_train_multi.py --dataset standard --pool_type flatten --loss_weight 0.5
python b_models/02_train_multi.py --dataset standard --pool_type flatten --loss_weight 1.0
python b_models/02_train_multi.py --dataset standard --pool_type flatten --loss_weight 2.0
python b_models/02_train_multi.py --dataset standard --pool_type gap --loss_weight 0.5
python b_models/02_train_multi.py --dataset standard --pool_type gap --loss_weight 1.0
python b_models/02_train_multi.py --dataset standard --pool_type gap --loss_weight 2.0
python b_models/02_train_multi.py --dataset contrast --pool_type flatten --loss_weight 0.5
python b_models/02_train_multi.py --dataset contrast --pool_type flatten --loss_weight 1.0
python b_models/02_train_multi.py --dataset contrast --pool_type flatten --loss_weight 2.0
python b_models/02_train_multi.py --dataset contrast --pool_type gap --loss_weight 0.5
python b_models/02_train_multi.py --dataset contrast --pool_type gap --loss_weight 1.0
python b_models/02_train_multi.py --dataset contrast --pool_type gap --loss_weight 2.0
```

### Evaluate All 16 Models

#### Single-Output Models

```bash
python b_models/03_evaluate_single.py --dataset standard --pool_type flatten
python b_models/03_evaluate_single.py --dataset standard --pool_type gap
python b_models/03_evaluate_single.py --dataset contrast --pool_type flatten
python b_models/03_evaluate_single.py --dataset contrast --pool_type gap
```

#### Multi-Output Models

```bash
python b_models/04_evaluate_multi.py --dataset standard --pool_type flatten --loss_weight 0.5
python b_models/04_evaluate_multi.py --dataset standard --pool_type flatten --loss_weight 1.0
python b_models/04_evaluate_multi.py --dataset standard --pool_type flatten --loss_weight 2.0
python b_models/04_evaluate_multi.py --dataset standard --pool_type gap --loss_weight 0.5
python b_models/04_evaluate_multi.py --dataset standard --pool_type gap --loss_weight 1.0
python b_models/04_evaluate_multi.py --dataset standard --pool_type gap --loss_weight 2.0
python b_models/04_evaluate_multi.py --dataset contrast --pool_type flatten --loss_weight 0.5
python b_models/04_evaluate_multi.py --dataset contrast --pool_type flatten --loss_weight 1.0
python b_models/04_evaluate_multi.py --dataset contrast --pool_type flatten --loss_weight 2.0
python b_models/04_evaluate_multi.py --dataset contrast --pool_type gap --loss_weight 0.5
python b_models/04_evaluate_multi.py --dataset contrast --pool_type gap --loss_weight 1.0
python b_models/04_evaluate_multi.py --dataset contrast --pool_type gap --loss_weight 2.0
```

### Extract Latent Features

```bash
python b_models/05_extract_latent_features.py
```

### Generate Training and PCA Analysis Report

`06_analyze.py` is an interactive `# %%` cell script. Run it in a Jupyter-compatible environment (e.g., VS Code with the Python extension) cell by cell, or execute it directly:

```bash
python b_models/06_analyze.py
```

### Sequential Run (All Configurations)

```bash
# Block 1: Single-output models
for dataset in standard contrast; do
    for pool_type in flatten gap; do
        python b_models/01_train_single.py --dataset "$dataset" --pool_type "$pool_type"
        python b_models/03_evaluate_single.py --dataset "$dataset" --pool_type "$pool_type"
    done
done

# Block 2: Multi-output models
for dataset in standard contrast; do
    for pool_type in flatten gap; do
        for weight in 0.5 1.0 2.0; do
            python b_models/02_train_multi.py --dataset "$dataset" --pool_type "$pool_type" --loss_weight "$weight"
            python b_models/04_evaluate_multi.py --dataset "$dataset" --pool_type "$pool_type" --loss_weight "$weight"
        done
    done
done

# Block 3: Latent feature extraction and analysis
python b_models/05_extract_latent_features.py
python b_models/06_analyze.py
```

---

## Related Documentation

- [Dataset pipeline (`a_datasets/`)](../a_datasets/README.md)
- [Adversarial attack pipeline (`c_attacks/`)](../c_attacks/README.md)
- [Data artifact directory specification (`data/README.md`)](../data/README.md)
- [Centralized path resolution (`shared/paths.py`)](../shared/paths.py)
- [Report generator (`shared/report_generator.py`)](../shared/report_generator.py)
- [GPU verification (`shared/check_gpu.py`)](../shared/check_gpu.py)