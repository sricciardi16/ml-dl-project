# Adversarial Robustness of Multi-Output CNNs on a Filtered dSprites Dataset

## 1. Research Problem

A multi-output neural network $f(x) = (f_{\rm main}(x), f_{\rm shape}(x), f_{\rm scale}(x))$ is trained to classify a composite target $y_{\rm main} = (y_{\rm shape}, y_{\rm scale}) \in {\rm Shape} \times {\rm Scale}$ ($9$ classes), where $y_{\rm main}$ is a deterministic function of two independent generative factors: ${\rm Shape}$ ($3$ classes) and ${\rm Scale}$ ($3$ classes). The network shares a convolutional trunk $\phi(x)$ feeding three parallel classification heads — the composite head $f_{\rm main}$ and two auxiliary factor heads $f_{\rm shape}$ and $f_{\rm scale}$.

An untargeted adversarial perturbation $\delta$ ($\|\delta\|_p \le \varepsilon$) is optimized exclusively against $f_{\rm main} = h_{\rm main} \circ \phi$, without any gradient signal from $f_{\rm shape}$ or $f_{\rm scale}$. Passing the perturbed input $x' = x + \delta$ through the full network $f(x')$ tests which of the following outcomes occurs:

- **Consistent alignment** — the auxiliary heads $f_{\rm shape}$ and $f_{\rm scale}$ shift in agreement with the fooled $f_{\rm main}$, indicating that the perturbation corrupts shared trunk features $\phi(x)$.
- **Retention of ground truth** — the auxiliary heads retain the correct labels while only $f_{\rm main}$ is fooled, indicating that the attack exploits a shortcut specific to the main classification head.
- **Divergence** — the auxiliary heads produce predictions that contradict both $f_{\rm main}$ and the ground truth, indicating internal inconsistency in the shared representation.

The experiment is repeated across controlled model architecture factors $m = (d, \pi, w) \in \mathcal{M}$ ($12$ multi-output models) and attack factors $a = (\alpha, p, \varepsilon) \in \mathcal{A}$ ($20$ attack settings), yielding $240$ evaluation slices.

---

## 2. Pipeline Stages

The project is organized into three sequential pipeline stages plus shared utilities and full methodological documentation.

### Stage 1 — Dataset Construction, Splitting & Validation (`a_datasets/`)

Constructs two controlled variants of the DeepMind dSprites dataset:

- A binary base dataset (`my_dsprites`) filtering the original 6 scales to indices `0, 3, 5` (remapped to `0, 1, 2`), retaining 3 shapes, 40 orientations, and 32 × 32 positions ($N = 368{,}640$ images), and engineering the 9-class composite target `shape_and_scale = shape × 3 + scale`.
- A $5\times$ contrast-augmented dataset (`my_dsprites_contrast`) with 5 `uint8` contrast classes on the $[0, 20]$ intensity scale ($N = 1{,}843{,}200$ images).
- Leak-free base-first 80/10/10 stratified splits (`seed=29`) ensuring the standard test set ($N = 36{,}864$) is an exact subset of the contrast test set with zero cross-dataset leakage.
- Statistical validation of marginal uniformity, Cartesian-product joint uniformity, pairwise independence ($\chi^2$, Cramér's $V$), and zero split overlap.

**Operational guide:** [`a_datasets/README.md`](a_datasets/README.md)

### Stage 2 — Model Training, Evaluation & Latent Feature Analysis (`b_models/`)

Trains and evaluates 16 CNN configurations:

- 4 single-output baseline models (`single` task): $d \in \{{\tt standard}, {\tt contrast}\} \times \pi \in \{{\tt flatten}, {\tt gap}\}$, predicting only `shape_and_scale` ($9$ classes).
- 12 multi-output models (`multi` task): $d \in \{{\tt standard}, {\tt contrast}\} \times \pi \in \{{\tt flatten}, {\tt gap}\} \times w \in \{0.5, 1.0, 2.0\}$, simultaneously predicting `shape_and_scale` ($9$), `shape` ($3$), and `scale` ($3$) with total loss $\mathcal{L}_{\rm total} = w\,\mathcal{L}_{\rm main} + 1.0\,\mathcal{L}_{\rm shape} + 1.0\,\mathcal{L}_{\rm scale}$.

All models share a 3-block convolutional encoder $\phi(x)$ (`Conv2D → BatchNormalization → ReLU → MaxPooling2D`) followed by `Flatten` + `Dropout(0.5)` or `GlobalAveragePooling2D`, with classification heads outputting raw linear logits (`from_logits=True`). Contrast-trained models undergo dual evaluation on both the contrast test split and the standard test split to establish a unified baseline. Trunk latent features are extracted and analyzed with 2D PCA alongside convergence diagnostics.

**Operational guide:** [`b_models/README.md`](b_models/README.md)

### Stage 3 — Untargeted Adversarial Attack Generation, Evaluation & Analysis (`c_attacks/`)

Isolates the primary branch $f_{\rm main}(x) = h_{\rm main}(\phi(x))$ of each of the 12 multi-output models and generates untargeted FGSM and PGD perturbations on the unified standard test set ($N = 36{,}864$) across:

- $L_\infty$ ($\varepsilon \in \{0.01, 0.05, 0.10, 0.15, 0.20\}$)
- $L_2$ ($\varepsilon \in \{0.10, 0.50, 1.00, 2.00, 3.00\}$)

This yields 20 attack settings per model and $12 \times 20 = 240$ evaluation slices. The full 3-head network $f(x')$ is evaluated on all perturbed images, and a comprehensive statistical and structural analysis is executed covering attack success rates, main-branch shift typologies, auxiliary-branch reaction codes, composite regimes (Full Alignment, Full Retention, Partial Retention, Divergence), confidence/entropy uncertainty metrics, sink-collapse detection, and Bernoulli KL-deviance variance attribution.

**Operational guide:** [`c_attacks/README.md`](c_attacks/README.md)

---

## 3. Repository Structure

```
.
├── a_datasets/          # Stage 1: dataset construction, splitting, validation
│   └── README.md        #    Operational guide for the dataset pipeline
├── b_models/            # Stage 2: model training, evaluation, latent features, analysis
│   └── README.md        #    Operational guide for the modeling pipeline
├── c_attacks/           # Stage 3: attack generation, evaluation, EDA, analysis
│   └── README.md        #    Operational guide for the adversarial pipeline
├── data/                # Generated artifacts (git-ignored, managed by shared/paths.py)
│   └── README.md        #    Directory hierarchy, schemas, and naming conventions
├── shared/              # Cross-stage utilities
│   ├── paths.py          # Centralized path resolution (PROJECT_ROOT, DATA_DIR, helpers)
│   ├── report_generator.py  # ReportGenerator: stdout-to-Markdown + Kroki SVG/Graphviz
│   ├── check_gpu.py      # TensorFlow physical GPU verification (exits non-zero if none)
│   ├── requirements.txt  # Lightweight CPU environment for a_datasets/ (pip)
│   └── environment.yml   # Conda GPU environment: ml-dl-project-training (Python 3.10)
└── README.md            # This file
```

---

## 4. Shared Utilities

| File | Purpose |
| :--- | :--- |
| [`shared/paths.py`](shared/paths.py) | Centralized absolute path constants (`DATASETS_DIR`, `MODELS_TRAINED_DIR`, `MODELS_EVAL_DIR`, `MODELS_LATENT_DIR`, `UNTARGETED_ATTACKS_DIR`, `UNTARGETED_EVAL_DIR`, `ANALYSES_DIR`) and helper functions (`get_model_paths`, `get_eval_paths`, `get_latent_features_path`, `get_untargeted_attack_path`, `get_attack_eval_paths`) used by all pipeline scripts. |
| [`shared/report_generator.py`](shared/report_generator.py) | `ReportGenerator` context manager that intercepts `sys.stdout` to capture printed output into Markdown code blocks and provides a `kroki()` method to embed matplotlib figures and Graphviz diagrams via the Kroki CLI. |
| [`shared/check_gpu.py`](shared/check_gpu.py) | Pre-check utility that queries `tf.config.list_physical_devices('GPU')`; prints detected GPU names and exits `0` on success, or prints a critical warning and exits `1` if no GPU is found. |
| [`shared/requirements.txt`](shared/requirements.txt) | Lightweight pip dependency manifest (`numpy`, `scikit-learn`, `scipy`, `matplotlib`, `pandas`, `tensorflow`, `ipywidgets`, `pydot`, `graphviz`) sufficient for running `a_datasets/` scripts. |
| [`shared/environment.yml`](shared/environment.yml) | Conda environment `ml-dl-project-training` (Python 3.10) with `tensorflow[and-cuda]`, `adversarial-robustness-toolbox`, and `TF_FORCE_GPU_ALLOW_GROWTH=true`; required for `b_models/` and `c_attacks/` GPU workloads. |

---

## 5. Artifact Directory

All generated artifacts (datasets, trained models, evaluation reports, adversarial examples, and analysis reports) are stored under `data/`, which is git-ignored and dynamically managed by `shared/paths.py`.

**Artifact specification:** [`data/README.md`](data/README.md)

---

## 6. Environment Setup

### CPU Environment (Dataset Pipeline — `a_datasets/`)

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows
pip install -r shared/requirements.txt
```

### GPU Environment (Model & Attack Pipelines — `b_models/`, `c_attacks/`)

```bash
conda env create -f shared/environment.yml
conda activate ml-dl-project-training
```

### GPU Verification

Before running GPU-dependent scripts, verify hardware availability:

```bash
python shared/check_gpu.py
```

The script exits with code `1` and a critical warning if no GPU is detected.

---

## 7. End-to-End Execution Sequence

All commands are executed from the project root. Each stage depends on the artifacts produced by the preceding stage.

### Stage 1 — Dataset Preparation

```bash
python a_datasets/01_build_my_dsprites.py
python a_datasets/02_build_my_dsprites_contrast.py
python a_datasets/03_split_my_dsprites.py
python a_datasets/04_split_my_dsprites_contrast.py
```

Scripts `a_datasets/05_analyze_my_dsprites.py` and `a_datasets/06_analyze_splits.py` are interactive `# %%` cell scripts using `ipywidgets`; run them in a Jupyter Notebook or IPython-compatible IDE. They generate analysis reports in `data/_analyses/`.

### Stage 2 — Model Training & Evaluation

```bash
# GPU verification
python shared/check_gpu.py

# Single-output baselines (4 models)
python b_models/01_train_single.py --dataset standard --pool_type flatten
python b_models/01_train_single.py --dataset standard --pool_type gap
python b_models/01_train_single.py --dataset contrast --pool_type flatten
python b_models/01_train_single.py --dataset contrast --pool_type gap

# Multi-output models (12 models)
for dataset in standard contrast; do
    for pool_type in flatten gap; do
        for weight in 0.5 1.0 2.0; do
            python b_models/02_train_multi.py --dataset "$dataset" --pool_type "$pool_type" --loss_weight "$weight"
        done
    done
done

# Evaluation (single-output)
for dataset in standard contrast; do
    for pool_type in flatten gap; do
        python b_models/03_evaluate_single.py --dataset "$dataset" --pool_type "$pool_type"
    done
done

# Evaluation (multi-output)
for dataset in standard contrast; do
    for pool_type in flatten gap; do
        for weight in 0.5 1.0 2.0; do
            python b_models/04_evaluate_multi.py --dataset "$dataset" --pool_type "$pool_type" --loss_weight "$weight"
        done
    done
done
```

### Stage 3 — Latent Feature Extraction

```bash
python b_models/05_extract_latent_features.py
```

### Stage 4 — Model Analysis (Interactive)

Run `b_models/06_analyze.py` in a Jupyter Notebook or IPython-compatible IDE. It generates `data/_analyses/05_models_training_analysis.md`.

### Stage 5 — Adversarial Attack Generation & Evaluation

```bash
for dataset in standard contrast; do
    for pool_type in flatten gap; do
        for weight in 0.5 1.0 2.0; do
            python c_attacks/01_untargeted_attacks.py --dataset "$dataset" --pool_type "$pool_type" --loss_weight "$weight"
            python c_attacks/02_evaluate_untargeted.py --dataset "$dataset" --pool_type "$pool_type" --loss_weight "$weight"
        done
    done
done
```

### Stage 6 — Post-Hoc Analysis

- `c_attacks/03_analyze_untargeted_eda.py` — interactive Phase-1 exploratory data analysis (`ipywidgets` Sankey diagram and image viewer). Run in a Jupyter Notebook or IPython-compatible IDE.
- `c_attacks/04_analyze_untargeted.py` — the primary target analysis script. Generates the complete analytical report at `data/_analyses/06_untargeted_attacks_analysis.md`.

```bash
python c_attacks/04_analyze_untargeted.py
```

### Generated Analysis Reports

All six analysis reports are written to `data/_analyses/`:

| Report | Generated by |
| :--- | :--- |
| `01_my_dsprites_analysis.md` | `a_datasets/05_analyze_my_dsprites.py` |
| `02_my_dsprites_contrast_analysis.md` | `a_datasets/05_analyze_my_dsprites.py` |
| `03_my_dsprites_split_analysis.md` | `a_datasets/06_analyze_splits.py` |
| `04_my_dsprites_contrast_split_analysis.md` | `a_datasets/06_analyze_splits.py` |
| `05_models_training_analysis.md` | `b_models/06_analyze.py` |
| `06_untargeted_attacks_analysis.md` | `c_attacks/04_analyze_untargeted.py` |
