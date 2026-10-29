# Technical Specification: Untargeted Adversarial Attack Generation, Evaluation, and Analysis Pipeline

The `c_attacks/` module evaluates the adversarial vulnerability and multi-head behavior of the 12 multi-output models $m = (d, \pi, w) \in \mathcal{M}$, where $d \in \{{\tt std}, {\tt con}\}$ is the training dataset variant, $\pi \in \{{\tt flat}, {\tt gap}\}$ is the spatial pooling type, and $w \in \{0.5, 1.0, 2.0\}$ is the main-head loss weight. Each model $f(x) = (f_{\rm main}(x), f_{\rm shape}(x), f_{\rm scale}(x))$ shares a common convolutional trunk $\phi(x)$ and branches into three classification heads.

The pipeline is composed of four scripts:

1. `c_attacks/01_untargeted_attacks.py` — generates untargeted FGSM and PGD adversarial examples against the main branch only.
2. `c_attacks/02_evaluate_untargeted.py` — runs full-network inference on the perturbed images and logs per-sample predictions, confidence, and entropy for all three heads.
3. `c_attacks/03_analyze_untargeted_eda.py` — Phase-1 exploratory data analysis with conditional probability tables and interactive visualization tools.
4. `c_attacks/04_analyze_untargeted.py` — the primary target analysis of the project, producing a structured report with 16 embedded figures.

---

## Pipeline Purpose & Unified Baseline Rationale

The objective of this module is to determine whether an untargeted adversarial perturbation crafted exclusively against the composite head $f_{\rm main}$ alters the internal representations of the underlying generative factors. Because the main target $y_{\rm main} = (y_{\rm shape}, y_{\rm scale})$ is a deterministic function of two independent generative factors, an attack on $f_{\rm main}$ provides a controlled probe: if the auxiliary heads shift consistently with the fooled main head, shared trunk features $\phi(x)$ have been corrupted; if they retain the ground truth, a main-head shortcut has been exploited; if they diverge into mutually contradictory predictions, a three-way inconsistency has been induced.

Attacks are always generated and evaluated on the **standard test set** $\mathcal{X}_{\rm test}$ (`MY_DSPRITES_SPLIT`, $N = 36{,}864$), regardless of the training dataset $d$ of the target model. The standard test set has binary $\{0, 1\}$ pixel values, which match the normalized float range $[0.0, 1.0]$ expected by both standard-trained models (pixels already $1.0$) and contrast-trained models ($\mathrm{Class\ 0}$ foreground $20$ maps to $20 / 20.0 = 1.0$ after normalization). Using a single unified image set guarantees that all 12 models are attacked from an identical clean starting point, and—since the standard test set is an exact subset of the contrast test set—introduces no cross-dataset leakage.

---

## Formal Summary

### Model & Attack Grids

- **Model grid** $m = (d, \pi, w) \in \mathcal{M}$ with $|\mathcal{M}| = 12$.
- **Attack grid** $a = (\alpha, p, \varepsilon) \in \mathcal{A}$ with $|\mathcal{A}| = 20$:
  - $\alpha \in \{{\tt fgsm}, {\tt pgd}\}$ (single-step gradient sign/step vs. 40-step iterative attack with random initialization).
  - $p \in \{\infty, 2\}$ ($L_\infty$ uniform per-pixel bound vs. $L_2$ Euclidean energy bound).
  - 5 ordinal budgets per norm: $\varepsilon_\infty \in \{0.01, 0.05, 0.10, 0.15, 0.20\}$ and $\varepsilon_2 \in \{0.10, 0.50, 1.00, 2.00, 3.00\}$.
- This yields $|\mathcal{M} \times \mathcal{A}| = 12 \times 20 = 240$ evaluation slices, each containing $N = 36{,}864$ samples.

### Attack Target

An untargeted perturbation $\delta$ with $\|\delta\|_p \le \varepsilon$ is crafted solely against $f_{\rm main}(x) = h_{\rm main}(\phi(x))$, clipped to $[0, 1]^{64 \times 64 \times 1}$, producing $x' = x + \delta$. No gradient signal from $h_{\rm shape}$ or $h_{\rm scale}$ is used during generation. Evaluation is performed on the full network $f(x') = (\hat{y}_{\rm main}, \hat{y}_{\rm shape}, \hat{y}_{\rm scale})$.

### Three-Stage Conditional Probability Hierarchy

Evaluated on $\mathcal{X}_{\rm test}$ and the full network $f(x')$:

- **Node 0 — Attack Outcome on `main`:**
  - Failed ($\mathcal{S}^c$): $\hat{y}_{\rm main} = y_{\rm main}$.
  - Successful ($\mathcal{S}$): $\hat{y}_{\rm main} \neq y_{\rm main}$, with attack success rate $\mathrm{ASR} = |\mathcal{S}| / N$.
- **Node 1 — Main-Branch Shift Typology on $\mathcal{S}$:**
  - Partition $\mathcal{S} = \mathcal{S}_{\rm shape} \cup \mathcal{S}_{\rm scale} \cup \mathcal{S}_{\rm joint}$ with $\Sigma = \{{\rm shape}, {\rm scale}, {\rm joint}\}$.
  - Shift rates $\mathrm{SR}_s = |\mathcal{S}_s| / |\mathcal{S}|$, single-factor rate $\mathrm{SR}_{\rm single} = \mathrm{SR}_{\rm shape} + \mathrm{SR}_{\rm scale}$, and effect size $\mathrm{cohen\_w}$ measured against the uniform random-error baseline $(0.25, 0.25, 0.50)$.
- **Node 2 — Auxiliary-Branch Reactions on $\mathcal{S}_s$:**
  - Single-branch code $o_k(x) \in \{\mathrm{A}, \mathrm{R}, \mathrm{D}\}$ for $k \in \{{\rm shape}, {\rm scale}\}$: $\mathrm{A}$ = aligned with the fooled main prediction, $\mathrm{R}$ = retains ground truth, $\mathrm{D}$ = diverged (neither).
  - Joint reaction pair $c(x) = (o_{\rm shape}(x), o_{\rm scale}(x)) \in \mathcal{K}_s$ with $|\mathcal{K}_{\rm shape}| = 6$, $|\mathcal{K}_{\rm scale}| = 6$, $|\mathcal{K}_{\rm joint}| = 9$ (21 states total).
  - Conditional rates $\mathrm{ARR}_{s,c} = |\mathcal{S}_{s,c}| / |\mathcal{S}_s|$, and the 4 mutually exclusive, exhaustive composite regimes over $\mathcal{S}$: Full Alignment ($\mathrm{FAR}$), Full Retention ($\mathrm{FRR}$), Partial Retention ($\mathrm{PRR}$), and Divergence ($\mathrm{DR}$), satisfying $\mathrm{FAR} + \mathrm{FRR} + \mathrm{PRR} + \mathrm{DR} = 1$.

---

## Script 01: Generate Untargeted Attacks (`c_attacks/01_untargeted_attacks.py`)

This script generates untargeted adversarial examples against a single multi-output model's main branch.

### Command-Line Arguments

| Argument | Type | Choices / Default | Meaning |
| --- | --- | --- | --- |
| `--dataset` | str | `standard`, `contrast` (default `standard`) | Training dataset variant $d$ |
| `--pool_type` | str | `flatten`, `gap` (default `flatten`) | Pooling type $\pi$ |
| `--loss_weight` | float | `0.5`, `1.0`, `2.0` (default `1.0`) | Main-head loss weight $w$ |
| `--batch_size` | int | default `4096` | Batch size for attack generation |

### Sub-Model Isolation

The full model is loaded via `get_model_paths('multi', args.dataset, args.pool_type, args.loss_weight)`. The main branch is isolated by constructing a new Keras model whose output is the layer named `'shape_and_scale'`:

```python
sub_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('shape_and_scale').output)
```

This detaches the `shape` and `scale` heads from gradient computation, so the perturbation is optimized against $f_{\rm main}$ only.

### ART Wrapper Configuration

```python
classifier = TensorFlowV2Classifier(
    model=sub_model,
    nb_classes=9,
    input_shape=(64, 64, 1),
    loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    clip_values=(0.0, 1.0)
)
```

### Attack Configurations

The script iterates over the following configurations, generating one `.npz` archive per $(\alpha, p, \varepsilon)$:

- **FGSM** (`FastGradientMethod`): `targeted=False`, `batch_size=args.batch_size`, across $L_\infty$ and $L_2$ epsilon grids.
- **PGD** (`ProjectedGradientDescent`): `targeted=False`, `max_iter=40`, `eps_step=(2.5 * eps) / 40.0` (Madry's heuristic $\alpha_{\rm step} = 2.5\,\varepsilon / N_{\rm iter}$), `num_random_init=1`, `batch_size=args.batch_size`.

Generation is executed in batches over `MY_DSPRITES_SPLIT` `X_test`, passing the ground-truth labels `y=y_shape_and_scale_test[start_idx:end_idx]` to `attack.generate(x=..., y=...)`.

### Outputs

The squeezed adversarial images `np.squeeze(adv_x, axis=-1)` of shape $(N, 64, 64)$ and all original test labels are written via `np.savez_compressed` to `data/c_attacks/untargeted/datasets/`, with the path constructed by `get_untargeted_attack_path(dataset, pool_type, attack, norm, eps, loss_weight)`. The archive contains `X_test` and the six label arrays (`y_shape_and_scale_test`, `y_shape_test`, `y_scale_test`, `y_orientation_test`, `y_pos_x_test`, `y_pos_y_test`). 240 archives are produced in total.

---

## Script 02: Evaluate Untargeted Attacks (`c_attacks/02_evaluate_untargeted.py`)

This script evaluates the **full** 3-output model $f(x')$ on every generated adversarial test set.

### Command-Line Arguments

| Argument | Type | Choices / Default | Meaning |
| --- | --- | --- | --- |
| `--pool_type` | str | `flatten`, `gap` (**required**) | Pooling type $\pi$ |
| `--dataset` | str | `standard`, `contrast` (default `standard`) | Training dataset variant $d$ |
| `--loss_weight` | float | `0.5`, `1.0`, `2.0` (default `1.0`) | Main-head loss weight $w$ |

The attack method $\alpha$ and norm $p$ are hardcoded internally in a `configs` list; the script iterates over the 4 $(\alpha, p)$ pairs, and for each pair loads the 5 epsilon-specific `.npz` files.

### Inference & Metrics

- Loads the full model, expands channels, and runs `model.predict(X_test, batch_size=4096, verbose=1)`.
- Unpacks logits `probs_main, probs_shape, probs_scale`, converts each via `tf.nn.softmax(..., axis=-1).numpy()`, and extracts class labels with `np.argmax(..., axis=1)`.
- Computes per-sample max softmax confidence and raw Shannon entropy $-\sum_c p_{t,c}(x') \ln(p_{t,c}(x') + 10^{-9})$ for all 3 branches.

### Outputs

Each epsilon DataFrame is concatenated, and the result is written as a gzip-compressed CSV via the path from `get_attack_eval_paths(dataset, pool_type, attack, norm, loss_weight)`. An idempotency check skips generation of an existing CSV.

Each of the 48 CSVs (`12 models × 4 attack-norm pairs`) contains $5 \times 36{,}864 = 184{,}320$ rows and 14 columns:

```
sample_index, epsilon,
y_shape_and_scale, y_shape, y_scale,
pred_main, pred_shape, pred_scale,
confidence, entropy,
confidence_shape, entropy_shape,
confidence_scale, entropy_scale
```

Output directory: `data/c_attacks/untargeted/evaluations/`.

---

## Script 03: Phase-1 Exploratory Data Analysis (`c_attacks/03_analyze_untargeted_eda.py`)

This is an interactive `# %%` cell script that loads the 48 evaluation CSVs and computes the conditional probability tree (Node 0 → Node 1 → Node 2), together with statistical summaries and interactive visual tools.

### Computed Metrics

- 26 legacy tree metrics (`success_rate`, `shape_shift`, `scale_shift`, `joint_shift`, `cohen_w`, and the 21 Node 2 leaf percentages prefixed with `ss_`, `sl_`, `js_`).
- Wilson 95% confidence intervals, Bernoulli KL-divergence $D^2\%$ shares, branch accuracies, softmax confidence (%), and normalized entropy (%).

### Printed Tables

- **Table 1**: Macro summary — global `ALL` and single-parameter variations (17 columns).
- **Tables 2.1–2.4**: Per attack-norm combination across all 12 models and 5 epsilon steps.
- **Tables 3A–3B**: Non-parametric medians/IQRs, Wilson `★★` counts, and 1-way / 2-way KL-deviance $D^2\%$ filters.
- **Mode Collapse Inspector**: slices where the main head predicts a single class for $\ge 99\%$ of images.
- **Table 4**: Branch accuracies, confidence, and normalized entropy by archetype (Failed, Full Aligned, Full Retained, Partial Retained, Diverged).

### Interactive Tools

- `plot_interactive_tree`: Sankey / proportional probability tree with statistical significance markers.
- `visualize_attacks`: clean / adversarial / perturbation image viewer with `ipywidgets` dropdowns.

---

## Script 04: Target Post-Hoc Analysis (`c_attacks/04_analyze_untargeted.py`)

This is the primary target analysis script of the project. It loads all 48 evaluation CSVs ($240$ slices, $8{,}847{,}360$ sample evaluations) plus the baseline evaluation JSONs from `b_models` in a single pass, and writes the structured analytical report to `data/_analyses/06_untargeted_attacks_analysis.md` via `ReportGenerator`.

### Structure

- **Single-pass data loader & pre-computation (Cell 1):** builds the 240-row `tree_df` (exact counts, $\mathrm{ASR}$, $\mathrm{SR}_s$, $\mathrm{SR}_{\rm single}$, $\mathrm{cohen\_w}$, all 21 $\mathrm{ARR}_{s,c}$, composite regime rates, archetype confidence/entropy sums, sink-collapse counts) and populates the per-image float stores `per_image_C` / `per_image_H`.
- **Part A — Main-Branch Vulnerability & Error Structure (Nodes 0 & 1):**
  - Section 1.1: Node 0 global $\mathrm{ASR}$ & 240-slice distribution → **Figure 1** (ASR histogram + $12 \times 20$ ASR heatmap).
  - Section 1.2: Node 0 factor marginals, $D^2$ KL-deviance attribution, and two-way interactions → **Figure 2A** (ASR trajectories by model family) and **Figure 2B** (factor marginal spread and $D^2\%$ shares).
  - Section 2: Node 1 shift typology $\mathrm{SR}_s$ & $\mathrm{cohen\_w}$ → **Figure 3A** (shift typology + pooling breakdown) and **Figure 3B** (deterministic image strip).
- **Part B — Auxiliary Branch Reactions, Decoupling & Uncertainty (Node 2):**
  - Section B0: failed-attack auxiliary integrity on $\mathcal{S}^c$ (silent auxiliary corruption).
  - Section 3: Node 2 full alignment & composite regimes → **Figure 4A** (3-stage Sankey) and **Figure 4B** (composite regimes + alignment by pooling).
  - Section 4: full retention $\mathrm{FRR}$ & $\mathrm{ARR}_{s,\mathrm{RR}}$ → **Figure 5**.
  - Section 5: partial retention $\mathrm{PRR}$ & collateral divergence → **Figure 6A**, **Figure 6B** ($12 \times 21$ $\mathrm{ARR}_{s,c}$ heatmap), **Figure 6C** (image strip).
  - Section 6: three-way divergence $\mathrm{DR}_s^{(k)}$ & $\mathrm{ARR}_{s,\mathrm{DD}}$ → **Figure 7A** and **Figure 7B** (image strip).
  - Section 7: is disagreement uncertain or confidently contradictory? → **Figure 8** (low-confidence tail shares).
- **Part C — High-Budget Regime & Global Variance Attribution Synthesis:**
  - Section 8: single-class sink collapse → **Figure 9B** (multi-class collapse image strip).
  - Section 9: factor $D^2$ attribution across all 23 tree metrics → **Figure 10** (attribution heatmap).

### Figures

16 matplotlib figures are embedded into the report via `save_and_show`, which calls `report.kroki(fig, title=title)`. The figures are numbered 1, 2A, 2B, 3A, 3B, 4A, 4B, 5, 6A, 6B, 6C, 7A, 7B, 8, 9B, 10.

---

## Execution Guide

### Environment Setup

Use the Conda environment configuration (do not use `requirements.txt` for this pipeline):

```bash
conda env create -f shared/environment.yml
conda activate ml-dl-project-training
python shared/check_gpu.py
```

### Pipeline Execution Commands

All commands are run from the project root.

#### Step 1: Generate Untargeted Attacks (`01_untargeted_attacks.py`)

```bash
python c_attacks/01_untargeted_attacks.py --dataset standard --pool_type flatten --loss_weight 1.0
python c_attacks/01_untargeted_attacks.py --dataset standard --pool_type gap     --loss_weight 1.0
python c_attacks/01_untargeted_attacks.py --dataset contrast --pool_type flatten --loss_weight 1.0
python c_attacks/01_untargeted_attacks.py --dataset contrast --pool_type gap     --loss_weight 1.0
```

#### Step 2: Evaluate Untargeted Attacks (`02_evaluate_untargeted.py`)

```bash
python c_attacks/02_evaluate_untargeted.py --dataset standard --pool_type flatten --loss_weight 1.0
python c_attacks/02_evaluate_untargeted.py --dataset standard --pool_type gap     --loss_weight 1.0
python c_attacks/02_evaluate_untargeted.py --dataset contrast --pool_type flatten --loss_weight 1.0
python c_attacks/02_evaluate_untargeted.py --dataset contrast --pool_type gap     --loss_weight 1.0
```

#### Step 3: Full Matrix Loop (All 12 Models)

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

#### Step 4: Phase-1 Exploratory Data Analysis (`03_analyze_untargeted_eda.py`)

Run as an interactive `# %%` cell script (Jupyter or a Python-interactive host) after all 48 evaluation CSVs exist:

```bash
python c_attacks/03_analyze_untargeted_eda.py
```

#### Step 5: Target Post-Hoc Analysis (`04_analyze_untargeted.py`)

Run as an interactive `# %%` cell script to produce `data/_analyses/06_untargeted_attacks_analysis.md`:

```bash
python c_attacks/04_analyze_untargeted.py
```