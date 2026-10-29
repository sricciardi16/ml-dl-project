#%% Setup and Imports
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import json
import pydot
from shared.paths import get_model_paths, get_eval_paths, MODELS_TRAINING_REPORT, MY_DSPRITES_SPLIT, get_latent_features_path
from shared.report_generator import ReportGenerator
from sklearn.decomposition import PCA

report = ReportGenerator(MODELS_TRAINING_REPORT)

#%% Single Models Training Curves
datasets = ['standard', 'contrast']
pool_types = ['flatten', 'gap']

fig_loss = plt.figure(figsize=(12, 10))
fig_acc = plt.figure(figsize=(12, 10))

for d_idx, dataset in enumerate(datasets):
    for p_idx, pool_type in enumerate(pool_types):
        row = d_idx
        col = p_idx
        history_path = get_model_paths('single', dataset, pool_type)[1]
        history = np.load(history_path, allow_pickle=True).item()
        
        # Loss figure
        ax_loss = fig_loss.add_subplot(2, 2, row*2 + col + 1)
        ax_loss.plot(history['loss'], label='Training Loss')
        ax_loss.plot(history['val_loss'], label='Validation Loss')
        ax_loss.set_title(f"{dataset} {pool_type}")
        ax_loss.legend()
        
        # Accuracy figure
        ax_acc = fig_acc.add_subplot(2, 2, row*2 + col + 1)
        ax_acc.plot(history.get('accuracy', history.get('acc')), label='Training Accuracy')
        ax_acc.plot(history.get('val_accuracy', history.get('val_acc')), label='Validation Accuracy')
        ax_acc.set_title(f"{dataset} {pool_type}")
        ax_acc.legend()

fig_loss.tight_layout()
fig_acc.tight_layout()

report.kroki(fig_loss, title="Single Models - Training Loss")
report.kroki(fig_acc, title="Single Models - Training Accuracy")

plt.close(fig_loss)
plt.close(fig_acc)

#%% Single Models - Metrics Verification
for dataset in datasets:
    for pool_type in pool_types:
        # Training History Summary + Convergence Analysis
        history_path = get_model_paths('single', dataset, pool_type)[1]
        history = np.load(history_path, allow_pickle=True).item()
        
        train_acc_key = 'accuracy' if 'accuracy' in history else 'acc'
        val_acc_key = 'val_accuracy' if 'val_accuracy' in history else 'val_acc'
        
        print(f"\n{'='*60}")
        print(f"Single Model - {dataset} {pool_type}")
        print(f"--- Training History Summary ---")
        print(f"Total Epochs: {len(history['loss'])}")
        print(f"Best Epoch: {history['val_loss'].index(min(history['val_loss'])) + 1}")
        print(f"Final Training Loss: {history['loss'][-1]}")
        print(f"Final Validation Loss: {history['val_loss'][-1]}")
        print(f"Final Training Accuracy: {history[train_acc_key][-1]}")
        print(f"Final Validation Accuracy: {history[val_acc_key][-1]}")
        
        print(f"--- Convergence Analysis ---")
        gen_gap = history['val_loss'][-1] - history['loss'][-1]
        acc_gap = history[train_acc_key][-1] - history[val_acc_key][-1]
        val_stability = np.std(history['val_loss'][-5:])
        print(f"Generalization Gap: {gen_gap}")
        print(f"Accuracy Gap: {acc_gap}")
        print(f"Validation Stability (last 5): {val_stability}")
        
        # Evaluation Metrics Summary + Accuracy Verification
        eval_paths = get_eval_paths('single', dataset, pool_type)
        for i, eval_path in enumerate(eval_paths):
            if not os.path.exists(eval_path):
                continue
            eval_label = "primary" if i == 0 else "standard test"
            with open(eval_path, 'r') as f:
                metrics = json.load(f)
            
            task = metrics['shape_and_scale_task']
            acc = task['accuracy']
            cr = task['classification_report']
            
            print(f"\n--- Evaluation Metrics Summary ({eval_label}) ---")
            print(f"Test Loss: {metrics['test_loss_total']}")
            print(f"Test Accuracy: {acc}")
            print(f"Macro Avg F1-Score: {cr['macro avg']['f1-score']}")
            print(f"Weighted Avg F1-Score: {cr['weighted avg']['f1-score']}")
            
            print(f"--- Accuracy Verification ({eval_label}) ---")
            if acc == 1.0:
                all_correct = True
                for cls, vals in cr.items():
                    if isinstance(vals, dict):
                        if vals['precision'] != 1.0 or vals['recall'] != 1.0 or vals['f1-score'] != 1.0:
                            all_correct = False
                            print(f"  ⚠️  Class {cls}: precision={vals['precision']}, recall={vals['recall']}, f1={vals['f1-score']}")
                if all_correct:
                    print(f"  ✅ Accuracy is 1.0 — all per-class precision, recall, and F1 are also 1.0. Confirmed.")
            else:
                print(f"  Accuracy is not 1.0 — per-class metrics are meaningful and expected to vary.")

#%% Multi Models Training Curves
datasets = ['standard', 'contrast']
pool_types = ['flatten', 'gap']
weights = [0.5, 1.0, 2.0]

fig_loss_m = plt.figure(figsize=(15, 16))
fig_acc_m = plt.figure(figsize=(15, 16))

for d_idx, dataset in enumerate(datasets):
    for p_idx, pool_type in enumerate(pool_types):
        for w_idx, weight in enumerate(weights):
            row = d_idx * 2 + p_idx
            col = w_idx
            history_path = get_model_paths('multi', dataset, pool_type, weight)[1]
            history = np.load(history_path, allow_pickle=True).item()
            
            # Loss figure
            ax_loss = fig_loss_m.add_subplot(4, 3, row*3 + col + 1)
            ax_loss.plot(history['loss'], label='Training Loss')
            ax_loss.plot(history['val_loss'], label='Validation Loss')
            ax_loss.set_title(f"{dataset} {pool_type} w={weight}")
            ax_loss.legend()
            
            # Accuracy figure
            ax_acc = fig_acc_m.add_subplot(4, 3, row*3 + col + 1)
            ax_acc.plot(history['shape_and_scale_accuracy'], label='Training Accuracy')
            ax_acc.plot(history['val_shape_and_scale_accuracy'], label='Validation Accuracy')
            ax_acc.set_title(f"{dataset} {pool_type} w={weight}")
            ax_acc.legend()

fig_loss_m.tight_layout()
fig_acc_m.tight_layout()

report.kroki(fig_loss_m, title="Multi Models - Training Loss")
report.kroki(fig_acc_m, title="Multi Models - Main Task Accuracy")

plt.close(fig_loss_m)
plt.close(fig_acc_m)

#%% Multi Models Separate Losses and Accuracies
datasets = ['standard', 'contrast']
pool_types = ['flatten', 'gap']
weights = [0.5, 1.0, 2.0]

fig_loss_detailed = plt.figure(figsize=(15, 16))
fig_acc_detailed = plt.figure(figsize=(15, 16))

for d_idx, dataset in enumerate(datasets):
    for p_idx, pool_type in enumerate(pool_types):
        for w_idx, weight in enumerate(weights):
            row = d_idx * 2 + p_idx
            col = w_idx
            history_path = get_model_paths('multi', dataset, pool_type, weight)[1]
            history = np.load(history_path, allow_pickle=True).item()
            
            # Loss subplot
            ax_loss = fig_loss_detailed.add_subplot(4, 3, row*3 + col + 1)
            # Shape and scale loss
            ax_loss.plot(history['shape_and_scale_loss'], label='Shape & Scale Loss (Train)', color='blue', linestyle='-')
            ax_loss.plot(history['val_shape_and_scale_loss'], label='Shape & Scale Loss (Val)', color='blue', linestyle='--')
            # Shape loss
            ax_loss.plot(history['shape_loss'], label='Shape Loss (Train)', color='orange', linestyle='-')
            ax_loss.plot(history['val_shape_loss'], label='Shape Loss (Val)', color='orange', linestyle='--')
            # Scale loss
            ax_loss.plot(history['scale_loss'], label='Scale Loss (Train)', color='green', linestyle='-')
            ax_loss.plot(history['val_scale_loss'], label='Scale Loss (Val)', color='green', linestyle='--')
            ax_loss.set_title(f"{dataset} {pool_type} w={weight}")
            ax_loss.legend()
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Loss')
            
            # Accuracy subplot
            ax_acc = fig_acc_detailed.add_subplot(4, 3, row*3 + col + 1)
            # Shape and scale accuracy
            ax_acc.plot(history['shape_and_scale_accuracy'], label='Shape & Scale Acc (Train)', color='blue', linestyle='-')
            ax_acc.plot(history['val_shape_and_scale_accuracy'], label='Shape & Scale Acc (Val)', color='blue', linestyle='--')
            # Shape accuracy
            ax_acc.plot(history['shape_accuracy'], label='Shape Acc (Train)', color='orange', linestyle='-')
            ax_acc.plot(history['val_shape_accuracy'], label='Shape Acc (Val)', color='orange', linestyle='--')
            # Scale accuracy
            ax_acc.plot(history['scale_accuracy'], label='Scale Acc (Train)', color='green', linestyle='-')
            ax_acc.plot(history['val_scale_accuracy'], label='Scale Acc (Val)', color='green', linestyle='--')
            ax_acc.set_title(f"{dataset} {pool_type} w={weight}")
            ax_acc.legend()
            ax_acc.set_xlabel('Epoch')
            ax_acc.set_ylabel('Accuracy')

fig_loss_detailed.tight_layout()
fig_acc_detailed.tight_layout()

report.kroki(fig_loss_detailed, title="Multi Models - Separate Losses")
report.kroki(fig_acc_detailed, title="Multi Models - Separate Accuracies")

plt.close(fig_loss_detailed)
plt.close(fig_acc_detailed)

#%% Multi Models - Metrics Verification
for dataset in datasets:
    for pool_type in pool_types:
        for weight in weights:
            # Training History Summary + All 12 Final Values
            history_path = get_model_paths('multi', dataset, pool_type, weight)[1]
            history = np.load(history_path, allow_pickle=True).item()
            
            print(f"\n{'='*60}")
            print(f"Multi Model - {dataset} {pool_type} w={weight}")
            print(f"--- Training History Summary ---")
            total_epochs = len(history.get('loss', history.get('shape_and_scale_loss', [])))
            best_epoch = np.argmin(history.get('val_loss', history.get('val_shape_and_scale_loss', []))) + 1
            print(f"Total Epochs: {total_epochs}")
            print(f"Best Epoch (min val_loss): {best_epoch}")
            
            keys_to_print = [
                'shape_and_scale_loss', 'val_shape_and_scale_loss',
                'shape_and_scale_accuracy', 'val_shape_and_scale_accuracy',
                'shape_loss', 'val_shape_loss',
                'shape_accuracy', 'val_shape_accuracy',
                'scale_loss', 'val_scale_loss',
                'scale_accuracy', 'val_scale_accuracy'
            ]
            for k in keys_to_print:
                if k in history:
                    print(f"Final {k}: {history[k][-1]:.4f}")
                else:
                    print(f"Final {k}: N/A")
            
            # Evaluation Metrics Summary + Accuracy Verification
            # For contrast models, check both primary (contrast test) and secondary (standard test) evals
            eval_paths = get_eval_paths('multi', dataset, pool_type, weight)
            for i, eval_path in enumerate(eval_paths):
                if not os.path.exists(eval_path):
                    continue
                eval_label = "primary" if i == 0 else "standard test"
                with open(eval_path, 'r') as f:
                    metrics = json.load(f)
                
                print(f"\n--- Evaluation Metrics Summary ({eval_label}) ---")
                print(f"Test Loss (total): {metrics['test_loss_total']}")
                
                for task_name in ['shape_and_scale_task', 'shape_task', 'scale_task']:
                    if task_name not in metrics:
                        continue
                    task = metrics[task_name]
                    acc = task['accuracy']
                    cr = task['classification_report']
                    
                    print(f"\n  {task_name}:")
                    print(f"    Loss: {task['loss']}")
                    print(f"    Accuracy: {acc}")
                    print(f"    Macro Avg F1-Score: {cr['macro avg']['f1-score']}")
                    print(f"    Weighted Avg F1-Score: {cr['weighted avg']['f1-score']}")
                    
                    if acc == 1.0:
                        all_correct = True
                        for cls, vals in cr.items():
                            if isinstance(vals, dict):
                                if vals['precision'] != 1.0 or vals['recall'] != 1.0 or vals['f1-score'] != 1.0:
                                    all_correct = False
                                    print(f"    ⚠️  Class {cls}: precision={vals['precision']}, recall={vals['recall']}, f1={vals['f1-score']}")
                        if all_correct:
                            print(f"    ✅ Accuracy is 1.0 — all per-class precision, recall, and F1 are also 1.0. Confirmed.")
                    else:
                        print(f"    Accuracy is not 1.0 — per-class metrics are meaningful and expected to vary.")

#%% Architecture Diagrams (standard only — architecture does not vary by dataset or loss weight)
for pool_type in pool_types:
    _, _, dot_path = get_model_paths('single', 'standard', pool_type)
    if os.path.exists(dot_path):
        graphs = pydot.graph_from_dot_file(dot_path)
        if graphs:
            report.kroki(graphs[0], title=f"Single - {pool_type} - Architecture")

for pool_type in pool_types:
    _, _, dot_path = get_model_paths('multi', 'standard', pool_type, 1.0)
    if os.path.exists(dot_path):
        graphs = pydot.graph_from_dot_file(dot_path)
        if graphs:
            report.kroki(graphs[0], title=f"Multi - {pool_type} - Architecture")

#%% Latent PCA Setup and Helpers
data_split = np.load(MY_DSPRITES_SPLIT)
y_shape_test_all = data_split['y_shape_test']
y_scale_test_all = data_split['y_scale_test']

pca_palette = {
    0: ['lightskyblue', 'dodgerblue', 'darkblue'],
    1: ['lightgreen', 'limegreen', 'darkgreen'],
    2: ['lightcoral', 'red', 'darkred']
}
shape_names = {0: 'Square', 1: 'Ellipse', 2: 'Heart'}
scale_names = {0: 'Small', 1: 'Medium', 2: 'Large'}

def plot_latent_pca(ax, task, dataset, pool, weight=None, title=None, show_legend=False):
    """Fit 2D PCA on pre-extracted trunk features, subsample 2000 points, and scatter plot."""
    feat_path = get_latent_features_path(task, dataset, pool, weight)
    if not os.path.exists(feat_path):
        ax.text(0.5, 0.5, f"Missing:\n{os.path.basename(feat_path)}",
                ha='center', va='center', fontsize=8, color='gray')
        return

    features = np.load(feat_path)['latent_test']
    coords = PCA(n_components=2).fit_transform(features)

    np.random.seed(29)
    n_pts = len(coords)
    idx = np.random.choice(n_pts, size=min(2000, n_pts), replace=False)
    coords_sub = coords[idx]
    sh_sub = y_shape_test_all[idx]
    sc_sub = y_scale_test_all[idx]

    for sh_val in sorted(np.unique(sh_sub)):
        for sc_val in sorted(np.unique(sc_sub)):
            mask = (sh_sub == sh_val) & (sc_sub == sc_val)
            ax.scatter(
                coords_sub[mask, 0], coords_sub[mask, 1],
                c=pca_palette[sh_val][sc_val], marker='o', s=8, alpha=0.7,
                label=f"{shape_names[sh_val]} {scale_names[sc_val]}"
            )

    ax_title = title or f"{dataset[:3]}|{pool}|{f'w={weight}' if weight is not None else 'single'}"
    ax.set_title(ax_title, fontsize=9.5, fontweight='bold')
    ax.tick_params(labelsize=7.5)
    ax.grid(True, linestyle=':', alpha=0.5)

    if show_legend:
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, framealpha=0.92)


#%% Latent PCA - Full Matrix Grids (Option B)
# 1. Single-Task Models (2x2 Grid)
fig_pca_single, axes_single = plt.subplots(2, 2, figsize=(11, 9), dpi=120)
for r_i, d_val in enumerate(['standard', 'contrast']):
    for c_i, p_val in enumerate(['flatten', 'gap']):
        ax = axes_single[r_i, c_i]
        is_last = (r_i == 0 and c_i == 1)
        plot_latent_pca(ax, 'single', d_val, p_val,
                        title=f"Single: {d_val.capitalize()} + {p_val.upper()}",
                        show_legend=is_last)
        if c_i == 0:
            ax.set_ylabel("PC 2", fontsize=9)
        if r_i == 1:
            ax.set_xlabel("PC 1", fontsize=9)

fig_pca_single.suptitle("Single-Task Models: Trunk Latent Space PCA", fontsize=12, fontweight='bold')
fig_pca_single.tight_layout()
report.kroki(fig_pca_single, title="Single Models - Latent Space PCA (2x2 Grid)")
plt.close(fig_pca_single)

# 2. Multi-Task Models (4x3 Grid)
fig_pca_multi, axes_multi = plt.subplots(4, 3, figsize=(15, 16), dpi=120)
multi_families = [
    ('standard', 'flatten', 'std|flat'),
    ('standard', 'gap',     'std|gap'),
    ('contrast', 'flatten', 'con|flat'),
    ('contrast', 'gap',     'con|gap'),
]
weights_list = [0.5, 1.0, 2.0]

for r_i, (d_val, p_val, f_lbl) in enumerate(multi_families):
    for c_i, w_val in enumerate(weights_list):
        ax = axes_multi[r_i, c_i]
        is_legend = (r_i == 0 and c_i == 2)
        plot_latent_pca(ax, 'multi', d_val, p_val, weight=w_val,
                        title=f"{f_lbl} (w={w_val:g})",
                        show_legend=is_legend)
        if c_i == 0:
            ax.set_ylabel(f"{f_lbl}\nPC 2", fontsize=9)
        if r_i == 3:
            ax.set_xlabel("PC 1", fontsize=9)

fig_pca_multi.suptitle("Multi-Task Models: Shared Trunk Latent Space PCA (4 Families x 3 Loss Weights)",
                       fontsize=12, fontweight='bold')
fig_pca_multi.tight_layout()
report.kroki(fig_pca_multi, title="Multi Models - Latent Space PCA (4x3 Grid)")
plt.close(fig_pca_multi)


#%% Latent PCA - Standard vs Contrast (GAP Only, w=0.5)
fig_dcomp, axes_dcomp = plt.subplots(1, 2, figsize=(13, 5.2), dpi=120)
plot_latent_pca(axes_dcomp[0], 'multi', 'standard', 'gap', weight=0.5,
                title="Standard Dataset (`std|gap|w=0.5`)")
plot_latent_pca(axes_dcomp[1], 'multi', 'contrast', 'gap', weight=0.5,
                title="Contrast Dataset (`con|gap|w=0.5`)",
                show_legend=True)

for ax in axes_dcomp:
    ax.set_xlabel("Principal Component 1", fontsize=9)
axes_dcomp[0].set_ylabel("Principal Component 2", fontsize=9)

fig_dcomp.suptitle("Dataset Comparison in Shared Trunk (GAP, w=1.0): Standard vs. Contrast",
                   fontsize=11.5, fontweight='bold')
fig_dcomp.tight_layout()
report.kroki(fig_dcomp, title="Multi Models - Standard vs Contrast Latent PCA (GAP Only)")
plt.close(fig_dcomp)


#%% Latent PCA - Multi-Task Loss Weight Trajectory (con|gap: w=0.5 -> 1.0 -> 2.0)
fig_wcomp, axes_wcomp = plt.subplots(1, 3, figsize=(17, 5.2), dpi=120)
for idx_w, w_val in enumerate(weights_list):
    ax = axes_wcomp[idx_w]
    is_legend = (idx_w == 2)
    plot_latent_pca(ax, 'multi', 'contrast', 'gap', weight=w_val,
                    title=f"`con|gap` (w={w_val:g})",
                    show_legend=is_legend)
    ax.set_xlabel("Principal Component 1", fontsize=9)

axes_wcomp[0].set_ylabel("Principal Component 2", fontsize=9)
fig_wcomp.suptitle("Loss Weight Trajectory in Shared Trunk (`con|gap`): w=0.5 -> 1.0 -> 2.0",
                   fontsize=11.5, fontweight='bold')
fig_wcomp.tight_layout()
report.kroki(fig_wcomp, title="Multi Models - Loss Weight Trajectory Latent PCA (con|gap)")
plt.close(fig_wcomp)


#%% Teardown
report.close()
# %%
