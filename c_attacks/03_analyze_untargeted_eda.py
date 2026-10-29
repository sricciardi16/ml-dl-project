"""
Adversarial Attack Analysis — Phase 1

For each model configuration and attack setting:
  1. How many attacks succeeded vs failed
  2. Among successful attacks: how many have both features (shape & scale)
     consistent with the fooled main prediction vs how many do not
"""

#%%
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from shared.paths import get_attack_eval_paths, UNTARGETED_EVAL_DIR

#%%
# Build the list of all 48 evaluation files
datasets = ['standard', 'contrast']
pool_types = ['flatten', 'gap']
loss_weights = [0.5, 1.0, 2.0]
attacks = ['fgsm', 'pgd']
norms = ['linf', 'l2']

configs = []
for dataset in datasets:
    for pool_type in pool_types:
        for weight in loss_weights:
            for attack in attacks:
                for norm in norms:
                    csv_path = get_attack_eval_paths(dataset, pool_type, attack, norm, weight)
                    if os.path.exists(csv_path):
                        configs.append({
                            'dataset': dataset,
                            'pool_type': pool_type,
                            'loss_weight': weight,
                            'attack': attack,
                            'norm': norm,
                            'csv_path': csv_path
                        })

print(f"Found {len(configs)} evaluation files out of 48 expected.")

#%%
# =============================================================================
# CONDITIONAL PROBABILITY TREE: Attack Analysis
# =============================================================================
# Node 0: Attack Success (Failed vs Success)
# Node 1: Shift Typology (Shape-Shift, Scale-Shift, Joint-Shift)
# Node 2: Aux Branch Reactions (Convention: <shape_reaction>_<scale_reaction>):
#   - aligned  : aux matches the fooled main prediction
#   - retained : aux matches the original ground-truth label
#   - diverged : aux matches neither original nor fooled main prediction
# =============================================================================

# Constants
N_SINGLE_EPS = 36864   # Exact test set size per epsilon
N_MIN_BRANCH = 100     # Minimum sample size for valid branch inference
Z_95 = 1.959963984540054  # Standard normal quantile for 95% confidence intervals

TREE_COLS = ['epsilon', 'y_shape_and_scale', 'pred_main', 'pred_shape', 'pred_scale']

SS_KEYS = [
    'ss_aligned_retained',  'ss_aligned_diverged',
    'ss_retained_retained', 'ss_retained_diverged',
    'ss_diverged_retained', 'ss_diverged_diverged',
]

SL_KEYS = [
    'sl_retained_aligned',  'sl_retained_retained', 'sl_retained_diverged',
    'sl_diverged_aligned',  'sl_diverged_retained', 'sl_diverged_diverged',
]

JS_KEYS = [
    'js_aligned_aligned',  'js_aligned_retained',  'js_aligned_diverged',
    'js_retained_aligned', 'js_retained_retained', 'js_retained_diverged',
    'js_diverged_aligned', 'js_diverged_retained', 'js_diverged_diverged',
]

# Canonical ordered list of all 26 tree metrics
ALL_26_METRICS = [
    # Node 0: Attack Outcome
    'success_rate',
    
    # Node 1: Shift Typology
    'shape_shift', 'scale_shift', 'joint_shift', 'cohen_w',
    
    # Node 2: Shape-Shift (% SS)
    *SS_KEYS,
    
    # Node 2: Scale-Shift (% SL)
    *SL_KEYS,
    
    # Node 2: Joint-Shift (% JS)
    *JS_KEYS
]

# 25 percentage metrics (excludes cohen_w)
PCT_25_METRICS = [m for m in ALL_26_METRICS if m != 'cohen_w']

# Display names for compact tables
DISPLAY_RENAME = {k: k[3:] for k in SS_KEYS + SL_KEYS + JS_KEYS}


def compute_node2_breakdowns(s_df, shape_shift_mask, scale_shift_mask, joint_shift_mask):
    sh_retained = s_df['pred_shape'] == s_df['true_shape']
    sh_aligned  = (s_df['pred_shape'] == s_df['fooled_shape']) & ~sh_retained
    sh_diverged = ~sh_retained & ~sh_aligned
    sh_unshifted_diverged = ~sh_retained

    sc_retained = s_df['pred_scale'] == s_df['true_scale']
    sc_aligned  = (s_df['pred_scale'] == s_df['fooled_scale']) & ~sc_retained
    sc_diverged = ~sc_retained & ~sc_aligned
    sc_unshifted_diverged = ~sc_retained

    ss_breakdown = {
        'ss_aligned_retained':  (shape_shift_mask & sh_aligned  & sc_retained).sum(),
        'ss_aligned_diverged':  (shape_shift_mask & sh_aligned  & sc_unshifted_diverged).sum(),
        'ss_retained_retained': (shape_shift_mask & sh_retained & sc_retained).sum(),
        'ss_retained_diverged': (shape_shift_mask & sh_retained & sc_unshifted_diverged).sum(),
        'ss_diverged_retained': (shape_shift_mask & sh_diverged & sc_retained).sum(),
        'ss_diverged_diverged': (shape_shift_mask & sh_diverged & sc_unshifted_diverged).sum(),
    }

    sl_breakdown = {
        'sl_retained_aligned':  (scale_shift_mask & sh_retained & sc_aligned).sum(),
        'sl_retained_retained': (scale_shift_mask & sh_retained & sc_retained).sum(),
        'sl_retained_diverged': (scale_shift_mask & sh_retained & sc_diverged).sum(),
        'sl_diverged_aligned':  (scale_shift_mask & sh_unshifted_diverged & sc_aligned).sum(),
        'sl_diverged_retained': (scale_shift_mask & sh_unshifted_diverged & sc_retained).sum(),
        'sl_diverged_diverged': (scale_shift_mask & sh_unshifted_diverged & sc_diverged).sum(),
    }

    js_breakdown = {
        'js_aligned_aligned':  (joint_shift_mask & sh_aligned  & sc_aligned).sum(),
        'js_aligned_retained': (joint_shift_mask & sh_aligned  & sc_retained).sum(),
        'js_aligned_diverged': (joint_shift_mask & sh_aligned  & sc_diverged).sum(),
        'js_retained_aligned': (joint_shift_mask & sh_retained & sc_aligned).sum(),
        'js_retained_retained': (joint_shift_mask & sh_retained & sc_retained).sum(),
        'js_retained_diverged': (joint_shift_mask & sh_retained & sc_diverged).sum(),
        'js_diverged_aligned': (joint_shift_mask & sh_diverged & sc_aligned).sum(),
        'js_diverged_retained': (joint_shift_mask & sh_diverged & sc_retained).sum(),
        'js_diverged_diverged': (joint_shift_mask & sh_diverged & sc_diverged).sum(),
    }

    return ss_breakdown, sl_breakdown, js_breakdown


def wilson_ci(p, n, z=Z_95):
    """Compute Wilson Score 95% Confidence Interval"""
    if n <= 0:
        return (0.0, 1.0)
    
    d = 1 + (z**2 / n)
    c = (p + (z**2 / (2*n))) / d
    h = (z * np.sqrt((p * (1 - p)) / n + (z**2 / (4 * n**2)))) / d
    
    lower = max(0.0, c - h)
    upper = min(1.0, c + h)
    
    return (lower, upper)


def classify_group_significance(vals_dict, n_branch):
    """Upgraded classification using Wilson Score confidence intervals"""
    k_states = len(vals_dict)
    if n_branch < N_MIN_BRANCH:
        return {k: 'noise' for k in vals_dict}
    
    # Primary classification
    null_p = 1.0 / k_states
    tiers = {}
    primary_mass = 0.0
    n_primary = 0

    for k, val_pct in vals_dict.items():
        p = val_pct / 100.0
        ci_low, _ = wilson_ci(p, n_branch)
        if ci_low > null_p:
            tiers[k] = 'primary'
            primary_mass += p
            n_primary += 1
        else:
            tiers[k] = 'noise'

    # Secondary classification
    k_rem = k_states - n_primary
    rem_mass = max(0.0, 1.0 - primary_mass)
    if k_rem > 0 and rem_mass > 0.05:
        res_null_p = rem_mass / k_rem
        for k, val_pct in vals_dict.items():
            if tiers[k] == 'noise' and val_pct >= 10.0:
                p = val_pct / 100.0
                ci_low, _ = wilson_ci(p, n_branch)
                if ci_low > res_null_p:
                    tiers[k] = 'secondary'
    return tiers


def aggregate_tree_subset(df_sub, n_per_row):
    """Pool raw sample counts across any subset of rows so 'ALL' is sample-weighted."""
    if {'n_total', 'n_succ', 'n_sh', 'n_sc', 'n_jt'}.issubset(df_sub.columns):
        n_total = int(df_sub['n_total'].sum())
        n_succ_arr = df_sub['n_succ'].to_numpy(dtype=float)
        n_sh_arr = df_sub['n_sh'].to_numpy(dtype=float)
        n_sc_arr = df_sub['n_sc'].to_numpy(dtype=float)
        n_jt_arr = df_sub['n_jt'].to_numpy(dtype=float)
        n_succ = int(df_sub['n_succ'].sum())
        n_sh = int(df_sub['n_sh'].sum())
        n_sc = int(df_sub['n_sc'].sum())
        n_jt = int(df_sub['n_jt'].sum())
    else:
        n_total = len(df_sub) * n_per_row
        n_succ_arr = n_per_row * (df_sub['success_rate'].to_numpy(dtype=float) / 100.0)
        n_sh_arr = n_succ_arr * (df_sub['shape_shift'].to_numpy(dtype=float) / 100.0)
        n_sc_arr = n_succ_arr * (df_sub['scale_shift'].to_numpy(dtype=float) / 100.0)
        n_jt_arr = n_succ_arr * (df_sub['joint_shift'].to_numpy(dtype=float) / 100.0)
        n_succ = int(round(n_succ_arr.sum()))
        n_sh = int(round(n_sh_arr.sum()))
        n_sc = int(round(n_sc_arr.sum()))
        n_jt = max(0, n_succ - n_sh - n_sc) if n_succ > 0 else 0

    agg = {
        'n_total': n_total,
        'n_succ': n_succ,
        'n_sh': n_sh,
        'n_sc': n_sc,
        'n_jt': n_jt,
        'success_rate': round(n_succ / n_total * 100.0, 2) if n_total > 0 else 0.0,
        'shape_shift': round(n_sh / n_succ * 100.0, 2) if n_succ > 0 else 0.0,
        'scale_shift': round(n_sc / n_succ * 100.0, 2) if n_succ > 0 else 0.0,
        'joint_shift': round(n_jt / n_succ * 100.0, 2) if n_succ > 0 else 0.0,
    }

    # Compute Cohen's w
    if n_succ >= N_MIN_BRANCH:
        p_sh, p_sc, p_jt = agg['shape_shift'] / 100.0, agg['scale_shift'] / 100.0, agg['joint_shift'] / 100.0
        cohen_w_raw = np.sqrt(((p_sh - 0.25) ** 2) / 0.25 + ((p_sc - 0.25) ** 2) / 0.25 + ((p_jt - 0.50) ** 2) / 0.50)
        agg['cohen_w'] = round(float(cohen_w_raw), 2)
    else:
        agg['cohen_w'] = 0.0

    for k in SS_KEYS:
        cnt = (n_sh_arr * (df_sub[k].to_numpy(dtype=float) / 100.0)).sum()
        agg[k] = round(cnt / n_sh * 100.0, 2) if n_sh > 0 else 0.0
    for k in SL_KEYS:
        cnt = (n_sc_arr * (df_sub[k].to_numpy(dtype=float) / 100.0)).sum()
        agg[k] = round(cnt / n_sc * 100.0, 2) if n_sc > 0 else 0.0
    for k in JS_KEYS:
        cnt = (n_jt_arr * (df_sub[k].to_numpy(dtype=float) / 100.0)).sum()
        agg[k] = round(cnt / n_jt * 100.0, 2) if n_jt > 0 else 0.0

    return agg


tree_results = []
summary_tree_rows = []

for i, config in enumerate(configs):
    print(f"[{i+1}/{len(configs)}] Processing tree: "
          f"{config['dataset']} {config['pool_type']} w={config['loss_weight']} "
          f"{config['attack']} {config['norm']}")
    
    df = pd.read_csv(config['csv_path'], compression='gzip', usecols=TREE_COLS)
    
    # --- Per-Epsilon Tree ---
    for step_idx, eps in enumerate(sorted(df['epsilon'].unique()), start=1):
        df_eps = df[df['epsilon'] == eps]
        total = len(df_eps)
        
        # === NODE 0: Attack Success ===
        success_mask = df_eps['pred_main'] != df_eps['y_shape_and_scale']
        success_df = df_eps[success_mask]
        successful = len(success_df)
        success_rate = (successful / total * 100.0) if total > 0 else 0.0
        
        if successful == 0:
            zero_row = {
                'dataset': config['dataset'], 'pool': config['pool_type'],
                'weight': config['loss_weight'], 'attack': config['attack'],
                'norm': config['norm'], 'epsilon': eps, 'eps_step': step_idx,
                'n_total': int(total), 'n_succ': 0, 'n_sh': 0, 'n_sc': 0, 'n_jt': 0,
                'success_rate': 0.0,
                'shape_shift': 0.0, 'scale_shift': 0.0, 'joint_shift': 0.0,
                'cohen_w': 0.0,
            }
            for k in SS_KEYS + SL_KEYS + JS_KEYS:
                zero_row[k] = 0.0
            tree_results.append(zero_row)
            continue
        
        # === NODE 1: Shift Typology ===
        s_df = success_df.copy()
        s_df['true_shape'] = s_df['y_shape_and_scale'] // 3
        s_df['true_scale'] = s_df['y_shape_and_scale'] % 3
        s_df['fooled_shape'] = s_df['pred_main'] // 3
        s_df['fooled_scale'] = s_df['pred_main'] % 3
        
        shape_changed = s_df['fooled_shape'] != s_df['true_shape']
        scale_changed = s_df['fooled_scale'] != s_df['true_scale']
        
        shape_shift_mask = shape_changed & ~scale_changed
        scale_shift_mask = ~shape_changed & scale_changed
        joint_shift_mask = shape_changed & scale_changed
        
        shape_shift_count = int(shape_shift_mask.sum())
        scale_shift_count = int(scale_shift_mask.sum())
        joint_shift_count = int(joint_shift_mask.sum())
        
        # Compute Cohen's w
        if successful >= N_MIN_BRANCH:
            p_sh = shape_shift_count / successful
            p_sc = scale_shift_count / successful
            p_jt = joint_shift_count / successful
            cohen_w_raw = np.sqrt(((p_sh - 0.25) ** 2) / 0.25 + ((p_sc - 0.25) ** 2) / 0.25 + ((p_jt - 0.50) ** 2) / 0.50)
            cohen_w = round(float(cohen_w_raw), 2)
        else:
            cohen_w = 0.0
        
        # === NODE 2: Aux Branch Reactions ===
        ss_breakdown, sl_breakdown, js_breakdown = compute_node2_breakdowns(
            s_df, shape_shift_mask, scale_shift_mask, joint_shift_mask
        )
        
        result = {
            'dataset': config['dataset'],
            'pool': config['pool_type'],
            'weight': config['loss_weight'],
            'attack': config['attack'],
            'norm': config['norm'],
            'epsilon': eps,
            'eps_step': step_idx,
            'n_total': int(total),
            'n_succ': int(successful),
            'n_sh': shape_shift_count,
            'n_sc': scale_shift_count,
            'n_jt': joint_shift_count,
            'success_rate': success_rate,
            'shape_shift': shape_shift_count / successful * 100.0,
            'scale_shift': scale_shift_count / successful * 100.0,
            'joint_shift': joint_shift_count / successful * 100.0,
            'cohen_w': cohen_w,
        }
        
        for k, v in ss_breakdown.items():
            result[k] = (v / shape_shift_count * 100.0) if shape_shift_count > 0 else 0.0
        for k, v in sl_breakdown.items():
            result[k] = (v / scale_shift_count * 100.0) if scale_shift_count > 0 else 0.0
        for k, v in js_breakdown.items():
            result[k] = (v / joint_shift_count * 100.0) if joint_shift_count > 0 else 0.0
        
        tree_results.append(result)

    # --- Aggregate Summary across all epsilons (computed in same pass) ---
    all_success_mask = df['pred_main'] != df['y_shape_and_scale']
    all_success_df = df[all_success_mask].copy()
    all_successful = int(len(all_success_df))
    all_total = int(len(df))
    
    if all_successful > 0:
        all_success_df['true_shape'] = all_success_df['y_shape_and_scale'] // 3
        all_success_df['true_scale'] = all_success_df['y_shape_and_scale'] % 3
        all_success_df['fooled_shape'] = all_success_df['pred_main'] // 3
        all_success_df['fooled_scale'] = all_success_df['pred_main'] % 3
        
        sh_chg = all_success_df['fooled_shape'] != all_success_df['true_shape']
        sc_chg = all_success_df['fooled_scale'] != all_success_df['true_scale']
        
        ss_mask = sh_chg & ~sc_chg
        sl_mask = ~sh_chg & sc_chg
        js_mask = sh_chg & sc_chg
        
        ss_cnt = int(ss_mask.sum())
        sl_cnt = int(sl_mask.sum())
        js_cnt = int(js_mask.sum())
        
        # Compute Cohen's w for summary
        if all_successful >= N_MIN_BRANCH:
            p_sh = ss_cnt / all_successful
            p_sc = sl_cnt / all_successful
            p_jt = js_cnt / all_successful
            cohen_w_raw = np.sqrt(((p_sh - 0.25) ** 2) / 0.25 + ((p_sc - 0.25) ** 2) / 0.25 + ((p_jt - 0.50) ** 2) / 0.50)
            cohen_w = round(float(cohen_w_raw), 2)
        else:
            cohen_w = 0.0
        
        ss_b, sl_b, js_b = compute_node2_breakdowns(all_success_df, ss_mask, sl_mask, js_mask)
        
        sum_row = {
            'dataset': config['dataset'],
            'pool': config['pool_type'],
            'weight': config['loss_weight'],
            'attack': config['attack'],
            'norm': config['norm'],
            'n_total': all_total,
            'n_succ': all_successful,
            'n_sh': ss_cnt,
            'n_sc': sl_cnt,
            'n_jt': js_cnt,
            'success_rate': all_successful / all_total * 100.0,
            'shape_shift': ss_cnt / all_successful * 100.0,
            'scale_shift': sl_cnt / all_successful * 100.0,
            'joint_shift': js_cnt / all_successful * 100.0,
            'cohen_w': cohen_w,
        }
        for k, v in ss_b.items():
            sum_row[k] = (v / ss_cnt * 100.0) if ss_cnt > 0 else 0.0
        for k, v in sl_b.items():
            sum_row[k] = (v / sl_cnt * 100.0) if sl_cnt > 0 else 0.0
        for k, v in js_b.items():
            sum_row[k] = (v / js_cnt * 100.0) if js_cnt > 0 else 0.0
        
        summary_tree_rows.append(sum_row)
    else:
        sum_row = {
            'dataset': config['dataset'],
            'pool': config['pool_type'],
            'weight': config['loss_weight'],
            'attack': config['attack'],
            'norm': config['norm'],
            'n_total': all_total,
            'n_succ': 0, 'n_sh': 0, 'n_sc': 0, 'n_jt': 0,
            'success_rate': 0.0,
            'shape_shift': 0.0, 'scale_shift': 0.0, 'joint_shift': 0.0,
            'cohen_w': 0.0,
        }
        for k in SS_KEYS + SL_KEYS + JS_KEYS:
            sum_row[k] = 0.0
        summary_tree_rows.append(sum_row)

tree_df = pd.DataFrame(tree_results)
summary_tree = pd.DataFrame(summary_tree_rows)

#%%
# =============================================================================
# TABLE SET 1: MACRO SUMMARY — GLOBAL 'ALL' & SINGLE-PARAMETER VARIATIONS (%)
# =============================================================================
def format_metric_value(metric, value):
    """Format metric value appropriately (2 decimals for percentages, cohen_w as is)"""
    if metric == 'cohen_w':
        return f"{value:6.2f}"
    else:
        return f"{value:6.2f}"

# Compute 17 column dictionaries
columns = {}

# Column 1: ALL
columns['ALL'] = aggregate_tree_subset(tree_df, N_SINGLE_EPS)

# Columns 2-3: DATASET
columns['std'] = aggregate_tree_subset(summary_tree[summary_tree['dataset'] == 'standard'], N_SINGLE_EPS * 5)
columns['con'] = aggregate_tree_subset(summary_tree[summary_tree['dataset'] == 'contrast'], N_SINGLE_EPS * 5)

# Columns 4-5: POOL
columns['flat'] = aggregate_tree_subset(summary_tree[summary_tree['pool'] == 'flatten'], N_SINGLE_EPS * 5)
columns['gap'] = aggregate_tree_subset(summary_tree[summary_tree['pool'] == 'gap'], N_SINGLE_EPS * 5)

# Columns 6-8: LOSS WEIGHT
for w in [0.5, 1.0, 2.0]:
    columns[f'w={w}'] = aggregate_tree_subset(summary_tree[summary_tree['weight'] == w], N_SINGLE_EPS * 5)

# Columns 9-10: ATTACK
columns['fgsm'] = aggregate_tree_subset(summary_tree[summary_tree['attack'] == 'fgsm'], N_SINGLE_EPS * 5)
columns['pgd'] = aggregate_tree_subset(summary_tree[summary_tree['attack'] == 'pgd'], N_SINGLE_EPS * 5)

# Columns 11-12: NORM
columns['linf'] = aggregate_tree_subset(summary_tree[summary_tree['norm'] == 'linf'], N_SINGLE_EPS * 5)
columns['l2'] = aggregate_tree_subset(summary_tree[summary_tree['norm'] == 'l2'], N_SINGLE_EPS * 5)

# Columns 13-17: EPSILON STEP (1 -> 5)
for step in [1, 2, 3, 4, 5]:
    columns[f'eps_{step}'] = aggregate_tree_subset(tree_df[tree_df['eps_step'] == step], N_SINGLE_EPS)

# Print Table 1
header1 = (
    f"{'Tree Node / Metric':<22} | {'GLOBAL':^6} | {'DATASET':^13} | {'POOL':^13} | "
    f"{'LOSS WEIGHT':^20} | {'ATTACK':^13} | {'NORM':^13} | {'EPSILON STEP (1 -> 5)':^34} |"
)
header2 = (
    f"{'':<22} | {'ALL':>6} | {'std':>6} {'con':>6} | {'flat':>6} {'gap':>6} | "
    f"{'w=0.5':>6} {'w=1.0':>6} {'w=2.0':>6} | {'fgsm':>6} {'pgd':>6} | "
    f"{'linf':>6} {'l2':>6} | {'eps_1':>6} {'eps_2':>6} {'eps_3':>6} {'eps_4':>6} {'eps_5':>6} |"
)
table1_width = len(header1)

print("=" * table1_width)
print("=== TABLE 1: MACRO SUMMARY — GLOBAL 'ALL' & SINGLE-PARAMETER VARIATIONS (%) ===")
print("=" * table1_width)
print(header1)
print(header2)
print("-" * table1_width)

# Tree sections
sections = [
    ("[NODE 0: Outcome]", ['success_rate']),
    ("[NODE 1: Shift Type]", ['shape_shift', 'scale_shift', 'joint_shift', 'cohen_w']),
    ("[NODE 2: Shape-Shift]", SS_KEYS),
    ("[NODE 2: Scale-Shift]", SL_KEYS),
    ("[NODE 2: Joint-Shift]", JS_KEYS)
]

col_groups_t1 = [
    ['ALL'],
    ['std', 'con'],
    ['flat', 'gap'],
    ['w=0.5', 'w=1.0', 'w=2.0'],
    ['fgsm', 'pgd'],
    ['linf', 'l2'],
    ['eps_1', 'eps_2', 'eps_3', 'eps_4', 'eps_5'],
]

for section_name, metrics in sections:
    print("-" * table1_width)
    print(f"{section_name:<22} |" + " " * (table1_width - 25) + "|")
    
    for metric in metrics:
        display_name = 'cohen_w (vs 25/25/50)' if metric == 'cohen_w' else DISPLAY_RENAME.get(metric, metric)
        group_strs = []
        for grp in col_groups_t1:
            grp_vals = " ".join(format_metric_value(metric, columns[c][metric]) for c in grp)
            group_strs.append(grp_vals)
        row = f"{display_name:<22} | " + " | ".join(group_strs) + " |"
        print(row)

print("=" * table1_width)

#%%
# =============================================================================
# TABLE SET 2: COMBINED SUMMARY + PER-EPSILON TABLES (GROUPED BY ATTACK-NORM)
# =============================================================================
attack_norm_pairs = [
    ('pgd', 'linf'),
    ('fgsm', 'linf'),
    ('fgsm', 'l2'),
    ('pgd', 'l2')
]

table_counter = 1
for attack, norm in attack_norm_pairs:
    # Filter data for this attack-norm pair
    sub_summary = summary_tree[(summary_tree['attack'] == attack) & (summary_tree['norm'] == norm)]
    sub_tree = tree_df[(tree_df['attack'] == attack) & (tree_df['norm'] == norm)]
    
    # Compute ATK_ALL column
    atk_all = aggregate_tree_subset(sub_summary, N_SINGLE_EPS * 5)
    
    # Build two-line aligned header (35 chars per model block)
    model_headers = []
    sub_headers = []
    sub_hdr_unit = f"{'ALL':>5} " + " ".join([f"{f'eps_{i}':>5}" for i in range(1, 6)])
    for dataset in ['standard', 'contrast']:
        ds_short = 'std' if dataset == 'standard' else 'con'
        for pool in ['flatten', 'gap']:
            pool_short = 'flat' if pool == 'flatten' else 'gap'
            for weight in [0.5, 1.0, 2.0]:
                lbl = f"{ds_short} | {pool_short} | w={weight}"
                model_headers.append(f"{lbl:^35}")
                sub_headers.append(sub_hdr_unit)
    
    header1 = f"{'Tree Node / Metric':<22} | {'ATK_ALL':>7} || " + " | ".join(model_headers) + " |"
    header2 = f"{'':<22} | {'(Pool)':>7} || " + " | ".join(sub_headers) + " |"
    table2_width = len(header1)

    print("=" * table2_width)
    print(f"=== TABLE 2.{table_counter}: ATTACK: {attack.upper()} | NORM: {norm.upper()} ===")
    print("=" * table2_width)
    print(header1)
    print(header2)
    print("-" * table2_width)
    
    # Sections
    sections = [
        ("[NODE 0: Outcome]", ['success_rate']),
        ("[NODE 1: Shift Type]", ['shape_shift', 'scale_shift', 'joint_shift', 'cohen_w']),
        ("[NODE 2: Shape-Shift]", SS_KEYS),
        ("[NODE 2: Scale-Shift]", SL_KEYS),
        ("[NODE 2: Joint-Shift]", JS_KEYS)
    ]
    
    for section_name, metrics in sections:
        print("-" * table2_width)
        print(f"{section_name:<22} |" + " " * (table2_width - 25) + "|")
        
        for metric in metrics:
            display_name = 'cohen_w (vs 25/25/50)' if metric == 'cohen_w' else DISPLAY_RENAME.get(metric, metric)
            atk_all_val = atk_all[metric]
            if metric == 'cohen_w':
                atk_all_str = f"{atk_all_val:7.2f}"
            else:
                atk_all_str = f"{atk_all_val:7.1f}"
            
            row = f"{display_name:<22} | {atk_all_str} ||"
            
            for dataset in ['standard', 'contrast']:
                for pool in ['flatten', 'gap']:
                    for weight in [0.5, 1.0, 2.0]:
                        model_summary = sub_summary[
                            (sub_summary['dataset'] == dataset) & 
                            (sub_summary['pool'] == pool) & 
                            (sub_summary['weight'] == weight)
                        ]
                        model_tree = sub_tree[
                            (sub_tree['dataset'] == dataset) & 
                            (sub_tree['pool'] == pool) & 
                            (sub_tree['weight'] == weight)
                        ]
                        
                        if len(model_summary) > 0:
                            all_vals = aggregate_tree_subset(model_summary, N_SINGLE_EPS * 5)
                            all_val = all_vals[metric]
                        else:
                            all_val = 0.0
                        
                        eps_vals = []
                        for step in [1, 2, 3, 4, 5]:
                            eps_df = model_tree[model_tree['eps_step'] == step]
                            if len(eps_df) > 0:
                                eps_vals_dict = aggregate_tree_subset(eps_df, N_SINGLE_EPS)
                                eps_val = eps_vals_dict[metric]
                            else:
                                eps_val = 0.0
                            eps_vals.append(eps_val)
                        
                        if metric == 'cohen_w':
                            model_str = f" {all_val:5.2f} {eps_vals[0]:5.2f} {eps_vals[1]:5.2f} {eps_vals[2]:5.2f} {eps_vals[3]:5.2f} {eps_vals[4]:5.2f}"
                        else:
                            model_str = f" {all_val:5.1f} {eps_vals[0]:5.1f} {eps_vals[1]:5.1f} {eps_vals[2]:5.1f} {eps_vals[3]:5.1f} {eps_vals[4]:5.1f}"
                        
                        row += model_str + " |"
            
            print(row)
    
    print("=" * table2_width)
    print()
    table_counter += 1

#%%
# =============================================================================
# TABLE SET 3: NON-PARAMETRIC DISTRIBUTIONS & VARIATION DRIVERS
# =============================================================================
def kl_divergence(p, q, eps=1e-12):
    """Compute Bernoulli KL divergence safely"""
    # Clip q to avoid log(0)
    q = np.clip(q, eps, 1 - eps)
    
    # Handle edge cases
    if p <= 0:
        return (1 - p) * np.log((1 - p) / (1 - q))
    elif p >= 1:
        return p * np.log(p / q)
    else:
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

def compute_branch_denominators(df, n_per_row=N_SINGLE_EPS):
    """Compute exact branch denominators for each metric in ALL_26_METRICS."""
    denominators = {}
    if {'n_total', 'n_succ', 'n_sh', 'n_sc', 'n_jt'}.issubset(df.columns):
        n_total = df['n_total'].to_numpy(dtype=int)
        n_succ = df['n_succ'].to_numpy(dtype=int)
        n_sh = df['n_sh'].to_numpy(dtype=int)
        n_sc = df['n_sc'].to_numpy(dtype=int)
        n_jt = df['n_jt'].to_numpy(dtype=int)
    else:
        n_total = np.full(len(df), n_per_row, dtype=int)
        n_succ = np.round(n_per_row * (df['success_rate'].to_numpy(dtype=float) / 100.0)).astype(int)
        n_sh = np.round(n_succ * (df['shape_shift'].to_numpy(dtype=float) / 100.0)).astype(int)
        n_sc = np.round(n_succ * (df['scale_shift'].to_numpy(dtype=float) / 100.0)).astype(int)
        n_jt = np.maximum(0, n_succ - n_sh - n_sc)

    # Node 0
    denominators['success_rate'] = n_total

    # Node 1 (including cohen_w)
    denominators['shape_shift'] = n_succ.copy()
    denominators['scale_shift'] = n_succ.copy()
    denominators['joint_shift'] = n_succ.copy()
    denominators['cohen_w'] = n_succ.copy()

    # Node 2
    for metric in SS_KEYS:
        denominators[metric] = n_sh.copy()
    for metric in SL_KEYS:
        denominators[metric] = n_sc.copy()
    for metric in JS_KEYS:
        denominators[metric] = n_jt.copy()

    return denominators

# Compute denominators for both per-epsilon (240 rows) and summary (48 rows)
denominators = compute_branch_denominators(tree_df, N_SINGLE_EPS)
summary_denominators = compute_branch_denominators(summary_tree, N_SINGLE_EPS * 5)

# Compute global ALL values
global_all = aggregate_tree_subset(tree_df, N_SINGLE_EPS)

# Header for Table 3A
header_3a = (
    f"{'Tree Node / Metric':<22} | {'ALL':>6} | {'Median [Q25 – Q75]':^19} | "
    f"{'Min - Max':^13} | {'★★ (48)':^12} || "
    f"{'data':>5} {'pool':>5} {'wght':>5} {'atk':>5} {'norm':>5} {'eps':>5} | "
    f"{'Top 2-Way Interaction':<21} |"
)
table3a_width = len(header_3a)

print("=" * table3a_width)
print("=== TABLE 3A: NON-PARAMETRIC DISTRIBUTIONS, WILSON ★★ RATE & KL-DEVIANCE (D²%) OVERVIEW ===")
print("=" * table3a_width)
print(header_3a)
print("-" * table3a_width)

# Sections for Table 3A
sections_3a = [
    ("[NODE 0: Outcome]", ['success_rate']),
    ("[NODE 1: Shift Type]", ['shape_shift', 'scale_shift', 'joint_shift', 'cohen_w']),
    ("[NODE 2: Shape-Shift]", SS_KEYS),
    ("[NODE 2: Scale-Shift]", SL_KEYS),
    ("[NODE 2: Joint-Shift]", JS_KEYS)
]

# Factors for analysis
factors = ['dataset', 'pool', 'weight', 'attack', 'norm', 'eps_step']
factor_display_names = ['data', 'pool', 'wght', 'atk', 'norm', 'eps']

# Compute significance counts for Node 2 leaves across the 48 summary configs
significance_counts = {k: 0 for k in SS_KEYS + SL_KEYS + JS_KEYS}
for row_pos, (_, row) in enumerate(summary_tree.iterrows()):
    for keys in (SS_KEYS, SL_KEYS, JS_KEYS):
        vals_dict = {k: float(row[k]) for k in keys}
        n_branch = int(summary_denominators[keys[0]][row_pos])
        tiers = classify_group_significance(vals_dict, n_branch)
        for k in keys:
            if tiers[k] == 'primary':
                significance_counts[k] += 1

for section_name, metrics in sections_3a:
    print("-" * table3a_width)
    print(f"{section_name:<22} |" + " " * (table3a_width - 25) + "|")
    
    # Print each metric row
    for metric in metrics:
        # Global ALL value
        all_val = global_all[metric]
        
        # Filter valid values for non-parametric stats
        valid_mask = denominators[metric] >= N_MIN_BRANCH
        valid_values = tree_df.loc[valid_mask, metric].to_numpy(dtype=float)
        
        if len(valid_values) > 0:
            # Non-parametric stats
            percentiles = np.percentile(valid_values, [0, 25, 50, 75, 100])
            if metric == 'cohen_w':
                min_max_str = f"{percentiles[0]:5.2f} - {percentiles[4]:5.2f}"
                median_iqr_str = f"{percentiles[2]:5.2f} [{percentiles[1]:4.2f} – {percentiles[3]:4.2f}]"
            else:
                min_max_str = f"{percentiles[0]:5.1f} - {percentiles[4]:5.1f}"
                median_iqr_str = f"{percentiles[2]:5.1f} [{percentiles[1]:4.1f} – {percentiles[3]:4.1f}]"
        else:
            min_max_str = f"{'--':>5} - {'--':>5}"
            median_iqr_str = f"{'--':>5} [{'--':>4} – {'--':>4}]"
        
        # ★★ Rate
        if metric in significance_counts:
            count = significance_counts[metric]
            pct = (count / 48) * 100
            star_str = f"{count:2d}/48 ({pct:3.0f}%)"
        else:
            star_str = f"{'--':^12}"
        
        # --- Compute 1F & 2F KL-Deviance (D²%) shares and significant variations ---
        if 'variations_3b' not in locals():
            variations_3b = []
        if section_name == "[NODE 0: Outcome]" and metric == "success_rate":
            variations_3b = []

        level_short = {
            'dataset': {'standard': 'std', 'contrast': 'con'},
            'pool': {'flatten': 'flat', 'gap': 'gap'},
            'weight': {0.5: 'w=0.5', 1.0: 'w=1.0', 2.0: 'w=2.0'},
            'attack': {'fgsm': 'fgsm', 'pgd': 'pgd'},
            'norm': {'linf': 'linf', 'l2': 'l2'},
            'eps_step': {1: 'eps_1', 2: 'eps_2', 3: 'eps_3', 4: 'eps_4', 5: 'eps_5'},
        }
        factor_short_map = dict(zip(factors, factor_display_names))

        n_arr = denominators[metric].astype(float)
        val_arr = tree_df[metric].to_numpy(dtype=float)
        pos_mask = (n_arr >= N_MIN_BRANCH) if metric == 'cohen_w' else (n_arr > 0)
        n_total_metric = float(n_arr[pos_mask].sum())

        if n_total_metric > 0:
            p_all_weighted = float((n_arr[pos_mask] * val_arr[pos_mask]).sum() / n_total_metric)
            if metric == 'cohen_w':
                d_tot = float((n_arr[pos_mask] * ((val_arr[pos_mask] - p_all_weighted) ** 2)).sum())
            else:
                p_all_prob = p_all_weighted / 100.0
                d_tot = float(sum(
                    n_i * kl_divergence(v_i / 100.0, p_all_prob)
                    for n_i, v_i in zip(n_arr[pos_mask], val_arr[pos_mask])
                ))
        else:
            p_all_weighted = 0.0
            d_tot = 0.0

        d1_dev = {}
        d1_pct = {}
        f1_level_vals = {}

        for f_col, f_disp in zip(factors, factor_display_names):
            dev_f = 0.0
            lvl_stats = []
            f1_level_vals[f_col] = {}

            for lvl in sorted(tree_df[f_col].unique()):
                l_mask = (tree_df[f_col].to_numpy() == lvl) & pos_mask
                n_lvl = float(n_arr[l_mask].sum())
                if n_lvl > 0:
                    v_lvl_mean = float((n_arr[l_mask] * val_arr[l_mask]).sum() / n_lvl)
                    if metric == 'cohen_w':
                        dev_f += n_lvl * ((v_lvl_mean - p_all_weighted) ** 2)
                        v_lvl_disp = float(aggregate_tree_subset(tree_df[tree_df[f_col] == lvl], N_SINGLE_EPS)['cohen_w'])
                    else:
                        dev_f += n_lvl * kl_divergence(v_lvl_mean / 100.0, p_all_weighted / 100.0)
                        v_lvl_disp = v_lvl_mean
                    f1_level_vals[f_col][lvl] = v_lvl_disp
                    if n_lvl >= N_MIN_BRANCH:
                        lvl_stats.append((lvl, v_lvl_disp, int(round(n_lvl))))

            d1_dev[f_col] = dev_f
            pct_share = (dev_f / d_tot * 100.0) if d_tot > 1e-12 else 0.0
            d1_pct[f_col] = pct_share

            # Check 1F variation for Table 3B
            if len(lvl_stats) >= 2:
                lvl_sorted = sorted(lvl_stats, key=lambda x: x[1])
                min_lvl, min_v, min_n = lvl_sorted[0]
                max_lvl, max_v, max_n = lvl_sorted[-1]
                delta = max_v - min_v

                if metric == 'cohen_w':
                    ci_ok = (min_n >= N_MIN_BRANCH) and (max_n >= N_MIN_BRANCH)
                    sig_1f = ci_ok and (pct_share >= 5.0) and (delta >= 0.10)
                    val_str = " -> ".join(
                        f"{level_short[f_col].get(lv, lv)}={vv:.2f}" for lv, vv, _ in lvl_sorted
                    )
                    delta_str = f"{delta:6.2f}"
                else:
                    _, ci_min_hi = wilson_ci(min_v / 100.0, min_n)
                    ci_max_lo, _ = wilson_ci(max_v / 100.0, max_n)
                    sig_1f = (ci_max_lo > ci_min_hi) and (pct_share >= 5.0) and (delta >= 5.0)
                    val_str = " -> ".join(
                        f"{level_short[f_col].get(lv, lv)}={vv:.1f}%" for lv, vv, _ in lvl_sorted
                    )
                    delta_str = f"{delta:6.1f}"

                if sig_1f:
                    variations_3b.append({
                        'metric': metric,
                        'type': '1F',
                        'factor': f_disp,
                        'd2_num': pct_share,
                        'delta_num': delta,
                        'd2_str': f"{pct_share:3.0f}%",
                        'delta_str': delta_str,
                        'values': val_str,
                    })

        # Format 6 single-factor D²% shares
        if d_tot > 1e-12:
            d2_str = " ".join(f"{d1_pct[f]:4.0f}%" for f in factors)
        else:
            d2_str = "  --   --   --   --   --   --"

        # Evaluate all 15 2-Way Factor Pairs for Top Interaction & Table 3B
        best_pair_name = None
        best_pair_d2 = 0.0

        for idx1 in range(len(factors)):
            for idx2 in range(idx1 + 1, len(factors)):
                f1, f2 = factors[idx1], factors[idx2]
                pair_label = f"{factor_short_map[f1]}×{factor_short_map[f2]}"
                dev_joint = 0.0
                cell_stats = []

                for l1 in sorted(tree_df[f1].unique()):
                    for l2 in sorted(tree_df[f2].unique()):
                        c_mask = (tree_df[f1].to_numpy() == l1) & (tree_df[f2].to_numpy() == l2) & pos_mask
                        n_cell = float(n_arr[c_mask].sum())
                        if n_cell > 0:
                            v_cell_mean = float((n_arr[c_mask] * val_arr[c_mask]).sum() / n_cell)
                            if metric == 'cohen_w':
                                dev_joint += n_cell * ((v_cell_mean - p_all_weighted) ** 2)
                                v_cell_disp = float(aggregate_tree_subset(
                                    tree_df[(tree_df[f1] == l1) & (tree_df[f2] == l2)], N_SINGLE_EPS
                                )['cohen_w'])
                                base_all = float(all_val)
                            else:
                                dev_joint += n_cell * kl_divergence(v_cell_mean / 100.0, p_all_weighted / 100.0)
                                v_cell_disp = v_cell_mean
                                base_all = p_all_weighted

                            if n_cell >= N_MIN_BRANCH and l1 in f1_level_vals[f1] and l2 in f1_level_vals[f2]:
                                exp_v = base_all + (f1_level_vals[f1][l1] - base_all) + (f1_level_vals[f2][l2] - base_all)
                                excess = v_cell_disp - exp_v
                                c_name = f"{level_short[f1].get(l1, l1)}×{level_short[f2].get(l2, l2)}"
                                cell_stats.append((c_name, v_cell_disp, int(round(n_cell)), excess))

                dev_int = max(0.0, dev_joint - d1_dev[f1] - d1_dev[f2])
                d2_int_pct = (dev_int / d_tot * 100.0) if d_tot > 1e-12 else 0.0

                if d2_int_pct > best_pair_d2:
                    best_pair_d2 = d2_int_pct
                    best_pair_name = pair_label

                # Check 2F interaction for Table 3B
                if len(cell_stats) >= 2:
                    cell_sorted = sorted(cell_stats, key=lambda x: x[1])
                    min_c, min_v, min_n, _ = cell_sorted[0]
                    max_c, max_v, max_n, _ = cell_sorted[-1]
                    delta_2f = max_v - min_v
                    peak_excess = max(cell_stats, key=lambda x: abs(x[3]))[3]

                    if metric == 'cohen_w':
                        sig_2f = (d2_int_pct >= 5.0) and (delta_2f >= 0.10) and (abs(peak_excess) >= 0.05)
                        val_2f_str = f"{min_c}={min_v:.2f} -> {max_c}={max_v:.2f} (excess Δ={peak_excess:+.2f})"
                        delta_2f_str = f"{delta_2f:6.2f}"
                    else:
                        _, ci_min_hi = wilson_ci(min_v / 100.0, min_n)
                        ci_max_lo, _ = wilson_ci(max_v / 100.0, max_n)
                        sig_2f = (ci_max_lo > ci_min_hi) and (d2_int_pct >= 5.0) and (delta_2f >= 5.0) and (abs(peak_excess) >= 3.0)
                        val_2f_str = f"{min_c}={min_v:.1f}% -> {max_c}={max_v:.1f}% (excess Δ={peak_excess:+.1f})"
                        delta_2f_str = f"{delta_2f:6.1f}"

                    if sig_2f:
                        variations_3b.append({
                            'metric': metric,
                            'type': '2F',
                            'factor': pair_label,
                            'd2_num': d2_int_pct,
                            'delta_num': delta_2f,
                            'd2_str': f"{d2_int_pct:3.0f}%",
                            'delta_str': delta_2f_str,
                            'values': val_2f_str,
                        })

        if best_pair_name is not None and best_pair_d2 >= 1.0:
            top_interaction_str = f"{best_pair_name:<10} ({best_pair_d2:4.1f}%)"
        else:
            top_interaction_str = f"{'--':^21}"
        
        display_name = 'cohen_w (vs 25/25/50)' if metric == 'cohen_w' else DISPLAY_RENAME.get(metric, metric)
        
        # Format row
        if metric == 'cohen_w':
            row = f"{display_name:<22} | {all_val:6.2f} | {median_iqr_str:>19} | {min_max_str:>13} | {star_str:>12} || {d2_str:<35} | {top_interaction_str:<21} |"
        else:
            row = f"{display_name:<22} | {all_val:6.1f} | {median_iqr_str:>19} | {min_max_str:>13} | {star_str:>12} || {d2_str:<35} | {top_interaction_str:<21} |"
        
        print(row)

print("=" * table3a_width)
print()

# Table 3B: Significant variations
header_3b = "Tree Node / Metric     | Type | Factor / Pair |  D²% | Δ (pp) | Concrete Pooled Subgroup Values"
table3b_width = max(115, len(header_3b))
print("=" * table3b_width)
print("=== TABLE 3B: ALL RELEVANT & SIGNIFICANT VARIATIONS (Sorted by Strength — Wilson 95% CI, D² >= 5%, Δ >= 5.0 pp) ===")
print("=" * table3b_width)
print(header_3b)
print("-" * table3b_width)

if len(variations_3b) > 0:
    variations_sorted = sorted(
        variations_3b,
        key=lambda x: (ALL_26_METRICS.index(x['metric']), -x['d2_num'], -x['delta_num'])
    )
    for var in variations_sorted:
        m_disp = 'cohen_w (vs 25/25/50)' if var['metric'] == 'cohen_w' else var['metric']
        print(
            f"{m_disp:<22} | {var['type']:<4} | {var['factor']:<13} | "
            f"{var['d2_str']:>4} | {var['delta_str']:>6} | {var['values']}"
        )
else:
    print(f"{'--':<22} | {'--':<4} | {'None (Stable)':<13} | {'--':>4} | {'<5.0':>6} | Stays stable across all factors")

print("=" * table3b_width)

#%%
# =============================================================================
# MODE COLLAPSE INSPECTOR: Which exact class did the model collapse into?
# =============================================================================
SHAPES = {0: 'Square', 1: 'Ellipse', 2: 'Heart'}
SCALES = {0: 'Small', 1: 'Medium', 2: 'Large'}

collapse_rows = []

for config in configs:
    df = pd.read_csv(
        config['csv_path'],
        compression='gzip',
        usecols=['epsilon', 'pred_main', 'pred_shape', 'pred_scale']
    )
    for eps in sorted(df['epsilon'].unique()):
        df_eps = df[df['epsilon'] == eps]
        total = len(df_eps)

        # Most frequent predicted class in pred_main across ALL test images
        vc_main = df_eps['pred_main'].value_counts()
        top_main_cls = int(vc_main.index[0])
        top_main_pct = round(vc_main.iloc[0] / total * 100, 2)

        # Filter for single-class collapse (>= 99% of all test images predicted as 1 class)
        if top_main_pct >= 99.0:
            sh_id, sc_id = top_main_cls // 3, top_main_cls % 3
            vc_sh = df_eps['pred_shape'].value_counts()
            vc_sc = df_eps['pred_scale'].value_counts()

            top_sh_cls = int(vc_sh.index[0])
            top_sh_pct = round(vc_sh.iloc[0] / total * 100, 2)
            top_sc_cls = int(vc_sc.index[0])
            top_sc_pct = round(vc_sc.iloc[0] / total * 100, 2)

            collapse_rows.append({
                'dataset': config['dataset'],
                'pool': config['pool_type'],
                'weight': config['loss_weight'],
                'attack': config['attack'],
                'norm': config['norm'],
                'epsilon': eps,
                'main_collapsed_to': f"[{top_main_cls}] {SHAPES[sh_id]}-{SCALES[sc_id]}",
                'main_%': top_main_pct,
                'aux_shape_top': f"{SHAPES[top_sh_cls]} ({top_sh_pct}%)",
                'aux_scale_top': f"{SCALES[top_sc_cls]} ({top_sc_pct}%)",
            })

collapse_df = pd.DataFrame(collapse_rows)
print(f"=== SINGLE-CLASS MODE COLLAPSE CONFIGURATIONS ({len(collapse_df)} found) ===")
print(collapse_df.to_string(index=False))






#%%
# =============================================================================
# TABLE 4: BRANCH ACCURACIES, CONFIDENCE (%) & NORMALIZED ENTROPY (%) BY ARCHETYPE
# =============================================================================
LN9 = float(np.log(9.0))
LN3 = float(np.log(3.0))

CONF_ENT_COLS = [
    'epsilon', 'y_shape_and_scale',
    'pred_main', 'confidence', 'entropy',
    'pred_shape', 'confidence_shape', 'entropy_shape',
    'pred_scale', 'confidence_scale', 'entropy_scale',
]

# Accumulate exact sample-weighted numerators & denominators per (config, eps_step)
ce_rows = []

for i, config in enumerate(configs):
    df = pd.read_csv(config['csv_path'], compression='gzip', usecols=CONF_ENT_COLS)

    for step_idx, eps in enumerate(sorted(df['epsilon'].unique()), start=1):
        dfe = df[df['epsilon'] == eps]
        n_tot = len(dfe)

        y_main = dfe['y_shape_and_scale'].to_numpy()
        y_sh = y_main // 3
        y_sc = y_main % 3

        p_main = dfe['pred_main'].to_numpy()
        p_sh = dfe['pred_shape'].to_numpy()
        p_sc = dfe['pred_scale'].to_numpy()

        f_sh = p_main // 3
        f_sc = p_main % 3

        # Confidence in % (0..100) and Normalized Entropy in % (0..100)
        c_m = dfe['confidence'].to_numpy(dtype=float) * 100.0
        c_sh = dfe['confidence_shape'].to_numpy(dtype=float) * 100.0
        c_sc = dfe['confidence_scale'].to_numpy(dtype=float) * 100.0

        e_m = (dfe['entropy'].to_numpy(dtype=float) / LN9) * 100.0
        e_sh = (dfe['entropy_shape'].to_numpy(dtype=float) / LN3) * 100.0
        e_sc = (dfe['entropy_scale'].to_numpy(dtype=float) / LN3) * 100.0

        # Masks
        m_fail = (p_main == y_main)
        m_succ = ~m_fail

        sh_changed = (f_sh != y_sh)
        sc_changed = (f_sc != y_sc)

        m_ss = m_succ & sh_changed & ~sc_changed
        m_sl = m_succ & ~sh_changed & sc_changed
        m_js = m_succ & sh_changed & sc_changed

        sh_ret = (p_sh == y_sh)
        sh_ali = (p_sh == f_sh) & ~sh_ret
        sh_div_shifted = ~sh_ret & ~sh_ali
        sh_div_unshifted = ~sh_ret

        sc_ret = (p_sc == y_sc)
        sc_ali = (p_sc == f_sc) & ~sc_ret
        sc_div_shifted = ~sc_ret & ~sc_ali
        sc_div_unshifted = ~sc_ret

        # 5 Archetype Masks
        # 1) Failed Attack (Main correct)
        arch_fail = m_fail
        # 2) Full Aligned (ss_aligned_retained | sl_retained_aligned | js_aligned_aligned)
        arch_ali = (m_ss & sh_ali & sc_ret) | (m_sl & sh_ret & sc_ali) | (m_js & sh_ali & sc_ali)
        # 3) Full Retained / Shortcut (both aux heads retain ground truth)
        arch_ret = m_succ & sh_ret & sc_ret
        # 4) Partial Retained in Joint-Shift (js_aligned_retained | js_retained_aligned)
        arch_js_sh_res = m_js & sh_ret & sc_ali  # Shape resists, Scale fooled
        arch_js_sc_res = m_js & sh_ali & sc_ret  # Scale resists, Shape fooled
        arch_part = arch_js_sh_res | arch_js_sc_res
        # 5) Any Diverged State (remaining successful attacks where >=1 aux head diverged)
        sh_is_div = (m_ss & sh_div_shifted) | (m_sl & sh_div_unshifted) | (m_js & sh_div_shifted)
        sc_is_div = (m_ss & sc_div_unshifted) | (m_sl & sc_div_shifted) | (m_js & sc_div_shifted)
        arch_div = m_succ & (sh_is_div | sc_is_div)

        row = {
            'dataset': config['dataset'], 'pool': config['pool_type'],
            'weight': config['loss_weight'], 'attack': config['attack'],
            'norm': config['norm'], 'eps_step': step_idx,
            # Counts & Accuracies
            'n_tot': n_tot,
            'cnt_acc_m': int(m_fail.sum()),
            'cnt_acc_sh': int(sh_ret.sum()),
            'cnt_acc_sc': int(sc_ret.sum()),
            'cnt_fail_aux_ok': int((m_fail & sh_ret & sc_ret).sum()),
        }

        for name, mask in [
            ('fail', arch_fail),
            ('ali', arch_ali),
            ('ret', arch_ret),
            ('part', arch_part),
            ('div', arch_div),
        ]:
            cnt = int(mask.sum())
            row[f'n_{name}'] = cnt
            row[f'sum_{name}_cm'] = float(c_m[mask].sum()) if cnt > 0 else 0.0
            row[f'sum_{name}_csh'] = float(c_sh[mask].sum()) if cnt > 0 else 0.0
            row[f'sum_{name}_csc'] = float(c_sc[mask].sum()) if cnt > 0 else 0.0
            row[f'sum_{name}_em'] = float(e_m[mask].sum()) if cnt > 0 else 0.0
            row[f'sum_{name}_esh'] = float(e_sh[mask].sum()) if cnt > 0 else 0.0
            row[f'sum_{name}_esc'] = float(e_sc[mask].sum()) if cnt > 0 else 0.0

        # Special tracking for Archetype 4 (Resisting Aux vs Fooled Aux in JS Partial Retention)
        row['sum_part_c_resist'] = float(c_sh[arch_js_sh_res].sum() + c_sc[arch_js_sc_res].sum())
        row['sum_part_c_fooled'] = float(c_sc[arch_js_sh_res].sum() + c_sh[arch_js_sc_res].sum())
        row['sum_part_e_resist'] = float(e_sh[arch_js_sh_res].sum() + e_sc[arch_js_sc_res].sum())
        row['sum_part_e_fooled'] = float(e_sc[arch_js_sh_res].sum() + e_sh[arch_js_sc_res].sum())

        # Special tracking for Archetype 5 (Specifically the Diverged Aux head(s))
        n_div_heads = int(sh_is_div.sum() + sc_is_div.sum())
        row['n_div_heads'] = n_div_heads
        row['sum_div_c_only'] = float(c_sh[sh_is_div].sum() + c_sc[sc_is_div].sum())
        row['sum_div_e_only'] = float(e_sh[sh_is_div].sum() + e_sc[sc_is_div].sum())

        ce_rows.append(row)

ce_df = pd.DataFrame(ce_rows)


def aggregate_ce_subset(sub):
    out = {}
    n_tot = sub['n_tot'].sum()
    n_fail = sub['n_fail'].sum()
    out['acc_main'] = sub['cnt_acc_m'].sum() / n_tot * 100.0 if n_tot > 0 else 0.0
    out['acc_shape'] = sub['cnt_acc_sh'].sum() / n_tot * 100.0 if n_tot > 0 else 0.0
    out['acc_scale'] = sub['cnt_acc_sc'].sum() / n_tot * 100.0 if n_tot > 0 else 0.0
    out['fail_aux_ok'] = sub['cnt_fail_aux_ok'].sum() / n_fail * 100.0 if n_fail > 0 else 0.0

    for name in ['fail', 'ali', 'ret', 'part', 'div']:
        n_k = sub[f'n_{name}'].sum()
        for stat in ['cm', 'csh', 'csc', 'em', 'esh', 'esc']:
            out[f'{name}_{stat}'] = sub[f'sum_{name}_{stat}'].sum() / n_k if n_k > 0 else 0.0

    n_part = sub['n_part'].sum()
    out['part_c_resist'] = sub['sum_part_c_resist'].sum() / n_part if n_part > 0 else 0.0
    out['part_c_fooled'] = sub['sum_part_c_fooled'].sum() / n_part if n_part > 0 else 0.0
    out['part_e_resist'] = sub['sum_part_e_resist'].sum() / n_part if n_part > 0 else 0.0
    out['part_e_fooled'] = sub['sum_part_e_fooled'].sum() / n_part if n_part > 0 else 0.0

    n_dh = sub['n_div_heads'].sum()
    out['div_c_only'] = sub['sum_div_c_only'].sum() / n_dh if n_dh > 0 else 0.0
    out['div_e_only'] = sub['sum_div_e_only'].sum() / n_dh if n_dh > 0 else 0.0
    return out


ce_cols = {
    'ALL': aggregate_ce_subset(ce_df),
    'std': aggregate_ce_subset(ce_df[ce_df['dataset'] == 'standard']),
    'con': aggregate_ce_subset(ce_df[ce_df['dataset'] == 'contrast']),
    'flat': aggregate_ce_subset(ce_df[ce_df['pool'] == 'flatten']),
    'gap': aggregate_ce_subset(ce_df[ce_df['pool'] == 'gap']),
    'w=0.5': aggregate_ce_subset(ce_df[ce_df['weight'] == 0.5]),
    'w=1.0': aggregate_ce_subset(ce_df[ce_df['weight'] == 1.0]),
    'w=2.0': aggregate_ce_subset(ce_df[ce_df['weight'] == 2.0]),
    'fgsm': aggregate_ce_subset(ce_df[ce_df['attack'] == 'fgsm']),
    'pgd': aggregate_ce_subset(ce_df[ce_df['attack'] == 'pgd']),
    'linf': aggregate_ce_subset(ce_df[ce_df['norm'] == 'linf']),
    'l2': aggregate_ce_subset(ce_df[ce_df['norm'] == 'l2']),
}
for s in range(1, 6):
    ce_cols[f'eps_{s}'] = aggregate_ce_subset(ce_df[ce_df['eps_step'] == s])

TABLE4_SECTIONS = [
    ('[0. OVERALL ACCURACIES (%)]', [
        ('acc_main', 'acc_main (9-cls)'),
        ('acc_shape', 'acc_shape (3-cls)'),
        ('acc_scale', 'acc_scale (3-cls)'),
        ('fail_aux_ok', 'failed_atk_aux_ok'),
    ]),
    ('[1. FAILED ATK (Main OK)]', [
        ('fail_cm', 'conf_main'), ('fail_csh', 'conf_shape'), ('fail_csc', 'conf_scale'),
        ('fail_em', 'ent_main (norm%)'), ('fail_esh', 'ent_shape (norm%)'), ('fail_esc', 'ent_scale (norm%)'),
    ]),
    ('[2. FULL ALIGNED (Trunk)]', [
        ('ali_cm', 'conf_main'), ('ali_csh', 'conf_shape'), ('ali_csc', 'conf_scale'),
        ('ali_em', 'ent_main (norm%)'), ('ali_esh', 'ent_shape (norm%)'), ('ali_esc', 'ent_scale (norm%)'),
    ]),
    ('[3. FULL RETAINED (Short)]', [
        ('ret_cm', 'conf_main'), ('ret_csh', 'conf_shape'), ('ret_csc', 'conf_scale'),
        ('ret_em', 'ent_main (norm%)'), ('ret_esh', 'ent_shape (norm%)'), ('ret_esc', 'ent_scale (norm%)'),
    ]),
    ('[4. PARTIAL RETAINED (JS)]', [
        ('part_cm', 'conf_main'), ('part_c_resist', 'conf_resisting_aux'), ('part_c_fooled', 'conf_fooled_aux'),
        ('part_em', 'ent_main (norm%)'), ('part_e_resist', 'ent_resisting_aux'), ('part_e_fooled', 'ent_fooled_aux'),
    ]),
    ('[5. DIVERGED (Shattered)]', [
        ('div_cm', 'conf_main'), ('div_c_only', 'conf_diverged_aux'),
        ('div_em', 'ent_main (norm%)'), ('div_e_only', 'ent_diverged_aux'),
    ]),
]

t4_w = 151
print("\n" + "=" * t4_w)
print("=== TABLE 4: BRANCH ACCURACIES, SOFTMAX CONFIDENCE (%) & NORMALIZED ENTROPY (%) BY TREE ARCHETYPE ===")
print("=" * t4_w)
print("Archetype / Metric     | GLOBAL |    DATASET    |     POOL      |     LOSS WEIGHT      |    ATTACK     |     NORM      |       EPSILON STEP (1 -> 5)        |")
print("                       |    ALL |    std    con |   flat    gap |  w=0.5  w=1.0  w=2.0 |   fgsm    pgd |   linf     l2 |  eps_1  eps_2  eps_3  eps_4  eps_5 |")
print("-" * t4_w)

for sec_title, rows_spec in TABLE4_SECTIONS:
    print("-" * t4_w)
    print(f"{sec_title:<22} |" + " " * (t4_w - 25) + "|")
    for key, label in rows_spec:
        c = ce_cols
        line = (
            f"{label:<22} | {c['ALL'][key]:6.2f} | "
            f"{c['std'][key]:6.2f} {c['con'][key]:6.2f} | "
            f"{c['flat'][key]:6.2f} {c['gap'][key]:6.2f} | "
            f"{c['w=0.5'][key]:6.2f} {c['w=1.0'][key]:6.2f} {c['w=2.0'][key]:6.2f} | "
            f"{c['fgsm'][key]:6.2f} {c['pgd'][key]:6.2f} | "
            f"{c['linf'][key]:6.2f} {c['l2'][key]:6.2f} | "
            f"{c['eps_1'][key]:6.2f} {c['eps_2'][key]:6.2f} {c['eps_3'][key]:6.2f} {c['eps_4'][key]:6.2f} {c['eps_5'][key]:6.2f} |"
        )
        print(line)

print("=" * t4_w)

#%%
# =============================================================================
# INTERACTIVE SANKEY & PROPORTIONAL TREE DIAGRAM (With Statistical Significance)
# =============================================================================
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import ipywidgets as widgets
from IPython.display import display


def plot_interactive_tree(dataset, pool_type, loss_weight, attack, norm, eps_choice):
    if eps_choice == 'ALL (Summary)':
        sub = summary_tree.copy()
        n_per_row = N_SINGLE_EPS * 5
        eps_label = "All Epsilons"
    else:
        eps_idx = int(eps_choice.split()[0].replace('eps_', '')) - 1
        sub = (
            tree_df.sort_values('epsilon')
            .groupby(['dataset', 'pool', 'weight', 'attack', 'norm'], as_index=False)
            .nth(eps_idx)
            .reset_index(drop=True)
        )
        n_per_row = N_SINGLE_EPS
        eps_vals = sorted(sub['epsilon'].unique())
        eps_label = f"ε = {eps_vals[0]}" if len(eps_vals) == 1 else f"Step {eps_idx+1} ({'/'.join(str(e) for e in eps_vals)})"

    if dataset != 'ALL':
        sub = sub[sub['dataset'] == dataset]
    if pool_type != 'ALL':
        sub = sub[sub['pool'] == pool_type]
    if loss_weight != 'ALL':
        sub = sub[sub['weight'] == float(loss_weight)]
    if attack != 'ALL':
        sub = sub[sub['attack'] == attack]
    if norm != 'ALL':
        sub = sub[sub['norm'] == norm]

    if sub.empty:
        print("No data for this configuration.")
        return

    row = aggregate_tree_subset(sub, n_per_row)
    n_total = row['n_total']
    n_succ = row['n_succ']
    n_fail = n_total - n_succ
    n_sh, n_sc, n_jt = row['n_sh'], row['n_sc'], row['n_jt']

    sr = float(row['success_rate'])
    fr = round(100.0 - sr, 2)
    sh_shift = float(row['shape_shift'])
    sc_shift = float(row['scale_shift'])
    jt_shift = float(row['joint_shift'])

    # Cohen's w Effect Size for Node 1 vs Null H0 = (25%, 25%, 50%)
    if n_succ >= N_MIN_BRANCH:
        p_sh, p_sc, p_jt = sh_shift / 100.0, sc_shift / 100.0, jt_shift / 100.0
        cohen_w = float(np.sqrt(
            ((p_sh - 0.25) ** 2) / 0.25 +
            ((p_sc - 0.25) ** 2) / 0.25 +
            ((p_jt - 0.50) ** 2) / 0.50
        ))
        if cohen_w >= 0.50:
            w_badge, w_col = f"Cohen's w = {cohen_w:.2f} (LARGE Bias vs 25/25/50)", '#B71C1C'
        elif cohen_w >= 0.30:
            w_badge, w_col = f"Cohen's w = {cohen_w:.2f} (Medium Bias vs 25/25/50)", '#E65100'
        elif cohen_w >= 0.10:
            w_badge, w_col = f"Cohen's w = {cohen_w:.2f} (Small Bias vs 25/25/50)", '#1565C0'
        else:
            w_badge, w_col = f"Cohen's w = {cohen_w:.2f} (Follows 25/25/50 Null)", '#555555'
    else:
        cohen_w = 0.0
        w_badge, w_col = f"LOW N = {n_succ} (< {N_MIN_BRANCH}: Insufficient Sample)", '#757575'

    fig, ax = plt.subplots(figsize=(19, 12.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Title
    ax.text(
        50, 97.8,
        f"Sankey Probability Tree | Dataset:{dataset.upper()} • Pool:{pool_type.upper()} • "
        f"w:{loss_weight} • Attack:{attack.upper()} • Norm:{norm.upper()} • {eps_label} "
        f"[{len(sub)} cfg | N={n_total:,}]",
        ha='center', va='center', fontsize=13, fontweight='bold'
    )

    # Column Headers
    ax.text(6.5,  93.8, "NODE 0: Attack Outcome\n(100% of Test Set)", ha='center', va='center', fontsize=10, fontweight='bold', color='#333333')
    ax.text(25.5, 94.3, "NODE 1: Shift Typology\n(100% of Successful)", ha='center', va='center', fontsize=10, fontweight='bold', color='#333333')
    ax.text(25.5, 90.8, w_badge, ha='center', va='center', fontsize=8.5, fontweight='bold', color=w_col,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#FAFAFA', edgecolor=w_col, lw=1.0))
    ax.text(45.5, 93.8, "NODE 2: Aux Proportions\n(Stacked by Shift)", ha='center', va='center', fontsize=10, fontweight='bold', color='#333333')
    ax.text(76.0, 93.8, "NODE 2 DETAILS: <ShapeAux>_<ScaleAux>\n(★★ Primary > 1/K Null  |  ★ Secondary > Residual Null  |  Faded = Noise)", ha='center', va='center', fontsize=9.2, fontweight='bold', color='#333333')

    # Helper: Draw smooth cubic-Bezier Sankey ribbon
    def draw_sankey_ribbon(x0, y0_bot, y0_top, x1, y1_bot, y1_top, facecolor, alpha=0.45):
        if (y0_top - y0_bot) <= 0.01 and (y1_top - y1_bot) <= 0.01:
            return
        xm = 0.5 * (x0 + x1)
        verts = [
            (x0, y0_top),
            (xm, y0_top), (xm, y1_top), (x1, y1_top),
            (x1, y1_bot),
            (xm, y1_bot), (xm, y0_bot), (x0, y0_bot),
            (x0, y0_top),
        ]
        codes = [
            Path.MOVETO,
            Path.CURVE4, Path.CURVE4, Path.CURVE4,
            Path.LINETO,
            Path.CURVE4, Path.CURVE4, Path.CURVE4,
            Path.CLOSEPOLY,
        ]
        patch = mpatches.PathPatch(Path(verts, codes), facecolor=facecolor, edgecolor='none', alpha=alpha)
        ax.add_patch(patch)

    # Color semantics for Node 2 states
    def get_leaf_style(short_name, shift_type):
        if (shift_type == 'ss' and short_name == 'aligned_retained') or \
           (shift_type == 'sl' and short_name == 'retained_aligned') or \
           (shift_type == 'js' and short_name == 'aligned_aligned'):
            return '#81C784', '#2E7D32'  # Green: Full Alignment w/ Main
        if short_name == 'retained_retained':
            return '#64B5F6', '#1565C0'  # Blue: Both Retained Original Truth
        if 'diverged' in short_name:
            return '#E57373', '#C62828'  # Red: Diverged
        return '#FFD54F', '#F57F17'      # Yellow/Orange: Partial Alignment

    # =========================================================================
    # BAR 0: Node 0 Vertical Proportional Bar (x = 4..9, y = 12..88, H = 76)
    # =========================================================================
    x0_l, x0_r = 4.0, 9.0
    y_bot_total, y_top_total = 12.0, 88.0
    H0 = y_top_total - y_bot_total

    h_succ = H0 * (sr / 100.0)
    h_fail = H0 - h_succ

    y0_succ_bot, y0_succ_top = y_bot_total, y_bot_total + h_succ
    y0_fail_bot, y0_fail_top = y0_succ_top, y_top_total

    # Failed segment (Top)
    if h_fail > 0:
        ax.add_patch(mpatches.Rectangle((x0_l, y0_fail_bot), x0_r - x0_l, h_fail, facecolor='#E0E0E0', edgecolor='#757575', lw=1.5))
        if h_fail >= 6.0:
            ax.text((x0_l + x0_r) / 2, (y0_fail_bot + y0_fail_top) / 2, f"Failed\n{fr:.1f}%\n(N={n_fail:,})", ha='center', va='center', fontsize=8.5, color='#333333', fontweight='bold')
        else:
            ax.text(x0_l - 0.5, (y0_fail_bot + y0_fail_top) / 2, f"Fail {fr:.1f}% (N={n_fail:,})", ha='right', va='center', fontsize=8, color='#555555')

    # Succeeded segment (Bottom)
    if h_succ > 0:
        ax.add_patch(mpatches.Rectangle((x0_l, y0_succ_bot), x0_r - x0_l, h_succ, facecolor='#FFB74D', edgecolor='#E65100', lw=1.5))
        if h_succ >= 6.0:
            ax.text((x0_l + x0_r) / 2, (y0_succ_bot + y0_succ_top) / 2, f"Success\n{sr:.1f}%\n(N={n_succ:,})", ha='center', va='center', fontsize=9, color='#111111', fontweight='bold')
        else:
            ax.text(x0_l - 0.5, (y0_succ_bot + y0_succ_top) / 2, f"Succ {sr:.1f}% (N={n_succ:,})", ha='right', va='center', fontsize=8, color='#E65100', fontweight='bold')

    # =========================================================================
    # BAR 1: Node 1 Vertical Proportional Bar (x = 23..28, y = 12..88)
    # =========================================================================
    x1_l, x1_r = 23.0, 28.0
    gap = 3.0
    H1_usable = H0 - 2 * gap

    h_jt = H1_usable * (jt_shift / 100.0)
    h_sc = H1_usable * (sc_shift / 100.0)
    h_sh = H1_usable * (sh_shift / 100.0)

    y1_jt_bot, y1_jt_top = y_bot_total, y_bot_total + h_jt
    y1_sc_bot, y1_sc_top = y1_jt_top + gap, y1_jt_top + gap + h_sc
    y1_sh_bot, y1_sh_top = y1_sc_top + gap, y1_sc_top + gap + h_sh

    s0_jt_bot = y0_succ_bot
    s0_jt_top = s0_jt_bot + h_succ * (jt_shift / 100.0)
    s0_sc_bot = s0_jt_top
    s0_sc_top = s0_sc_bot + h_succ * (sc_shift / 100.0)
    s0_sh_bot = s0_sc_top
    s0_sh_top = y0_succ_top

    n1_blocks = [
        ('Shape-Shift', sh_shift, n_sh, s0_sh_bot, s0_sh_top, y1_sh_bot, y1_sh_top, '#90CAF9', '#1565C0', SS_KEYS, 'ss', 88.5, 67.5),
        ('Scale-Shift', sc_shift, n_sc, s0_sc_bot, s0_sc_top, y1_sc_bot, y1_sc_top, '#FFE082', '#EF6C00', SL_KEYS, 'sl', 60.5, 39.5),
        ('Joint-Shift', jt_shift, n_jt, s0_jt_bot, s0_jt_top, y1_jt_bot, y1_jt_top, '#CE93D8', '#6A1B9A', JS_KEYS, 'js', 32.5,  3.5),
    ]

    x2_l, x2_r = 43.0, 48.0
    bar_x, bar_w, bar_h = 57.0, 38.0, 2.25

    for label, pct, n_br, s0_b, s0_t, y1_b, y1_t, fcol, ecol, keys, stype, leaf_top, leaf_bot in n1_blocks:
        k_states = len(keys)
        null_pct = 100.0 / k_states
        low_n = n_br < N_MIN_BRANCH

        # 1) Draw Sankey ribbon: Node 0 (Success) -> Node 1 (Shift block)
        if sr > 0 and pct > 0:
            draw_sankey_ribbon(x0_r, s0_b, s0_t, x1_l, y1_b, y1_t, facecolor=fcol, alpha=0.5 if not low_n else 0.2)

        # 2) Draw Node 1 vertical block
        block_h = max(0.4, y1_t - y1_b)
        ax.add_patch(mpatches.Rectangle(
            (x1_l, y1_b), x1_r - x1_l, block_h,
            facecolor=fcol, edgecolor=ecol, lw=1.5, alpha=0.95 if not low_n else 0.4
        ))

        ymid = (y1_b + y1_t) / 2
        n_tag = f"N={n_br:,}" if not low_n else f"LOW N={n_br}"
        if block_h >= 7.0:
            ax.text((x1_l + x1_r) / 2, ymid, f"{label.split('-')[0]}\n{pct:.1f}%\n({n_tag})", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#111111')
        else:
            ax.text(x1_l - 0.6, ymid, f"{label.split('-')[0]}: {pct:.1f}% ({n_tag})", ha='right', va='center', fontsize=8, fontweight='bold', color=ecol if not low_n else '#757575')

        # 3) Draw BAR 2: Node 2 Stacked Vertical Proportions
        g_bot, g_top = leaf_bot - 1.0, leaf_top + 1.0
        g_h = g_top - g_bot

        if pct > 0:
            draw_sankey_ribbon(x1_r, y1_b, y1_t, x2_l, g_bot, g_top, facecolor=fcol, alpha=0.25 if not low_n else 0.1)

        cur_y2_top = g_top
        leaf_ys = np.linspace(leaf_top, leaf_bot, k_states)

        # Draw 1/K Uniform Null Threshold vertical dashed line across the Node 2 detail bars
        null_x = bar_x + (null_pct / 100.0) * bar_w
        ax.plot([null_x, null_x], [g_bot - 0.3, g_top + 0.3], linestyle='--', color='#444444', lw=1.2, alpha=0.75, zorder=5)
        ax.text(null_x, g_top + 0.7, f"1/{k_states} Null ({null_pct:.1f}%)", ha='center', va='bottom', fontsize=7.5, color='#444444', fontweight='bold')

        if low_n:
            ax.text(bar_x + bar_w / 2, g_top + 0.7, f"⚠ LOW SAMPLE (N={n_br} < {N_MIN_BRANCH}): Ignore %", ha='center', va='bottom', fontsize=8, color='#C62828', fontweight='bold')

        vals_dict = {k: float(row[k]) for k in keys}
        tiers = classify_group_significance(vals_dict, n_br)

        for k, y_leaf in zip(keys, leaf_ys):
            val = vals_dict[k]
            short_name = k[3:]
            lf_fcol, lf_ecol = get_leaf_style(short_name, stype)
            tier = tiers[k]  # 'primary', 'secondary', or 'noise'
            sig = tier in ('primary', 'secondary')
            bar_alpha = 0.95 if tier == 'primary' else (0.78 if tier == 'secondary' else 0.25)
            rib_alpha = 0.50 if tier == 'primary' else (0.35 if tier == 'secondary' else 0.10)

            seg_h2 = g_h * (val / 100.0)
            seg_y2_bot = cur_y2_top - seg_h2

            if seg_h2 > 0.05 and pct > 0:
                ax.add_patch(mpatches.Rectangle(
                    (x2_l, seg_y2_bot), x2_r - x2_l, seg_h2,
                    facecolor=lf_fcol, edgecolor=lf_ecol, lw=0.8,
                    alpha=bar_alpha
                ))
                draw_sankey_ribbon(
                    x2_r, seg_y2_bot, cur_y2_top, bar_x, y_leaf - 1.05, y_leaf + 1.05,
                    facecolor=lf_fcol, alpha=rib_alpha
                )

            cur_y2_top = seg_y2_bot

            # 4) Right-side Horizontal Leaf Bar
            ax.add_patch(mpatches.FancyBboxPatch(
                (bar_x, y_leaf - bar_h / 2), bar_w, bar_h,
                boxstyle="round,pad=0.05,rounding_size=0.25",
                facecolor='#F5F5F5', edgecolor='#DDDDDD', lw=0.8
            ))
            if val > 0:
                fill_w = max(0.4, (val / 100.0) * bar_w)
                ax.add_patch(mpatches.FancyBboxPatch(
                    (bar_x, y_leaf - bar_h / 2), fill_w, bar_h,
                    boxstyle="round,pad=0.05,rounding_size=0.25",
                    facecolor=lf_fcol, edgecolor=lf_ecol,
                    lw=1.3 if sig else 0.7,
                    alpha=bar_alpha
                ))

            txt_col = '#111111' if sig else '#777777'
            star_prefix = "★★ " if tier == 'primary' else ("★  " if tier == 'secondary' else "   ")
            ax.text(
                bar_x + 0.6, y_leaf, f"{star_prefix}{short_name}",
                ha='left', va='center', fontsize=8.8,
                fontweight='bold' if sig else 'normal', color=txt_col
            )
            ax.text(
                bar_x + bar_w - 0.8, y_leaf, f"{val:6.2f}%",
                ha='right', va='center', fontsize=8.8,
                fontweight='bold' if sig else 'normal', color=txt_col
            )

        # Outline around Bar 2 group
        ax.add_patch(mpatches.Rectangle((x2_l, g_bot), x2_r - x2_l, g_h, facecolor='none', edgecolor=ecol, lw=1.4, alpha=0.9 if not low_n else 0.35))

    # Legend at bottom-left
    legend_patches = [
        mpatches.Patch(facecolor='#81C784', edgecolor='#2E7D32', label='Full Alignment w/ Main (Core Features Broken)'),
        mpatches.Patch(facecolor='#64B5F6', edgecolor='#1565C0', label='Both Retained Truth (Pure Main-Head Shortcut)'),
        mpatches.Patch(facecolor='#FFD54F', edgecolor='#F57F17', label='Partial Alignment (1 Aligned, 1 Retained)'),
        mpatches.Patch(facecolor='#E57373', edgecolor='#C62828', label='Diverged (Conflicting / Shattered Shortcuts)'),
        mpatches.Patch(facecolor='#EEEEEE', edgecolor='#999999', label='★★ Primary (> 1/K)  |  ★ Secondary (> Residual)  |  Faded = Noise'),
    ]
    ax.legend(
        handles=legend_patches, loc='lower left', bbox_to_anchor=(0.01, 0.005),
        frameon=True, facecolor='white', edgecolor='#CCCCCC', fontsize=9,
        title="Node 2 Semantic & Statistical Legend", title_fontsize=9.5
    )

    plt.tight_layout()
    plt.show()


tree_ui = widgets.HBox([
    widgets.Dropdown(options=['ALL', 'standard', 'contrast'], value='ALL', description='Dataset:', layout=widgets.Layout(width='170px')),
    widgets.Dropdown(options=['ALL', 'flatten', 'gap'], value='ALL', description='Pooling:', layout=widgets.Layout(width='160px')),
    widgets.Dropdown(options=['ALL', 0.5, 1.0, 2.0], value='ALL', description='Weight:', layout=widgets.Layout(width='150px')),
    widgets.Dropdown(options=['ALL', 'fgsm', 'pgd'], value='ALL', description='Attack:', layout=widgets.Layout(width='150px')),
    widgets.Dropdown(options=['ALL', 'linf', 'l2'], value='ALL', description='Norm:', layout=widgets.Layout(width='140px')),
    widgets.Dropdown(
        options=[
            'ALL (Summary)',
            'eps_1 (0.01 / 0.1)',
            'eps_2 (0.05 / 0.5)',
            'eps_3 (0.10 / 1.0)',
            'eps_4 (0.15 / 2.0)',
            'eps_5 (0.20 / 3.0)',
        ],
        value='ALL (Summary)',
        description='Epsilon:',
        layout=widgets.Layout(width='220px')
    ),
])

tree_out = widgets.interactive_output(
    plot_interactive_tree,
    {
        'dataset': tree_ui.children[0],
        'pool_type': tree_ui.children[1],
        'loss_weight': tree_ui.children[2],
        'attack': tree_ui.children[3],
        'norm': tree_ui.children[4],
        'eps_choice': tree_ui.children[5],
    }
)

display(tree_ui, tree_out)
# %%




#%%
#%%
# Visualize Original vs Adversarial Images (Interactive)
from ipywidgets import interact_manual, Dropdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from shared.paths import (
    MY_DSPRITES_SPLIT, 
    get_untargeted_attack_path, 
    get_attack_eval_paths,
)

# Load original images once (outside the widget function)
orig_split = np.load(MY_DSPRITES_SPLIT)
X_orig = orig_split['X_test']

epsilons_linf = [0.01, 0.05, 0.1, 0.15, 0.2]
epsilons_l2 = [0.1, 0.5, 1.0, 2.0, 3.0]

@interact_manual(
    dataset=Dropdown(options=['standard', 'contrast'], value='standard', description='Dataset'),
    pool=Dropdown(options=['flatten', 'gap'], value='flatten', description='Pool'),
    weight=Dropdown(options=[('0.5', 0.5), ('1.0', 1.0), ('2.0', 2.0)], value=1.0, description='Weight'),
    attack=Dropdown(options=['fgsm', 'pgd'], value='pgd', description='Attack'),
    norm=Dropdown(options=['linf', 'l2'], value='linf', description='Norm'),
    epsilon=Dropdown(options=[0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0], value=0.1, description='Epsilon'),
)
def visualize_attacks(dataset, pool, weight, attack, norm, epsilon):
    # Validate epsilon matches norm
    if norm == 'linf' and epsilon not in epsilons_linf:
        print(f"⚠️ Epsilon {epsilon} is not valid for L∞. Use one of: {epsilons_linf}")
        return
    if norm == 'l2' and epsilon not in epsilons_l2:
        print(f"⚠️ Epsilon {epsilon} is not valid for L2. Use one of: {epsilons_l2}")
        return
    
    # Load adversarial images
    adv_path = get_untargeted_attack_path(dataset, pool, attack, norm, epsilon, weight)
    if not os.path.exists(adv_path):
        print(f"❌ File not found:\n{adv_path}")
        return
    adv_data = np.load(adv_path)
    X_adv = adv_data['X_test']
    
    # Load predictions
    csv_path = get_attack_eval_paths(dataset, pool, attack, norm, weight)
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found:\n{csv_path}")
        return
    df = pd.read_csv(csv_path, compression='gzip')
    
    # Filter for this epsilon and only successful attacks
    df_eps = df[df['epsilon'] == epsilon].copy()
    successful_attacks = df_eps[df_eps['pred_main'] != df_eps['y_shape_and_scale']].copy()
    
    if len(successful_attacks) == 0:
        print("No successful attacks for this configuration/epsilon.")
        return
    
    # Decode shape/scale
    successful_attacks['true_shape'] = successful_attacks['y_shape_and_scale'] // 3
    successful_attacks['true_scale'] = successful_attacks['y_shape_and_scale'] % 3
    successful_attacks['fooled_shape'] = successful_attacks['pred_main'] // 3
    successful_attacks['fooled_scale'] = successful_attacks['pred_main'] % 3
    
    def get_shift_type(row):
        if row['true_shape'] != row['fooled_shape'] and row['true_scale'] == row['fooled_scale']:
            return "Shape-Shift"
        elif row['true_shape'] == row['fooled_shape'] and row['true_scale'] != row['fooled_scale']:
            return "Scale-Shift"
        elif row['true_shape'] != row['fooled_shape'] and row['true_scale'] != row['fooled_scale']:
            return "Joint-Shift"
        return "No-Shift"
    
    successful_attacks['shift_type'] = successful_attacks.apply(get_shift_type, axis=1)
    
    # Select samples: 1 of each shift type, plus 2 extra Joint-Shift
    samples_to_plot = []
    for shift in ['Shape-Shift', 'Scale-Shift', 'Joint-Shift']:
        match = successful_attacks[successful_attacks['shift_type'] == shift]
        if len(match) > 0:
            samples_to_plot.append(match.iloc[0]['sample_index'])
    
    joint_matches = successful_attacks[successful_attacks['shift_type'] == 'Joint-Shift']
    if len(joint_matches) > 1:
        samples_to_plot.append(joint_matches.iloc[1]['sample_index'])
    if len(joint_matches) > 2:
        samples_to_plot.append(joint_matches.iloc[2]['sample_index'])
    
    if len(samples_to_plot) == 0:
        print("No samples to plot.")
        return
    
    print(f"Config: {dataset} {pool} w={weight} {attack} {norm} eps={epsilon}")
    print(f"Successful attacks: {len(successful_attacks)} / {len(df_eps)}")
    print(f"Plotting {len(samples_to_plot)} samples...")
    
    n_samples = len(samples_to_plot)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample_idx in enumerate(samples_to_plot):
        row = successful_attacks[successful_attacks['sample_index'] == sample_idx].iloc[0]
        
        orig_img = X_orig[int(sample_idx)]
        adv_img = X_adv[int(sample_idx)]
        
        perturbation = adv_img.astype(np.float32) - orig_img.astype(np.float32)
        perturbation_vis = np.clip(np.abs(perturbation) * 5, 0, 1)
        
        axes[i, 0].imshow(orig_img, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f"ORIGINAL\nTrue: S{int(row['true_shape'])} Sc{int(row['true_scale'])} (Class {int(row['y_shape_and_scale'])})")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(adv_img, cmap='gray', vmin=0, vmax=1)
        aux_status = f"Main: {int(row['pred_main'])}\nShape Aux: {int(row['pred_shape'])}\nScale Aux: {int(row['pred_scale'])}"
        axes[i, 1].set_title(f"ADVERSARIAL\nFooled: S{int(row['fooled_shape'])} Sc{int(row['fooled_scale'])}\n({row['shift_type']})\n{aux_status}")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(perturbation_vis, cmap='hot', vmin=0, vmax=1)
        axes[i, 2].set_title("PERTURBATION (x5)")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
# %%