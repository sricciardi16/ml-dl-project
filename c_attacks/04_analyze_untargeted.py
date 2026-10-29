# %% Cell 1 — Setup, Notation Constants, Statistical Helpers
import os, sys, json, random, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
import matplotlib.colors as mcolors

# --- Determinism (matches new_anal2's lineage) ---
os.environ['PYTHONHASHSEED'] = '29'
random.seed(29)
np.random.seed(29)

# --- Resolve project root so `shared.*` imports work in Jupyter or script mode ---
try:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:  # interactive / cell execution
    PROJECT_ROOT = os.path.abspath('..')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared.paths import (
    ANALYSES_DIR, UNTARGETED_ATTACKS_REPORT, MY_DSPRITES_SPLIT,
    get_eval_paths, get_untargeted_attack_path, get_attack_eval_paths,
)
from shared.report_generator import ReportGenerator

os.makedirs(ANALYSES_DIR, exist_ok=True)
if os.path.exists(UNTARGETED_ATTACKS_REPORT):
    os.remove(UNTARGETED_ATTACKS_REPORT)
report = ReportGenerator(UNTARGETED_ATTACKS_REPORT)

# --- Model grid M = (d, pi, w) ---
DATASETS    = ['standard', 'contrast']
POOL_TYPES  = ['flatten', 'gap']
LOSS_WEIGHTS = [0.5, 1.0, 2.0]
DS_SHORT   = {'standard': 'std', 'contrast': 'con'}
POOL_SHORT = {'flatten': 'flat', 'gap': 'gap'}

MODELS = [  # ordered 12: std|flat|w0.5 ... con|gap|w2.0
    (d, p, w) for d in DATASETS for p in POOL_TYPES for w in LOSS_WEIGHTS
]
MODEL_LABELS = [f"{DS_SHORT[d]}|{POOL_SHORT[p]}|w={w:g}" for (d, p, w) in MODELS]

# --- Attack grid A = (alpha, p, epsilon) ---
ATTACKS   = ['fgsm', 'pgd']
NORMS     = ['linf', 'l2']
EPS_STEPS = [1, 2, 3, 4, 5]
EPS_LINF  = {1: 0.01, 2: 0.05, 3: 0.10, 4: 0.15, 5: 0.20}
EPS_L2    = {1: 0.10, 2: 0.50, 3: 1.00, 4: 2.00, 5: 3.00}
ATTACK_NORMS = [(a, n) for a in ATTACKS for n in NORMS]  # 4 attack-norm pairs

# --- Inference / statistics constants ---
N_TEST       = 36864
N_MIN_BRANCH = 100
Z_95         = 1.959963984540054
LN9          = float(np.log(9.0))
LN3          = float(np.log(3.0))

CSV_COLS = [  # note: y_shape/y_scale exist in the CSV but we re-derive them for parity with new_anal2
    'epsilon', 'y_shape_and_scale',
    'pred_main', 'confidence', 'entropy',
    'pred_shape', 'confidence_shape', 'entropy_shape',
    'pred_scale', 'confidence_scale', 'entropy_scale',
]

# --- Shift types Sigma and auxiliary reaction state spaces ---
SHIFT_TYPES = ['shape', 'scale', 'joint']
K_SHAPE = ['AR', 'AD', 'RR', 'RD', 'DR', 'DD']                       # |K| = 6
K_SCALE = ['RA', 'RR', 'RD', 'DA', 'DR', 'DD']                       # |K| = 6
K_JOINT = ['AA', 'AR', 'AD', 'RA', 'RR', 'RD', 'DA', 'DR', 'DD']     # |K| = 9
K_SPACES = {'shape': K_SHAPE, 'scale': K_SCALE, 'joint': K_JOINT}
# Tail state (both auxiliaries diverged) per shift type:
K_DD = {'shape': 'DD', 'scale': 'DD', 'joint': 'DD'}
# Aligned state (both follow main) per shift type:
K_ALIGNED = {'shape': 'AR', 'scale': 'RA', 'joint': 'AA'}

# --- Label dictionaries ---
SHAPE_NAMES = {0: 'Square', 1: 'Ellipse', 2: 'Heart'}
SCALE_NAMES = {0: 'Small', 1: 'Medium', 2: 'Large'}
CLASS_NAMES_9 = {i: f'[{i}] {SHAPE_NAMES[i // 3]}-{SCALE_NAMES[i % 3]}' for i in range(9)}

# --- Visual styling: single-branch states, composite regimes, model families ---
STATE_COLORS  = {'A': '#2ca02c', 'R': '#1f77b4', 'D': '#d62728'}
REGIME_COLORS = {'FAR': '#1a9850', 'FRR': '#762a83', 'PRR': '#f4a736', 'DR': '#d73027'}
REGIME_ORDER  = ['FAR', 'FRR', 'PRR', 'DR']
FAMILY_COLORS = {
    'std|flat': '#ff7f0e', 'std|gap': '#d62728',
    'con|flat': '#2ca02c', 'con|gap': '#1f77b4',
}
FAMILY_MARKERS = {'std|flat': 's', 'std|gap': '^', 'con|flat': 'D', 'con|gap': 'v'}
# %%

# %%

def save_and_show(fig, title):
    """Embed in the markdown report via Kroki, render inline, then close."""
    report.kroki(fig, title=title)
    plt.show()
    plt.close(fig)


def wilson_ci(p_pct, n, z=Z_95):
    """95% Wilson score CI for a percentage p_pct in [0,100]; returns (lo, hi) in [0,100]."""
    if n <= 0:
        return (np.nan, np.nan)
    p = p_pct / 100.0
    d = 1.0 + (z ** 2 / n)
    c = (p + (z ** 2 / (2.0 * n))) / d
    h = (z * np.sqrt((p * (1.0 - p)) / n + (z ** 2 / (4.0 * n ** 2)))) / d
    return (max(0.0, c - h) * 100.0, min(1.0, c + h) * 100.0)


def compute_cohen_w(n_sh, n_sc, n_jt, min_n=N_MIN_BRANCH):
    """Cohen's w vs the (0.25, 0.25, 0.50) random-error reference; NaN below min_n."""
    n_succ = n_sh + n_sc + n_jt
    if n_succ < min_n or n_succ == 0:
        return np.nan
    p_sh, p_sc, p_jt = n_sh / n_succ, n_sc / n_succ, n_jt / n_succ
    return float(np.sqrt(
        ((p_sh - 0.25) ** 2) / 0.25
        + ((p_sc - 0.25) ** 2) / 0.25
        + ((p_jt - 0.50) ** 2) / 0.50
    ))


def kl_divergence(p, q, eps=1e-12):
    """Bernoulli KL divergence D_KL(p || q) for proportions p, q in [0,1]."""
    q = min(max(q, eps), 1.0 - eps)
    if p <= 0.0:
        return float((1.0 - p) * np.log((1.0 - p) / (1.0 - q)))
    if p >= 1.0:
        return float(p * np.log(p / q))
    return float(p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q)))


def branch_reaction_code(pred_aux, pred_main_factor, y_true_factor):
    """Vectorised o_k(x) in {'A','R','D'}:
       'A' if aux == main-implied != truth; 'R' if aux == truth; else 'D'."""
    return np.where(
        (pred_aux == pred_main_factor) & (pred_main_factor != y_true_factor),
        'A',
        np.where(pred_aux == y_true_factor, 'R', 'D'),
    )


def classify_wilson_dominance(vals_pct_dict, n_branch):
    """Two-tier Wilson test vs uniform chance 1/|K_s|.
       Returns {state: 'primary'|'secondary'|'noise'|'low_n'}."""
    if n_branch < N_MIN_BRANCH:
        return {k: 'low_n' for k in vals_pct_dict}
    k_states = len(vals_pct_dict)
    null_p = 1.0 / k_states
    tiers, primary_mass, n_primary = {}, 0.0, 0
    for k, val_pct in vals_pct_dict.items():
        p = val_pct / 100.0
        ci_lo, _ = wilson_ci(val_pct, n_branch)
        if ci_lo / 100.0 > null_p:
            tiers[k] = 'primary'; primary_mass += p; n_primary += 1
        else:
            tiers[k] = 'noise'
    k_rem = k_states - n_primary
    rem_mass = max(0.0, 1.0 - primary_mass)
    if k_rem > 0 and rem_mass > 0.05:
        res_null_p = rem_mass / k_rem
        for k, val_pct in vals_pct_dict.items():
            if tiers[k] == 'noise' and val_pct >= 10.0:
                ci_lo, _ = wilson_ci(val_pct, n_branch)
                if ci_lo / 100.0 > res_null_p:
                    tiers[k] = 'secondary'
    return tiers
# %%


# %% Cell 1 (continued) — Single-Pass Data Loader: 240 slices + per-image arrays
# Discover the 48 evaluation CSVs (12 models x 4 attack-norm pairs).
configs = []
for d in DATASETS:
    for p in POOL_TYPES:
        for w in LOSS_WEIGHTS:
            for a in ATTACKS:
                for n in NORMS:
                    csv_path = get_attack_eval_paths(d, p, a, n, w)
                    if os.path.exists(csv_path):
                        configs.append(dict(dataset=d, pool=p, weight=w,
                                            attack=a, norm=n, csv_path=csv_path))

# Per-image stores: key = (archetype, role) -> list of float32 arrays
per_image_H   = {}   # normalized entropy  H~_t(x) in [0,100]
per_image_C   = {}   # confidence         Conf_t(x) in [0,100]

def _collect(store, key, arr):
    store.setdefault(key, []).append(np.asarray(arr, dtype=np.float32))

tree_rows = []
for cfg in configs:
    df_raw = pd.read_csv(cfg['csv_path'], compression='gzip', usecols=CSV_COLS)
    for step_idx, eps_val in enumerate(sorted(df_raw['epsilon'].unique()), start=1):
        sub = df_raw[df_raw['epsilon'] == eps_val]
        n_tot = int(len(sub))

        # --- Ground truth and predictions ---
        y_m  = sub['y_shape_and_scale'].to_numpy()
        y_sh, y_sc = y_m // 3, y_m % 3

        p_m  = sub['pred_main'].to_numpy()
        p_sh = sub['pred_shape'].to_numpy()
        p_sc = sub['pred_scale'].to_numpy()
        p_m_sh, p_m_sc = p_m // 3, p_m % 3          # main-implied factor predictions

        # --- Confidence (x100) and normalized entropy (raw nats -> /ln K -> x100) ---
        c_m  = sub['confidence'].to_numpy(dtype=float) * 100.0
        c_sh = sub['confidence_shape'].to_numpy(dtype=float) * 100.0
        c_sc = sub['confidence_scale'].to_numpy(dtype=float) * 100.0
        e_m  = (sub['entropy'].to_numpy(dtype=float) / LN9) * 100.0
        e_sh = (sub['entropy_shape'].to_numpy(dtype=float) / LN3) * 100.0
        e_sc = (sub['entropy_scale'].to_numpy(dtype=float) / LN3) * 100.0

        # --- Node 0: S (success) vs S^c (failed) ---
        m_succ = p_m != y_m
        m_fail = ~m_succ
        n_succ, n_fail = int(m_succ.sum()), int(m_fail.sum())

        sh_ok, sc_ok = p_sh == y_sh, p_sc == y_sc
        n_fail_sh_ok  = int((m_fail & sh_ok).sum())
        n_fail_sc_ok  = int((m_fail & sc_ok).sum())
        n_fail_aux_ok = int((m_fail & sh_ok & sc_ok).sum())

        # --- Node 1: shift subtypes S_shape, S_scale, S_joint ---
        sh_err = p_m_sh != y_sh
        sc_err = p_m_sc != y_sc
        m_sh = m_succ & sh_err & (~sc_err)
        m_sc = m_succ & (~sh_err) & sc_err
        m_jt = m_succ & sh_err & sc_err
        n_sh, n_sc, n_jt = int(m_sh.sum()), int(m_sc.sum()), int(m_jt.sum())

        # --- Cohen's w (pooled slice) ---
        cohen_w = compute_cohen_w(n_sh, n_sc, n_jt)

        # --- Node 2: single-branch reaction codes o_shape, o_scale and pair c ---
        o_sh = branch_reaction_code(p_sh, p_m_sh, y_sh)
        o_sc = branch_reaction_code(p_sc, p_m_sc, y_sc)
        c_pair = np.char.add(o_sh, o_sc)

        row = dict(
            dataset=cfg['dataset'], pool=cfg['pool'], weight=float(cfg['weight']),
            attack=cfg['attack'], norm=cfg['norm'],
            epsilon=float(eps_val), eps_step=step_idx,
            n_total=n_tot, n_succ=n_succ, n_fail=n_fail,
            n_fail_sh_ok=n_fail_sh_ok, n_fail_sc_ok=n_fail_sc_ok,
            n_fail_aux_ok=n_fail_aux_ok,
            n_sh=n_sh, n_sc=n_sc, n_jt=n_jt,
            ASR=(n_succ / n_tot * 100.0) if n_tot else 0.0,
            SR_shape=(n_sh / n_succ * 100.0) if n_succ else 0.0,
            SR_scale=(n_sc / n_succ * 100.0) if n_succ else 0.0,
            SR_joint=(n_jt / n_succ * 100.0) if n_succ else 0.0,
            SR_single=((n_sh + n_sc) / n_succ * 100.0) if n_succ else 0.0,
            cohen_w=cohen_w,
        )

        # --- Per-shift state counts and ARR ---
        for s_name, s_mask, s_cnt, k_space in [
            ('shape', m_sh, n_sh, K_SHAPE),
            ('scale', m_sc, n_sc, K_SCALE),
            ('joint', m_jt, n_jt, K_JOINT),
        ]:
            s_pairs = c_pair[s_mask]
            for code in k_space:
                cnt = int((s_pairs == code).sum())
                row[f'n_{s_name}_{code}']  = cnt
                row[f'ARR_{s_name}_{code}'] = (cnt / s_cnt * 100.0) if s_cnt else 0.0

        # --- The 5 archetype masks ---
        sh_ali, sc_ali = o_sh == 'A', o_sc == 'A'
        sh_div, sc_div = o_sh == 'D', o_sc == 'D'

        arch_fail = m_fail
        arch_far  = (m_sh & sh_ali & sc_ok) | (m_sc & sh_ok & sc_ali) | (m_jt & sh_ali & sc_ali)
        arch_frr  = m_succ & sh_ok & sc_ok
        js_sh_res = m_jt & sh_ok & sc_ali        # joint shift: shape retained, scale aligned(fooled)
        js_sc_res = m_jt & sh_ali & sc_ok        # joint shift: scale retained, shape aligned(fooled)
        arch_prr  = js_sh_res | js_sc_res
        arch_dr   = m_succ & (sh_div | sc_div)

        # Archetype counts + confidence/entropy SUMS (for pooled weighted means)
        for name, mask in [('fail', arch_fail), ('far', arch_far), ('frr', arch_frr),
                           ('prr', arch_prr), ('dr', arch_dr)]:
            cnt = int(mask.sum())
            row[f'n_arch_{name}']  = cnt
            row[f'sum_{name}_cm']  = float(c_m[mask].sum())
            row[f'sum_{name}_csh'] = float(c_sh[mask].sum())
            row[f'sum_{name}_csc'] = float(c_sc[mask].sum())
            row[f'sum_{name}_em']  = float(e_m[mask].sum())
            row[f'sum_{name}_esh'] = float(e_sh[mask].sum())
            row[f'sum_{name}_esc'] = float(e_sc[mask].sum())

        # PRR role splits: resisting vs fooled auxiliary
        row['sum_prr_c_resist'] = float(c_sh[js_sh_res].sum() + c_sc[js_sc_res].sum())
        row['sum_prr_c_fooled'] = float(c_sc[js_sh_res].sum() + c_sh[js_sc_res].sum())
        row['sum_prr_e_resist'] = float(e_sh[js_sh_res].sum() + e_sc[js_sc_res].sum())
        row['sum_prr_e_fooled'] = float(e_sc[js_sh_res].sum() + e_sh[js_sc_res].sum())

        # DR role split: divergent head(s) only
        sh_div_succ, sc_div_succ = m_succ & sh_div, m_succ & sc_div
        row['n_div_heads']   = int(sh_div_succ.sum() + sc_div_succ.sum())
        row['sum_dr_c_only'] = float(c_sh[sh_div_succ].sum() + c_sc[sc_div_succ].sum())
        row['sum_dr_e_only'] = float(e_sh[sh_div_succ].sum() + e_sc[sc_div_succ].sum())

        # --- Sink-class collapse statistics ---
        m_cnts = np.bincount(p_m, minlength=9)
        order_m = np.argsort(m_cnts)[::-1]
        c1, c2 = int(order_m[0]), int(order_m[1])
        sh_cnts, sc_cnts = np.bincount(p_sh, minlength=3), np.bincount(p_sc, minlength=3)
        sh_top, sc_top = int(np.argmax(sh_cnts)), int(np.argmax(sc_cnts))
        row.update(
            main_top1_cls=c1, main_top1_pct=float(m_cnts[c1] / n_tot * 100.0),
            main_top2_cls=c2, main_top2_pct=float(m_cnts[c2] / n_tot * 100.0),
            aux_sh_top_cls=sh_top, aux_sh_top_pct=float(sh_cnts[sh_top] / n_tot * 100.0),
            aux_sc_top_cls=sc_top, aux_sc_top_pct=float(sc_cnts[sc_top] / n_tot * 100.0),
            sh_match_sink_pct=float((p_sh == c1 // 3).mean() * 100.0),   # shape aux agreeing with sink shape
            sc_match_sink_pct=float((p_sc == c1 % 3).mean() * 100.0),    # scale aux agreeing with sink scale
        )
        tree_rows.append(row)

        # --- Per-image arrays grouped by archetype & head role (Section 7) ---
        # Failed / FAR / FRR: roles = main, shape, scale
        for name, mask in [('fail', arch_fail), ('far', arch_far), ('frr', arch_frr)]:
            _collect(per_image_H, (name, 'main'),  e_m[mask]); _collect(per_image_C, (name, 'main'),  c_m[mask])
            _collect(per_image_H, (name, 'shape'), e_sh[mask]); _collect(per_image_C, (name, 'shape'), c_sh[mask])
            _collect(per_image_H, (name, 'scale'), e_sc[mask]); _collect(per_image_C, (name, 'scale'), c_sc[mask])
        # PRR: roles = main, resisting_aux, fooled_aux
        _collect(per_image_H, ('prr', 'main'), e_m[arch_prr]); _collect(per_image_C, ('prr', 'main'), c_m[arch_prr])
        _collect(per_image_H, ('prr', 'resisting_aux'), np.concatenate([e_sh[js_sh_res], e_sc[js_sc_res]]))
        _collect(per_image_C, ('prr', 'resisting_aux'), np.concatenate([c_sh[js_sh_res], c_sc[js_sc_res]]))
        _collect(per_image_H, ('prr', 'fooled_aux'),    np.concatenate([e_sc[js_sh_res], e_sh[js_sc_res]]))
        _collect(per_image_C, ('prr', 'fooled_aux'),    np.concatenate([c_sc[js_sh_res], c_sh[js_sc_res]]))
        # DR: roles = main, divergent_aux
        _collect(per_image_H, ('dr', 'main'), e_m[arch_dr]); _collect(per_image_C, ('dr', 'main'), c_m[arch_dr])
        _collect(per_image_H, ('dr', 'divergent_aux'), np.concatenate([e_sh[sh_div_succ], e_sc[sc_div_succ]]))
        _collect(per_image_C, ('dr', 'divergent_aux'), np.concatenate([c_sh[sh_div_succ], c_sc[sc_div_succ]]))

tree_df = pd.DataFrame(tree_rows)

# Flatten per-image stores into contiguous float32 arrays
per_image_H = {k: np.concatenate(v) for k, v in per_image_H.items()}
per_image_C = {k: np.concatenate(v) for k, v in per_image_C.items()}

print(f"Loaded {len(configs)}/48 evaluation files -> {len(tree_df)} slices "
      f"(expected 240; N={N_TEST:,} per slice).")
print(f"Per-image archetype role arrays: {len(per_image_H)} entropy / {len(per_image_C)} confidence groups.")
# %%


# %% Cell 1 (continued) — Derived Slice-Level Composite Columns, pool_slices, and compute_d2_attribution

# Attach composite regime columns to tree_df so slice-level D^2 attribution works on every metric
_n_succ_s = tree_df['n_succ'].replace(0, np.nan)
tree_df['FAR_shape'] = tree_df['ARR_shape_AR']
tree_df['FAR_scale'] = tree_df['ARR_scale_RA']
tree_df['FAR_joint'] = tree_df['ARR_joint_AA']
tree_df['FAR'] = (
    (tree_df['n_shape_AR'] + tree_df['n_scale_RA'] + tree_df['n_joint_AA']) / _n_succ_s * 100.0
).fillna(0.0)
tree_df['FRR'] = (
    (tree_df['n_shape_RR'] + tree_df['n_scale_RR'] + tree_df['n_joint_RR']) / _n_succ_s * 100.0
).fillna(0.0)
tree_df['PRR_joint'] = tree_df['ARR_joint_RA'] + tree_df['ARR_joint_AR']
tree_df['PRR'] = (
    (tree_df['n_joint_RA'] + tree_df['n_joint_AR']) / _n_succ_s * 100.0
).fillna(0.0)

tree_df['DR_shape_sh'] = tree_df['ARR_shape_DR'] + tree_df['ARR_shape_DD']
tree_df['DR_shape_sc'] = tree_df['ARR_shape_AD'] + tree_df['ARR_shape_RD'] + tree_df['ARR_shape_DD']
tree_df['DR_shape']    = (
    tree_df['ARR_shape_AD'] + tree_df['ARR_shape_RD'] + tree_df['ARR_shape_DR'] + tree_df['ARR_shape_DD']
)

tree_df['DR_scale_sh'] = tree_df['ARR_scale_DA'] + tree_df['ARR_scale_DR'] + tree_df['ARR_scale_DD']
tree_df['DR_scale_sc'] = tree_df['ARR_scale_RD'] + tree_df['ARR_scale_DD']
tree_df['DR_scale']    = (
    tree_df['ARR_scale_RD'] + tree_df['ARR_scale_DA'] + tree_df['ARR_scale_DR'] + tree_df['ARR_scale_DD']
)

tree_df['DR_joint_sh'] = tree_df['ARR_joint_DA'] + tree_df['ARR_joint_DR'] + tree_df['ARR_joint_DD']
tree_df['DR_joint_sc'] = tree_df['ARR_joint_AD'] + tree_df['ARR_joint_RD'] + tree_df['ARR_joint_DD']
tree_df['DR_joint']    = (
    tree_df['ARR_joint_AD'] + tree_df['ARR_joint_RD']
    + tree_df['ARR_joint_DA'] + tree_df['ARR_joint_DR'] + tree_df['ARR_joint_DD']
)

tree_df['DR'] = (tree_df['n_arch_dr'] / _n_succ_s * 100.0).fillna(0.0)


def pool_slices(sub_df):
    """Sum raw counts across a subset of slices first, then compute exact pooled rates."""
    n_tot         = int(sub_df['n_total'].sum())
    n_succ        = int(sub_df['n_succ'].sum())
    n_fail        = int(sub_df['n_fail'].sum())
    n_fail_sh_ok  = int(sub_df['n_fail_sh_ok'].sum())
    n_fail_sc_ok  = int(sub_df['n_fail_sc_ok'].sum())
    n_fail_aux_ok = int(sub_df['n_fail_aux_ok'].sum())
    n_sh          = int(sub_df['n_sh'].sum())
    n_sc          = int(sub_df['n_sc'].sum())
    n_jt          = int(sub_df['n_jt'].sum())

    cohen_w = compute_cohen_w(n_sh, n_sc, n_jt)

    out = dict(
        n_total=n_tot, n_succ=n_succ, n_fail=n_fail,
        n_fail_sh_ok=n_fail_sh_ok, n_fail_sc_ok=n_fail_sc_ok, n_fail_aux_ok=n_fail_aux_ok,
        n_sh=n_sh, n_sc=n_sc, n_jt=n_jt,
        ASR=(n_succ / n_tot * 100.0) if n_tot > 0 else 0.0,
        Acc_main=(n_fail / n_tot * 100.0) if n_tot > 0 else 0.0,
        AuxShapeOK_fail=(n_fail_sh_ok / n_fail * 100.0) if n_fail > 0 else 0.0,
        AuxScaleOK_fail=(n_fail_sc_ok / n_fail * 100.0) if n_fail > 0 else 0.0,
        AuxOK_fail=(n_fail_aux_ok / n_fail * 100.0) if n_fail > 0 else 0.0,
        SilentCorr_fail=((n_fail - n_fail_aux_ok) / n_fail * 100.0) if n_fail > 0 else 0.0,
        SR_shape=(n_sh / n_succ * 100.0) if n_succ > 0 else 0.0,
        SR_scale=(n_sc / n_succ * 100.0) if n_succ > 0 else 0.0,
        SR_joint=(n_jt / n_succ * 100.0) if n_succ > 0 else 0.0,
        SR_single=((n_sh + n_sc) / n_succ * 100.0) if n_succ > 0 else 0.0,
        cohen_w=cohen_w,
    )

    for s_name, s_cnt, k_space in [
        ('shape', n_sh, K_SHAPE),
        ('scale', n_sc, K_SCALE),
        ('joint', n_jt, K_JOINT),
    ]:
        for code in k_space:
            cnt = int(sub_df[f'n_{s_name}_{code}'].sum())
            out[f'n_{s_name}_{code}']   = cnt
            out[f'ARR_{s_name}_{code}'] = (cnt / s_cnt * 100.0) if s_cnt > 0 else 0.0

    # Shift-conditional composite rates
    out['FAR_shape'] = out['ARR_shape_AR']
    out['FAR_scale'] = out['ARR_scale_RA']
    out['FAR_joint'] = out['ARR_joint_AA']
    out['PRR_joint'] = out['ARR_joint_RA'] + out['ARR_joint_AR']

    out['DR_shape_sh'] = out['ARR_shape_DR'] + out['ARR_shape_DD']
    out['DR_shape_sc'] = out['ARR_shape_AD'] + out['ARR_shape_RD'] + out['ARR_shape_DD']
    out['DR_shape']    = out['DR_shape_sh'] + out['DR_shape_sc'] - out['ARR_shape_DD']

    out['DR_scale_sh'] = out['ARR_scale_DA'] + out['ARR_scale_DR'] + out['ARR_scale_DD']
    out['DR_scale_sc'] = out['ARR_scale_RD'] + out['ARR_scale_DD']
    out['DR_scale']    = out['DR_scale_sh'] + out['DR_scale_sc'] - out['ARR_scale_DD']

    out['DR_joint_sh'] = out['ARR_joint_DA'] + out['ARR_joint_DR'] + out['ARR_joint_DD']
    out['DR_joint_sc'] = out['ARR_joint_AD'] + out['ARR_joint_RD'] + out['ARR_joint_DD']
    out['DR_joint']    = out['DR_joint_sh'] + out['DR_joint_sc'] - out['ARR_joint_DD']

    # Overall auxiliary accuracy across all N_total samples (failed + retained on S)
    n_sh_ok_succ = (
        out['n_shape_RR'] + out['n_shape_RD']
        + out['n_scale_RA'] + out['n_scale_RR'] + out['n_scale_RD']
        + out['n_joint_RA'] + out['n_joint_RR'] + out['n_joint_RD']
    )
    n_sc_ok_succ = (
        out['n_shape_AR'] + out['n_shape_RR'] + out['n_shape_DR']
        + out['n_scale_RR'] + out['n_scale_DR']
        + out['n_joint_AR'] + out['n_joint_RR'] + out['n_joint_DR']
    )
    out['Acc_shape'] = ((n_fail_sh_ok + n_sh_ok_succ) / n_tot * 100.0) if n_tot > 0 else 0.0
    out['Acc_scale'] = ((n_fail_sc_ok + n_sc_ok_succ) / n_tot * 100.0) if n_tot > 0 else 0.0

    # Overall composite regime counts and rates on S
    n_far = out['n_shape_AR'] + out['n_scale_RA'] + out['n_joint_AA']
    n_frr = out['n_shape_RR'] + out['n_scale_RR'] + out['n_joint_RR']
    n_prr = out['n_joint_RA'] + out['n_joint_AR']

    n_dr_sh_s = out['n_shape_DR'] + out['n_shape_DD']
    n_dr_sc_s = out['n_shape_AD'] + out['n_shape_RD'] + out['n_shape_DD']
    n_dr_sh_l = out['n_scale_DA'] + out['n_scale_DR'] + out['n_scale_DD']
    n_dr_sc_l = out['n_scale_RD'] + out['n_scale_DD']
    n_dr_sh_j = out['n_joint_DA'] + out['n_joint_DR'] + out['n_joint_DD']
    n_dr_sc_j = out['n_joint_AD'] + out['n_joint_RD'] + out['n_joint_DD']

    n_dr_sh_tot = n_dr_sh_s + n_dr_sh_l + n_dr_sh_j
    n_dr_sc_tot = n_dr_sc_s + n_dr_sc_l + n_dr_sc_j
    n_dd_tot    = out['n_shape_DD'] + out['n_scale_DD'] + out['n_joint_DD']
    n_dr_tot    = n_dr_sh_tot + n_dr_sc_tot - n_dd_tot

    out['n_far'], out['n_frr'], out['n_prr'], out['n_dr'] = n_far, n_frr, n_prr, n_dr_tot
    out['n_dr_sh_tot'], out['n_dr_sc_tot'], out['n_dd_tot'] = n_dr_sh_tot, n_dr_sc_tot, n_dd_tot

    if n_succ > 0:
        out['FAR']          = n_far / n_succ * 100.0
        out['FRR']          = n_frr / n_succ * 100.0
        out['PRR']          = n_prr / n_succ * 100.0
        out['DR']           = n_dr_tot / n_succ * 100.0
        out['DR_shape_aux'] = n_dr_sh_tot / n_succ * 100.0
        out['DR_scale_aux'] = n_dr_sc_tot / n_succ * 100.0
        out['DR_DD_total']  = n_dd_tot / n_succ * 100.0
    else:
        for k in ['FAR', 'FRR', 'PRR', 'DR', 'DR_shape_aux', 'DR_scale_aux', 'DR_DD_total']:
            out[k] = 0.0

    # Pooled mean confidence & normalized entropy per archetype
    for name in ['fail', 'far', 'frr', 'prr', 'dr']:
        nk = int(sub_df[f'n_arch_{name}'].sum())
        out[f'n_arch_{name}'] = nk
        for st_k in ['cm', 'csh', 'csc', 'em', 'esh', 'esc']:
            out[f'{name}_{st_k}'] = float(sub_df[f'sum_{name}_{st_k}'].sum() / nk) if nk > 0 else 0.0

    n_p = out['n_arch_prr']
    out['prr_c_resist'] = float(sub_df['sum_prr_c_resist'].sum() / n_p) if n_p > 0 else 0.0
    out['prr_c_fooled'] = float(sub_df['sum_prr_c_fooled'].sum() / n_p) if n_p > 0 else 0.0
    out['prr_e_resist'] = float(sub_df['sum_prr_e_resist'].sum() / n_p) if n_p > 0 else 0.0
    out['prr_e_fooled'] = float(sub_df['sum_prr_e_fooled'].sum() / n_p) if n_p > 0 else 0.0

    n_dh = int(sub_df['n_div_heads'].sum())
    out['n_div_heads'] = n_dh
    out['dr_c_only']   = float(sub_df['sum_dr_c_only'].sum() / n_dh) if n_dh > 0 else 0.0
    out['dr_e_only']   = float(sub_df['sum_dr_e_only'].sum() / n_dh) if n_dh > 0 else 0.0

    return out


def compute_d2_attribution(df, metric_col, denom_col, is_continuous=False):
    """Compute 1-way and 2-way Bernoulli KL-deviance (or weighted SS if is_continuous=True)
       share D^2 (%), max-min spread Delta, and level stats across the 6 factors."""
    factors = ['dataset', 'pool', 'weight', 'attack', 'norm', 'eps_step']
    f_names = ['data', 'pool', 'wght', 'atk', 'norm', 'eps']
    f_map   = dict(zip(factors, f_names))
    lvl_short = {
        'dataset':  {'standard': 'std', 'contrast': 'con'},
        'pool':     {'flatten': 'flat', 'gap': 'gap'},
        'weight':   {0.5: 'w=0.5', 1.0: 'w=1.0', 2.0: 'w=2.0'},
        'attack':   {'fgsm': 'fgsm', 'pgd': 'pgd'},
        'norm':     {'linf': 'linf', 'l2': 'l2'},
        'eps_step': {s: f'eps_{s}' for s in EPS_STEPS},
    }

    n_arr   = df[denom_col].to_numpy(dtype=float)
    val_arr = df[metric_col].to_numpy(dtype=float)
    if is_continuous or denom_col != 'n_total':
        pos_mask = (n_arr >= N_MIN_BRANCH) & (~np.isnan(val_arr))
    else:
        pos_mask = (n_arr > 0) & (~np.isnan(val_arr))

    n_tot_m = float(n_arr[pos_mask].sum())
    p_all_w = float((n_arr[pos_mask] * val_arr[pos_mask]).sum() / n_tot_m) if n_tot_m > 0 else 0.0
    g_all_val = pool_slices(df).get(metric_col, p_all_w)

    if is_continuous:
        d_tot = float((n_arr[pos_mask] * ((val_arr[pos_mask] - p_all_w) ** 2)).sum())
    else:
        d_tot = float(sum(
            n_i * kl_divergence(v_i / 100.0, p_all_w / 100.0)
            for n_i, v_i in zip(n_arr[pos_mask], val_arr[pos_mask])
        )) if n_tot_m > 0 else 0.0

    d1_dev, d1_pct, d1_delta, f1_vals, level_table = {}, {}, {}, {}, {}
    for f_col, f_disp in zip(factors, f_names):
        dev_f = 0.0
        lvl_stats = []
        f1_vals[f_col] = {}
        for lvl in sorted(df[f_col].unique()):
            l_mask = (df[f_col].to_numpy() == lvl) & pos_mask
            n_lvl  = float(n_arr[l_mask].sum())
            if n_lvl > 0:
                v_mean = float((n_arr[l_mask] * val_arr[l_mask]).sum() / n_lvl)
                if is_continuous:
                    dev_f += n_lvl * ((v_mean - p_all_w) ** 2)
                    v_disp = float(pool_slices(df[df[f_col] == lvl])[metric_col])
                else:
                    dev_f += n_lvl * kl_divergence(v_mean / 100.0, p_all_w / 100.0)
                    v_disp = v_mean
                f1_vals[f_col][lvl] = v_disp
                lvl_stats.append((lvl, lvl_short[f_col][lvl], v_disp, int(round(n_lvl))))
        d1_dev[f_col]    = dev_f
        pct_sh           = (dev_f / d_tot * 100.0) if d_tot > 1e-12 else 0.0
        d1_pct[f_disp]   = pct_sh
        d1_pct[f_col]    = pct_sh
        vals_only        = [x[2] for x in lvl_stats if not np.isnan(x[2])]
        d1_delta[f_disp] = (max(vals_only) - min(vals_only)) if len(vals_only) >= 2 else 0.0
        d1_delta[f_col]  = d1_delta[f_disp]
        level_table[f_col] = lvl_stats

    # All 15 two-way interactions
    inter_list = []
    for f1, f2 in itertools.combinations(factors, 2):
        p_lbl = f'{f_map[f1]}×{f_map[f2]}'
        dev_j = 0.0
        c_vals = []
        for l1 in sorted(df[f1].unique()):
            for l2 in sorted(df[f2].unique()):
                c_mask = (df[f1].to_numpy() == l1) & (df[f2].to_numpy() == l2) & pos_mask
                n_c = float(n_arr[c_mask].sum())
                if n_c > 0:
                    v_c_mean = float((n_arr[c_mask] * val_arr[c_mask]).sum() / n_c)
                    if is_continuous:
                        dev_j += n_c * ((v_c_mean - p_all_w) ** 2)
                    else:
                        dev_j += n_c * kl_divergence(v_c_mean / 100.0, p_all_w / 100.0)
                    c_vals.append(v_c_mean)
        dev_int = max(0.0, dev_j - d1_dev[f1] - d1_dev[f2])
        d2_int  = (dev_int / d_tot * 100.0) if d_tot > 1e-12 else 0.0
        del_2f  = (max(c_vals) - min(c_vals)) if len(c_vals) >= 2 else 0.0
        inter_list.append((p_lbl, d2_int, del_2f))

    inter_list.sort(key=lambda x: -x[1])
    top2_name, top2_d2, top2_delta = inter_list[0] if inter_list else ('--', 0.0, 0.0)

    atk_d2  = d1_pct['eps'] + d1_pct['norm'] + d1_pct['atk']
    arch_d2 = d1_pct['data'] + d1_pct['pool'] + d1_pct['wght']

    return dict(
        d_tot=d_tot,
        d1_pct=d1_pct,
        d1_delta=d1_delta,
        level_table=level_table,
        atk_d2=atk_d2,
        arch_d2=arch_d2,
        top2=(top2_name, top2_d2, top2_delta),
        interactions=inter_list,
    )
# %%


# %% PART A — Section 1.1: How Often Does the Attack Succeed?
print("=" * 128)
print("=== PART A — SECTION 1.1: HOW OFTEN DOES THE ATTACK SUCCEED? (NODE 0: GLOBAL ASR & 240-SLICE DISTRIBUTION) ===")
print("=" * 128)
print("Observed result:")
print("  - Clean accuracy is perfect on the unified standard test split (N=36,864) for all 12 models and all 3 branches,")
print("    so every successful attack is caused by the perturbation.")
print("  - Across the 240 (model x attack) slices, ASR is strongly left-skewed: the mean lies well below the median, a")
print("    sizeable share of slices is almost untouched (small budgets, L_2, contrast models), and the upper quartile sits")
print("    exactly at the 8/9 = 88.89% ceiling. Many slices cluster around this ceiling and a few reach ~100%.")
print("-" * 128)

# 1. Clean evaluation JSON verification across all 12 models
print("\n[1] Clean Baseline Verification on MY_DSPRITES_SPLIT (from b_models evaluation JSONs):")
print(f"{'Model Label':<18} | {'Primary Eval (Main / Shape / Scale)':<38} | {'Standard-Test Eval (Main / Shape / Scale)':<42}")
print("-" * 106)
for (d, p, w), m_lbl in zip(MODELS, MODEL_LABELS):
    prim_path, sec_path = get_eval_paths('multi', d, p, w)
    prim_str, sec_str = "N/A", "N/A (primary is standard)"
    if os.path.exists(prim_path):
        with open(prim_path, 'r', encoding='utf-8') as f:
            jd = json.load(f)
        a_m  = jd['shape_and_scale_task']['accuracy']
        a_sh = jd['shape_task']['accuracy']
        a_sc = jd['scale_task']['accuracy']
        prim_str = f"{a_m:.4f} / {a_sh:.4f} / {a_sc:.4f}"
    if d == 'contrast' and os.path.exists(sec_path):
        with open(sec_path, 'r', encoding='utf-8') as f:
            jd2 = json.load(f)
        a2_m  = jd2['shape_and_scale_task']['accuracy']
        a2_sh = jd2['shape_task']['accuracy']
        a2_sc = jd2['scale_task']['accuracy']
        sec_str = f"{a2_m:.4f} / {a2_sh:.4f} / {a2_sc:.4f}"
    print(f"{m_lbl:<18} | {prim_str:<38} | {sec_str:<42}")
print("-" * 106)

# 2. Pooled ASR vs unweighted slice mean + slice distribution quantiles
g_all = pool_slices(tree_df)
asr_ci_lo, asr_ci_hi = wilson_ci(g_all['ASR'], g_all['n_total'])
asr_arr = tree_df['ASR'].to_numpy(dtype=float)
unweighted_mean = float(np.mean(asr_arr))
p_min, q25, med, q75, p_max = np.percentile(asr_arr, [0, 25, 50, 75, 100])
iqr_val = float(q75 - q25)
std_val = float(np.std(asr_arr, ddof=1))

print("\n[2] Global Pooled ASR vs. 240-Slice Distribution Summary:")
print(f"  Pooled ASR (sum |S| / sum N) : {g_all['ASR']:6.2f}%  [95% Wilson CI: {asr_ci_lo:6.2f}%, {asr_ci_hi:6.2f}%]  "
      f"(|S| = {g_all['n_succ']:,} / N = {g_all['n_total']:,})")
print(f"  Unweighted 240-Slice Mean    : {unweighted_mean:6.2f}%  (Exact match because N = {N_TEST:,} is constant per slice)")
print(f"  Slice Quantiles (n=240)      : Min = {p_min:5.2f}% | Q25 = {q25:5.2f}% | Median = {med:5.2f}% | "
      f"Q75 = {q75:5.2f}% | Max = {p_max:5.2f}%")
print(f"  Slice Spread                 : IQR = {iqr_val:5.2f} pp | Std (sample) = {std_val:5.2f} pp")

# 3. Dynamic slice counts across meaningful ASR bands
n_slices = len(asr_arr)
bands = [
    ("Near-zero (< 1.0%)",                           asr_arr < 1.0),
    ("Untouched / very low (< 5.0%)",                asr_arr < 5.0),
    ("Intermediate (5.0% <= ASR < 80.0%)",           (asr_arr >= 5.0) & (asr_arr < 80.0)),
    ("High / below ceiling (80.0% <= ASR < 88.0%)",  (asr_arr >= 80.0) & (asr_arr < 88.0)),
    ("Around 8/9 ceiling (88.0% <= ASR <= 90.0%)",   (asr_arr >= 88.0) & (asr_arr <= 90.0)),
    ("  -> Tight 8/9 ceiling (|ASR - 88.89%| < 0.5)", np.abs(asr_arr - 800.0 / 9.0) < 0.5),
    ("Above 8/9 ceiling (90.0% < ASR < 99.0%)",      (asr_arr > 90.0) & (asr_arr < 99.0)),
    ("Near / at 100% (ASR >= 99.0%)",                asr_arr >= 99.0),
    ("  -> Exact 100.0% (ASR == 100.0%)",            np.isclose(asr_arr, 100.0)),
]
print("\n[3] Slice Counts Across ASR Regimes (out of 240 slices):")
print(f"{'ASR Band / Regime':<48} | {'Slices (k / 240)':>18} | {'Share (%)':>10}")
print("-" * 84)
for b_lbl, b_mask in bands:
    cnt = int(np.sum(b_mask))
    print(f"{b_lbl:<48} | {f'{cnt:3d} / {n_slices}':>18} | {cnt / n_slices * 100.0:9.2f}%")
print("-" * 84)

# 4. Full 12 models x 20 attack settings ASR table
atk_norm_order = [('fgsm', 'l2'), ('pgd', 'l2'), ('fgsm', 'linf'), ('pgd', 'linf')]
col_tuples = [(a, n, s) for (a, n) in atk_norm_order for s in EPS_STEPS]
col_short  = [f"{a.upper()}-{n.upper()} e{s}" for (a, n, s) in col_tuples]

asr_grid = np.zeros((len(MODELS), len(col_tuples)), dtype=float)
for i, (d, p, w) in enumerate(MODELS):
    for j, (a, n, s) in enumerate(col_tuples):
        sub_cell = tree_df[
            (tree_df['dataset'] == d) & (tree_df['pool'] == p) & (tree_df['weight'] == w)
            & (tree_df['attack'] == a) & (tree_df['norm'] == n) & (tree_df['eps_step'] == s)
        ]
        asr_grid[i, j] = float(sub_cell['ASR'].iloc[0]) if len(sub_cell) > 0 else np.nan

print("\n[4] Full 12 Models x 20 Attack Settings ASR (%) Matrix (Columns grouped: FGSM-L2 | PGD-L2 | FGSM-Linf | PGD-Linf):")
hdr_blocks = " || ".join(
    f"{a.upper() + '-' + n.upper():^34}" for (a, n) in atk_norm_order
)
sub_hdr = " || ".join(
    " ".join(f"{'e'+str(s):>6}" for s in EPS_STEPS) for _ in atk_norm_order
)
print(f"{'Model':<16} || {hdr_blocks} || {'Mean':>6}")
print(f"{'(d|pi|w)':<16} || {sub_hdr} || {'ASR%':>6}")
print("-" * 176)
for i, m_lbl in enumerate(MODEL_LABELS):
    if i in (3, 6, 9):
        print("-" * 176)
    blocks_str = " || ".join(
        " ".join(f"{asr_grid[i, 5*b + k]:6.1f}" for k in range(5))
        for b in range(4)
    )
    row_mean = float(np.nanmean(asr_grid[i, :]))
    print(f"{m_lbl:<16} || {blocks_str} || {row_mean:6.1f}")
print("-" * 176)
col_means_str = " || ".join(
    " ".join(f"{np.nanmean(asr_grid[:, 5*b + k]):6.1f}" for k in range(5))
    for b in range(4)
)
print(f"{'Col Mean':<16} || {col_means_str} || {float(np.nanmean(asr_grid)):6.1f}")
print("=" * 128 + "\n")

# --- Figure 1: 2-Panel (Panel A: 240-Slice ASR Histogram | Panel B: 12 x 20 ASR Heatmap) ---
fig1, (ax_hist, ax_hm) = plt.subplots(
    1, 2, figsize=(16.5, 5.8), dpi=130, gridspec_kw={'width_ratios': [1.0, 1.85]}
)

# Panel A: Histogram over the 240 slices
bins = np.linspace(0, 100, 21)
ax_hist.hist(asr_arr, bins=bins, color='#4a6fe3', edgecolor='#1b263b', alpha=0.85)
ax_hist.axvline(g_all['ASR'], color='#d62728', linestyle='-', linewidth=2.2,
                label=f"Pooled / Slice Mean = {g_all['ASR']:.1f}%")
ax_hist.axvline(med, color='#2ca02c', linestyle='--', linewidth=2.2,
                label=f"Slice Median = {med:.1f}% (IQR: {q25:.1f}–{q75:.1f}%)")
ax_hist.axvline(800.0 / 9.0, color='#111111', linestyle='-.', linewidth=2.0,
                label="8/9 Single-Class Ceiling = 88.9%")
ax_hist.set_xlim(-2, 102)
ax_hist.set_xlabel("Attack Success Rate — $\\mathrm{ASR}$ (%)", fontsize=10.5)
ax_hist.set_ylabel("Number of Slices (out of 240)", fontsize=10.5)
ax_hist.set_title("A. Distribution of $\\mathrm{ASR}$ Across the 240 $(m, a)$ Slices", fontsize=11, fontweight='bold')
ax_hist.grid(True, linestyle=':', alpha=0.6)
ax_hist.legend(loc='upper left', fontsize=8.6, framealpha=0.92)

# Panel B: 12 models x 20 attacks heatmap
im1 = ax_hm.imshow(asr_grid, cmap='YlOrRd', aspect='auto', vmin=0.0, vmax=100.0)
ax_hm.set_yticks(np.arange(len(MODEL_LABELS)))
ax_hm.set_yticklabels(MODEL_LABELS, fontsize=8.6, fontweight='bold')

x_tick_lbls = []
for (a, n, s) in col_tuples:
    eps_real = EPS_LINF[s] if n == 'linf' else EPS_L2[s]
    x_tick_lbls.append(f"$\\varepsilon_{s}$\n({eps_real:g})")
ax_hm.set_xticks(np.arange(len(col_tuples)))
ax_hm.set_xticklabels(x_tick_lbls, fontsize=7.2)

# Annotate each cell with exact ASR
for i in range(asr_grid.shape[0]):
    for j in range(asr_grid.shape[1]):
        v = asr_grid[i, j]
        if not np.isnan(v):
            txt_c = 'white' if v >= 55.0 else '#111111'
            ax_hm.text(j, i, f"{v:.0f}", ha='center', va='center',
                       color=txt_c, fontsize=7.0, fontweight='bold')

# Visual separators between attack x norm blocks (every 5 columns) and model families (every 3 rows)
for x_sep in [4.5, 9.5, 14.5]:
    ax_hm.axvline(x_sep, color='#111111', linewidth=1.8)
for y_sep in [2.5, 5.5, 8.5]:
    ax_hm.axhline(y_sep, color='#333333', linestyle='--', linewidth=1.0)

# Group headers above the 4 attack x norm blocks (positioned cleanly below the title)
for b_idx, (a, n) in enumerate(atk_norm_order):
    norm_sym = "$L_2$" if n == 'l2' else "$L_\\infty$"
    ax_hm.text(b_idx * 5 + 2.0, -0.58, f"{a.upper()} ({norm_sym})",
               ha='center', va='bottom', fontsize=9.0, fontweight='bold', color='#111111')

ax_hm.set_title("B. $\\mathrm{ASR}$ (%) Matrix: 12 Models $\\times$ 20 Attack Settings", fontsize=11, fontweight='bold', pad=26)
plt.colorbar(im1, ax=ax_hm, fraction=0.026, pad=0.02, label="$\\mathrm{ASR}$ (%)")
plt.tight_layout()
save_and_show(fig1, "Figure 1: Global ASR Distribution (240 Slices) and 12 Models x 20 Attacks Heatmap")
# %%



# %% PART A — Section 1.2: What Does the Success Depend On?
print("=" * 128)
print("=== PART A — SECTION 1.2: WHAT DOES THE SUCCESS DEPEND ON? (NODE 0: FACTOR MARGINALS, D^2 & INTERACTIONS) ===")
print("=" * 128)
print("Observed result:")
print("  - Attack parameters (budget, norm, optimizer) explain most of the variation in ASR; L_inf is more effective than")
print("    L_2 and PGD more effective than FGSM.")
print("  - The budget effect depends on the norm: in L_inf ASR jumps sharply from eps_1 to eps_2 and then saturates,")
print("    whereas in L_2 it keeps growing up to eps_4 (the pooled budget marginal mixes the two norms).")
print("  - Architecture plays a secondary role: contrast training (con) and flatten pooling (flat) make `main` harder to")
print("    fool overall, while loss weight w has a negligible effect.")
print("  - `con | flat` strongly resists single-step FGSM-L2 at every budget, yields to multi-step PGD-L2 only at higher")
print("    budgets, and yields almost completely under PGD-L_inf from eps_2 onward.")
print("-" * 128)

d2_asr = compute_d2_attribution(tree_df, 'ASR', 'n_total', is_continuous=False)

factor_specs = [
    ('eps_step', 'eps',  'Budget Step (eps)',   EPS_STEPS, [f'eps_{s}' for s in EPS_STEPS]),
    ('norm',     'norm', 'Norm (p)',            ['linf', 'l2'], ['L_inf', 'L_2']),
    ('attack',   'atk',  'Optimizer (alpha)',   ['fgsm', 'pgd'], ['FGSM', 'PGD']),
    ('dataset',  'data', 'Dataset (d)',         ['standard', 'contrast'], ['std', 'con']),
    ('pool',     'pool', 'Pooling (pi)',        ['flatten', 'gap'], ['flat', 'gap']),
    ('weight',   'wght', 'Loss Weight (w)',     [0.5, 1.0, 2.0], ['w=0.5', 'w=1.0', 'w=2.0']),
]

# Build marginal stats and sort by decreasing max-min spread Delta (pp)
factor_rows = []
for col, f_key, f_lbl, lvls, short_lvls in factor_specs:
    lvl_entries = []
    for lv, s_lv in zip(lvls, short_lvls):
        st = pool_slices(tree_df[tree_df[col] == lv])
        lo, hi = wilson_ci(st['ASR'], st['n_total'])
        lvl_entries.append((s_lv, st['ASR'], lo, hi, st['n_total']))
    vals = [x[1] for x in lvl_entries]
    delta_pp = max(vals) - min(vals)
    factor_rows.append((delta_pp, d2_asr['d1_pct'][f_key], f_lbl, col, f_key, lvl_entries))

factor_rows.sort(key=lambda x: -x[0])

# 1. Marginal ASR table for all 6 factors ordered by decreasing Delta (pp)
print("\n[1] Node 0 Marginal ASR (%) Across All 6 Experimental Factors (Ordered by Decreasing Spread Delta):")
print(f"{'Factor':<20} | {'Delta (pp)':>10} | {'D^2 (%)':>8} | {'Level Pooled ASR (%) [95% Wilson CI] (N per level)':<80}")
print("-" * 128)
for delta_pp, d2_val, f_lbl, col, f_key, lvl_entries in factor_rows:
    n_per_lvl = lvl_entries[0][4]
    lvl_str = " | ".join(f"{s_lv}: {v:5.2f}% [{lo:5.2f},{hi:5.2f}]" for s_lv, v, lo, hi, _ in lvl_entries)
    print(f"{f_lbl:<20} | {delta_pp:10.2f} | {d2_val:7.2f}% | {lvl_str}  (N={n_per_lvl:,})")
print("-" * 128)

# Separate eps_1..eps_5 marginals by norm (L_inf vs L_2) so physical scales are explicit
print("\n[1b] Budget Marginals Split by Norm (Explicit Physical Scales eps_inf vs eps_2):")
for n_code, n_lbl, eps_map in [('linf', 'L_inf', EPS_LINF), ('l2', 'L_2', EPS_L2)]:
    sub_n = tree_df[tree_df['norm'] == n_code]
    parts = []
    vals_n = []
    for s in EPS_STEPS:
        st_s = pool_slices(sub_n[sub_n['eps_step'] == s])
        lo, hi = wilson_ci(st_s['ASR'], st_s['n_total'])
        vals_n.append(st_s['ASR'])
        parts.append(f"e{s}({eps_map[s]:g}): {st_s['ASR']:5.2f}% [{lo:5.2f},{hi:5.2f}]")
    del_n = max(vals_n) - min(vals_n)
    n_cell = pool_slices(sub_n[sub_n['eps_step'] == 1])['n_total']
    print(f"  {n_lbl:<6} (Delta={del_n:5.2f} pp, N/step={n_cell:,}) -> " + " | ".join(parts))

# 2. Grouped D^2 summary (Attack Parameters vs Architecture) + top 2-way interactions
print("\n[2] Grouped Bernoulli KL-Deviance Attribution (D^2%) & Top 2-Way Interactions:")
print(f"  Attack Parameters (eps + norm + optimizer) : D^2 = {d2_asr['atk_d2']:6.2f}%  "
      f"(eps={d2_asr['d1_pct']['eps']:.2f}%, norm={d2_asr['d1_pct']['norm']:.2f}%, atk={d2_asr['d1_pct']['atk']:.2f}%)")
print(f"  Model Architecture (dataset + pool + w)    : D^2 = {d2_asr['arch_d2']:6.2f}%  "
      f"(data={d2_asr['d1_pct']['data']:.2f}%, pool={d2_asr['d1_pct']['pool']:.2f}%, wght={d2_asr['d1_pct']['wght']:.2f}%)")
print("  Top 5 Two-Way Interactions on Node 0 ASR:")
for rank_i, (p_lbl, d2_i, del_i) in enumerate(d2_asr['interactions'][:5], start=1):
    print(f"    {rank_i}. {p_lbl:<12} : D^2 = {d2_i:5.2f}%  |  Cell Spread Delta = {del_i:5.2f} pp")

# 3. ASR across eps_1..eps_5 for the 4 model families across the 4 attack x norm combinations
families = [
    ('std|gap',  'standard', 'gap'),
    ('std|flat', 'standard', 'flatten'),
    ('con|gap',  'contrast', 'gap'),
    ('con|flat', 'contrast', 'flatten'),
]
print("\n[3] Pooled ASR (%) Across eps_1..eps_5 for the 4 Model Families x 4 Attack-Norm Combinations (with 95% Wilson CIs):")
print(f"{'Attack-Norm':<12} | {'Family':<10} | {'N / step':>9} | {'Pooled':>7} || "
      + " | ".join(f"{'eps_' + str(s) + ' [95% CI]':^21}" for s in EPS_STEPS))
print("-" * 168)
for (a, n) in [('fgsm', 'linf'), ('pgd', 'linf'), ('fgsm', 'l2'), ('pgd', 'l2')]:
    for f_lbl, d_val, p_val in families:
        sub_f = tree_df[(tree_df['attack'] == a) & (tree_df['norm'] == n)
                        & (tree_df['dataset'] == d_val) & (tree_df['pool'] == p_val)]
        st_all = pool_slices(sub_f)
        step_strs = []
        for s in EPS_STEPS:
            st_s = pool_slices(sub_f[sub_f['eps_step'] == s])
            lo, hi = wilson_ci(st_s['ASR'], st_s['n_total'])
            step_strs.append(f"{st_s['ASR']:5.1f}% [{lo:5.1f},{hi:5.1f}]")
        n_step = pool_slices(sub_f[sub_f['eps_step'] == 1])['n_total']
        print(f"{a.upper()+'-'+n.upper():<12} | {f_lbl:<10} | {n_step:9,d} | {st_all['ASR']:6.2f}% || " + " | ".join(step_strs))
    print("-" * 168)

# 4. Detailed table comparing con|flat vs con|gap under FGSM-L2 and PGD-L2 for each weight w in {0.5, 1.0, 2.0}
print("\n[4] Detailed Caution Comparison: `con | flat` vs `con | gap` Under FGSM-L2 and PGD-L2 by Loss Weight w (N=36,864 per cell):")
print(f"{'Configuration (con | pool | w | attack-L2)':<40} | {'Pooled ASR [95% CI]':>23} || "
      + " | ".join(f"{'e'+str(s)+' ('+str(EPS_L2[s])+')':>18}" for s in EPS_STEPS))
print("-" * 176)
for p_val, p_short in [('flatten', 'flat'), ('gap', 'gap')]:
    for a_val in ['fgsm', 'pgd']:
        for w_val in LOSS_WEIGHTS:
            sub_c = tree_df[(tree_df['dataset'] == 'contrast') & (tree_df['pool'] == p_val)
                            & (tree_df['weight'] == w_val) & (tree_df['attack'] == a_val)
                            & (tree_df['norm'] == 'l2')]
            st_c = pool_slices(sub_c)
            lo_c, hi_c = wilson_ci(st_c['ASR'], st_c['n_total'])
            e_cells = []
            for s in EPS_STEPS:
                st_s = pool_slices(sub_c[sub_c['eps_step'] == s])
                lo_s, hi_s = wilson_ci(st_s['ASR'], st_s['n_total'])
                e_cells.append(f"{st_s['ASR']:5.2f}% [{lo_s:4.1f},{hi_s:4.1f}]")
            lbl_c = f"con | {p_short:<4} | w={w_val:<3g} | {a_val.upper()}-L2"
            print(f"{lbl_c:<40} | {st_c['ASR']:6.2f}% [{lo_c:5.2f},{hi_c:5.2f}] || " + " | ".join(e_cells))
        print("-" * 176)
print("=" * 128 + "\n")

# --- Figure 2A: Standalone 2x2 Grid (FGSM-Linf, PGD-Linf, FGSM-L2, PGD-L2) With Shared Y-Axes ---
fig2a, axes2a = plt.subplots(2, 2, figsize=(12.4, 7.2), dpi=130, sharey=True)

panel_a_specs = [
    (0, 0, 'fgsm', 'linf', "A1. FGSM ($L_\\infty$)"),
    (0, 1, 'pgd',  'linf', "A2. PGD ($L_\\infty$)"),
    (1, 0, 'fgsm', 'l2',   "A3. FGSM ($L_2$) — `con|flat` Resists"),
    (1, 1, 'pgd',  'l2',   "A4. PGD ($L_2$) — `con|flat` Yields at High $\\varepsilon$"),
]

x_steps = np.arange(1, 6)
for r_i, c_i, a_val, n_val, p_title in panel_a_specs:
    ax_p = axes2a[r_i, c_i]
    eps_map = EPS_LINF if n_val == 'linf' else EPS_L2
    for f_lbl, d_val, p_val in families:
        sub_f = tree_df[(tree_df['attack'] == a_val) & (tree_df['norm'] == n_val)
                        & (tree_df['dataset'] == d_val) & (tree_df['pool'] == p_val)]
        y_vals, y_lo, y_hi = [], [], []
        for s in EPS_STEPS:
            st_s = pool_slices(sub_f[sub_f['eps_step'] == s])
            lo_s, hi_s = wilson_ci(st_s['ASR'], st_s['n_total'])
            y_vals.append(st_s['ASR'])
            y_lo.append(lo_s)
            y_hi.append(hi_s)
        y_vals = np.array(y_vals)
        err_low  = np.maximum(0.0, y_vals - np.array(y_lo))
        err_high = np.maximum(0.0, np.array(y_hi) - y_vals)
        ax_p.errorbar(
            x_steps, y_vals, yerr=[err_low, err_high],
            label=f_lbl, color=FAMILY_COLORS[f_lbl], marker=FAMILY_MARKERS[f_lbl],
            linewidth=2.1, markersize=6.0, capsize=2.8,
        )
    ax_p.axhline(800.0 / 9.0, color='#444444', linestyle='--', linewidth=1.2, label='8/9 Ceiling (88.9%)')
    ax_p.set_xticks(x_steps)
    ax_p.set_xticklabels([f"$\\varepsilon_{s}$\n({eps_map[s]:g})" for s in EPS_STEPS], fontsize=8.8)
    ax_p.set_ylim(-4, 104)
    if c_i == 0:
        ax_p.set_ylabel("$\\mathrm{ASR}$ (%)", fontsize=10.2)
    ax_p.set_title(p_title, fontsize=10.6, fontweight='bold')
    ax_p.grid(True, linestyle=':', alpha=0.6)
    if r_i == 0 and c_i == 0:
        ax_p.legend(loc='lower right', fontsize=8.4, framealpha=0.92)

plt.tight_layout()
save_and_show(fig2a, "Figure 2A: Node 0 ASR Trajectories Across Budget Steps by Optimizer, Norm, and Model Family")

# --- Figure 2B: Separate Factor Marginal Spread (Delta pp) Plot With [0, 100] X-Axis ---
fig2b, ax_b = plt.subplots(figsize=(10.6, 5.4), dpi=130)

dot_rows = []
for delta_pp, d2_val, f_lbl, col, f_key, lvl_entries in factor_rows:
    dot_rows.append((f_lbl, delta_pp, d2_val, lvl_entries))
    if col == 'eps_step':
        for n_code, n_lbl in [('linf', '  eps ($L_\\infty$ only)'), ('l2', '  eps ($L_2$ only)')]:
            sub_n = tree_df[tree_df['norm'] == n_code]
            entries_n = []
            for s in EPS_STEPS:
                st_s = pool_slices(sub_n[sub_n['eps_step'] == s])
                lo_s, hi_s = wilson_ci(st_s['ASR'], st_s['n_total'])
                entries_n.append((f"e{s}", st_s['ASR'], lo_s, hi_s, st_s['n_total']))
            vals_n = [x[1] for x in entries_n]
            dot_rows.append((n_lbl, max(vals_n) - min(vals_n), np.nan, entries_n))

dot_rows_rev = dot_rows[::-1]
y_positions = np.arange(len(dot_rows_rev))
palette_dots = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e']

y_tick_labels_2b = []
for y_idx, (r_lbl, del_v, d2_v, entries) in enumerate(dot_rows_rev):
    stat_suffix = f"$\\Delta={del_v:.1f}$ pp" + (f", $D^2={d2_v:.1f}\\%$" if not np.isnan(d2_v) else "")
    y_tick_labels_2b.append(f"{r_lbl}\n({stat_suffix})")

    entries_sorted = sorted(entries, key=lambda x: x[1])
    v_min_lbl, v_min = entries_sorted[0][0],  entries_sorted[0][1]
    v_max_lbl, v_max = entries_sorted[-1][0], entries_sorted[-1][1]
    ax_b.plot([v_min, v_max], [y_idx, y_idx], color='#555555', linewidth=2.8, zorder=2)
    for k_i, (s_lv, v, lo, hi, _) in enumerate(entries):
        c_dot = palette_dots[k_i % len(palette_dots)]
        ax_b.scatter(v, y_idx, color=c_dot, edgecolor='#111111', s=64, zorder=3)
    # Place min/max endpoint labels slightly above the dots so the x-axis stays strictly within [0, 100]%
    if del_v >= 2.5:
        ax_b.text(v_min, y_idx + 0.22, f"{v_min_lbl}: {v_min:.1f}%",
                  ha='left' if v_min < 12.0 else 'center', va='bottom',
                  fontsize=7.8, color='#222222', fontweight='bold')
        ax_b.text(v_max, y_idx + 0.22, f"{v_max_lbl}: {v_max:.1f}%",
                  ha='right' if v_max > 88.0 else 'center', va='bottom',
                  fontsize=7.8, color='#222222', fontweight='bold')
    else:
        ax_b.text(v_max + 1.5, y_idx, f"{v_min:.1f}–{v_max:.1f}%",
                  ha='left', va='center', fontsize=7.8, color='#222222', fontweight='bold')

ax_b.axvline(g_all['ASR'], color='#d62728', linestyle=':', linewidth=1.6, label=f"Global Pooled ASR ({g_all['ASR']:.1f}%)")
ax_b.set_yticks(y_positions)
ax_b.set_yticklabels(y_tick_labels_2b, fontsize=8.8, fontweight='bold')
ax_b.set_xlim(0, 100)
ax_b.set_xticks([0, 20, 40, 60, 80, 100])
ax_b.set_ylim(-0.6, len(dot_rows_rev) - 0.15)
ax_b.set_xlabel("Marginal Pooled $\\mathrm{ASR}$ (%)", fontsize=10.2)
ax_b.set_title("Figure 2B: Node 0 Factor Marginal Spreads ($\\Delta$ pp) and Explained Deviance ($D^2\\%$)", fontsize=10.8, fontweight='bold')
ax_b.grid(True, axis='x', linestyle=':', alpha=0.6)
ax_b.legend(loc='lower left', fontsize=8.4, framealpha=0.92)

plt.tight_layout()
save_and_show(fig2b, "Figure 2B: Node 0 Factor Marginal Spreads (Delta pp) and D^2 Share")
# %%



# %% PART A — Section 2: How Does the Main Make Mistakes?
print("=" * 128)
print("=== PART A — SECTION 2: HOW DOES THE MAIN MAKE MISTAKES? (NODE 1: SHIFT TYPOLOGY SR_s & COHEN'S w) ===")
print("=" * 128)
print("Observed result:")
print("  - At eps_1 most successful attacks alter a single factor; as the budget grows, the share of single-factor errors")
print("    and the effect size against the (25%, 25%, 50%) reference both decrease steadily toward the random-error profile.")
print("  - At eps_1, `gap` breaks scale first in both norms. In `flat`, shape exceeds scale in both norms, but under")
print("    L_inf the most frequent shift is still joint, and the L_2 result rests on only ~2,200 images (moreover, within")
print("    `std | flat` under PGD-L2 at eps_1, 73.7% of shape-shifts are full retention RR and only 26.3% are aligned AR,")
print("    indicating that this L_2 shape-first excess is a near-boundary phenomenon).")
print("  - Pooled effect size at eps_1 is smaller than the slice-median one, consistent with opposite pooling biases")
print("    cancelling in pooled counts (the slice-median at eps_1 rests on few slices).")
print("  - Contrast models (`con`) shift error mass toward joint errors relative to standard models (`std`).")
print("-" * 128)

def _slice_cohen_w_quantiles(sub_df):
    """Return (median, q25, q75, n_valid) of slice-level cohen_w over slices with n_succ >= N_MIN_BRANCH."""
    valid = sub_df.loc[sub_df['n_succ'] >= N_MIN_BRANCH, 'cohen_w'].dropna().to_numpy(dtype=float)
    if len(valid) == 0:
        return (np.nan, np.nan, np.nan, 0)
    q25_v, med_v, q75_v = np.percentile(valid, [25, 50, 75])
    return (float(med_v), float(q25_v), float(q75_v), int(len(valid)))

# 1. Pooled SR_shape, SR_scale, SR_joint, SR_single over all successful attacks S
g_all = pool_slices(tree_df)
med_w_all, q25_w_all, q75_w_all, n_val_all = _slice_cohen_w_quantiles(tree_df)

print("\n[1] Global Shift Typology Over All Successful Attacks S (Compared to 25% / 25% / 50% Uniform Random Reference):")
print(f"{'Shift Metric':<28} | {'Event Count (|S_s| / |S|)':>28} | {'Pooled Rate [95% Wilson CI]':>28} | {'Random Ref':>12} | {'Diff vs Ref':>12}")
print("-" * 120)
for s_lbl, k_col, r_col, ref_v in [
    ("SR_shape  (Shape-Only)",   'n_sh', 'SR_shape',  25.0),
    ("SR_scale  (Scale-Only)",   'n_sc', 'SR_scale',  25.0),
    ("SR_joint  (Joint-Shift)",  'n_jt', 'SR_joint',  50.0),
    ("SR_single (Shape + Scale)", None,  'SR_single', 50.0),
]:
    k_cnt = (g_all['n_sh'] + g_all['n_sc']) if k_col is None else g_all[k_col]
    r_val = g_all[r_col]
    lo, hi = wilson_ci(r_val, g_all['n_succ'])
    ratio_str = f"{k_cnt:,} / {g_all['n_succ']:,}"
    print(
        f"{s_lbl:<28} | {ratio_str:>28} | "
        f"{r_val:6.2f}% [{lo:6.2f}%, {hi:6.2f}%] | {ref_v:11.2f}% | {r_val - ref_v:+11.2f} pp"
    )
print("-" * 120)
print(f"  Global Pooled cohen_w = {g_all['cohen_w']:.3f}  |  "
      f"Slice-Median cohen_w = {med_w_all:.3f} [IQR: {q25_w_all:.3f}–{q75_w_all:.3f}] (n={n_val_all}/240 valid slices >= {N_MIN_BRANCH})")

# 2. SR_shape, SR_scale, SR_joint, |S|, pooled cohen_w, and slice-median cohen_w [Q25-Q75] across eps_1..eps_5
print("\n[2] Shift Typology and Effect Size cohen_w Across Budget Steps eps_1..eps_5:")
print("  [Note: At eps_1, pooled cohen_w is smaller than slice-median cohen_w, consistent with `gap` (scale-biased) and `flat`")
print("   (shape/joint-biased) pulling in opposite directions when summed before computing w.]")
print(f"{'Step':<8} | {'|S| (Succ.)':>12} | {'SR_shape [95% CI]':>21} | {'SR_scale [95% CI]':>21} | "
      f"{'SR_joint [95% CI]':>21} | {'SR_single':>9} | {'Pooled w':>8} | {'Slice-Med w [Q25-Q75] (n)':>26}")
print("-" * 144)
eps_stats = []
for s in EPS_STEPS:
    sub_s = tree_df[tree_df['eps_step'] == s]
    st_s = pool_slices(sub_s)
    med_w, q25_w, q75_w, n_v = _slice_cohen_w_quantiles(sub_s)
    eps_stats.append((s, st_s, med_w, q25_w, q75_w, n_v))
    sh_lo, sh_hi = wilson_ci(st_s['SR_shape'], st_s['n_succ'])
    sc_lo, sc_hi = wilson_ci(st_s['SR_scale'], st_s['n_succ'])
    jt_lo, jt_hi = wilson_ci(st_s['SR_joint'], st_s['n_succ'])
    print(f"{'eps_'+str(s):<8} | {st_s['n_succ']:12,d} | "
          f"{st_s['SR_shape']:5.2f}% [{sh_lo:5.2f},{sh_hi:5.2f}] | "
          f"{st_s['SR_scale']:5.2f}% [{sc_lo:5.2f},{sc_hi:5.2f}] | "
          f"{st_s['SR_joint']:5.2f}% [{jt_lo:5.2f},{jt_hi:5.2f}] | "
          f"{st_s['SR_single']:8.2f}% | {st_s['cohen_w']:8.3f} | "
          f"{med_w:5.3f} [{q25_w:5.3f}-{q75_w:5.3f}] ({n_v:2d})")
print("-" * 144)

# 3. Shift typology by pooling (flat vs gap), dataset (std vs con), and the 4 families {std, con} x {flat, gap}
print("\n[3] Shift Typology by Pooling, Dataset, and Model Family (Pooled Across All Attacks):")
print(f"{'Subgroup':<22} | {'|S| (Succ.)':>12} | {'SR_shape [95% CI]':>21} | {'SR_scale [95% CI]':>21} | "
      f"{'SR_joint [95% CI]':>21} | {'Pooled w':>8} | {'Slice-Med w [Q25-Q75]':>22}")
print("-" * 140)
subgroups_s2 = [
    ("Pooling: flat", tree_df['pool'] == 'flatten'),
    ("Pooling: gap",  tree_df['pool'] == 'gap'),
    ("Dataset: std",  tree_df['dataset'] == 'standard'),
    ("Dataset: con",  tree_df['dataset'] == 'contrast'),
    ("Family: std|flat", (tree_df['dataset'] == 'standard') & (tree_df['pool'] == 'flatten')),
    ("Family: std|gap",  (tree_df['dataset'] == 'standard') & (tree_df['pool'] == 'gap')),
    ("Family: con|flat", (tree_df['dataset'] == 'contrast') & (tree_df['pool'] == 'flatten')),
    ("Family: con|gap",  (tree_df['dataset'] == 'contrast') & (tree_df['pool'] == 'gap')),
]
for idx, (g_lbl, g_mask) in enumerate(subgroups_s2):
    if idx in (2, 4):
        print("-" * 140)
    sub_g = tree_df[g_mask]
    st_g = pool_slices(sub_g)
    med_w, q25_w, q75_w, _ = _slice_cohen_w_quantiles(sub_g)
    sh_lo, sh_hi = wilson_ci(st_g['SR_shape'], st_g['n_succ'])
    sc_lo, sc_hi = wilson_ci(st_g['SR_scale'], st_g['n_succ'])
    jt_lo, jt_hi = wilson_ci(st_g['SR_joint'], st_g['n_succ'])
    print(f"{g_lbl:<22} | {st_g['n_succ']:12,d} | "
          f"{st_g['SR_shape']:5.2f}% [{sh_lo:5.2f},{sh_hi:5.2f}] | "
          f"{st_g['SR_scale']:5.2f}% [{sc_lo:5.2f},{sc_hi:5.2f}] | "
          f"{st_g['SR_joint']:5.2f}% [{jt_lo:5.2f},{jt_hi:5.2f}] | "
          f"{st_g['cohen_w']:8.3f} | {med_w:5.3f} [{q25_w:5.3f}-{q75_w:5.3f}]")
print("-" * 140)

# 4. At eps_1: {flat, gap} x {linf, l2} and std|gap vs std|flat under PGD-L2 at eps_1
print("\n[4] Initial Boundary Breakdown at eps_1: {flat, gap} x {L_inf, L_2} and `std|gap` vs `std|flat` Under PGD-L2:")
print(f"{'eps_1 Subgroup':<34} | {'|S| (Succ.)':>12} | {'SR_shape [95% CI]':>21} | "
      f"{'SR_scale [95% CI]':>21} | {'SR_joint [95% CI]':>21} | {'Pooled w':>8}")
print("-" * 128)
e1_df = tree_df[tree_df['eps_step'] == 1]
e1_groups = [
    ("eps_1: flat x L_inf", (e1_df['pool'] == 'flatten') & (e1_df['norm'] == 'linf')),
    ("eps_1: gap  x L_inf", (e1_df['pool'] == 'gap')     & (e1_df['norm'] == 'linf')),
    ("eps_1: flat x L_2",   (e1_df['pool'] == 'flatten') & (e1_df['norm'] == 'l2')),
    ("eps_1: gap  x L_2",   (e1_df['pool'] == 'gap')     & (e1_df['norm'] == 'l2')),
    ("eps_1 PGD-L2: std|gap  (all w)", (e1_df['dataset'] == 'standard') & (e1_df['pool'] == 'gap')     & (e1_df['attack'] == 'pgd') & (e1_df['norm'] == 'l2')),
    ("eps_1 PGD-L2: std|flat (all w)", (e1_df['dataset'] == 'standard') & (e1_df['pool'] == 'flatten') & (e1_df['attack'] == 'pgd') & (e1_df['norm'] == 'l2')),
]
for idx, (lbl_e1, mask_e1) in enumerate(e1_groups):
    if idx in (2, 4):
        print("-" * 128)
    st_e1 = pool_slices(e1_df[mask_e1])
    sh_lo, sh_hi = wilson_ci(st_e1['SR_shape'], st_e1['n_succ'])
    sc_lo, sc_hi = wilson_ci(st_e1['SR_scale'], st_e1['n_succ'])
    jt_lo, jt_hi = wilson_ci(st_e1['SR_joint'], st_e1['n_succ'])
    print(f"{lbl_e1:<34} | {st_e1['n_succ']:12,d} | "
          f"{st_e1['SR_shape']:5.2f}% [{sh_lo:5.2f},{sh_hi:5.2f}] | "
          f"{st_e1['SR_scale']:5.2f}% [{sc_lo:5.2f},{sc_hi:5.2f}] | "
          f"{st_e1['SR_joint']:5.2f}% [{jt_lo:5.2f},{jt_hi:5.2f}] | "
          f"{st_e1['cohen_w']:8.3f}")
print("=" * 128 + "\n")

# --- Figure 3A: Multi-panel without dual y-axes (Left Top: Stacked SR_s | Left Bottom: cohen_w | Right: PGD-L2 eps_1) ---
fig3a = plt.figure(figsize=(15.6, 6.4), dpi=130)
gs3a = fig3a.add_gridspec(2, 2, height_ratios=[1.55, 1.0], width_ratios=[1.35, 1.0], wspace=0.26, hspace=0.28)

ax_top = fig3a.add_subplot(gs3a[0, 0])
ax_bot = fig3a.add_subplot(gs3a[1, 0], sharex=ax_top)
ax_rgt = fig3a.add_subplot(gs3a[:, 1])

SHIFT_COLORS = {'SR_shape': '#e66101', 'SR_scale': '#5e3c99', 'SR_joint': '#b2abd2'}

x_eps = np.arange(1, 6)
sr_sh_vals = np.array([st['SR_shape'] for _, st, _, _, _, _ in eps_stats])
sr_sc_vals = np.array([st['SR_scale'] for _, st, _, _, _, _ in eps_stats])
sr_jt_vals = np.array([st['SR_joint'] for _, st, _, _, _, _ in eps_stats])
n_s_vals   = [st['n_succ'] for _, st, _, _, _, _ in eps_stats]

bw_eps = 0.56
ax_top.bar(x_eps, sr_sh_vals, bw_eps, label="$\\mathrm{SR}_{\\mathrm{shape}}$ (Shape-Only)", color=SHIFT_COLORS['SR_shape'], edgecolor='#222222')
ax_top.bar(x_eps, sr_sc_vals, bw_eps, bottom=sr_sh_vals, label="$\\mathrm{SR}_{\\mathrm{scale}}$ (Scale-Only)", color=SHIFT_COLORS['SR_scale'], edgecolor='#222222')
ax_top.bar(x_eps, sr_jt_vals, bw_eps, bottom=sr_sh_vals + sr_sc_vals, label="$\\mathrm{SR}_{\\mathrm{joint}}$ (Joint-Shift)", color=SHIFT_COLORS['SR_joint'], edgecolor='#222222')

ax_top.axhline(25.0, color='#111111', linestyle=':', linewidth=1.2, label="Random Ref (25% / 50%)")
ax_top.axhline(50.0, color='#111111', linestyle='--', linewidth=1.2)

for i, x_v in enumerate(x_eps):
    ax_top.text(x_v, 101.8, f"$|\\mathcal{{S}}|={n_s_vals[i]//1000:,}$k", ha='center', va='bottom', fontsize=8.0, fontweight='bold')
    if sr_sh_vals[i] >= 8.0:
        ax_top.text(x_v, sr_sh_vals[i] / 2.0, f"{sr_sh_vals[i]:.1f}%", ha='center', va='center', color='white', fontsize=8.0, fontweight='bold')
    if sr_sc_vals[i] >= 8.0:
        ax_top.text(x_v, sr_sh_vals[i] + sr_sc_vals[i] / 2.0, f"{sr_sc_vals[i]:.1f}%", ha='center', va='center', color='white', fontsize=8.0, fontweight='bold')
    if sr_jt_vals[i] >= 8.0:
        ax_top.text(x_v, sr_sh_vals[i] + sr_sc_vals[i] + sr_jt_vals[i] / 2.0, f"{sr_jt_vals[i]:.1f}%", ha='center', va='center', color='#111111', fontsize=8.0, fontweight='bold')

ax_top.set_ylim(0, 113)
ax_top.set_ylabel("Shift Share Over $\\mathcal{S}$ — $\\mathrm{SR}_s$ (%)", fontsize=10.0)
ax_top.set_title("A1. Shift Typology $\\mathrm{SR}_s$ Across $\\varepsilon_1\\dots\\varepsilon_5$ (Single-Factor $\\to$ 25/25/50)", fontsize=10.6, fontweight='bold')
ax_top.legend(loc='upper right', fontsize=8.0, ncol=2, framealpha=0.92)
ax_top.grid(True, axis='y', linestyle=':', alpha=0.6)

# Left Bottom Panel: Pooled vs Slice-Median cohen_w (with Q25-Q75 IQR band)
w_pool_vals = np.array([st['cohen_w'] for _, st, _, _, _, _ in eps_stats])
w_med_vals  = np.array([med_w for _, _, med_w, _, _, _ in eps_stats])
w_q25_vals  = np.array([q25_w for _, _, _, q25_w, _, _ in eps_stats])
w_q75_vals  = np.array([q75_w for _, _, _, _, q75_w, _ in eps_stats])

ax_bot.fill_between(x_eps, w_q25_vals, w_q75_vals, color='#d95f02', alpha=0.22, label="Slice IQR [Q25–Q75]")
ax_bot.plot(x_eps, w_med_vals, color='#d95f02', marker='s', linestyle='-', linewidth=2.2, markersize=6.0,
            label="Slice-Median $\\mathrm{cohen\\_w}$")
ax_bot.plot(x_eps, w_pool_vals, color='#1b9e77', marker='o', linestyle='--', linewidth=2.2, markersize=6.0,
            label="Pooled $\\mathrm{cohen\\_w}$ (Consistent With Opposite Pooling Biases at $\\varepsilon_1$)")
for i, x_v in enumerate(x_eps):
    ax_bot.text(x_v, w_med_vals[i] + 0.05, f"{w_med_vals[i]:.2f}", ha='center', va='bottom', fontsize=7.8, color='#a63603', fontweight='bold')
    ax_bot.text(x_v, max(0.01, w_pool_vals[i] - 0.11), f"{w_pool_vals[i]:.2f}", ha='center', va='top', fontsize=7.8, color='#006d2c', fontweight='bold')

ax_bot.set_xticks(x_eps)
ax_bot.set_xticklabels([f"$\\varepsilon_{s}$" for s in EPS_STEPS], fontsize=9.5)
ax_bot.set_ylim(-0.05, max(1.15, float(np.nanmax(w_q75_vals)) + 0.18))
ax_bot.set_xlabel("Perturbation Budget Step ($\\varepsilon_1 \\to \\varepsilon_5$)", fontsize=10.0)
ax_bot.set_ylabel("Effect Size $\\mathrm{cohen\\_w}$", fontsize=10.0)
ax_bot.set_title("A2. Deviation From Random Reference ($\\mathrm{cohen\\_w}$)", fontsize=10.4, fontweight='bold')
ax_bot.grid(True, linestyle=':', alpha=0.6)
ax_bot.legend(loc='upper right', fontsize=8.0, framealpha=0.92)

# Right Panel: All 4 subgroups {gap, flat} x {L_inf, L_2} at eps_1
st_gap_li_e1 = pool_slices(e1_df[(e1_df['pool'] == 'gap')     & (e1_df['norm'] == 'linf')])
st_flt_li_e1 = pool_slices(e1_df[(e1_df['pool'] == 'flatten') & (e1_df['norm'] == 'linf')])
st_gap_l2_e1 = pool_slices(e1_df[(e1_df['pool'] == 'gap')     & (e1_df['norm'] == 'l2')])
st_flt_l2_e1 = pool_slices(e1_df[(e1_df['pool'] == 'flatten') & (e1_df['norm'] == 'l2')])

r_stats = [st_gap_li_e1, st_flt_li_e1, st_gap_l2_e1, st_flt_l2_e1]
x_r = np.arange(len(r_stats))
bw_r = 0.54
r_labels = [
    f"gap $\\times L_\\infty$\n$|\\mathcal{{S}}|={st_gap_li_e1['n_succ']:,}$\n$w={st_gap_li_e1['cohen_w']:.2f}$",
    f"flat $\\times L_\\infty$\n$|\\mathcal{{S}}|={st_flt_li_e1['n_succ']:,}$\n$w={st_flt_li_e1['cohen_w']:.2f}$",
    f"gap $\\times L_2$\n$|\\mathcal{{S}}|={st_gap_l2_e1['n_succ']:,}$\n$w={st_gap_l2_e1['cohen_w']:.2f}$",
    f"flat $\\times L_2$\n$|\\mathcal{{S}}|={st_flt_l2_e1['n_succ']:,}$\n$w={st_flt_l2_e1['cohen_w']:.2f}$",
]
sh_r = np.array([s['SR_shape'] for s in r_stats])
sc_r = np.array([s['SR_scale'] for s in r_stats])
jt_r = np.array([s['SR_joint'] for s in r_stats])

ax_rgt.bar(x_r, sh_r, bw_r, label="$\\mathrm{SR}_{\\mathrm{shape}}$", color=SHIFT_COLORS['SR_shape'], edgecolor='#222222')
ax_rgt.bar(x_r, sc_r, bw_r, bottom=sh_r, label="$\\mathrm{SR}_{\\mathrm{scale}}$", color=SHIFT_COLORS['SR_scale'], edgecolor='#222222')
ax_rgt.bar(x_r, jt_r, bw_r, bottom=sh_r + sc_r, label="$\\mathrm{SR}_{\\mathrm{joint}}$", color=SHIFT_COLORS['SR_joint'], edgecolor='#222222')

for idx_b, st_b in enumerate(r_stats):
    cum_b = 0.0
    for s_key, col_txt in [('SR_shape', 'white'), ('SR_scale', 'white'), ('SR_joint', '#111111')]:
        val_b = st_b[s_key]
        lo_b, hi_b = wilson_ci(val_b, st_b['n_succ'])
        if val_b >= 7.0:
            ax_rgt.text(
                idx_b, cum_b + val_b / 2.0,
                f"{val_b:.1f}%\n[{lo_b:.1f},{hi_b:.1f}]",
                ha='center', va='center', color=col_txt, fontsize=7.6, fontweight='bold',
            )
        cum_b += val_b

ax_rgt.axvline(1.5, color='#333333', linestyle='--', linewidth=1.1)
ax_rgt.set_xticks(x_r)
ax_rgt.set_xticklabels(r_labels, fontsize=8.4, fontweight='bold')
ax_rgt.set_ylim(0, 108)
ax_rgt.set_ylabel("Shift Share Over $\\mathcal{S}$ — $\\mathrm{SR}_s$ (%)", fontsize=10.0)
ax_rgt.set_title("B. Initial Shift Breakdown at $\\varepsilon_1$: `{gap, flat}` $\\times$ `{$L_\\infty, L_2$}`", fontsize=10.4, fontweight='bold')
ax_rgt.grid(True, axis='y', linestyle=':', alpha=0.6)
ax_rgt.legend(loc='upper right', fontsize=8.0, framealpha=0.92)

plt.tight_layout()
save_and_show(fig3a, "Figure 3A: Node 1 Shift Typology Across Budget Steps, Cohen's w, and Pooling x Norm Breakdown at eps_1")

# --- Load Clean Test Images Once + Helper to Fetch Slice Evaluation & Adversarial Images for Image Strips ---
with np.load(MY_DSPRITES_SPLIT) as _split_data:
    X_CLEAN_TEST = _split_data['X_test'].astype(np.float32)

def load_slice_eval_and_adv(dataset, pool, weight, attack, norm, eps_val):
    """Return (df_eps_indexed_by_sample, X_adv_array) for a specific (m, a) slice."""
    csv_path = get_attack_eval_paths(dataset, pool, attack, norm, weight)
    df_full = pd.read_csv(csv_path, compression='gzip')
    df_eps = df_full[np.isclose(df_full['epsilon'], eps_val)].sort_values('sample_index').reset_index(drop=True)
    adv_path = get_untargeted_attack_path(dataset, pool, attack, norm, eps_val, weight)
    with np.load(adv_path) as adv_npz:
        X_adv = adv_npz['X_test'].astype(np.float32)
    return df_eps, X_adv

# --- Figure 3B: Deterministically Sampled Image Strip (Scale-Shift RA in std|gap vs Boundary Shape-Shift RR in std|flat) ---
rng_s2 = np.random.default_rng(29)

df_gap_s2, X_adv_gap_s2 = load_slice_eval_and_adv('standard', 'gap',     1.0, 'pgd', 'l2', EPS_L2[1])
df_flt_s2, X_adv_flt_s2 = load_slice_eval_and_adv('standard', 'flatten', 1.0, 'pgd', 'l2', EPS_L2[1])

# Candidate pool 1: Scale-shift in dominant state RA in std|gap|w=1.0 (PGD-L2, eps_1=0.10)
y_m_g  = df_gap_s2['y_shape_and_scale'].to_numpy()
p_m_g  = df_gap_s2['pred_main'].to_numpy()
p_sh_g = df_gap_s2['pred_shape'].to_numpy()
p_sc_g = df_gap_s2['pred_scale'].to_numpy()
cand_sc_all = np.where((p_m_g != y_m_g) & ((p_m_g // 3) == (y_m_g // 3)) & ((p_m_g % 3) != (y_m_g % 3)))[0]
cand_scale_shift = np.where(
    (p_m_g != y_m_g)
    & ((p_m_g // 3) == (y_m_g // 3)) & ((p_m_g % 3) != (y_m_g % 3))
    & (p_sh_g == (y_m_g // 3)) & (p_sc_g == (p_m_g % 3))
)[0]
idx_pick_gap = int(rng_s2.choice(cand_scale_shift))

# Candidate pool 2: Shape-shift in std|flat|w=1.0 (PGD-L2, eps_1=0.10) — majority state is RR (73.7% RR vs 26.3% AR)
y_m_f  = df_flt_s2['y_shape_and_scale'].to_numpy()
p_m_f  = df_flt_s2['pred_main'].to_numpy()
p_sh_f = df_flt_s2['pred_shape'].to_numpy()
p_sc_f = df_flt_s2['pred_scale'].to_numpy()
cand_sh_all = np.where((p_m_f != y_m_f) & ((p_m_f // 3) != (y_m_f // 3)) & ((p_m_f % 3) == (y_m_f % 3)))[0]
cand_sh_rr  = np.where(
    (p_m_f != y_m_f)
    & ((p_m_f // 3) != (y_m_f // 3)) & ((p_m_f % 3) == (y_m_f % 3))
    & (p_sh_f == (y_m_f // 3)) & (p_sc_f == (y_m_f % 3))
)[0]
cand_sh_ar  = np.where(
    (p_m_f != y_m_f)
    & ((p_m_f // 3) != (y_m_f // 3)) & ((p_m_f % 3) == (y_m_f % 3))
    & (p_sh_f == (p_m_f // 3)) & (p_sc_f == (y_m_f % 3))
)[0]
idx_pick_flt = int(rng_s2.choice(cand_sh_rr))

print("[Figure 3B Deterministic Sampling Log — np.random.default_rng(29)]:")
print(f"  Row 1 (Scale-Shift | std|gap|w=1.0  | PGD-L2 eps_1=0.10): total scale-shifts = {len(cand_sc_all):,}, "
      f"dominant state RA = {len(cand_scale_shift):,} ({len(cand_scale_shift)/max(1,len(cand_sc_all))*100:.1f}%) -> selected sample_index = {idx_pick_gap}")
print(f"  Row 2 (Shape-Shift | std|flat|w=1.0 | PGD-L2 eps_1=0.10): total shape-shifts = {len(cand_sh_all):,}, "
      f"majority state RR (boundary retention) = {len(cand_sh_rr):,} ({len(cand_sh_rr)/max(1,len(cand_sh_all))*100:.1f}%) vs "
      f"minority AR = {len(cand_sh_ar):,} ({len(cand_sh_ar)/max(1,len(cand_sh_all))*100:.1f}%) -> selected RR sample_index = {idx_pick_flt}\n")

fig3b, axes3b = plt.subplots(2, 3, figsize=(12.4, 6.8), dpi=130)

strip_cases_s2 = [
    ("Row 1: Scale-Shift, Dominant State $\\mathrm{RA}$ (`std|gap|w=1.0`, PGD-$L_2$, $\\varepsilon_1=0.10$)", df_gap_s2.iloc[idx_pick_gap], X_CLEAN_TEST[idx_pick_gap], X_adv_gap_s2[idx_pick_gap], idx_pick_gap),
    ("Row 2: Shape-Shift, Majority State $\\mathrm{RR}$ (73.7% Boundary Retention in `std|flat|w=1.0`, PGD-$L_2$, $\\varepsilon_1=0.10$)", df_flt_s2.iloc[idx_pick_flt], X_CLEAN_TEST[idx_pick_flt], X_adv_flt_s2[idx_pick_flt], idx_pick_flt),
]

for row_i, (row_title, r_s, img_clean, img_adv, s_idx) in enumerate(strip_cases_s2):
    y_true_c = int(r_s['y_shape_and_scale'])
    p_main_c = int(r_s['pred_main'])
    p_sh_c   = int(r_s['pred_shape'])
    p_sc_c   = int(r_s['pred_scale'])
    abs_diff = np.abs(img_adv - img_clean)
    delta_20x = np.clip(abs_diff * 20.0, 0.0, 1.0)
    l2_norm  = float(np.linalg.norm(img_adv - img_clean))

    axes3b[row_i, 0].imshow(img_clean, cmap='gray', vmin=0.0, vmax=1.0)
    axes3b[row_i, 0].set_title(
        f"{row_title}\nClean (idx={s_idx}): True = {CLASS_NAMES_9[y_true_c]}",
        fontsize=8.6, fontweight='bold',
    )
    axes3b[row_i, 0].axis('off')

    axes3b[row_i, 1].imshow(img_adv, cmap='gray', vmin=0.0, vmax=1.0)
    axes3b[row_i, 1].set_title(
        f"Adversarial ($\\|\\delta\\|_2={l2_norm:.3f}$)\n"
        f"Main: {CLASS_NAMES_9[p_main_c]} ({r_s['confidence']*100:.1f}%)\n"
        f"Aux Shape: {SHAPE_NAMES[p_sh_c]} ({r_s['confidence_shape']*100:.1f}%) | "
        f"Aux Scale: {SCALE_NAMES[p_sc_c]} ({r_s['confidence_scale']*100:.1f}%)",
        fontsize=8.6, fontweight='bold',
    )
    axes3b[row_i, 1].axis('off')

    im_d = axes3b[row_i, 2].imshow(delta_20x, cmap='magma', vmin=0.0, vmax=1.0)
    axes3b[row_i, 2].set_title(
        f"Perturbation Magnitude $|\\delta| \\times 20$\n(max $|\\delta| = {np.max(abs_diff):.3f}$)",
        fontsize=9.0, fontweight='bold',
    )
    axes3b[row_i, 2].axis('off')
    plt.colorbar(im_d, ax=axes3b[row_i, 2], fraction=0.046, pad=0.04)

plt.tight_layout()
save_and_show(fig3b, "Figure 3B: Deterministic Sample Strip — Scale-Shift (Dominant State RA in std|gap) vs Boundary Shape-Shift (Majority State RR in std|flat)")
# %%



# %% PART B — Section B0: Preliminary Check — What Happens If the Attack Fails?
print("=" * 128)
print("=== PART B — SECTION B0: PRELIMINARY CHECK — WHAT HAPPENS IF THE ATTACK FAILS? (ON S^c WHERE main IS CORRECT) ===")
print("=" * 128)
print("Observed result:")
print("  - On failed attacks (main correct), both auxiliary branches usually stay correct, but a minority of samples show")
print("    'silent auxiliary corruption', mostly on the scale head.")
print("  - This corruption grows steadily with the budget and is larger under L_inf, standard training (`std`) and")
print("    single-step FGSM.")
print("-" * 128)

b0_groups = [
    ("GLOBAL ALL (240 slices)", tree_df['n_total'] > 0),
    ("Budget: eps_1",           tree_df['eps_step'] == 1),
    ("Budget: eps_2",           tree_df['eps_step'] == 2),
    ("Budget: eps_3",           tree_df['eps_step'] == 3),
    ("Budget: eps_4",           tree_df['eps_step'] == 4),
    ("Budget: eps_5",           tree_df['eps_step'] == 5),
    ("Norm: L_inf",             tree_df['norm'] == 'linf'),
    ("Norm: L_2",               tree_df['norm'] == 'l2'),
    ("Dataset: std",            tree_df['dataset'] == 'standard'),
    ("Dataset: con",            tree_df['dataset'] == 'contrast'),
    ("Optimizer: FGSM",         tree_df['attack'] == 'fgsm'),
    ("Optimizer: PGD",          tree_df['attack'] == 'pgd'),
]

print(f"{'Subgroup on S^c (Failed Attacks)':<28} | {'|S^c| (Failed N)':>16} | {'Shape Aux OK [95% CI]':>23} | "
      f"{'Scale Aux OK [95% CI]':>23} | {'Both Aux OK [95% CI]':>23} | {'Silent Corr. [95% CI]':>23}")
print("-" * 148)
for idx, (b_lbl, b_mask) in enumerate(b0_groups):
    if idx in (1, 6, 8, 10):
        print("-" * 148)
    st_b = pool_slices(tree_df[b_mask])
    nf = st_b['n_fail']
    sh_lo, sh_hi = wilson_ci(st_b['AuxShapeOK_fail'], nf)
    sc_lo, sc_hi = wilson_ci(st_b['AuxScaleOK_fail'], nf)
    bo_lo, bo_hi = wilson_ci(st_b['AuxOK_fail'],      nf)
    si_lo, si_hi = wilson_ci(st_b['SilentCorr_fail'], nf)
    print(f"{b_lbl:<28} | {nf:16,d} | "
          f"{st_b['AuxShapeOK_fail']:6.2f}% [{sh_lo:5.2f},{sh_hi:5.2f}] | "
          f"{st_b['AuxScaleOK_fail']:6.2f}% [{sc_lo:5.2f},{sc_hi:5.2f}] | "
          f"{st_b['AuxOK_fail']:6.2f}% [{bo_lo:5.2f},{bo_hi:5.2f}] | "
          f"{st_b['SilentCorr_fail']:6.2f}% [{si_lo:5.2f},{si_hi:5.2f}]")
print("=" * 128 + "\n")


# %% PART B — Section 3: Do the Auxiliaries Follow the Main?
print("=" * 128)
print("=== PART B — SECTION 3: DO THE AUXILIARIES FOLLOW THE MAIN? (NODE 2: FULL ALIGNMENT & COMPOSITE REGIMES) ===")
print("=" * 128)
print("Observed result:")
print("  - In most successful attacks the auxiliary branches follow `main` (Full Alignment is the dominant regime).")
print("  - In single-factor shifts the shifted auxiliary aligns with `main` while the unshifted one retains the truth;")
print("    this is consistent with a partially factorised trunk but does not distinguish it from holistic class prototypes.")
print("  - When the shifted auxiliary errs, it almost always picks the same wrong class as `main`, far above the 50%")
print("    conditional random wrong-class chance.")
print("  - Alignment is higher overall under `gap` than `flat` and lowest in `con | flat`, but it varies substantially")
print("    within pools (e.g. `con | gap` drops sharply as w increases from 0.5 to 2.0).")
print("  - The aligned state is the most frequent one in most of the 48 (model x attack-norm) configurations; the Wilson")
print("    star criterion is passed almost everywhere because large N makes CI_low > 1/|K_s| easy, so frequencies are")
print("    more informative.")
print("-" * 128)

g_all = pool_slices(tree_df)

# 1. Pooled rates of the 4 mutually exclusive, exhaustive composite regimes over S
sum_4_regimes = g_all['FAR'] + g_all['FRR'] + g_all['PRR'] + g_all['DR']
print("\n[1] Global Partition of Successful Attacks S Into the 4 Mutually Exclusive Composite Regimes:")
print(f"{'Composite Regime':<36} | {'Event Count (k / |S|)':>28} | {'Pooled Rate (%) [95% Wilson CI]':>32} | {'Definition Summary'}")
print("-" * 134)
for r_code, r_name, k_key, desc_s in [
    ('FAR', 'Full Alignment Regime',     'n_far', 'Shifted aux(s) follow main; unshifted aux (if any) retains truth'),
    ('FRR', 'Full Retention Regime',     'n_frr', 'Both aux branches retain ground truth despite main error (RR)'),
    ('PRR', 'Partial Retention Regime',  'n_prr', 'Joint-shift where one aux aligns with main and one retains truth (RA+AR)'),
    ('DR',  'Divergence Regime',         'n_dr',  'At least one aux predicts a class != truth and != main-implied factor'),
]:
    cnt_r = g_all[k_key]
    val_r = g_all[r_code]
    lo_r, hi_r = wilson_ci(val_r, g_all['n_succ'])
    label_str = f"{r_code} ({r_name})"
    ratio_str = f"{cnt_r:,} / {g_all['n_succ']:,}"

    print(
        f"{label_str:<36} | {ratio_str:>28} | "
        f"{val_r:6.2f}% [{lo_r:6.2f}%, {hi_r:6.2f}%] | {desc_s}"
    )
print("-" * 134)
print(f"  Exhaustiveness Check: FAR + FRR + PRR + DR = {sum_4_regimes:.6f}%  "
      f"(Sum of counts = {g_all['n_far'] + g_all['n_frr'] + g_all['n_prr'] + g_all['n_dr']:,} == |S| = {g_all['n_succ']:,})")

# 2. FAR_s per shift type compared against uniform 1/|K_s| + conditional wrong-class alignment given auxiliary error
print("\n[2] Shift-Specific Full Alignment (FAR_s) and Conditional Alignment Given Auxiliary Error on the Shifted Factor:")
print("  Benchmark (a): Uniform state-space chance 1/|K_s| (1/6 = 16.67% for shape/scale; 1/9 = 11.11% for joint) — compares to FAR_s.")
print("  Benchmark (b): Conditional wrong-class chance given the shifted auxiliary errs (50.00% for 1 factor; 25.00% for 2 factors)")
print("                 — compares to P(Aux aligns with Main | Aux errs on shifted factor), NOT directly to unconditional FAR_s.")
print(f"{'Shift Type & Aligned State':<34} | {'FAR_s [95% Wilson CI]':>24} | {'1/|K_s|':>8} || "
      f"{'P(Align | Aux Errs on Shifted)':>32} | {'Cond. Chance':>12}")
print("-" * 124)
sh_err_align = (g_all['n_shape_AR'] + g_all['n_shape_AD']) / max(1, g_all['n_sh'] - g_all['n_shape_RR'] - g_all['n_shape_RD']) * 100.0
sc_err_align = (g_all['n_scale_RA'] + g_all['n_scale_DA']) / max(1, g_all['n_sc'] - g_all['n_scale_RR'] - g_all['n_scale_DR']) * 100.0
jt_err_align = g_all['n_joint_AA'] / max(1, g_all['n_joint_AA'] + g_all['n_joint_AD'] + g_all['n_joint_DA'] + g_all['n_joint_DD']) * 100.0
for s_lbl, k_key, n_key, r_key, unif_c, cond_emp, cond_c in [
    ("Shape-Shift: FAR_shape (ARR_shape,AR)", 'n_shape_AR', 'n_sh', 'FAR_shape', 100.0 / 6.0, sh_err_align, 50.0),
    ("Scale-Shift: FAR_scale (ARR_scale,RA)", 'n_scale_RA', 'n_sc', 'FAR_scale', 100.0 / 6.0, sc_err_align, 50.0),
    ("Joint-Shift: FAR_joint (ARR_joint,AA)", 'n_joint_AA', 'n_jt', 'FAR_joint', 100.0 / 9.0, jt_err_align, 25.0),
]:
    v_s = g_all[r_key]
    lo_s, hi_s = wilson_ci(v_s, g_all[n_key])
    print(f"{s_lbl:<34} | {v_s:6.2f}% [{lo_s:5.2f}%,{hi_s:5.2f}%] | {unif_c:7.2f}% || "
          f"{cond_emp:31.2f}% | {cond_c:11.2f}%")
print("-" * 124)

# 3. FAR_s and overall FAR by pooling (gap vs flat), dataset (std vs con), and each of the 12 models
print("\n[3] Full Alignment (FAR_s & Overall FAR) and Composite Regimes by Pooling, Dataset, and All 12 Models:")
print(f"{'Group / Model':<18} | {'|S| (Succ.)':>11} | {'FAR_shape (|S_sh|)':>24} | {'FAR_scale (|S_sc|)':>24} | "
      f"{'FAR_joint (|S_jt|)':>24} || {'FAR [95% CI]':>19} | {'FRR%':>6} | {'PRR%':>6} | {'DR%':>6}")
print("-" * 162)

summary_groups_s3 = [
    ("Pool: gap",  tree_df['pool'] == 'gap'),
    ("Pool: flat", tree_df['pool'] == 'flatten'),
    ("Data: std",  tree_df['dataset'] == 'standard'),
    ("Data: con",  tree_df['dataset'] == 'contrast'),
]
for idx, (g_lbl, g_mask) in enumerate(summary_groups_s3):
    if idx == 2:
        print("-" * 162)
    st_g = pool_slices(tree_df[g_mask])
    lo_f, hi_f = wilson_ci(st_g['FAR'], st_g['n_succ'])
    print(f"{g_lbl:<18} | {st_g['n_succ']:11,d} | "
          f"{st_g['FAR_shape']:6.2f}% ({st_g['n_sh']:8,d}) | "
          f"{st_g['FAR_scale']:6.2f}% ({st_g['n_sc']:8,d}) | "
          f"{st_g['FAR_joint']:6.2f}% ({st_g['n_jt']:8,d}) || "
          f"{st_g['FAR']:5.2f}% [{lo_f:4.1f},{hi_f:4.1f}] | "
          f"{st_g['FRR']:5.2f}% | {st_g['PRR']:5.2f}% | {st_g['DR']:5.2f}%")
print("-" * 162)

model_pooled_stats = []
for m_idx, ((d, p, w), m_lbl) in enumerate(zip(MODELS, MODEL_LABELS)):
    if m_idx in (3, 6, 9):
        print("-" * 162)
    st_m = pool_slices(tree_df[(tree_df['dataset'] == d) & (tree_df['pool'] == p) & (tree_df['weight'] == w)])
    model_pooled_stats.append((m_lbl, st_m))
    lo_f, hi_f = wilson_ci(st_m['FAR'], st_m['n_succ'])
    print(f"{m_lbl:<18} | {st_m['n_succ']:11,d} | "
          f"{st_m['FAR_shape']:6.2f}% ({st_m['n_sh']:8,d}) | "
          f"{st_m['FAR_scale']:6.2f}% ({st_m['n_sc']:8,d}) | "
          f"{st_m['FAR_joint']:6.2f}% ({st_m['n_jt']:8,d}) || "
          f"{st_m['FAR']:5.2f}% [{lo_f:4.1f},{hi_f:4.1f}] | "
          f"{st_m['FRR']:5.2f}% | {st_m['PRR']:5.2f}% | {st_m['DR']:5.2f}%")
print("-" * 162)

# 4. Dominance count out of the 48 (model x attack-norm) configurations (pooled across the 5 eps steps)
dom_counts = {s: {'plurality': 0, 'criterion_2star': 0, 'valid_n': 0} for s in SHIFT_TYPES}
for (d, p, w) in MODELS:
    for (a, n) in ATTACK_NORMS:
        st_48 = pool_slices(tree_df[(tree_df['dataset'] == d) & (tree_df['pool'] == p)
                                    & (tree_df['weight'] == w) & (tree_df['attack'] == a) & (tree_df['norm'] == n)])
        for s_name, n_key, k_space, ali_code in [
            ('shape', 'n_sh', K_SHAPE, 'AR'),
            ('scale', 'n_sc', K_SCALE, 'RA'),
            ('joint', 'n_jt', K_JOINT, 'AA'),
        ]:
            vals_d = {code: st_48[f'ARR_{s_name}_{code}'] for code in k_space}
            if st_48[n_key] >= N_MIN_BRANCH:
                dom_counts[s_name]['valid_n'] += 1
            top_code = max(vals_d.items(), key=lambda x: x[1])[0]
            if top_code == ali_code and st_48[n_key] > 0:
                dom_counts[s_name]['plurality'] += 1
            tiers = classify_wilson_dominance(vals_d, st_48[n_key])
            if tiers[ali_code] == 'primary':
                dom_counts[s_name]['criterion_2star'] += 1

print("\n[4] Dominance Counts Out of the 48 (12 Models x 4 Attack-Norm) Configurations (Pooled Over eps_1..eps_5):")
for s_name, ali_code in [('shape', 'AR'), ('scale', 'RA'), ('joint', 'AA')]:
    dc = dom_counts[s_name]
    print(f"  {s_name.capitalize()}-Shift Aligned State ({ali_code}) : "
          f"Plurality #1 in {dc['plurality']:2d}/48 configs  |  "
          f"Wilson Criterion ★★ (CI_low > 1/|K_s|) in {dc['criterion_2star']:2d}/48 configs "
          f"(valid N >= {N_MIN_BRANCH} in {dc['valid_n']}/48)")
print("=" * 128 + "\n")

# --- Figure 4A: Single Global 3-Stage 4-Regime Sankey Diagram (Node 0 -> Node 1 -> Node 2 Composite Regimes) ---
fig4a, ax_sk = plt.subplots(figsize=(15.4, 7.2), dpi=130)
ax_sk.set_xlim(0, 100)
ax_sk.set_ylim(0, 100)
ax_sk.axis('off')

def _draw_bezier_ribbon(ax, x0, y0_b, y0_t, x1, y1_b, y1_t, color, alpha=0.48):
    if (y0_t - y0_b) <= 0.02 and (y1_t - y1_b) <= 0.02:
        return
    xm = 0.5 * (x0 + x1)
    verts = [
        (x0, y0_t), (xm, y0_t), (xm, y1_t), (x1, y1_t),
        (x1, y1_b), (xm, y1_b), (xm, y0_b), (x0, y0_b), (x0, y0_t),
    ]
    codes = [
        MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY,
    ]
    ax.add_patch(mpatches.PathPatch(MplPath(verts, codes), facecolor=color, edgecolor='none', alpha=alpha))

ax_sk.text(
    50.0, 96.5,
    f"Global 3-Stage Conditional Probability Tree (Pooled Over All 240 Slices, $N={g_all['n_total']:,}$; "
    f"Weighted Toward Higher-$|\\mathcal{{S}}|$ Slices)",
    ha='center', va='center', fontsize=11.2, fontweight='bold', color='#111111',
)
ax_sk.text(10.0, 90.5, "NODE 0: Attack Outcome\n(All Test Evaluations $N$)", ha='center', va='center', fontsize=9.8, fontweight='bold')
ax_sk.text(46.0, 90.5, "NODE 1: Main Shift Typology\n(Over Successful Attacks $\\mathcal{S}$)", ha='center', va='center', fontsize=9.8, fontweight='bold')
ax_sk.text(84.0, 90.5, "NODE 2: Auxiliary Composite Regimes\n(4 Mutually Exclusive Regimes on $\\mathcal{S}$)", ha='center', va='center', fontsize=9.8, fontweight='bold')

# Stage 0 geometry (Failed S^c on top, Successful S on bottom)
y_bot, y_top = 10.0, 84.0
H_tot = y_top - y_bot
x0_l, x0_r = 6.5, 13.5
h_succ = H_tot * (g_all['ASR'] / 100.0)
h_fail = H_tot - h_succ
y0_s_b, y0_s_t = y_bot, y_bot + h_succ
y0_f_b, y0_f_t = y0_s_t, y_top

ax_sk.add_patch(mpatches.Rectangle((x0_l, y0_f_b), x0_r - x0_l, h_fail, facecolor='#cfd8dc', edgecolor='#455a64', lw=1.5))
ax_sk.text((x0_l + x0_r) / 2, (y0_f_b + y0_f_t) / 2,
           f"Failed ($\\mathcal{{S}}^c$)\n{g_all['Acc_main']:.1f}%\n($N={g_all['n_fail']:,}$)\n[Aux OK: {g_all['AuxOK_fail']:.1f}%]",
           ha='center', va='center', fontsize=8.5, fontweight='bold', color='#263238')

ax_sk.add_patch(mpatches.Rectangle((x0_l, y0_s_b), x0_r - x0_l, h_succ, facecolor='#ffb74d', edgecolor='#e65100', lw=1.5))
ax_sk.text((x0_l + x0_r) / 2, (y0_s_b + y0_s_t) / 2,
           f"Successful ($\\mathcal{{S}}$)\n$\\mathrm{{ASR}}={g_all['ASR']:.1f}\\%$\n($|\\mathcal{{S}}|={g_all['n_succ']:,}$)",
           ha='center', va='center', fontsize=8.8, fontweight='bold', color='#111111')

# Stage 1 geometry (Joint-Shift bottom, Scale-Shift middle, Shape-Shift top)
x1_l, x1_r = 42.5, 49.5
gap1 = 3.2
H1_use = H_tot - 2 * gap1
h1_jt = H1_use * (g_all['SR_joint'] / 100.0)
h1_sc = H1_use * (g_all['SR_scale'] / 100.0)
h1_sh = H1_use * (g_all['SR_shape'] / 100.0)

y1_jt_b, y1_jt_t = y_bot, y_bot + h1_jt
y1_sc_b, y1_sc_t = y1_jt_t + gap1, y1_jt_t + gap1 + h1_sc
y1_sh_b, y1_sh_t = y1_sc_t + gap1, y1_sc_t + gap1 + h1_sh

# Source slices on Node 0 Successful block
s0_jt_b, s0_jt_t = y0_s_b, y0_s_b + h_succ * (g_all['SR_joint'] / 100.0)
s0_sc_b, s0_sc_t = s0_jt_t, s0_jt_t + h_succ * (g_all['SR_scale'] / 100.0)
s0_sh_b, s0_sh_t = s0_sc_t, y0_s_t

n1_specs = [
    ('shape', 'Shape-Shift ($\\mathcal{S}_{\\mathrm{shape}}$)', g_all['SR_shape'], g_all['n_sh'], s0_sh_b, s0_sh_t, y1_sh_b, y1_sh_t, '#fdae61', '#d95f02'),
    ('scale', 'Scale-Shift ($\\mathcal{S}_{\\mathrm{scale}}$)', g_all['SR_scale'], g_all['n_sc'], s0_sc_b, s0_sc_t, y1_sc_b, y1_sc_t, '#b2abd2', '#5e3c99'),
    ('joint', 'Joint-Shift ($\\mathcal{S}_{\\mathrm{joint}}$)', g_all['SR_joint'], g_all['n_jt'], s0_jt_b, s0_jt_t, y1_jt_b, y1_jt_t, '#90caf9', '#1565c0'),
]

for s_key, s_lbl, sr_v, n_v, y0b, y0t, y1b, y1t, fc, ec in n1_specs:
    _draw_bezier_ribbon(ax_sk, x0_r, y0b, y0t, x1_l, y1b, y1t, fc, alpha=0.50)
    ax_sk.add_patch(mpatches.Rectangle((x1_l, y1b), x1_r - x1_l, y1t - y1b, facecolor=fc, edgecolor=ec, lw=1.5))
    ax_sk.text((x1_l + x1_r) / 2, (y1b + y1t) / 2,
               f"{s_lbl}\n{sr_v:.1f}%\n($N={n_v:,}$)",
               ha='center', va='center', fontsize=8.4, fontweight='bold', color='#111111')

# Stage 2 geometry: 4 composite regimes (DR bottom, PRR, FRR, FAR top)
x2_l, x2_r = 79.0, 86.0
gap2 = 2.6
H2_use = H_tot - 3 * gap2
# Give small regimes a minimum visual height while keeping labels exact
raw_h2 = {r: H2_use * (g_all[r] / 100.0) for r in ['DR', 'PRR', 'FRR', 'FAR']}
min_h = 3.2
adj_h2 = {r: max(min_h, raw_h2[r]) for r in raw_h2}
scale_h2 = H2_use / sum(adj_h2.values())
adj_h2 = {r: v * scale_h2 for r, v in adj_h2.items()}

y2_bounds = {}
cur_y2 = y_bot
for r_code in ['DR', 'PRR', 'FRR', 'FAR']:
    y2_bounds[r_code] = (cur_y2, cur_y2 + adj_h2[r_code])
    cur_y2 += adj_h2[r_code] + gap2

# Decompose each shift block's contribution to (FAR, FRR, PRR, DR)
shift_regime_counts = {
    'shape': {
        'FAR': g_all['n_shape_AR'],
        'FRR': g_all['n_shape_RR'],
        'PRR': 0,
        'DR':  g_all['n_shape_AD'] + g_all['n_shape_RD'] + g_all['n_shape_DR'] + g_all['n_shape_DD'],
    },
    'scale': {
        'FAR': g_all['n_scale_RA'],
        'FRR': g_all['n_scale_RR'],
        'PRR': 0,
        'DR':  g_all['n_scale_RD'] + g_all['n_scale_DA'] + g_all['n_scale_DR'] + g_all['n_scale_DD'],
    },
    'joint': {
        'FAR': g_all['n_joint_AA'],
        'FRR': g_all['n_joint_RR'],
        'PRR': g_all['n_joint_RA'] + g_all['n_joint_AR'],
        'DR':  (g_all['n_joint_AD'] + g_all['n_joint_RD'] + g_all['n_joint_DA']
                + g_all['n_joint_DR'] + g_all['n_joint_DD']),
    },
}

# Track current top offsets on Node 1 blocks and Node 2 blocks (filling top-down: shape -> scale -> joint)
n1_top_cursor = {'shape': y1_sh_t, 'scale': y1_sc_t, 'joint': y1_jt_t}
n1_heights    = {'shape': y1_sh_t - y1_sh_b, 'scale': y1_sc_t - y1_sc_b, 'joint': y1_jt_t - y1_jt_b}
n1_denoms     = {'shape': g_all['n_sh'], 'scale': g_all['n_sc'], 'joint': g_all['n_jt']}

n2_top_cursor = {r: y2_bounds[r][1] for r in ['FAR', 'FRR', 'PRR', 'DR']}
n2_heights    = {r: y2_bounds[r][1] - y2_bounds[r][0] for r in ['FAR', 'FRR', 'PRR', 'DR']}
n2_denoms     = {'FAR': g_all['n_far'], 'FRR': g_all['n_frr'], 'PRR': g_all['n_prr'], 'DR': g_all['n_dr']}

for s_key in ['shape', 'scale', 'joint']:
    for r_code in ['FAR', 'FRR', 'PRR', 'DR']:
        cnt_sr = shift_regime_counts[s_key][r_code]
        if cnt_sr <= 0:
            continue
        dh1 = n1_heights[s_key] * (cnt_sr / n1_denoms[s_key])
        dh2 = n2_heights[r_code] * (cnt_sr / n2_denoms[r_code])
        y1_seg_t = n1_top_cursor[s_key]
        y1_seg_b = y1_seg_t - dh1
        n1_top_cursor[s_key] = y1_seg_b

        y2_seg_t = n2_top_cursor[r_code]
        y2_seg_b = y2_seg_t - dh2
        n2_top_cursor[r_code] = y2_seg_b

        _draw_bezier_ribbon(ax_sk, x1_r, y1_seg_b, y1_seg_t, x2_l, y2_seg_b, y2_seg_t, REGIME_COLORS[r_code], alpha=0.46)

regime_titles = {
    'FAR': f"Full Alignment (FAR)\n{g_all['FAR']:.1f}% ($N={g_all['n_far']:,}$)",
    'FRR': f"Full Retention (FRR)\n{g_all['FRR']:.1f}% ($N={g_all['n_frr']:,}$)",
    'PRR': f"Partial Retention (PRR)\n{g_all['PRR']:.1f}% ($N={g_all['n_prr']:,}$)",
    'DR':  f"Divergence (DR)\n{g_all['DR']:.1f}% ($N={g_all['n_dr']:,}$)",
}
for r_code in ['FAR', 'FRR', 'PRR', 'DR']:
    yb, yt = y2_bounds[r_code]
    ax_sk.add_patch(mpatches.Rectangle((x2_l, yb), x2_r - x2_l, yt - yb,
                                       facecolor=REGIME_COLORS[r_code], edgecolor='#111111', lw=1.5))
    ax_sk.text(x2_r + 1.0, (yb + yt) / 2, regime_titles[r_code],
               ha='left', va='center', fontsize=8.8, fontweight='bold', color='#111111')

plt.tight_layout()
save_and_show(fig4a, "Figure 4A: Global 3-Stage Conditional Probability Sankey Diagram (Stopped at the 4 Composite Regimes)")

# --- Figure 4B: 2-Panel (Panel A: 12-Model Stacked Composite Regimes | Panel B: FAR_s & FAR for gap vs flat) ---
fig4b, (ax_m12, ax_gf) = plt.subplots(
    1, 2, figsize=(16.4, 5.8), dpi=130, gridspec_kw={'width_ratios': [1.45, 1.0]}
)

# Panel A: Stacked bar chart of (FAR, FRR, PRR, DR) across the 12 models
x_m = np.arange(len(MODEL_LABELS))
bw_m = 0.64
bottom_m = np.zeros(len(MODEL_LABELS), dtype=float)

for r_code, r_lbl in [
    ('FAR', 'FAR (Full Alignment)'),
    ('PRR', 'PRR (Partial Retention)'),
    ('FRR', 'FRR (Full Retention)'),
    ('DR',  'DR (Divergence)'),
]:
    vals_r = np.array([st[r_code] for _, st in model_pooled_stats], dtype=float)
    ax_m12.bar(x_m, vals_r, bw_m, bottom=bottom_m, label=r_lbl, color=REGIME_COLORS[r_code], edgecolor='#222222')
    for i, v in enumerate(vals_r):
        if v >= 7.0:
            txt_c = 'white' if r_code in ('FAR', 'FRR', 'DR') else '#111111'
            ax_m12.text(i, bottom_m[i] + v / 2.0, f"{v:.0f}%", ha='center', va='center',
                        color=txt_c, fontsize=7.5, fontweight='bold')
    bottom_m += vals_r

for i, (_, st_m) in enumerate(model_pooled_stats):
    ax_m12.text(i, 101.5, f"$|\\mathcal{{S}}|={st_m['n_succ']//1000:,}$k",
                ha='center', va='bottom', fontsize=7.2, fontweight='bold', rotation=25)

for x_sep in [2.5, 5.5, 8.5]:
    ax_m12.axvline(x_sep, color='#333333', linestyle='--', linewidth=1.1)

ax_m12.set_xticks(x_m)
ax_m12.set_xticklabels(MODEL_LABELS, rotation=30, ha='right', fontsize=8.5, fontweight='bold')
ax_m12.set_ylim(0, 115)
ax_m12.set_ylabel("Share of Successful Attacks $\\mathcal{S}$ (%)", fontsize=10.0)
ax_m12.set_title("A. Composite Regimes Across All 12 Models (Pooled Over 20 Attacks)", fontsize=10.8, fontweight='bold')
ax_m12.legend(loc='lower left', fontsize=8.2, ncol=2, framealpha=0.92)
ax_m12.grid(True, axis='y', linestyle=':', alpha=0.6)

# Panel B: Grouped bar chart comparing FAR_shape, FAR_scale, FAR_joint, and overall FAR between gap and flat
st_gap_all = pool_slices(tree_df[tree_df['pool'] == 'gap'])
st_flt_all = pool_slices(tree_df[tree_df['pool'] == 'flatten'])

far_metrics = [
    ('FAR_shape', 'n_sh', "$\\mathrm{FAR}_{\\mathrm{shape}}$\n($\\mathrm{ARR}_{\\mathrm{shape,AR}}$)"),
    ('FAR_scale', 'n_sc', "$\\mathrm{FAR}_{\\mathrm{scale}}$\n($\\mathrm{ARR}_{\\mathrm{scale,RA}}$)"),
    ('FAR_joint', 'n_jt', "$\\mathrm{FAR}_{\\mathrm{joint}}$\n($\\mathrm{ARR}_{\\mathrm{joint,AA}}$)"),
    ('FAR',       'n_succ', "Overall $\\mathrm{FAR}$\n(on $\\mathcal{S}$)"),
]
x_f = np.arange(len(far_metrics))
bw_f = 0.36

for offset, st_p, p_lbl, p_col in [
    (-bw_f / 2, st_gap_all, "GAP Pooling ($\\pi=\\mathtt{gap}$)",     '#1f78b4'),
    ( bw_f / 2, st_flt_all, "Flatten Pooling ($\\pi=\\mathtt{flat}$)", '#ff7f00'),
]:
    vals_p, err_lo, err_hi, n_denoms = [], [], [], []
    for m_k, n_k, _ in far_metrics:
        v = st_p[m_k]
        lo, hi = wilson_ci(v, st_p[n_k])
        vals_p.append(v)
        err_lo.append(max(0.0, v - lo))
        err_hi.append(max(0.0, hi - v))
        n_denoms.append(st_p[n_k])
    bars_p = ax_gf.bar(x_f + offset, vals_p, bw_f, yerr=[err_lo, err_hi], capsize=3.0,
                       label=p_lbl, color=p_col, edgecolor='#222222')
    for bar, v, n_d in zip(bars_p, vals_p, n_denoms):
        ax_gf.text(bar.get_x() + bar.get_width() / 2, v + 1.8,
                   f"{v:.1f}%\n($N={n_d//1000}k$)",
                   ha='center', va='bottom', fontsize=7.4, fontweight='bold', color='#111111')

# Unconditional uniform state-space benchmark 1/|K_s| (16.7% for shape/scale, 11.1% for joint)
ax_gf.hlines(100.0 / 6.0, -0.45, 1.45, colors='#555555', linestyles=':', linewidth=1.4, label="Uniform $1/|\\mathcal{K}_s|$ (16.7% / 11.1%)")
ax_gf.hlines(100.0 / 9.0,  1.55, 2.45, colors='#555555', linestyles=':', linewidth=1.4)

ax_gf.set_xticks(x_f)
ax_gf.set_xticklabels([m[2] for m in far_metrics], fontsize=8.8, fontweight='bold')
ax_gf.set_ylim(0, 108)
ax_gf.set_ylabel("Full Alignment Rate (%)", fontsize=10.0)
ax_gf.set_title("B. Internal Coherence ($\\mathrm{FAR}_s$ & $\\mathrm{FAR}$): `gap` vs `flat`", fontsize=10.8, fontweight='bold')
ax_gf.legend(loc='lower right', fontsize=7.8, framealpha=0.92)
ax_gf.grid(True, axis='y', linestyle=':', alpha=0.6)

plt.tight_layout()
save_and_show(fig4b, "Figure 4B: Composite Regimes Across the 12 Models (Panel A) and Full Alignment by Pooling (Panel B)")
# %%


# %% PART B — Section 4: Full Retention: Main Fooled, Both Auxiliaries Keep the Truth
print("=" * 128)
print("=== PART B — SECTION 4: FULL RETENTION: MAIN FOOLED, BOTH AUXILIARIES KEEP THE TRUTH (FRR & ARR_{s,RR}) ===")
print("=" * 128)
print("Observed result:")
print("  - Full retention (both auxiliaries keep the truth although `main` is fooled) is rare and splits into two optimizer")
print("    regimes:")
print("    * Under PGD it is a near-boundary phenomenon: present at eps_1, then very rare from eps_2 onward.")
print("    * Under FGSM it persists at a low but stable level across budgets; in scale-shift under FGSM it increases in the")
print("      `con` models, while `std` remains stable from eps_3 onward.")
print("  - At eps_1, `con` shows much higher retention than `std`, but it rests on few successful attacks, almost all")
print("    PGD-L_inf (only 9 come from FGSM).")
print("  - Retention is more frequent in scale-shift than in shape-shift or joint-shift.")
print("-" * 128)

g_all = pool_slices(tree_df)

# 1. Pooled FRR and ARR_{s,RR} for s in {shape, scale, joint} with |S_s| and 95% Wilson CIs
print("\n[1] Global Full Retention Rates (Overall FRR and Shift-Conditional ARR_{s,RR}):")
print(f"{'Retention Metric':<36} | {'Event Count (k / Denom)':>28} | {'Pooled Rate [95% Wilson CI]':>30}")
print("-" * 102)
for r_lbl, k_key, d_key, m_key in [
    ("Overall FRR (over S)",                'n_frr',      'n_succ', 'FRR'),
    ("Shape-Shift: ARR_shape,RR (over S_sh)", 'n_shape_RR', 'n_sh',   'ARR_shape_RR'),
    ("Scale-Shift: ARR_scale,RR (over S_sc)", 'n_scale_RR', 'n_sc',   'ARR_scale_RR'),
    ("Joint-Shift: ARR_joint,RR (over S_jt)", 'n_joint_RR', 'n_jt',   'ARR_joint_RR'),
]:
    v = g_all[m_key]
    lo, hi = wilson_ci(v, g_all[d_key])
    print(f"{r_lbl:<36} | {f'{g_all[k_key]:,} / {g_all[d_key]:,}':>28} | {v:6.2f}% [{lo:6.2f}%, {hi:6.2f}%]")
print("-" * 102)

def _fmt_rate_ci_n(val, denom):
    """Format 'rate% [lo,hi] (N=denom)' and flag if denom < N_MIN_BRANCH."""
    if denom <= 0:
        return f"   --   [  -- ,  -- ] (N=0!)"
    lo, hi = wilson_ci(val, denom)
    flag = "*" if denom < N_MIN_BRANCH else " "
    return f"{val:5.2f}% [{lo:5.2f},{hi:5.2f}] ({denom:7,d}){flag}"

# 2. Table of ARR_{s,RR} and FRR across eps_1..eps_5 for FGSM vs PGD
print("\n[2] Full Retention Across Budget Steps eps_1..eps_5 by Optimizer (FGSM vs PGD) (* flags |S_s| < 100):")
print(f"{'Opt x Step':<14} | {'ARR_shape,RR [95% CI] (|S_sh|)':>34} | {'ARR_scale,RR [95% CI] (|S_sc|)':>34} | "
      f"{'ARR_joint,RR [95% CI] (|S_jt|)':>34} | {'Overall FRR [95% CI] (|S|)':>34}")
print("-" * 162)
for a_val in ['fgsm', 'pgd']:
    for s in EPS_STEPS:
        st_as = pool_slices(tree_df[(tree_df['attack'] == a_val) & (tree_df['eps_step'] == s)])
        print(f"{a_val.upper() + ' eps_' + str(s):<14} | "
              f"{_fmt_rate_ci_n(st_as['ARR_shape_RR'], st_as['n_sh']):>34} | "
              f"{_fmt_rate_ci_n(st_as['ARR_scale_RR'], st_as['n_sc']):>34} | "
              f"{_fmt_rate_ci_n(st_as['ARR_joint_RR'], st_as['n_jt']):>34} | "
              f"{_fmt_rate_ci_n(st_as['FRR'],          st_as['n_succ']):>34}")
    st_a = pool_slices(tree_df[tree_df['attack'] == a_val])
    print(f"{a_val.upper() + ' Pooled':<14} | "
          f"{_fmt_rate_ci_n(st_a['ARR_shape_RR'], st_a['n_sh']):>34} | "
          f"{_fmt_rate_ci_n(st_a['ARR_scale_RR'], st_a['n_sc']):>34} | "
          f"{_fmt_rate_ci_n(st_a['ARR_joint_RR'], st_a['n_jt']):>34} | "
          f"{_fmt_rate_ci_n(st_a['FRR'],          st_a['n_succ']):>34}")
    print("-" * 162)

print("\n[2b] FGSM Scale-Shift Retention Split by Dataset (`FGSM x std` vs `FGSM x con` Across eps_1..eps_5):")
for d_val, d_short in [('standard', 'std'), ('contrast', 'con')]:
    sub_fd = tree_df[(tree_df['attack'] == 'fgsm') & (tree_df['dataset'] == d_val)]
    cells_fd = [
        f"e{s}: {_fmt_rate_ci_n(pool_slices(sub_fd[sub_fd['eps_step'] == s])['ARR_scale_RR'], pool_slices(sub_fd[sub_fd['eps_step'] == s])['n_sc'])}"
        for s in EPS_STEPS
    ]
    print(f"  FGSM x {d_short:<3} -> " + " | ".join(cells_fd))

# 3. Table at eps_1 comparing std vs con (overall and split by FGSM vs PGD and norm)
print("\n[3] Boundary Retention at eps_1: Standard (`std`) vs Contrast (`con`) (Overall & Split by Optimizer and Norm):")
print(f"{'eps_1 Subgroup':<26} | {'ARR_shape,RR [95% CI] (|S_sh|)':>34} | {'ARR_scale,RR [95% CI] (|S_sc|)':>34} | "
      f"{'ARR_joint,RR [95% CI] (|S_jt|)':>34} | {'Overall FRR [95% CI] (|S|)':>34}")
print("-" * 174)
e1_df = tree_df[tree_df['eps_step'] == 1]
e1_ret_groups = [
    ("eps_1: std (all)",          e1_df['dataset'] == 'standard'),
    ("eps_1: con (all)",          e1_df['dataset'] == 'contrast'),
    ("eps_1: std x FGSM",         (e1_df['dataset'] == 'standard') & (e1_df['attack'] == 'fgsm')),
    ("eps_1: con x FGSM",         (e1_df['dataset'] == 'contrast') & (e1_df['attack'] == 'fgsm')),
    ("eps_1: std x PGD",          (e1_df['dataset'] == 'standard') & (e1_df['attack'] == 'pgd')),
    ("eps_1: con x PGD",          (e1_df['dataset'] == 'contrast') & (e1_df['attack'] == 'pgd')),
    ("eps_1: std x L_inf",        (e1_df['dataset'] == 'standard') & (e1_df['norm'] == 'linf')),
    ("eps_1: con x L_inf",        (e1_df['dataset'] == 'contrast') & (e1_df['norm'] == 'linf')),
    ("eps_1: std x L_2",          (e1_df['dataset'] == 'standard') & (e1_df['norm'] == 'l2')),
    ("eps_1: con x L_2",          (e1_df['dataset'] == 'contrast') & (e1_df['norm'] == 'l2')),
]
for idx, (g_lbl, g_mask) in enumerate(e1_ret_groups):
    if idx in (2, 6):
        print("-" * 174)
    st_r = pool_slices(e1_df[g_mask])
    print(f"{g_lbl:<26} | "
          f"{_fmt_rate_ci_n(st_r['ARR_shape_RR'], st_r['n_sh']):>34} | "
          f"{_fmt_rate_ci_n(st_r['ARR_scale_RR'], st_r['n_sc']):>34} | "
          f"{_fmt_rate_ci_n(st_r['ARR_joint_RR'], st_r['n_jt']):>34} | "
          f"{_fmt_rate_ci_n(st_r['FRR'],          st_r['n_succ']):>34}")
print("=" * 128 + "\n")

# --- Figure 5: 3-Panel Retention Trajectories (ARR_{s,RR} vs eps_1..eps_5 for s in {shape, scale, joint}) ---
fig5, axes5 = plt.subplots(1, 3, figsize=(16.5, 5.2), dpi=130, sharey=False)

panel5_specs = [
    (0, 'shape', 'ARR_shape_RR', 'n_sh', "A. Shape-Shift Full Retention ($\\mathrm{ARR}_{\\mathrm{shape,RR}}$)"),
    (1, 'scale', 'ARR_scale_RR', 'n_sc', "B. Scale-Shift Full Retention ($\\mathrm{ARR}_{\\mathrm{scale,RR}}$)"),
    (2, 'joint', 'ARR_joint_RR', 'n_jt', "C. Joint-Shift Full Retention ($\\mathrm{ARR}_{\\mathrm{joint,RR}}$)"),
]

traj_lines = [
    ('fgsm', 'contrast', "FGSM $\\times$ `con`", '#762a83', 's', '-'),
    ('fgsm', 'standard', "FGSM $\\times$ `std`", '#af8dc3', 'o', '--'),
    ('pgd',  'contrast', "PGD $\\times$ `con`",  '#1b7837', '^', '-'),
    ('pgd',  'standard', "PGD $\\times$ `std`",  '#7fbf7b', 'v', '--'),
]

x_steps = np.arange(1, 6)
for p_idx, s_name, m_col, n_col, p_title in panel5_specs:
    ax_r = axes5[p_idx]
    for a_val, d_val, l_lbl, col_l, mk_l, ls_l in traj_lines:
        sub_ld = tree_df[(tree_df['attack'] == a_val) & (tree_df['dataset'] == d_val)]
        y_vals, err_lo, err_hi, n_list = [], [], [], []
        for s in EPS_STEPS:
            st_s = pool_slices(sub_ld[sub_ld['eps_step'] == s])
            denom_s = st_s[n_col]
            n_list.append(denom_s)
            if denom_s >= N_MIN_BRANCH:
                v_s = st_s[m_col]
                lo_s, hi_s = wilson_ci(v_s, denom_s)
                y_vals.append(v_s)
                err_lo.append(max(0.0, v_s - lo_s))
                err_hi.append(max(0.0, hi_s - v_s))
            else:
                y_vals.append(np.nan)
                err_lo.append(0.0)
                err_hi.append(0.0)
        y_arr = np.array(y_vals, dtype=float)
        valid_m = ~np.isnan(y_arr)
        if np.any(valid_m):
            ax_r.errorbar(
                x_steps[valid_m], y_arr[valid_m],
                yerr=[np.array(err_lo)[valid_m], np.array(err_hi)[valid_m]],
                label=l_lbl, color=col_l, marker=mk_l, linestyle=ls_l,
                linewidth=2.1, markersize=6.0, capsize=3.0,
            )
            # Annotate eps_1 and eps_2 values & sample counts for the top curves
            for s_i in [0, 1]:
                if valid_m[s_i] and y_arr[s_i] >= 2.0:
                    ax_r.annotate(
                        f"{y_arr[s_i]:.1f}%\n($N={n_list[s_i]:,}$)",
                        (x_steps[s_i], y_arr[s_i]),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=7.2, fontweight='bold', color=col_l,
                    )

    ax_r.set_xticks(x_steps)
    ax_r.set_xticklabels([f"$\\varepsilon_{s}$" for s in EPS_STEPS], fontsize=9.5)
    ax_r.set_xlabel("Perturbation Budget Step ($\\varepsilon_1 \\to \\varepsilon_5$)", fontsize=9.8)
    if p_idx == 0:
        ax_r.set_ylabel("Full Retention Rate $\\mathrm{ARR}_{s,\\mathrm{RR}}$ (%)", fontsize=10.0)
    ax_r.set_title(p_title, fontsize=10.2, fontweight='bold')
    ax_r.grid(True, linestyle=':', alpha=0.6)
    ax_r.set_ylim(bottom=-1.0)
    if p_idx == 0:
        ax_r.legend(loc='upper right', fontsize=8.4, framealpha=0.92)

plt.tight_layout()
save_and_show(fig5, "Figure 5: Full Retention Trajectories (ARR_{s,RR}) Across Budget Steps by Optimizer and Dataset (Masked if |S_s| < 100)")
# %%


# %% PART B — Section 5: When Only One Auxiliary Decouples From Main — Partial Retention & Collateral Divergence
print("=" * 128)
print("=== PART B — SECTION 5: WHEN ONLY ONE AUXILIARY DECOUPLES FROM MAIN — PARTIAL RETENTION (PRR) & COLLATERAL DIVERGENCE ===")
print("=" * 128)
print("Observed result:")
print("  - Partial retention and collateral divergence are governed by a `dataset x pooling` interaction rather than by")
print("    pooling alone:")
print("    * Within `con`, a mirror pattern appears: in `con | flat` the scale head resists in joint-shift and diverges")
print("      in shape-shift; in `con | gap` the shape head resists in joint-shift and diverges in scale-shift.")
print("    * Within `std`, the pattern reverses: `std | flat` has the shape head resisting more often than the scale head")
print("      in joint-shift, and collateral divergence of the scale head in shape-shift is more frequent in `std | gap`.")
print("  - Error inversion on the unshifted factor is virtually absent when pooled; within `con | gap` shape-shift, both")
print("    collateral divergence and inversion are near zero.")
print("-" * 128)

g_all = pool_slices(tree_df)

# 1. Pooled PRR and its decomposition into ARR_{joint,RA} and ARR_{joint,AR}
print("\n[1] Global Partial Retention (PRR Over S and Decomposition Over Joint-Shift S_joint):")
print(f"{'Metric':<42} | {'Event Count (k / Denom)':>28} | {'Pooled Rate [95% Wilson CI]':>30}")
print("-" * 108)
for p_lbl, k_cnt, d_cnt, r_val in [
    ("Overall PRR (over S)",                          g_all['n_prr'],      g_all['n_succ'], g_all['PRR']),
    ("PRR_joint = ARR_joint,RA + ARR_joint,AR (S_jt)", g_all['n_prr'],      g_all['n_jt'],   g_all['PRR_joint']),
    ("  -> ARR_joint,RA (Shape retains, Scale aligns)", g_all['n_joint_RA'], g_all['n_jt'],   g_all['ARR_joint_RA']),
    ("  -> ARR_joint,AR (Scale retains, Shape aligns)", g_all['n_joint_AR'], g_all['n_jt'],   g_all['ARR_joint_AR']),
]:
    lo, hi = wilson_ci(r_val, d_cnt)
    print(f"{p_lbl:<42} | {f'{k_cnt:,} / {d_cnt:,}':>28} | {r_val:6.2f}% [{lo:6.2f}%, {hi:6.2f}%]")
print("-" * 108)

# 2. Mirror symmetry table across the 4 cells {std, con} x {flat, gap}
print("\n[2] `Dataset x Pooling` Mirror Symmetry: Partial Retention (S_joint) & Collateral Divergence (S_shape, S_scale):")
print(f"{'Family (d | pi)':<16} | {'ARR_joint,RA (|S_jt|)':>32} | {'ARR_joint,AR (|S_jt|)':>32} | "
      f"{'ARR_shape,AD (|S_sh|)':>32} | {'ARR_scale,DA (|S_sc|)':>32}")
print("-" * 158)
family_order_s5 = [
    ('std|flat', 'standard', 'flatten'),
    ('std|gap',  'standard', 'gap'),
    ('con|flat', 'contrast', 'flatten'),
    ('con|gap',  'contrast', 'gap'),
]
fam_stats_s5 = {}
for f_lbl, d_v, p_v in family_order_s5:
    st_f = pool_slices(tree_df[(tree_df['dataset'] == d_v) & (tree_df['pool'] == p_v)])
    fam_stats_s5[f_lbl] = st_f
    print(f"{f_lbl:<16} | "
          f"{_fmt_rate_ci_n(st_f['ARR_joint_RA'], st_f['n_jt']):>32} | "
          f"{_fmt_rate_ci_n(st_f['ARR_joint_AR'], st_f['n_jt']):>32} | "
          f"{_fmt_rate_ci_n(st_f['ARR_shape_AD'], st_f['n_sh']):>32} | "
          f"{_fmt_rate_ci_n(st_f['ARR_scale_DA'], st_f['n_sc']):>32}")
print("-" * 158)

# 3. Non-inversion comparison table across {std, con} x {flat, gap} and pooled
print("\n[3] Non-Inversion Check: Collateral Divergence Along Aligned Shifted Factor (AD, DA) vs Inverted Retention (RD, DR):")
print(f"{'Group':<16} | {'Shape-Shift |S_sh|':>12} | {'ARR_shape,AD [95% CI]':>23} | {'ARR_shape,RD [95% CI]':>23} || "
      f"{'Scale-Shift |S_sc|':>12} | {'ARR_scale,DA [95% CI]':>23} | {'ARR_scale,DR [95% CI]':>23}")
print("-" * 152)
for f_lbl, d_v, p_v in family_order_s5 + [('GLOBAL ALL', None, None)]:
    st_c = g_all if d_v is None else fam_stats_s5[f_lbl]
    if d_v is None:
        print("-" * 152)
    ad_lo, ad_hi = wilson_ci(st_c['ARR_shape_AD'], st_c['n_sh'])
    rd_lo, rd_hi = wilson_ci(st_c['ARR_shape_RD'], st_c['n_sh'])
    da_lo, da_hi = wilson_ci(st_c['ARR_scale_DA'], st_c['n_sc'])
    dr_lo, dr_hi = wilson_ci(st_c['ARR_scale_DR'], st_c['n_sc'])
    print(f"{f_lbl:<16} | {st_c['n_sh']:12,d} | "
          f"{st_c['ARR_shape_AD']:6.2f}% [{ad_lo:5.2f},{ad_hi:5.2f}] | "
          f"{st_c['ARR_shape_RD']:6.2f}% [{rd_lo:5.2f},{rd_hi:5.2f}] || "
          f"{st_c['n_sc']:12,d} | "
          f"{st_c['ARR_scale_DA']:6.2f}% [{da_lo:5.2f},{da_hi:5.2f}] | "
          f"{st_c['ARR_scale_DR']:6.2f}% [{dr_lo:5.2f},{dr_hi:5.2f}]")
print("-" * 152)

# 4. Full 12 models x 21 states table of ARR_{s,c} with |S_s| per model and Criterion ★★ markers
print("\n[4] Complete 12 Models x 21 States Auxiliary Reaction Matrix ARR_{s,c} (%) (★★ = Wilson CI_low > 1/|K_s|):")
for s_name, n_key, k_space in [
    ('shape', 'n_sh', K_SHAPE),
    ('scale', 'n_sc', K_SCALE),
    ('joint', 'n_jt', K_JOINT),
]:
    print(f"\n  --- Shift Type s = '{s_name}' (|K_{s_name}| = {len(k_space)}, Uniform Null = {100.0/len(k_space):.2f}%) ---")
    col_hdr = " | ".join(f"{code:>10}" for code in k_space)
    print(f"  {'Model (d|pi|w)':<16} | {'|S_' + s_name[:2] + '|':>10} || {col_hdr}")
    print("  " + "-" * (34 + 13 * len(k_space)))
    for i, ((d, p, w), m_lbl) in enumerate(zip(MODELS, MODEL_LABELS)):
        if i in (3, 6, 9):
            print("  " + "-" * (34 + 13 * len(k_space)))
        st_m = pool_slices(tree_df[(tree_df['dataset'] == d) & (tree_df['pool'] == p) & (tree_df['weight'] == w)])
        denom_m = st_m[n_key]
        vals_d = {code: st_m[f'ARR_{s_name}_{code}'] for code in k_space}
        tiers_m = classify_wilson_dominance(vals_d, denom_m)
        cells_str = []
        for code in k_space:
            if denom_m < N_MIN_BRANCH:
                cells_str.append(f"{'LOW_N':>10}")
            else:
                star = "★★" if tiers_m[code] == 'primary' else ("★ " if tiers_m[code] == 'secondary' else "  ")
                cells_str.append(f"{vals_d[code]:6.2f}%{star}")
        print(f"  {m_lbl:<16} | {denom_m:10,d} || " + " | ".join(cells_str))
    print("  " + "-" * (34 + 13 * len(k_space)))
print("=" * 128 + "\n")

# --- Figure 6A: 2x2 Interaction Plots Across Pooling (flat vs gap) and Dataset (std vs con) ---
fig6a, axes6a = plt.subplots(2, 2, figsize=(13.6, 8.2), dpi=130)

inter_specs = [
    (0, 0, 'ARR_joint_RA', 'n_jt', "A. Joint-Shift: Shape Resists ($\\mathrm{ARR}_{\\mathrm{joint,RA}}$)"),
    (0, 1, 'ARR_joint_AR', 'n_jt', "B. Joint-Shift: Scale Resists ($\\mathrm{ARR}_{\\mathrm{joint,AR}}$)"),
    (1, 0, 'ARR_shape_AD', 'n_sh', "C. Shape-Shift: Unshifted Scale Diverges ($\\mathrm{ARR}_{\\mathrm{shape,AD}}$)"),
    (1, 1, 'ARR_scale_DA', 'n_sc', "D. Scale-Shift: Unshifted Shape Diverges ($\\mathrm{ARR}_{\\mathrm{scale,DA}}$)"),
]

x_pool = np.array([0, 1])
for r_i, c_i, m_col, n_col, p_title in inter_specs:
    ax_i = axes6a[r_i, c_i]
    for d_val, d_lbl, col_d, mk_d in [
        ('standard', "Standard (`std`)", '#d95f02', 'o'),
        ('contrast', "Contrast (`con`)", '#1b9e77', 's'),
    ]:
        y_v, e_lo, e_hi, n_v = [], [], [], []
        for p_val in ['flatten', 'gap']:
            st_cell = pool_slices(tree_df[(tree_df['dataset'] == d_val) & (tree_df['pool'] == p_val)])
            denom_c = st_cell[n_col]
            n_v.append(denom_c)
            if denom_c >= N_MIN_BRANCH:
                val_c = st_cell[m_col]
                lo_c, hi_c = wilson_ci(val_c, denom_c)
                y_v.append(val_c)
                e_lo.append(max(0.0, val_c - lo_c))
                e_hi.append(max(0.0, hi_c - val_c))
            else:
                y_v.append(np.nan)
                e_lo.append(0.0)
                e_hi.append(0.0)
        y_arr = np.array(y_v, dtype=float)
        valid_c = ~np.isnan(y_arr)
        ax_i.errorbar(
            x_pool[valid_c], y_arr[valid_c],
            yerr=[np.array(e_lo)[valid_c], np.array(e_hi)[valid_c]],
            label=d_lbl, color=col_d, marker=mk_d, linewidth=2.3, markersize=7.0, capsize=3.5,
        )
        for x_k, val_k, nk in zip(x_pool, y_arr, n_v):
            if not np.isnan(val_k):
                other_d = 'standard' if d_val == 'contrast' else 'contrast'
                p_code  = 'flatten' if x_k == 0 else 'gap'
                other_v = pool_slices(tree_df[(tree_df['dataset'] == other_d) & (tree_df['pool'] == p_code)])[m_col]
                is_upper = (val_k >= other_v)
                y_offset = 1.5 if is_upper else -2.4
                ha_align = 'left' if x_k == 0 else 'right'
                x_nudge  = x_k + 0.04 if x_k == 0 else x_k - 0.04
                ax_i.text(
                    x_nudge, val_k + y_offset,
                    f"{val_k:.2f}% ($N={nk:,}$)",
                    ha=ha_align, va='bottom' if is_upper else 'top',
                    fontsize=8.0, fontweight='bold', color=col_d,
                )

    ax_i.set_xticks(x_pool)
    ax_i.set_xticklabels(["Flatten (`flat`)", "GAP (`gap`)"], fontsize=9.8, fontweight='bold')
    ax_i.set_xlim(-0.25, 1.25)
    ax_i.set_ylabel("Conditional Rate $\\mathrm{ARR}_{s,c}$ (%)", fontsize=9.8)
    ax_i.set_title(p_title, fontsize=10.4, fontweight='bold')
    ax_i.grid(True, linestyle=':', alpha=0.6)
    ax_i.set_ylim(bottom=-4.5, top=max(27.0, float(ax_i.get_ylim()[1]) * 1.25))
    if r_i == 0 and c_i == 0:
        ax_i.legend(loc='upper left', fontsize=8.8, framealpha=0.92)

plt.tight_layout()
save_and_show(fig6a, "Figure 6A: Dataset x Pooling Mirror Symmetry in Partial Retention (PRR) and Collateral Divergence")

# --- Figure 6B: 3-Panel 12 Models x 21 States Heatmap of ARR_{s,c} (with Low-N Hatched Masking) ---
fig6b, axes6b = plt.subplots(
    1, 3, figsize=(17.2, 6.2), dpi=130, sharey=True,
    gridspec_kw={'width_ratios': [6.0, 6.0, 9.0], 'wspace': 0.12},
)

cmap_arr = plt.cm.YlGnBu.copy()
cmap_arr.set_bad(color='#e0e0e0')

hm_specs = [
    (0, 'shape', 'n_sh', K_SHAPE, "A. Shape-Shift $\\mathcal{K}_{\\mathrm{shape}}$ (6 States)"),
    (1, 'scale', 'n_sc', K_SCALE, "B. Scale-Shift $\\mathcal{K}_{\\mathrm{scale}}$ (6 States)"),
    (2, 'joint', 'n_jt', K_JOINT, "C. Joint-Shift $\\mathcal{K}_{\\mathrm{joint}}$ (9 States)"),
]

im_shared = None
for p_i, s_name, n_key, k_space, h_title in hm_specs:
    ax_h = axes6b[p_i]
    mat_arr = np.full((len(MODELS), len(k_space)), np.nan, dtype=float)
    denom_vec = np.zeros(len(MODELS), dtype=int)
    for r_idx, (d, p, w) in enumerate(MODELS):
        st_m = pool_slices(tree_df[(tree_df['dataset'] == d) & (tree_df['pool'] == p) & (tree_df['weight'] == w)])
        denom_vec[r_idx] = st_m[n_key]
        if st_m[n_key] >= N_MIN_BRANCH:
            for c_idx, code in enumerate(k_space):
                mat_arr[r_idx, c_idx] = st_m[f'ARR_{s_name}_{code}']

    im_shared = ax_h.imshow(mat_arr, cmap=cmap_arr, vmin=0.0, vmax=100.0, aspect='auto')
    ax_h.set_xticks(np.arange(len(k_space)))
    ax_h.set_xticklabels(k_space, fontsize=9.0, fontweight='bold')
    ax_h.set_title(h_title, fontsize=10.4, fontweight='bold')

    if p_i == 0:
        ax_h.set_yticks(np.arange(len(MODEL_LABELS)))
        ax_h.set_yticklabels(MODEL_LABELS, fontsize=8.8, fontweight='bold')

    for y_sep in [2.5, 5.5, 8.5]:
        ax_h.axhline(y_sep, color='#222222', linestyle='--', linewidth=1.0)

    # Annotate values or render hatched low-N patch
    for r_idx in range(len(MODELS)):
        for c_idx in range(len(k_space)):
            val_rc = mat_arr[r_idx, c_idx]
            if np.isnan(val_rc):
                ax_h.add_patch(mpatches.Rectangle(
                    (c_idx - 0.5, r_idx - 0.5), 1.0, 1.0,
                    facecolor='#e0e0e0', edgecolor='#666666', hatch='///', lw=0.8,
                ))
                ax_h.text(c_idx, r_idx, "LOW N", ha='center', va='center',
                          fontsize=6.8, fontweight='bold', color='#444444')
            else:
                txt_col = 'white' if val_rc >= 50.0 else '#111111'
                ax_h.text(c_idx, r_idx, f"{val_rc:.1f}", ha='center', va='center',
                          fontsize=7.6, fontweight='bold', color=txt_col)

cbar6b = fig6b.colorbar(im_shared, ax=axes6b.ravel().tolist(), fraction=0.018, pad=0.02)
cbar6b.set_label("Conditional Auxiliary Reaction Rate $\\mathrm{ARR}_{s,c}$ (%)", fontsize=10.0)
save_and_show(fig6b, "Figure 6B: 12 Models x 21 States Heatmap of ARR_{s,c} Across Shape, Scale, and Joint Shifts")

# --- Figure 6C: Deterministically Sampled Image Strip — Collateral Divergence AD During Shape-Shift ---
rng_s5 = np.random.default_rng(29)

# Sample from std|flat|w=1.0 under PGD-Linf eps_2=0.05 (where shape-shift collateral scale divergence AD is prominent)
df_ad_s5, X_adv_ad_s5 = load_slice_eval_and_adv('standard', 'flatten', 1.0, 'pgd', 'linf', EPS_LINF[2])
y_m_ad  = df_ad_s5['y_shape_and_scale'].to_numpy()
y_sh_ad = y_m_ad // 3
y_sc_ad = y_m_ad % 3
p_m_ad  = df_ad_s5['pred_main'].to_numpy()
p_sh_ad = df_ad_s5['pred_shape'].to_numpy()
p_sc_ad = df_ad_s5['pred_scale'].to_numpy()

# State AD in S_shape: main shifts shape only (p_m//3 != y_sh and p_m%3 == y_sc),
# shape aux aligns with main (p_sh == p_m//3), scale aux diverges from true scale (p_sc != y_sc)
cand_sh_ad = np.where(
    (p_m_ad != y_m_ad)
    & ((p_m_ad // 3) != y_sh_ad) & ((p_m_ad % 3) == y_sc_ad)
    & (p_sh_ad == (p_m_ad // 3))
    & (p_sc_ad != y_sc_ad)
)[0]
idx_pick_ad = int(rng_s5.choice(cand_sh_ad))
r_ad = df_ad_s5.iloc[idx_pick_ad]

print("[Figure 6C Deterministic Sampling Log — np.random.default_rng(29)]:")
print(f"  Collateral Divergence State AD in S_shape (`std|flat|w=1.0`, PGD-Linf eps_2=0.05): "
      f"pool = {len(cand_sh_ad):,} candidates -> selected sample_index = {idx_pick_ad}\n")

fig6c, axes6c = plt.subplots(1, 3, figsize=(12.2, 3.8), dpi=130)
img_clean_ad = X_CLEAN_TEST[idx_pick_ad]
img_adv_ad   = X_adv_ad_s5[idx_pick_ad]
delta_ad_5x  = np.clip(np.abs(img_adv_ad - img_clean_ad) * 5.0, 0.0, 1.0)
y_true_ad    = int(r_ad['y_shape_and_scale'])
p_main_ad    = int(r_ad['pred_main'])

axes6c[0].imshow(img_clean_ad, cmap='gray', vmin=0.0, vmax=1.0)
axes6c[0].set_title(
    f"Clean Input (sample_index={idx_pick_ad})\nTrue: {CLASS_NAMES_9[y_true_ad]}",
    fontsize=9.2, fontweight='bold',
)
axes6c[0].axis('off')

axes6c[1].imshow(img_adv_ad, cmap='gray', vmin=0.0, vmax=1.0)
axes6c[1].set_title(
    f"Adversarial (`std|flat|w=1.0`, PGD-$L_\\infty$, $\\varepsilon_2=0.05$)\n"
    f"Main (Shape-Shift): {CLASS_NAMES_9[p_main_ad]} ({r_ad['confidence']*100:.1f}%)\n"
    f"Shape Aux [A]: {SHAPE_NAMES[int(r_ad['pred_shape'])]} ({r_ad['confidence_shape']*100:.1f}%) | "
    f"Scale Aux [D!]: {SCALE_NAMES[int(r_ad['pred_scale'])]} ({r_ad['confidence_scale']*100:.1f}%)",
    fontsize=8.6, fontweight='bold',
)
axes6c[1].axis('off')

im6c = axes6c[2].imshow(delta_ad_5x, cmap='magma', vmin=0.0, vmax=1.0)
axes6c[2].set_title(
    f"Perturbation $|\\delta| \\times 5$ (State $\\mathrm{{AD}} \\in \\mathcal{{S}}_{{\\mathrm{{shape}}}}$)\n"
    f"(max $|\\delta| = {np.max(np.abs(img_adv_ad - img_clean_ad)):.3f}$)",
    fontsize=9.2, fontweight='bold',
)
axes6c[2].axis('off')
plt.colorbar(im6c, ax=axes6c[2], fraction=0.046, pad=0.04)

plt.tight_layout()
save_and_show(fig6c, "Figure 6C: Deterministic Sample Strip — Collateral Divergence (State AD During Shape-Shift)")
# %%


# %% PART B — Section 6: When the Heads Contradict Each Other — Three-Way Divergence
print("=" * 128)
print("=== PART B — SECTION 6: WHEN THE HEADS CONTRADICT EACH OTHER — THREE-WAY DIVERGENCE (DR_s^{(k)} & ARR_{s,DD}) ===")
print("=" * 128)
print("Observed result:")
print("  - On a shifted factor, state 'D' is a strict 3-way contradiction: truth, main's prediction and the auxiliary's")
print("    prediction are all different classes.")
print("  - This happens about twice as often on shape (categorical) as on scale (ordinal).")
print("  - Double divergence is essentially exclusive to `flat`: it is absent in `gap`, although total divergence is")
print("    present in `gap` too (e.g. in `con | gap` at w >= 1).")
print("  - Pooled across models, shape-head divergence increases with w in all three shift types (one training run per w);")
print("    within `flat` the trend is not monotonic in shape-shift and scale-shift.")
print("-" * 128)

g_all = pool_slices(tree_df)

# 1. Table of DR_s^{(shape)}, DR_s^{(scale)}, DR_s, and ARR_{s,DD} for each s in {shape, scale, joint} and overall on S
print("\n[1] Global Divergence Breakdown by Shift Type s and Auxiliary Branch (Shape vs Scale vs Double Divergence DD):")
print(f"{'Scope / Shift Type':<26} | {'Denom |S_s|':>12} | {'DR_s^(shape) [95% CI]':>24} | "
      f"{'DR_s^(scale) [95% CI]':>24} | {'Total DR_s [95% CI]':>24} | {'Double Div DD [95% CI]':>24}")
print("-" * 148)
div_rows_s6 = [
    ("Shape-Shift (s=shape)", g_all['n_sh'],   g_all['DR_shape_sh'],   g_all['DR_shape_sc'],   g_all['DR_shape'], g_all['ARR_shape_DD']),
    ("Scale-Shift (s=scale)", g_all['n_sc'],   g_all['DR_scale_sh'],   g_all['DR_scale_sc'],   g_all['DR_scale'], g_all['ARR_scale_DD']),
    ("Joint-Shift (s=joint)", g_all['n_jt'],   g_all['DR_joint_sh'],   g_all['DR_joint_sc'],   g_all['DR_joint'], g_all['ARR_joint_DD']),
    ("Overall on S (All)",    g_all['n_succ'], g_all['DR_shape_aux'],  g_all['DR_scale_aux'],  g_all['DR'],       g_all['DR_DD_total']),
]
for idx, (s_lbl, denom_v, v_sh, v_sc, v_tot, v_dd) in enumerate(div_rows_s6):
    if idx == 3:
        print("-" * 148)
    lo_sh, hi_sh   = wilson_ci(v_sh,  denom_v)
    lo_sc, hi_sc   = wilson_ci(v_sc,  denom_v)
    lo_tot, hi_tot = wilson_ci(v_tot, denom_v)
    lo_dd, hi_dd   = wilson_ci(v_dd,  denom_v)
    print(f"{s_lbl:<26} | {denom_v:12,d} | "
          f"{v_sh:6.2f}% [{lo_sh:5.2f},{hi_sh:5.2f}] | "
          f"{v_sc:6.2f}% [{lo_sc:5.2f},{hi_sc:5.2f}] | "
          f"{v_tot:6.2f}% [{lo_tot:5.2f},{hi_tot:5.2f}] | "
          f"{v_dd:6.2f}% [{lo_dd:5.2f},{hi_dd:5.2f}]")
print("-" * 148)

# 2. Direct comparison of shifted-factor 3-way divergence: DR_shape^{(shape)} vs DR_scale^{(scale)}
n_3way_sh = g_all['n_shape_DR'] + g_all['n_shape_DD']
n_3way_sc = g_all['n_scale_RD'] + g_all['n_scale_DD']
sh3_lo, sh3_hi = wilson_ci(g_all['DR_shape_sh'], g_all['n_sh'])
sc3_lo, sc3_hi = wilson_ci(g_all['DR_scale_sc'], g_all['n_sc'])
ratio_3way = g_all['DR_shape_sh'] / g_all['DR_scale_sc'] if g_all['DR_scale_sc'] > 0 else np.nan

print("\n[2] Direct Shifted-Factor 3-Way Contradiction Comparison (Categorical Shape vs Ordinal Scale):")
print(f"  DR_shape^(shape) = ARR_shape,DR + ARR_shape,DD : {g_all['DR_shape_sh']:6.2f}% [{sh3_lo:5.2f}%, {sh3_hi:5.2f}%]  "
      f"(k = {n_3way_sh:,} / |S_shape| = {g_all['n_sh']:,})")
print(f"  DR_scale^(scale) = ARR_scale,RD + ARR_scale,DD : {g_all['DR_scale_sc']:6.2f}% [{sc3_lo:5.2f}%, {sc3_hi:5.2f}%]  "
      f"(k = {n_3way_sc:,} / |S_scale| = {g_all['n_sc']:,})")
print(f"  Ratio DR_shape^(shape) / DR_scale^(scale)      : {ratio_3way:5.2f}x  "
      f"(Difference = {g_all['DR_shape_sh'] - g_all['DR_scale_sc']:+5.2f} pp)")

# 3. Table of ARR_{s,DD} and DR_s by pooling (flat vs gap)
print("\n[3] Double Divergence (ARR_{s,DD}) and Total Divergence (DR_s) by Pooling (`flat` vs `gap`):")
print(f"{'Pooling':<10} | {'ARR_shape,DD (|S_sh|)':>32} | {'ARR_scale,DD (|S_sc|)':>32} | "
      f"{'ARR_joint,DD (|S_jt|)':>32} || {'Overall DD on S':>22} | {'Overall DR on S':>22}")
print("-" * 164)
for p_val, p_lbl in [('flatten', 'flat'), ('gap', 'gap')]:
    st_p = pool_slices(tree_df[tree_df['pool'] == p_val])
    dd_lo, dd_hi = wilson_ci(st_p['DR_DD_total'], st_p['n_succ'])
    dr_lo, dr_hi = wilson_ci(st_p['DR'],          st_p['n_succ'])
    print(f"{p_lbl:<10} | "
          f"{_fmt_rate_ci_n(st_p['ARR_shape_DD'], st_p['n_sh']):>32} | "
          f"{_fmt_rate_ci_n(st_p['ARR_scale_DD'], st_p['n_sc']):>32} | "
          f"{_fmt_rate_ci_n(st_p['ARR_joint_DD'], st_p['n_jt']):>32} || "
          f"{st_p['DR_DD_total']:5.2f}% [{dd_lo:4.2f},{dd_hi:4.2f}] | "
          f"{st_p['DR']:5.2f}% [{dr_lo:5.2f},{dr_hi:5.2f}]")
print("-" * 164)

# 4. Table of DR_s^{(shape)} across w in {0.5, 1.0, 2.0} (both pooled across all models and within flat models)
print("\n[4] Shape Auxiliary Divergence DR_s^{(shape)} Across Loss Weights w in {0.5, 1.0, 2.0} (Observed Trend Across 1 Run/w):")
print(f"{'Scope & Loss Weight':<24} | {'DR_shape^(shape) (|S_sh|)':>34} | {'DR_scale^(shape) (|S_sc|)':>34} | "
      f"{'DR_joint^(shape) (|S_jt|)':>34} | {'Overall DR^(shape) (|S|)':>34}")
print("-" * 172)
for scope_lbl, base_mask in [
    ("All Models",  tree_df['n_total'] > 0),
    ("Flat Models", tree_df['pool'] == 'flatten'),
]:
    for w_val in LOSS_WEIGHTS:
        st_w = pool_slices(tree_df[base_mask & (tree_df['weight'] == w_val)])
        print(f"{scope_lbl + f' (w={w_val:g})':<24} | "
              f"{_fmt_rate_ci_n(st_w['DR_shape_sh'],  st_w['n_sh']):>34} | "
              f"{_fmt_rate_ci_n(st_w['DR_scale_sh'],  st_w['n_sc']):>34} | "
              f"{_fmt_rate_ci_n(st_w['DR_joint_sh'],  st_w['n_jt']):>34} | "
              f"{_fmt_rate_ci_n(st_w['DR_shape_aux'], st_w['n_succ']):>34}")
    print("-" * 172)
print("=" * 128 + "\n")

# --- Figure 7A: Grouped Bar Chart of DR_s^{(shape)} Across Loss Weights w in {0.5, 1.0, 2.0} ---
fig7a, (ax_w_all, ax_w_flt) = plt.subplots(1, 2, figsize=(15.6, 5.4), dpi=130, sharey=False)

dr_sh_metrics = [
    ('DR_shape_sh',  'n_sh',   "Shape-Shift\n$\\mathrm{DR}_{\\mathrm{shape}}^{(\\mathrm{shape})}$"),
    ('DR_scale_sh',  'n_sc',   "Scale-Shift\n$\\mathrm{DR}_{\\mathrm{scale}}^{(\\mathrm{shape})}$"),
    ('DR_joint_sh',  'n_jt',   "Joint-Shift\n$\\mathrm{DR}_{\\mathrm{joint}}^{(\\mathrm{shape})}$"),
    ('DR_shape_aux', 'n_succ', "Overall on $\\mathcal{S}$\n$\\mathrm{DR}^{(\\mathrm{shape})}$"),
]
w_colors = {0.5: '#6baed6', 1.0: '#2171b5', 2.0: '#08306b'}
x_g = np.arange(len(dr_sh_metrics))
bw_w = 0.25

for ax_w, base_mask, p_title in [
    (ax_w_all, tree_df['n_total'] > 0,         "A. All Models Pooled (Observed Upward Trend With $w$)"),
    (ax_w_flt, tree_df['pool'] == 'flatten',   "B. Flatten (`flat`) Models Only (Strongest Divergence)"),
]:
    for idx_w, w_val in enumerate(LOSS_WEIGHTS):
        st_w = pool_slices(tree_df[base_mask & (tree_df['weight'] == w_val)])
        offset = (idx_w - 1) * bw_w
        vals_w, err_lo, err_hi, n_den = [], [], [], []
        for m_k, n_k, _ in dr_sh_metrics:
            v = st_w[m_k]
            lo, hi = wilson_ci(v, st_w[n_k])
            vals_w.append(v)
            err_lo.append(max(0.0, v - lo))
            err_hi.append(max(0.0, hi - v))
            n_den.append(st_w[n_k])
        bars_w = ax_w.bar(
            x_g + offset, vals_w, bw_w, yerr=[err_lo, err_hi], capsize=2.8,
            label=f"$w = {w_val:g}$", color=w_colors[w_val], edgecolor='#111111',
        )
        for bar, v, nd in zip(bars_w, vals_w, n_den):
            ax_w.text(
                bar.get_x() + bar.get_width() / 2.0, v + 0.35,
                f"{v:.1f}%\n({nd//1000}k)",
                ha='center', va='bottom', fontsize=6.9, fontweight='bold', color='#111111',
            )
    ax_w.set_xticks(x_g)
    ax_w.set_xticklabels([m[2] for m in dr_sh_metrics], fontsize=9.0, fontweight='bold')
    ax_w.set_ylabel("Shape Auxiliary Divergence Rate (%)", fontsize=9.8)
    ax_w.set_title(p_title, fontsize=10.4, fontweight='bold')
    ax_w.set_ylim(0, max(16.0, float(ax_w.get_ylim()[1]) * 1.22))
    ax_w.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax_w.legend(loc='upper right', fontsize=8.8, title="Main Loss Weight $w$", framealpha=0.92)

plt.tight_layout()
save_and_show(fig7a, "Figure 7A: Shape Auxiliary Divergence DR_s^{(shape)} by Loss Weight w Across Shift Types")

# --- Figure 7B: Deterministically Sampled Image Strip — Three-Way Shape Divergence ---
rng_s6 = np.random.default_rng(29)

# Sample from std|flat|w=2.0 under FGSM-Linf eps_2=0.05 (where 3-way shape divergence DR in shape-shift is prominent)
df_3w_s6, X_adv_3w_s6 = load_slice_eval_and_adv('standard', 'flatten', 2.0, 'fgsm', 'linf', EPS_LINF[2])
y_m_3w  = df_3w_s6['y_shape_and_scale'].to_numpy()
y_sh_3w = y_m_3w // 3
y_sc_3w = y_m_3w % 3
p_m_3w  = df_3w_s6['pred_main'].to_numpy()
p_sh_3w = df_3w_s6['pred_shape'].to_numpy()
p_sc_3w = df_3w_s6['pred_scale'].to_numpy()

# 3-way shape contradiction in S_shape: y_sh != p_m//3, p_sh != y_sh, and p_sh != p_m//3 (with scale retained)
cand_3way_sh = np.where(
    (p_m_3w != y_m_3w)
    & ((p_m_3w // 3) != y_sh_3w) & ((p_m_3w % 3) == y_sc_3w)
    & (p_sh_3w != y_sh_3w) & (p_sh_3w != (p_m_3w // 3))
    & (p_sc_3w == y_sc_3w)
)[0]
idx_pick_3w = int(rng_s6.choice(cand_3way_sh))
r_3w = df_3w_s6.iloc[idx_pick_3w]

print("[Figure 7B Deterministic Sampling Log — np.random.default_rng(29)]:")
print(f"  Three-Way Shape Divergence State DR in S_shape (`std|flat|w=2.0`, FGSM-Linf eps_2=0.05): "
      f"pool = {len(cand_3way_sh):,} candidates -> selected sample_index = {idx_pick_3w}\n")

fig7b, axes7b = plt.subplots(1, 3, figsize=(12.4, 3.9), dpi=130)
img_clean_3w = X_CLEAN_TEST[idx_pick_3w]
img_adv_3w   = X_adv_3w_s6[idx_pick_3w]
delta_3w_5x  = np.clip(np.abs(img_adv_3w - img_clean_3w) * 5.0, 0.0, 1.0)
y_true_3w    = int(r_3w['y_shape_and_scale'])
p_main_3w    = int(r_3w['pred_main'])
p_sh_3w_c    = int(r_3w['pred_shape'])
p_sc_3w_c    = int(r_3w['pred_scale'])

axes7b[0].imshow(img_clean_3w, cmap='gray', vmin=0.0, vmax=1.0)
axes7b[0].set_title(
    f"Clean Input (sample_index={idx_pick_3w})\nTrue: {CLASS_NAMES_9[y_true_3w]}\n(True Shape = {SHAPE_NAMES[y_true_3w // 3]})",
    fontsize=9.0, fontweight='bold',
)
axes7b[0].axis('off')

axes7b[1].imshow(img_adv_3w, cmap='gray', vmin=0.0, vmax=1.0)
axes7b[1].set_title(
    f"3-Way Shape Contradiction (`std|flat|w=2.0`, FGSM-$L_\\infty$, $\\varepsilon_2=0.05$)\n"
    f"Main Shape = {SHAPE_NAMES[p_main_3w // 3]} ({CLASS_NAMES_9[p_main_3w]}, {r_3w['confidence']*100:.1f}%)\n"
    f"Shape Aux [D!] = {SHAPE_NAMES[p_sh_3w_c]} ({r_3w['confidence_shape']*100:.1f}%) | "
    f"Scale Aux [R] = {SCALE_NAMES[p_sc_3w_c]} ({r_3w['confidence_scale']*100:.1f}%)",
    fontsize=8.4, fontweight='bold',
)
axes7b[1].axis('off')

im7b = axes7b[2].imshow(delta_3w_5x, cmap='magma', vmin=0.0, vmax=1.0)
axes7b[2].set_title(
    f"Perturbation $|\\delta| \\times 5$ (State $\\mathrm{{DR}} \\in \\mathcal{{S}}_{{\\mathrm{{shape}}}}$)\n"
    f"(Truth $\\neq$ Main $\\neq$ Shape Aux)",
    fontsize=9.0, fontweight='bold',
)
axes7b[2].axis('off')
plt.colorbar(im7b, ax=axes7b[2], fraction=0.046, pad=0.04)

plt.tight_layout()
save_and_show(fig7b, "Figure 7B: Deterministic Sample Strip — Three-Way Shape Contradiction (Truth != Main != Shape Aux)")
# %%



# %% PART B — Section 7: Check — Is the Disagreement Uncertain or Confidently Contradictory?
print("=" * 128)
print("=== PART B — SECTION 7: CHECK — IS THE DISAGREEMENT UNCERTAIN OR CONFIDENTLY CONTRADICTORY? ===")
print("=" * 128)
print("Observed result:")
print("  - Compared with failed attacks, aligned (FAR), partially retained (PRR) and divergent (DR) heads are")
print("    overwhelmingly high-confidence: median confidence is ~100% and median entropy ~0 with IQR [0, 0], so quartiles")
print("    are uninformative and tail shares or means must be used.")
print("  - Divergent auxiliaries are confidently contradictory from eps_2 onward; at eps_1 they are noticeably less")
print("    confident. In PRR the resisting auxiliary is slightly less confident than the fooled one.")
print("  - Lower confidence in Full Retention (FRR) is a tail effect, not the typical sample: mean confidence of `main`")
print("    drops (more under PGD) and the scale head is more uncertain than the shape head, but medians stay very high.")
print("  - Aligned heads are as confident as unfooled predictions, so confidence alone does not separate them from failed")
print("    attacks (the comparison here is with failed attacks, not with clean images).")
print("-" * 128)

def _per_img_summary(arr):
    """Return (n, mean, med, q25, q75) from a 1D float32 array of per-image values."""
    n = int(len(arr))
    if n == 0:
        return (0, np.nan, np.nan, np.nan, np.nan)
    q25, med, q75 = np.percentile(arr, [25, 50, 75])
    return (n, float(np.mean(arr)), float(med), float(q25), float(q75))

# 1. Comprehensive archetype table across the 5 regimes organized by head role
archetype_roles = [
    ('fail', 'Failed (S^c)', 'main',          'Main Branch (9-class)'),
    ('fail', 'Failed (S^c)', 'shape',         'Shape Aux (3-class)'),
    ('fail', 'Failed (S^c)', 'scale',         'Scale Aux (3-class)'),
    ('far',  'FAR (Aligned)', 'main',         'Main Branch (9-class)'),
    ('far',  'FAR (Aligned)', 'shape',        'Shape Aux (3-class)'),
    ('far',  'FAR (Aligned)', 'scale',        'Scale Aux (3-class)'),
    ('frr',  'FRR (Retained)', 'main',        'Main Branch (9-class)'),
    ('frr',  'FRR (Retained)', 'shape',       'Shape Aux (3-class)'),
    ('frr',  'FRR (Retained)', 'scale',       'Scale Aux (3-class)'),
    ('prr',  'PRR (Partial)',  'main',        'Main Branch (9-class)'),
    ('prr',  'PRR (Partial)',  'resisting_aux', 'Resisting Aux (Retains Truth)'),
    ('prr',  'PRR (Partial)',  'fooled_aux',    'Fooled Aux (Follows Main)'),
    ('dr',   'DR (Diverged)',  'main',        'Main Branch (9-class)'),
    ('dr',   'DR (Diverged)',  'divergent_aux', 'Divergent Aux (3-Way / Collateral)'),
]

print("\n[1] Per-Image Confidence (%) & Normalized Shannon Entropy H~_t (%) Across the 5 Archetypes and Head Roles:")
print(f"{'Archetype':<16} | {'Head Role':<34} | {'Instances N':>13} | {'Mean Conf%':>10} | "
      f"{'Median Conf [Q25-Q75]':>24} || {'Mean H~_t%':>10} | {'Median H~_t [Q25-Q75]':>24}")
print("-" * 148)
last_arch = None
role_stats_cache = {}
for a_key, a_lbl, r_key, r_lbl in archetype_roles:
    if last_arch is not None and a_key != last_arch:
        print("-" * 148)
    last_arch = a_key
    c_arr = per_image_C[(a_key, r_key)]
    h_arr = per_image_H[(a_key, r_key)]
    nc, c_mean, c_med, c_q25, c_q75 = _per_img_summary(c_arr)
    nh, h_mean, h_med, h_q25, h_q75 = _per_img_summary(h_arr)
    role_stats_cache[(a_key, r_key)] = dict(
        n=nh, c_mean=c_mean, c_med=c_med, c_q25=c_q25, c_q75=c_q75,
        h_mean=h_mean, h_med=h_med, h_q25=h_q25, h_q75=h_q75,
    )
    print(f"{a_lbl:<16} | {r_lbl:<34} | {nc:13,d} | {c_mean:9.2f}% | "
          f"{c_med:6.2f}% [{c_q25:6.2f}, {c_q75:6.2f}] || {h_mean:9.2f}% | "
          f"{h_med:6.2f}% [{h_q25:6.2f}, {h_q75:6.2f}]")
print("-" * 148)

# 2. Breakdown of FRR and DR confidence and normalized entropy across eps_1..eps_5
print("\n[2] Boundary vs Divergence Uncertainty Across Budget Steps eps_1..eps_5 (FRR vs DR Pooled Means):")
print(f"{'Step':<8} | {'FRR N':>9} | {'FRR Main Conf / H~':>20} | {'FRR Shape Conf / H~':>20} | {'FRR Scale Conf / H~':>20} || "
      f"{'DR N (Heads)':>16} | {'DR Main Conf / H~':>20} | {'DR DivAux Conf / H~':>20}")
print("-" * 154)
eps_frr_dr = []
for s in EPS_STEPS:
    st_s = pool_slices(tree_df[tree_df['eps_step'] == s])
    eps_frr_dr.append((s, st_s))
    n_frr_s = st_s['n_arch_frr']
    n_dr_s  = st_s['n_arch_dr']
    n_dh_s  = st_s['n_div_heads']
    frr_m_str  = f"{st_s['frr_cm']:5.1f}% / {st_s['frr_em']:5.1f}%"  if n_frr_s >= N_MIN_BRANCH else "  LOW_N (<100)  "
    frr_sh_str = f"{st_s['frr_csh']:5.1f}% / {st_s['frr_esh']:5.1f}%" if n_frr_s >= N_MIN_BRANCH else "  LOW_N (<100)  "
    frr_sc_str = f"{st_s['frr_csc']:5.1f}% / {st_s['frr_esc']:5.1f}%" if n_frr_s >= N_MIN_BRANCH else "  LOW_N (<100)  "
    dr_m_str   = f"{st_s['dr_cm']:5.1f}% / {st_s['dr_em']:5.1f}%"    if n_dr_s  >= N_MIN_BRANCH else "  LOW_N (<100)  "
    dr_au_str  = f"{st_s['dr_c_only']:5.1f}% / {st_s['dr_e_only']:5.1f}%" if n_dh_s >= N_MIN_BRANCH else "  LOW_N (<100)  "
    print(f"{'eps_'+str(s):<8} | {n_frr_s:9,d} | {frr_m_str:>20} | {frr_sh_str:>20} | {frr_sc_str:>20} || "
          f"{f'{n_dr_s:,} ({n_dh_s:,})':>16} | {dr_m_str:>20} | {dr_au_str:>20}")
print("-" * 154)

# 3. Breakdown of FRR and DR confidence and normalized entropy by optimizer (FGSM vs PGD) and dataset (std vs con)
print("\n[3] FRR and DR Confidence (%) / Normalized Entropy H~_t (%) by Optimizer (FGSM vs PGD) and Dataset (`std` vs `con`):")
print(f"{'Subgroup':<16} | {'FRR N':>9} | {'FRR Main Conf / H~':>20} | {'FRR Shape Conf / H~':>20} | {'FRR Scale Conf / H~':>20} || "
      f"{'DR N (Heads)':>16} | {'DR Main Conf / H~':>20} | {'DR DivAux Conf / H~':>20}")
print("-" * 154)
for idx, (g_lbl, g_mask) in enumerate([
    ("Opt: FGSM", tree_df['attack'] == 'fgsm'),
    ("Opt: PGD",  tree_df['attack'] == 'pgd'),
    ("Data: std", tree_df['dataset'] == 'standard'),
    ("Data: con", tree_df['dataset'] == 'contrast'),
]):
    if idx == 2:
        print("-" * 154)
    st_g = pool_slices(tree_df[g_mask])
    n_frr_g, n_dr_g, n_dh_g = st_g['n_arch_frr'], st_g['n_arch_dr'], st_g['n_div_heads']
    print(f"{g_lbl:<16} | {n_frr_g:9,d} | "
          f"{st_g['frr_cm']:5.1f}% / {st_g['frr_em']:5.1f}% | "
          f"{st_g['frr_csh']:5.1f}% / {st_g['frr_esh']:5.1f}% | "
          f"{st_g['frr_csc']:5.1f}% / {st_g['frr_esc']:5.1f}% || "
          f"{f'{n_dr_g:,} ({n_dh_g:,})':>16} | "
          f"{st_g['dr_cm']:5.1f}% / {st_g['dr_em']:5.1f}% | "
          f"{st_g['dr_c_only']:5.1f}% / {st_g['dr_e_only']:5.1f}%")
print("=" * 128 + "\n")

# --- Figure 8: Full-Width Bar Chart of Low-Confidence Tail Share Pr(Conf_t < 90%) (%) Across Archetypes & Head Roles ---
fig8, ax_box = plt.subplots(figsize=(12.2, 5.4), dpi=130)

box_groups = [
    ('fail', "Failed ($\\mathcal{S}^c$)", [
        ('main',  'Main',      '#377eb8'),
        ('shape', 'Shape Aux', '#4daf4a'),
        ('scale', 'Scale Aux', '#984ea3'),
    ]),
    ('far', "FAR (Aligned)", [
        ('main',  'Main',      '#377eb8'),
        ('shape', 'Shape Aux', '#4daf4a'),
        ('scale', 'Scale Aux', '#984ea3'),
    ]),
    ('frr', "FRR (Retained)", [
        ('main',  'Main',      '#377eb8'),
        ('shape', 'Shape Aux', '#4daf4a'),
        ('scale', 'Scale Aux', '#984ea3'),
    ]),
    ('prr', "PRR (Partial)", [
        ('main',          'Main',          '#377eb8'),
        ('resisting_aux', 'Resisting Aux', '#1b9e77'),
        ('fooled_aux',    'Fooled Aux',    '#ff7f00'),
    ]),
    ('dr', "DR (Diverged)", [
        ('main',          'Main',          '#377eb8'),
        ('divergent_aux', 'Divergent Aux', '#e41a1c'),
    ]),
]

xtick_centers, xtick_labels = [], []
for g_idx, (a_key, a_title, roles_spec) in enumerate(box_groups):
    base_x = g_idx * 1.15
    k_roles = len(roles_spec)
    offsets = np.linspace(-0.26, 0.26, k_roles) if k_roles == 3 else np.array([-0.15, 0.15])
    n_arch = len(per_image_C[(a_key, 'main')])
    xtick_centers.append(base_x)
    xtick_labels.append(f"{a_title}\n($N={n_arch:,}$)")
    for off, (r_key, r_lbl, r_col) in zip(offsets, roles_spec):
        c_arr = per_image_C[(a_key, r_key)]
        low_conf_pct = float(np.mean(c_arr < 90.0) * 100.0) if len(c_arr) > 0 else 0.0
        lo_c, hi_c = wilson_ci(low_conf_pct, len(c_arr))
        ax_box.bar(
            base_x + off, low_conf_pct, width=0.22,
            yerr=[[max(0.0, low_conf_pct - lo_c)], [max(0.0, hi_c - low_conf_pct)]],
            capsize=2.8, facecolor=r_col, edgecolor='#111111', alpha=0.88,
        )
        ax_box.text(
            base_x + off, low_conf_pct + 0.45, f"{low_conf_pct:.1f}%",
            ha='center', va='bottom', fontsize=8.0, fontweight='bold', color='#111111',
        )

ax_box.set_xticks(xtick_centers)
ax_box.set_xticklabels(xtick_labels, fontsize=9.2, fontweight='bold')
ax_box.set_ylim(0, max(28.0, float(ax_box.get_ylim()[1]) * 1.18))
ax_box.set_ylabel("Share of Predictions With $\\mathrm{Conf}_t(x) < 90\\%$ (%)", fontsize=10.2)
ax_box.set_title(
    "Figure 8: Low-Confidence Tail Share ($\\mathrm{Conf}_t(x) < 90\\%$) Across Archetypes and Head Roles",
    fontsize=10.8, fontweight='bold',
)
ax_box.grid(True, axis='y', linestyle=':', alpha=0.6)

legend_roles = [
    mpatches.Patch(facecolor='#377eb8', edgecolor='#111111', label='Main Branch'),
    mpatches.Patch(facecolor='#4daf4a', edgecolor='#111111', label='Shape Aux (in Failed/FAR/FRR)'),
    mpatches.Patch(facecolor='#984ea3', edgecolor='#111111', label='Scale Aux (in Failed/FAR/FRR)'),
    mpatches.Patch(facecolor='#1b9e77', edgecolor='#111111', label='PRR: Resisting Aux (Retains Truth)'),
    mpatches.Patch(facecolor='#ff7f00', edgecolor='#111111', label='PRR: Fooled Aux (Follows Main)'),
    mpatches.Patch(facecolor='#e41a1c', edgecolor='#111111', label='DR: Divergent Aux ($o_k=\\mathrm{D}$)'),
]
ax_box.legend(handles=legend_roles, loc='upper right', fontsize=8.4, ncol=2, framealpha=0.92)

plt.tight_layout()
save_and_show(fig8, "Figure 8: Low-Confidence Tail Share (Conf < 90%) Across Archetypes and Head Roles")
# %%

# %% PART C — Section 8: What Happens at High Budget — Collapse Onto a Single Class
print("=" * 128)
print("=== PART C — SECTION 8: WHAT HAPPENS AT HIGH BUDGET — COLLAPSE ONTO A SINGLE CLASS (SINK ATTRACTORS) ===")
print("=" * 128)
print("Observed result:")
print("  - At the strict >= 99% single-class threshold, 23 / 240 slices collapse onto a `Large`-scale sink class")
print("    (`[8] Heart-Large` in 20, `[2] Square-Large` in 3). A constant prediction is right on 1/9 of a balanced test")
print("    set, which yields ASR = 8/9 = 88.89%, a (25, 25, 50) shift profile and cohen_w ~ 0.")
print("  - The 8/9 ceiling is broader than this detector: 49 slices sit near it, including `std | gap` under PGD-L2 at")
print("    eps_2..eps_5, where no single class exceeds 99%.")
print("  - Among the 15 slices with ASR >= 99.5%, 9 concentrate all predictions into two classes, while the others")
print("    spread over more classes. `con | flat` never collapses onto a single class but still reaches ~100% ASR under")
print("    PGD-L_inf, sometimes onto small-scale sinks.")
print("  - Auxiliaries collapse synchronously with `main` in 16/23 slices; scale lags in `std | flat` under FGSM and")
print("    shape lags in `con | gap` Square-Large slices.")
print("-" * 128)

# 1. Detailed table of all single-class collapsed slices (main_top1_pct >= 99.0%)
coll_df = tree_df[tree_df['main_top1_pct'] >= 99.0].sort_values(
    ['dataset', 'pool', 'weight', 'attack', 'norm', 'eps_step']
).reset_index(drop=True)

print(f"\n[1] All Single-Class Collapsed Slices (main_top1_pct >= 99.0%): {len(coll_df)} / 240 Slices:")
print(f"{'Model (d|pi|w)':<16} | {'Attack-Norm':<11} | {'Step (eps)':<13} | {'Sink Class':<18} | "
      f"{'Main Sink%':>10} | {'Shape Sink%':>11} | {'Scale Sink%':>11} | {'ASR%':>7} | {'cohen_w':>7} | {'3-Head Sync Status'}")
print("-" * 154)
for _, r_c in coll_df.iterrows():
    m_lbl_c = f"{DS_SHORT[r_c['dataset']]}|{POOL_SHORT[r_c['pool']]}|w={r_c['weight']:g}"
    an_lbl  = f"{r_c['attack'].upper()}-{r_c['norm'].upper()}"
    st_lbl  = f"e{int(r_c['eps_step'])} ({r_c['epsilon']:g})"
    sink_c  = int(r_c['main_top1_cls'])
    sink_nm = CLASS_NAMES_9[sink_c]
    m_sh_p  = float(r_c['main_top1_pct'])
    sh_p    = float(r_c['sh_match_sink_pct'])
    sc_p    = float(r_c['sc_match_sink_pct'])
    if m_sh_p >= 99.0 and sh_p >= 99.0 and sc_p >= 99.0:
        sync_str = "Synchronous (All >=99%)"
    elif sc_p < 99.0 and sh_p >= 99.0:
        sync_str = f"Desync: Scale lags ({sc_p:.1f}%)"
    elif sh_p < 99.0 and sc_p >= 99.0:
        sync_str = f"Desync: Shape lags ({sh_p:.1f}%)"
    else:
        sync_str = f"Desync: Sh={sh_p:.1f}%, Sc={sc_p:.1f}%"
    print(f"{m_lbl_c:<16} | {an_lbl:<11} | {st_lbl:<13} | {sink_nm:<18} | "
          f"{m_sh_p:9.2f}% | {sh_p:10.2f}% | {sc_p:10.2f}% | "
          f"{r_c['ASR']:6.2f}% | {r_c['cohen_w']:7.3f} | {sync_str}")
print("-" * 154)

# 2. Compact table grouping consecutive eps steps for identical (dataset, pool, weight, attack, norm, sink_class)
compact_groups = []
for (d_v, p_v, w_v, a_v, n_v, sink_c), grp in coll_df.groupby(
    ['dataset', 'pool', 'weight', 'attack', 'norm', 'main_top1_cls'], sort=False
):
    grp_s = grp.sort_values('eps_step')
    steps_g = grp_s['eps_step'].astype(int).tolist()
    eps_g   = grp_s['epsilon'].astype(float).tolist()
    step_str = f"e{steps_g[0]}" if len(steps_g) == 1 else f"e{steps_g[0]}..e{steps_g[-1]}"
    eps_str  = f"{eps_g[0]:g}"  if len(eps_g) == 1   else f"{eps_g[0]:g}..{eps_g[-1]:g}"
    m_mean, m_min, m_max     = float(grp_s['main_top1_pct'].mean()), float(grp_s['main_top1_pct'].min()), float(grp_s['main_top1_pct'].max())
    sh_mean, sh_min, sh_max  = float(grp_s['sh_match_sink_pct'].mean()), float(grp_s['sh_match_sink_pct'].min()), float(grp_s['sh_match_sink_pct'].max())
    sc_mean, sc_min, sc_max  = float(grp_s['sc_match_sink_pct'].mean()), float(grp_s['sc_match_sink_pct'].min()), float(grp_s['sc_match_sink_pct'].max())
    sync_all = (m_min >= 99.0) and (sh_min >= 99.0) and (sc_min >= 99.0)
    compact_groups.append(dict(
        dataset=d_v, pool=p_v, weight=float(w_v), attack=a_v, norm=n_v, sink_cls=int(sink_c),
        label=f"{DS_SHORT[d_v]}|{POOL_SHORT[p_v]}|w={w_v:g} | {a_v.upper()}-{n_v.upper()} ({step_str})",
        n_slices=len(grp_s), step_str=step_str, eps_str=eps_str,
        sink_name=CLASS_NAMES_9[int(sink_c)],
        m_mean=m_mean, m_min=m_min, m_max=m_max,
        sh_mean=sh_mean, sh_min=sh_min, sh_max=sh_max,
        sc_mean=sc_mean, sc_min=sc_min, sc_max=sc_max,
        sync_all=sync_all,
    ))

print(f"\n[2] Compact Grouped Collapse Table (Consecutive eps Steps Merged -> {len(compact_groups)} Groups):")
print(f"{'Configuration Group':<36} | {'Slices':>6} | {'eps Range':<12} | {'Sink Class':<18} | "
      f"{'Main Mean [Range]':>21} | {'Shape Aux Mean [Range]':>23} | {'Scale Aux Mean [Range]':>23}")
print("-" * 156)
for cg in compact_groups:
    m_rng  = f"{cg['m_mean']:6.2f}% [{cg['m_min']:5.1f},{cg['m_max']:5.1f}]"
    sh_rng = f"{cg['sh_mean']:6.2f}% [{cg['sh_min']:5.1f},{cg['sh_max']:5.1f}]"
    sc_rng = f"{cg['sc_mean']:6.2f}% [{cg['sc_min']:5.1f},{cg['sc_max']:5.1f}]"
    print(f"{cg['label']:<36} | {cg['n_slices']:6d} | {cg['eps_str']:<12} | {cg['sink_name']:<18} | "
          f"{m_rng:>21} | {sh_rng:>23} | {sc_rng:>23}")
print("-" * 156)

# 3. Summary counts by dataset x pooling family and sink class + mini-table of ASR >= 99.5% two-sink collapse slices
print("\n[3] Summary Counts of Single-Class Collapsed Slices by Model Family and by Sink Class:")
for f_lbl, d_v, p_v in [
    ('std|gap',  'standard', 'gap'),
    ('std|flat', 'standard', 'flatten'),
    ('con|gap',  'contrast', 'gap'),
    ('con|flat', 'contrast', 'flatten'),
]:
    sub_fc = coll_df[(coll_df['dataset'] == d_v) & (coll_df['pool'] == p_v)]
    sink_breakdown = ", ".join(
        f"{CLASS_NAMES_9[int(k)]}: {int(v)}" for k, v in sub_fc['main_top1_cls'].value_counts().items()
    ) or "None (0 collapsed slices)"
    print(f"  Family {f_lbl:<10} : {len(sub_fc):2d} / 60 slices collapsed  ({sink_breakdown})")

print("\n  Sink Class Totals Across All Collapsed Slices:")
for sink_k, cnt_k in coll_df['main_top1_cls'].value_counts().items():
    print(f"    {CLASS_NAMES_9[int(sink_k)]:<20} : {int(cnt_k):2d} / {len(coll_df)} collapsed slices")

high_asr_df = tree_df[tree_df['ASR'] >= 99.5].sort_values(
    ['dataset', 'pool', 'weight', 'attack', 'norm', 'eps_step']
).reset_index(drop=True)
print(f"\n[3b] Two-Sink Collapse Inspection: Slices with ASR >= 99.5% ({len(high_asr_df)} slices evade the 88.89% single-sink ceiling):")
print(f"{'Model (d|pi|w)':<16} | {'Attack-Norm':<11} | {'Step (eps)':<13} | {'ASR%':>7} | "
      f"{'Top-1 Class (Share%)':<28} | {'Top-2 Class (Share%)':<28} | {'Top1+Top2%':>10}")
print("-" * 128)
for _, r_h in high_asr_df.iterrows():
    m_lbl_h = f"{DS_SHORT[r_h['dataset']]}|{POOL_SHORT[r_h['pool']]}|w={r_h['weight']:g}"
    an_lbl  = f"{r_h['attack'].upper()}-{r_h['norm'].upper()}"
    st_lbl  = f"e{int(r_h['eps_step'])} ({r_h['epsilon']:g})"
    c1_str  = f"{CLASS_NAMES_9[int(r_h['main_top1_cls'])]} ({r_h['main_top1_pct']:.1f}%)"
    c2_str  = f"{CLASS_NAMES_9[int(r_h['main_top2_cls'])]} ({r_h['main_top2_pct']:.1f}%)"
    sum_12  = float(r_h['main_top1_pct'] + r_h['main_top2_pct'])
    print(f"{m_lbl_h:<16} | {an_lbl:<11} | {st_lbl:<13} | {r_h['ASR']:6.2f}% | "
          f"{c1_str:<28} | {c2_str:<28} | {sum_12:9.2f}%")
print("=" * 128 + "\n")

# --- Figure 9B: Deterministically Sampled Image Strip — 4 Distinct Ground-Truth Classes Collapsing Into One Sink ---
rng_s8 = np.random.default_rng(29)

# Use representative synchronous collapsed slice: std|gap|w=1.0 under PGD-Linf eps_4=0.15 (collapses to [8] Heart-Large)
rep_row = coll_df.iloc[0]
for _, cand_r in coll_df.iterrows():
    if (cand_r['dataset'] == 'standard' and cand_r['pool'] == 'gap' and cand_r['weight'] == 1.0
            and cand_r['attack'] == 'pgd' and cand_r['norm'] == 'linf' and int(cand_r['eps_step']) == 4):
        rep_row = cand_r
        break

rep_d, rep_p, rep_w = rep_row['dataset'], rep_row['pool'], float(rep_row['weight'])
rep_a, rep_n, rep_eps = rep_row['attack'], rep_row['norm'], float(rep_row['epsilon'])
sink_target = int(rep_row['main_top1_cls'])
sink_sh_t, sink_sc_t = sink_target // 3, sink_target % 3

df_col_s8, X_adv_col_s8 = load_slice_eval_and_adv(rep_d, rep_p, rep_w, rep_a, rep_n, rep_eps)
y_m_col  = df_col_s8['y_shape_and_scale'].to_numpy()
p_m_col  = df_col_s8['pred_main'].to_numpy()
p_sh_col = df_col_s8['pred_shape'].to_numpy()
p_sc_col = df_col_s8['pred_scale'].to_numpy()

# Pick 4 distinct non-sink ground-truth classes that all map to sink_target across all 3 heads
distinct_true_classes = [c for c in range(9) if c != sink_target][:4]
picked_indices_s8 = []
print(f"[Figure 9B Deterministic Sampling Log — np.random.default_rng(29) on `{DS_SHORT[rep_d]}|{POOL_SHORT[rep_p]}|w={rep_w:g}` "
      f"{rep_a.upper()}-{rep_n.upper()} eps={rep_eps:g} -> Sink {CLASS_NAMES_9[sink_target]}]:")
for c_true in distinct_true_classes:
    cands_c = np.where(
        (y_m_col == c_true)
        & (p_m_col == sink_target)
        & (p_sh_col == sink_sh_t)
        & (p_sc_col == sink_sc_t)
    )[0]
    pick_i = int(rng_s8.choice(cands_c))
    picked_indices_s8.append((c_true, pick_i, len(cands_c)))
    print(f"  True Class {CLASS_NAMES_9[c_true]:<18} -> pool = {len(cands_c):,} candidates -> selected sample_index = {pick_i}")
print()

fig9b, axes9b = plt.subplots(2, 4, figsize=(15.2, 7.0), dpi=130)
for col_i, (c_true, s_idx, pool_sz) in enumerate(picked_indices_s8):
    r_s8 = df_col_s8.iloc[s_idx]
    img_c = X_CLEAN_TEST[s_idx]
    img_a = X_adv_col_s8[s_idx]

    axes9b[0, col_i].imshow(img_c, cmap='gray', vmin=0.0, vmax=1.0)
    axes9b[0, col_i].set_title(
        f"Clean (idx={s_idx})\nTrue: {CLASS_NAMES_9[c_true]}",
        fontsize=9.0, fontweight='bold',
    )
    axes9b[0, col_i].axis('off')

    axes9b[1, col_i].imshow(img_a, cmap='gray', vmin=0.0, vmax=1.0)
    axes9b[1, col_i].set_title(
        f"Perturbed ($\\varepsilon={rep_eps:g}$)\n"
        f"Main: {CLASS_NAMES_9[int(r_s8['pred_main'])]} ({r_s8['confidence']*100:.0f}%)\n"
        f"Aux: {SHAPE_NAMES[int(r_s8['pred_shape'])]} ({r_s8['confidence_shape']*100:.0f}%), "
        f"{SCALE_NAMES[int(r_s8['pred_scale'])]} ({r_s8['confidence_scale']*100:.0f}%)",
        fontsize=8.3, fontweight='bold', color='#b71c1c',
    )
    axes9b[1, col_i].axis('off')

fig9b.suptitle(
    f"Figure 9B: Multi-Class Collapse Into Single Sink {CLASS_NAMES_9[sink_target]} "
    f"(`{DS_SHORT[rep_d]}|{POOL_SHORT[rep_p]}|w={rep_w:g}`, {rep_a.upper()}-$L_\\infty$, $\\varepsilon={rep_eps:g}$)",
    fontsize=11.0, fontweight='bold', y=1.01,
)
plt.tight_layout()
save_and_show(fig9b, "Figure 9B: Deterministic Sample Strip — Four Distinct Classes Collapsing Into a Single Sink Class")
# %%

# %% PART C — Section 9: Which Factors Explain the Variation at Each Node
print("=" * 128)
print("=== PART C — SECTION 9: WHICH FACTORS EXPLAIN THE VARIATION AT EACH NODE (D^2 ATTRIBUTION & SYNTHESIS) ===")
print("=" * 128)
print("Observed result:")
print("  - D^2 is a descriptive attribution of variation, not a causal effect.")
print("  - Node 0 (ASR): attack parameters explain much more than architecture.")
print("  - Node 1: mixed picture. Budget is the strongest single factor for SR_scale, SR_joint and cohen_w, and cohen_w is")
print("    explained mainly by attack factors; at group level, architecture explains more of SR_shape and SR_joint, and")
print("    attack factors more of SR_scale.")
print("  - Node 2, dominant regimes (FAR_s, PRR, collateral divergence): with few exceptions (e.g. ARR_{joint,RA}, where")
print("    optimizer main effect exceeds architecture main effect, but the data x pool interaction is 15.3%), pooling,")
print("    dataset and their interaction explain more than attack factors, though total main-effect D^2 on overall FAR is modest.")
print("  - Node 2, boundary regimes (FRR, ARR_{s,RR}): optimizer and budget explain most of the variation. DR remains")
print("    mostly architectural, with loss weight relevant for shape divergence.")
print("-" * 128)

tree_metrics_spec = [
    # (Node Group, Display Label, Math Label, DataFrame Column, Denominator Column, is_continuous)
    ("Node 0", "ASR",                 "$\\mathrm{ASR}$",                                   'ASR',          'n_total', False),
    ("Node 1", "SR_shape",            "$\\mathrm{SR}_{\\mathrm{shape}}$",                  'SR_shape',     'n_succ',  False),
    ("Node 1", "SR_scale",            "$\\mathrm{SR}_{\\mathrm{scale}}$",                  'SR_scale',     'n_succ',  False),
    ("Node 1", "SR_joint",            "$\\mathrm{SR}_{\\mathrm{joint}}$",                  'SR_joint',     'n_succ',  False),
    ("Node 1", "cohen_w",             "$\\mathrm{cohen\\_w}$",                             'cohen_w',      'n_succ',  True),
    ("Node 2", "FAR",                 "$\\mathrm{FAR}$ (Overall)",                         'FAR',          'n_succ',  False),
    ("Node 2", "FAR_shape",           "$\\mathrm{FAR}_{\\mathrm{shape}}$",                 'FAR_shape',    'n_sh',    False),
    ("Node 2", "FAR_scale",           "$\\mathrm{FAR}_{\\mathrm{scale}}$",                 'FAR_scale',    'n_sc',    False),
    ("Node 2", "FAR_joint",           "$\\mathrm{FAR}_{\\mathrm{joint}}$",                 'FAR_joint',    'n_jt',    False),
    ("Node 2", "FRR",                 "$\\mathrm{FRR}$ (Overall)",                         'FRR',          'n_succ',  False),
    ("Node 2", "ARR_{shape,RR}",      "$\\mathrm{ARR}_{\\mathrm{shape,RR}}$",              'ARR_shape_RR', 'n_sh',    False),
    ("Node 2", "ARR_{scale,RR}",      "$\\mathrm{ARR}_{\\mathrm{scale,RR}}$",              'ARR_scale_RR', 'n_sc',    False),
    ("Node 2", "ARR_{joint,RR}",      "$\\mathrm{ARR}_{\\mathrm{joint,RR}}$",              'ARR_joint_RR', 'n_jt',    False),
    ("Node 2", "PRR",                 "$\\mathrm{PRR}$ (Overall)",                         'PRR',          'n_succ',  False),
    ("Node 2", "ARR_{joint,RA}",      "$\\mathrm{ARR}_{\\mathrm{joint,RA}}$",              'ARR_joint_RA', 'n_jt',    False),
    ("Node 2", "ARR_{joint,AR}",      "$\\mathrm{ARR}_{\\mathrm{joint,AR}}$",              'ARR_joint_AR', 'n_jt',    False),
    ("Node 2", "ARR_{shape,AD}",      "$\\mathrm{ARR}_{\\mathrm{shape,AD}}$",              'ARR_shape_AD', 'n_sh',    False),
    ("Node 2", "ARR_{scale,DA}",      "$\\mathrm{ARR}_{\\mathrm{scale,DA}}$",              'ARR_scale_DA', 'n_sc',    False),
    ("Node 2", "DR",                  "$\\mathrm{DR}$ (Overall)",                          'DR',           'n_succ',  False),
    ("Node 2", "DR_shape^{(shape)}",  "$\\mathrm{DR}_{\\mathrm{shape}}^{(\\mathrm{shape})}$", 'DR_shape_sh', 'n_sh',  False),
    ("Node 2", "DR_scale^{(scale)}",  "$\\mathrm{DR}_{\\mathrm{scale}}^{(\\mathrm{scale})}$", 'DR_scale_sc', 'n_sc',  False),
    ("Node 2", "DR_joint^{(shape)}",  "$\\mathrm{DR}_{\\mathrm{joint}}^{(\\mathrm{shape})}$", 'DR_joint_sh', 'n_jt',  False),
    ("Node 2", "DR_joint^{(scale)}",  "$\\mathrm{DR}_{\\mathrm{joint}}^{(\\mathrm{scale})}$", 'DR_joint_sc', 'n_jt',  False),
]

# 1. Complete D^2 (%) attribution table across all 23 tree metrics
print("\n[1] Complete Explained Deviance Share D^2 (%) Across All 23 Tree Metrics (Single Factors, Grouped Totals, Top 2F):")
print(f"{'Node':<6} | {'Metric':<20} || {'eps':>6} | {'norm':>6} | {'atk':>6} | {'data':>6} | {'pool':>6} | {'wght':>6} || "
      f"{'Attack D2%':>10} | {'Arch D2%':>9} || {'Top 2-Way Interaction (D2%)':<26}")
print("-" * 142)

d2_matrix = np.zeros((len(tree_metrics_spec), 7), dtype=float)
top2_names = []
last_node = None

for row_i, (node_lbl, disp_lbl, math_lbl, col_m, denom_m, is_cont) in enumerate(tree_metrics_spec):
    if last_node is not None and node_lbl != last_node:
        print("-" * 142)
    last_node = node_lbl
    res_d2 = compute_d2_attribution(tree_df, col_m, denom_m, is_continuous=is_cont)
    p_eps  = res_d2['d1_pct']['eps']
    p_norm = res_d2['d1_pct']['norm']
    p_atk  = res_d2['d1_pct']['atk']
    p_data = res_d2['d1_pct']['data']
    p_pool = res_d2['d1_pct']['pool']
    p_wght = res_d2['d1_pct']['wght']
    t2_nm, t2_val, _ = res_d2['top2']

    d2_matrix[row_i, :] = [p_eps, p_norm, p_atk, p_data, p_pool, p_wght, t2_val]
    top2_names.append(t2_nm)

    print(f"{node_lbl:<6} | {disp_lbl:<20} || "
          f"{p_eps:5.1f}% | {p_norm:5.1f}% | {p_atk:5.1f}% | {p_data:5.1f}% | {p_pool:5.1f}% | {p_wght:5.1f}% || "
          f"{res_d2['atk_d2']:9.1f}% | {res_d2['arch_d2']:8.1f}% || {f'{t2_nm} ({t2_val:.1f}%)':<26}")
print("-" * 142)

# 2. Factor x Node Directional Synthesis Table (computed dynamically from pooled marginals)
st_e1, st_e5     = pool_slices(tree_df[tree_df['eps_step'] == 1]),      pool_slices(tree_df[tree_df['eps_step'] == 5])
st_li, st_l2     = pool_slices(tree_df[tree_df['norm'] == 'linf']),     pool_slices(tree_df[tree_df['norm'] == 'l2'])
st_fg, st_pg     = pool_slices(tree_df[tree_df['attack'] == 'fgsm']),   pool_slices(tree_df[tree_df['attack'] == 'pgd'])
st_st, st_co     = pool_slices(tree_df[tree_df['dataset'] == 'standard']), pool_slices(tree_df[tree_df['dataset'] == 'contrast'])
st_fl, st_ga     = pool_slices(tree_df[tree_df['pool'] == 'flatten']),  pool_slices(tree_df[tree_df['pool'] == 'gap'])
st_w05, st_w20   = pool_slices(tree_df[tree_df['weight'] == 0.5]),      pool_slices(tree_df[tree_df['weight'] == 2.0])
st_cf            = pool_slices(tree_df[(tree_df['dataset'] == 'contrast') & (tree_df['pool'] == 'flatten')])
st_cg            = pool_slices(tree_df[(tree_df['dataset'] == 'contrast') & (tree_df['pool'] == 'gap')])
st_sg            = pool_slices(tree_df[(tree_df['dataset'] == 'standard') & (tree_df['pool'] == 'gap')])
st_cf_fg2        = pool_slices(tree_df[(tree_df['dataset'] == 'contrast') & (tree_df['pool'] == 'flatten')
                                       & (tree_df['attack'] == 'fgsm') & (tree_df['norm'] == 'l2')])

synthesis_rows = [
    (
        "Budget (eps_1 -> eps_5)",
        f"Strongest driver: ASR jumps {st_e1['ASR']:.1f}% -> {st_e5['ASR']:.1f}%; single-class collapses start at eps_2",
        f"SR_single drops {st_e1['SR_single']:.1f}% -> {st_e5['SR_single']:.1f}%; cohen_w {st_e1['cohen_w']:.2f} -> {st_e5['cohen_w']:.2f} (25/25/50)",
        "FRR and FAR both drop between eps_1 and eps_2, then stay roughly constant",
    ),
    (
        "Norm (L_inf vs L_2)",
        f"L_inf > L_2 ({st_li['ASR']:.1f}% vs {st_l2['ASR']:.1f}%); L_inf drives most sink collapses",
        f"L_2 sharpens single-factor bias at eps_1; L_inf reaches 25/25/50 faster",
        "Negligible main effect on Node 2 regimes (norm D^2 <= 1.5% across all metrics)",
    ),
    (
        "Optimizer (FGSM vs PGD)",
        f"Iterative PGD > single-step FGSM ({st_pg['ASR']:.1f}% vs {st_fg['ASR']:.1f}%); breaks `con|flat` in L_2",
        f"PGD pushes SR_joint higher ({st_pg['SR_joint']:.1f}% vs {st_fg['SR_joint']:.1f}%)",
        f"FGSM sustains retention FRR ({st_fg['FRR']:.1f}% vs {st_pg['FRR']:.2f}%) & scale lag in `std|flat`",
    ),
    (
        "Dataset (`std` vs `con`)",
        f"`con` lowers ASR ({st_co['ASR']:.1f}% vs {st_st['ASR']:.1f}%) and prevents `con|flat` >=99% collapse",
        f"`con` shifts errors to SR_joint ({st_co['SR_joint']:.1f}% vs {st_st['SR_joint']:.1f}%)",
        f"`con` elevates PRR ({st_co['PRR']:.1f}% vs {st_st['PRR']:.1f}%) and FRR ({st_co['FRR']:.1f}% vs {st_st['FRR']:.1f}%)",
    ),
    (
        "Pooling (`flat` vs `gap`)",
        f"`flat` more resistant overall ({st_fl['ASR']:.1f}% vs `gap` {st_ga['ASR']:.1f}%)",
        f"Different biases: `gap` breaks scale ({st_ga['SR_scale']:.1f}%), `flat` has higher shape ({st_fl['SR_shape']:.1f}%)",
        f"`gap` higher FAR ({st_ga['FAR']:.1f}% vs {st_fl['FAR']:.1f}%); `flat` owns DD ({st_fl['DR_DD_total']:.2f}% vs {st_ga['DR_DD_total']:.2f}%)",
    ),
    (
        "Loss Weight (w: 0.5->2.0)",
        f"Negligible on ASR ({st_w05['ASR']:.1f}% vs {st_w20['ASR']:.1f}%, D^2 < 0.1%)",
        "No monotone effect on SR_joint (peaks at w=1.0)",
        f"Pooled upward trend in DR^(shape): {st_w05['DR_shape_aux']:.2f}% -> {st_w20['DR_shape_aux']:.2f}% (non-monotonic in `flat` shape- & scale-shift)",
    ),
    (
        "Interaction `data x pool`",
        f"`con|flat` resists FGSM-L2 (ASR={st_cf_fg2['ASR']:.1f}% pooled, 1.2-3.0% per w) while `con|gap` yields",
        f"`con|flat` strongly joint-shifted ({st_cf['SR_joint']:.1f}%) vs `std|gap` scale-shifted ({st_sg['SR_scale']:.1f}%)",
        f"In `con`: `con|flat` -> ARR_joint,AR={st_cf['ARR_joint_AR']:.1f}%, `con|gap` -> ARR_joint,RA={st_cg['ARR_joint_RA']:.1f}% (reversed in `std`)",
    ),
]

print("\n[2] Factor x Node Directional Synthesis Across the Conditional Probability Tree:")
print(f"{'Factor / Interaction':<25} | {'Node 0 (Attack Success ASR)':<44} | "
      f"{'Node 1 (Shift Typology SR_s & cohen_w)':<44} | {'Node 2 (Auxiliary Regimes FAR/FRR/PRR/DR)'}")
print("-" * 168)
for f_nm, n0_txt, n1_txt, n2_txt in synthesis_rows:
    print(f"{f_nm:<25} | {n0_txt:<44} | {n1_txt:<44} | {n2_txt}")
print("=" * 128 + "\n")

# --- Figure 10: D^2 (%) Attribution Heatmap Across the Tree (23 Metrics x [6 Factors + Top 2-Way Interaction]) ---
fig10, ax10 = plt.subplots(figsize=(13.8, 10.2), dpi=130)

im10 = ax10.imshow(d2_matrix, cmap='YlOrBr', vmin=0.0, vmax=80.0, aspect='auto')

col_labels_10 = [
    "Budget\n($\\varepsilon$)",
    "Norm\n($p$)",
    "Optimizer\n($\\alpha$)",
    "Dataset\n($d$)",
    "Pooling\n($\\pi$)",
    "Weight\n($w$)",
    "Dominant 2-Way\nInteraction (Top 2F)",
]
ax10.set_xticks(np.arange(len(col_labels_10)))
ax10.set_xticklabels(col_labels_10, fontsize=9.2, fontweight='bold')

y_labels_10 = [f"[{n_g}]  {m_l}" for (n_g, _, m_l, _, _, _) in tree_metrics_spec]
ax10.set_yticks(np.arange(len(y_labels_10)))
ax10.set_yticklabels(y_labels_10, fontsize=8.8, fontweight='bold')

for r_i in range(d2_matrix.shape[0]):
    for c_i in range(d2_matrix.shape[1]):
        val_rc = d2_matrix[r_i, c_i]
        txt_col = 'white' if val_rc >= 38.0 else '#111111'
        if c_i < 6:
            cell_txt = f"{val_rc:.1f}%"
        else:
            cell_txt = f"{top2_names[r_i]}: {val_rc:.1f}%"
        ax10.text(c_i, r_i, cell_txt, ha='center', va='center',
                  color=txt_col, fontsize=8.0, fontweight='bold')

# Group dividers between Node 0 (row 0), Node 1 (rows 1..4), and Node 2 sub-blocks
for y_div, lw_d, ls_d in [
    (0.5,  2.2, '-'),   # Node 0 | Node 1
    (4.5,  2.2, '-'),   # Node 1 | Node 2 (FAR)
    (8.5,  1.2, '--'),  # FAR    | FRR
    (12.5, 1.2, '--'),  # FRR    | PRR & Collateral
    (17.5, 1.2, '--'),  # PRR    | DR
]:
    ax10.axhline(y_div, color='#111111', linewidth=lw_d, linestyle=ls_d)

# Vertical dividers separating Attack Factors (0..2), Architecture Factors (3..5), and Top 2F (6)
ax10.axvline(2.5, color='#111111', linewidth=1.8, linestyle='--')
ax10.axvline(5.5, color='#111111', linewidth=2.0, linestyle='-')

ax10.set_title(
    "Figure 10: Explained Deviance Share $D^2$ (%) Across the Conditional Tree (Node 0 $\\to$ Node 1 $\\to$ Node 2)",
    fontsize=11.2, fontweight='bold', pad=14,
)
cbar10 = plt.colorbar(im10, ax=ax10, fraction=0.028, pad=0.02)
cbar10.set_label("Explained Deviance Share $D^2$ (%)", fontsize=10.0)

plt.tight_layout()
save_and_show(fig10, "Figure 10: Global D^2 (%) Attribution Heatmap Across All 23 Tree Metrics")


# %% Teardown
report.close()
# %%
