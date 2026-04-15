"""
Phase 1 analysis: find the best failure signal to replace pred_score.

Reads per-sequence diagnostic logs and outputs:
  1. Signal separation table (AUC-ranked)
  2. Per-sequence 3-panel diagnostic plots (<seq>_signals.png)
  3. Threshold recommendation

Usage:
    cd AMTTrack_v2/
    python tracking/analysis_signals.py
"""

import numpy as np
import os
import matplotlib.pyplot as plt

RESULTS_DIR = 'output/test/tracking_results/amttrack/amttrack_fe108'

SEQUENCES = {
    'airplane_mul222': 0.6802,
    'bike333':         0.5642,
    'tower':           0.9921,
    'ball_2hz':        0.4118,
    'ball_4hz':        0.0187,
    'ball_8hz':        0.1015,
}

# "bad" = clearly failed tracking; "good" = clearly working
BAD_THRESHOLD  = 0.2   # P < 0.2
GOOD_THRESHOLD = 0.5   # P > 0.5

SIGNALS = ['pred_score', 'norm_displacement', 'displacement_ema',
           'response_entropy', 'event_density']

# For each signal: True if higher = worse (failure), False if lower = worse
SIGNAL_HIGH_IS_BAD = {
    'pred_score':       False,  # low score = bad
    'norm_displacement': True,  # high displacement = bad
    'displacement_ema':  True,
    'response_entropy':  True,  # high entropy = diffuse = bad
    'event_density':     True,  # (unclear — included for completeness)
}


def load(seq, key):
    path = os.path.join(RESULTS_DIR, f'{seq}_{key}.txt')
    if not os.path.isfile(path):
        return None
    d = np.loadtxt(path)
    return d if d.ndim > 0 else d.reshape(1)


def auc_threshold_sweep(good_vals, bad_vals, high_is_bad):
    """Binary AUC: label bad=1, good=0. Sweep threshold over pooled distribution."""
    all_vals = np.concatenate([good_vals, bad_vals])
    labels   = np.concatenate([np.zeros(len(good_vals)), np.ones(len(bad_vals))])
    thresholds = np.percentile(all_vals, np.linspace(0, 100, 200))

    tprs, fprs = [], []
    for t in thresholds:
        if high_is_bad:
            pred = all_vals >= t
        else:
            pred = all_vals <= t
        tp = np.sum(pred & (labels == 1))
        fp = np.sum(pred & (labels == 0))
        fn = np.sum(~pred & (labels == 1))
        tn = np.sum(~pred & (labels == 0))
        tprs.append(tp / (tp + fn + 1e-8))
        fprs.append(fp / (fp + tn + 1e-8))

    # sort by fpr for trapezoidal integration
    pairs = sorted(zip(fprs, tprs))
    fprs_s = [p[0] for p in pairs]
    tprs_s = [p[1] for p in pairs]
    auc = float(np.trapz(tprs_s, fprs_s))
    return max(auc, 1 - auc)   # ensure AUC >= 0.5


def best_threshold(good_vals, bad_vals, high_is_bad):
    """Return threshold with best (TPR - FPR) on frame-level labels."""
    all_vals = np.concatenate([good_vals, bad_vals])
    labels   = np.concatenate([np.zeros(len(good_vals)), np.ones(len(bad_vals))])
    thresholds = np.percentile(all_vals, np.linspace(1, 99, 300))

    best_t, best_margin, best_tpr, best_fpr = None, -1, 0, 1
    for t in thresholds:
        pred = (all_vals >= t) if high_is_bad else (all_vals <= t)
        tp = np.sum(pred & (labels == 1))
        fp = np.sum(pred & (labels == 0))
        fn = np.sum(~pred & (labels == 1))
        tn = np.sum(~pred & (labels == 0))
        tpr = tp / (tp + fn + 1e-8)
        fpr = fp / (fp + tn + 1e-8)
        margin = tpr - fpr
        if margin > best_margin:
            best_margin, best_t, best_tpr, best_fpr = margin, t, tpr, fpr
    return best_t, best_tpr, best_fpr


# ── Load all data ─────────────────────────────────────────────────────────────
data = {}
for seq in SEQUENCES:
    data[seq] = {}
    for sig in SIGNALS:
        data[seq][sig] = load(seq, sig)

# ── OUTPUT 1: Signal separation table ────────────────────────────────────────
print(f'\n{"Signal":<22} {"Good mean":>10} {"Bad mean":>10} {"Ratio":>8} {"AUC":>6}')
print('─' * 62)

results_table = []
for sig in SIGNALS:
    good_all, bad_all = [], []
    for seq, prec in SEQUENCES.items():
        d = data[seq][sig]
        if d is None:
            continue
        if prec > GOOD_THRESHOLD:
            good_all.append(d)
        elif prec < BAD_THRESHOLD:
            bad_all.append(d)

    if not good_all or not bad_all:
        print(f'{sig:<22}  [insufficient data]')
        continue

    good_vals = np.concatenate(good_all)
    bad_vals  = np.concatenate(bad_all)
    good_mean = good_vals.mean()
    bad_mean  = bad_vals.mean()
    high_is_bad = SIGNAL_HIGH_IS_BAD[sig]
    ratio = bad_mean / good_mean if high_is_bad else good_mean / bad_mean
    auc   = auc_threshold_sweep(good_vals, bad_vals, high_is_bad)

    results_table.append((sig, good_mean, bad_mean, ratio, auc, good_vals, bad_vals))
    print(f'{sig:<22} {good_mean:>10.4f} {bad_mean:>10.4f} {ratio:>8.3f} {auc:>6.3f}')

# ── OUTPUT 2: Per-sequence 3-panel plots ─────────────────────────────────────
for seq, prec in SEQUENCES.items():
    score     = data[seq].get('pred_score')
    entropy   = data[seq].get('response_entropy')
    norm_disp = data[seq].get('norm_displacement')
    disp_ema  = data[seq].get('displacement_ema')
    density   = data[seq].get('event_density')

    if score is None:
        print(f'[MISSING] {seq} — run tracking first')
        continue

    frames = np.arange(len(score))
    label  = 'FLICKER' if prec < GOOD_THRESHOLD else 'control'
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    ax = axes[0]
    ax.plot(frames, score,   lw=0.7, label='pred_score',       color='steelblue')
    if entropy is not None:
        ax2 = ax.twinx()
        ax2.plot(frames, entropy, lw=0.7, label='response_entropy', color='darkorange', alpha=0.7)
        ax2.set_ylabel('entropy', color='darkorange', fontsize=8)
        ax2.tick_params(axis='y', labelcolor='darkorange', labelsize=7)
    ax.set_ylabel('pred_score')
    ax.legend(loc='upper left', fontsize=7)
    ax.set_title(f'{seq}  P@20px={prec:.4f}  [{label}]', fontsize=10)

    ax = axes[1]
    if norm_disp is not None:
        ax.plot(frames[:len(norm_disp)], norm_disp, lw=0.7,
                label='norm_displacement', color='green', alpha=0.8)
    if disp_ema is not None:
        ax.plot(frames[:len(disp_ema)], disp_ema, lw=1.5,
                label='displacement_ema', color='darkgreen')
    ax.set_ylabel('normalised displacement')
    ax.legend(loc='upper left', fontsize=7)

    ax = axes[2]
    if density is not None:
        ax.plot(frames[:len(density)], density, lw=0.7,
                label='event_density', color='purple', alpha=0.8)
    ax.set_ylabel('event density')
    ax.set_xlabel('Frame')
    ax.legend(loc='upper left', fontsize=7)

    plt.tight_layout()
    out_path = f'{seq}_signals.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'Saved: {out_path}')

# ── OUTPUT 3: Threshold recommendation ───────────────────────────────────────
if results_table:
    best = max(results_table, key=lambda r: r[4])  # sort by AUC
    sig, good_mean, bad_mean, ratio, auc, good_vals, bad_vals = best
    high_is_bad = SIGNAL_HIGH_IS_BAD[sig]
    t, tpr, fpr = best_threshold(good_vals, bad_vals, high_is_bad)

    print(f'\n{"─"*62}')
    print(f'BEST FAILURE SIGNAL   : {sig}')
    print(f'RECOMMENDED THRESHOLD : {t:.4f}')
    print(f'TRUE POSITIVE RATE    : {tpr*100:.1f}%')
    print(f'FALSE POSITIVE RATE   : {fpr*100:.1f}%')
    print(f'AUC                   : {auc:.3f}')
    print(f'Direction             : {"high = bad" if high_is_bad else "low = bad"}')
