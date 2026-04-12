"""
Experiment 1: Flicker spike detector validation.

Reads per-sequence diagnostic logs produced by the tracker and outputs:
  - Per-sequence console summary (spike %, run length stats)
  - Plot of event_density and event_density_ema over time
  - Vertical shading where event_spike == 1
  - Spike duration distribution

Usage:
    cd AMTTrack_v2/
    python tracking/analysis_flicker.py
"""

import numpy as np
import os
import matplotlib.pyplot as plt

RESULTS_DIR = 'output/test/tracking_results/amttrack/amttrack_felt'

# Flicker sequences to validate the spike detector against
TARGET_SEQUENCES = ['ball_2hz', 'ball_4hz', 'ball_8hz']

# Normal sequences to check false positive rate (fill in 2-3 FELT sequence names)
CONTROL_SEQUENCES = []


def load_txt(path):
    if not os.path.isfile(path):
        return None
    data = np.loadtxt(path)
    return data if data.ndim > 0 else data.reshape(1)


def extract_runs(binary_array):
    """Return list of lengths of consecutive-1 runs."""
    runs, count = [], 0
    for v in binary_array:
        if v == 1:
            count += 1
        else:
            if count > 0:
                runs.append(count)
            count = 0
    if count > 0:
        runs.append(count)
    return runs


def analyse_sequence(seq_name, results_dir):
    density = load_txt(os.path.join(results_dir, f'{seq_name}_event_density.txt'))
    ema     = load_txt(os.path.join(results_dir, f'{seq_name}_event_density_ema.txt'))
    spikes  = load_txt(os.path.join(results_dir, f'{seq_name}_event_spike.txt'))

    if density is None:
        print(f'[MISSING] {seq_name} — run tracking first')
        return

    spikes = spikes.astype(int)
    spike_runs = extract_runs(spikes)
    spike_pct  = spikes.mean() * 100
    mean_run   = float(np.mean(spike_runs)) if spike_runs else 0.0
    max_run    = int(np.max(spike_runs))    if spike_runs else 0

    print(f'\n{seq_name}:')
    print(f'  Total frames     : {len(density)}')
    print(f'  Spike frames     : {spikes.sum()} ({spike_pct:.1f}%)')
    print(f'  Distinct runs    : {len(spike_runs)}')
    print(f'  Mean run length  : {mean_run:.1f} frames')
    print(f'  Max run length   : {max_run} frames')
    print(f'  Density range    : [{density.min():.2f}, {density.max():.2f}]  '
          f'mean={density.mean():.2f}')

    # plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    ax = axes[0]
    ax.plot(density, label='event_density', alpha=0.7, linewidth=0.8)
    ax.plot(ema,     label='event_density_ema', linewidth=1.5)
    ax.fill_between(range(len(spikes)), density.min(), density.max(),
                    where=spikes == 1, alpha=0.25, color='red',
                    label='spike detected')
    ax.set_ylabel('Mean pixel intensity')
    ax.set_title(f'{seq_name} — event density over time')
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.step(range(len(spikes)), spikes, where='mid', color='red', linewidth=0.8)
    ax2.set_ylim(-0.1, 1.4)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['normal', 'spike'])
    ax2.set_xlabel('Frame')
    ax2.set_title('Spike flag')

    plt.tight_layout()
    out_path = f'{seq_name}_flicker_analysis.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  Plot saved       : {out_path}')

    # spike run distribution histogram (only if there are runs)
    if spike_runs:
        fig2, ax3 = plt.subplots(figsize=(7, 3))
        ax3.hist(spike_runs, bins=max(1, min(30, max_run)), edgecolor='black')
        ax3.set_xlabel('Spike run length (frames)')
        ax3.set_ylabel('Count')
        ax3.set_title(f'{seq_name} — spike run length distribution')
        plt.tight_layout()
        hist_path = f'{seq_name}_spike_runs.png'
        plt.savefig(hist_path, dpi=150)
        plt.close()
        print(f'  Run dist saved   : {hist_path}')


if __name__ == '__main__':
    print(f'Results dir: {RESULTS_DIR}')
    print('=' * 60)

    print('\n--- Flicker sequences ---')
    for seq in TARGET_SEQUENCES:
        analyse_sequence(seq, RESULTS_DIR)

    if CONTROL_SEQUENCES:
        print('\n--- Control sequences (false positive check) ---')
        for seq in CONTROL_SEQUENCES:
            analyse_sequence(seq, RESULTS_DIR)
    else:
        print('\n[INFO] No CONTROL_SEQUENCES defined — add normal FELT sequence '
              'names to check false positive rate.')
