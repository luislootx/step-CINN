"""
Section 6.4 -- Periodic Advection with Square-Wave IC
=====================================================
Step-CIELM on a periodic advection with a discontinuous (square-wave) IC.
Two step neurons placed at the two initial discontinuities make the jump
representation exact; the tanh block fits the constant plateaus.

PDE:   u_t + v * u_x = 0,                              v in {1, 5, 10, 20}
BC:    u(0, t) = u(2*pi, t)                            (periodic)
IC:    u(x, 0) = 1 if pi/2 < x < 3*pi/2 else 0
Exact: u(x, t) = IC((x - v*t) mod 2*pi)

Ablation: the same solver is run with and without the step neurons so that
the contribution of the step block is isolated.

Artifacts produced
------------------
figures/fig04_square_wave_v5.png        snapshots, v = 5
figures/fig04_square_wave_v20.png       snapshots, v = 20
figures/fig04_step_ablation.png         step-neuron ablation bar chart
results/results_04_square_wave.json     numerical results, all v
"""

import json
import time
import numpy as np
import matplotlib.pyplot as plt

from _core import (
    generate_tanh_weights, hidden_matrix, solve_ridge, compute_errors,
    style_ax, add_legend, fig_path, result_path,
    C_BLUE, C_RED, C_GREEN, C_ORANGE, C_TEXT, BG,
)


# ----------------------------------------------------------------------------
# Problem setup
# ----------------------------------------------------------------------------

L = 2 * np.pi
T_FINAL = 1.0
SQUARE_DISC = [np.pi / 2, 3 * np.pi / 2]


def square_wave_ic(x):
    return np.where((x > SQUARE_DISC[0]) & (x < SQUARE_DISC[1]), 1.0, 0.0)


def exact_periodic(x, t, v):
    xi = np.mod(x - v * t, L)
    return square_wave_ic(xi)


# ----------------------------------------------------------------------------
# Step-CIELM solver (periodic)
# ----------------------------------------------------------------------------

def cielm_periodic(config, v, snap_times, use_steps=True):
    N_tanh = config['N_tanh']
    kappa = config['kappa']
    seed = config['seed']
    positions = SQUARE_DISC if use_steps else []

    W_tanh, b_tanh = generate_tanh_weights(N_tanh, seed,
                                           scale=2.5, domain_scale=L)

    x_ic = np.linspace(0, L, config['n_ic'], endpoint=False)
    y_ic = square_wave_ic(x_ic)
    H_ic = hidden_matrix(x_ic, W_tanh, b_tanh, positions, kappa)
    beta = solve_ridge(H_ic, y_ic, config['lam'])
    ic_rmse = float(np.sqrt(np.mean((H_ic @ beta - y_ic) ** 2)))

    x_eval = np.linspace(0, L, config['n_eval'], endpoint=False)
    snapshots = {}
    t0 = time.time()
    for ts in snap_times:
        xi = np.mod(x_eval - v * ts, L)
        H = hidden_matrix(xi, W_tanh, b_tanh, positions, kappa)
        u_pred = H @ beta
        u_ref = exact_periodic(x_eval, ts, v)
        l1, l2 = compute_errors(u_pred, u_ref)
        snapshots[f"t={ts:.2f}"] = {
            't': float(ts),
            'u_pred': u_pred, 'u_ref': u_ref,
            'l1_error': l1, 'l2_error': l2,
        }
    elapsed = time.time() - t0

    return {
        'ic_rmse': ic_rmse, 'elapsed_s': elapsed,
        'snapshots': snapshots, 'x_eval': x_eval,
    }


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def plot_snapshots(x_eval, snapshots, title, fname):
    keys = sorted(snapshots.keys(), key=lambda k: snapshots[k]['t'])
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    fig.patch.set_facecolor(BG)
    if n == 1:
        axes = [axes]
    for i, key in enumerate(keys):
        s = snapshots[key]
        ax = axes[i]
        style_ax(ax)
        ax.plot(x_eval, s['u_ref'], color=C_BLUE, linewidth=2.5, zorder=3,
                label='Exact')
        ax.plot(x_eval, s['u_pred'], color=C_RED, linewidth=2, linestyle='--',
                zorder=4, label='Step-CIELM')
        ax.set_title(f"t = {s['t']:.2f}   L$_2$ = {s['l2_error']:.4f}",
                     color=C_TEXT, fontsize=11, fontfamily='serif')
        ax.set_xlabel('x', color=C_TEXT, fontsize=10)
        if i == 0:
            ax.set_ylabel('u(x,t)', color=C_TEXT, fontsize=10)
        add_legend(ax, loc='best')
    fig.suptitle(title, color=C_TEXT, fontsize=14, fontfamily='serif', y=1.02)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"  Saved: {fname}")
    plt.close(fig)


def plot_ablation(results_with, results_without, fname):
    velocities = sorted(results_with.keys())
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG)
    style_ax(ax)
    x_pos = np.arange(len(velocities))
    width = 0.3
    ax.bar(x_pos - width / 2,
           [results_with[v]['l2_mean'] for v in velocities], width,
           yerr=[results_with[v]['l2_std'] for v in velocities],
           capsize=4, color=C_GREEN, edgecolor='#555', linewidth=0.8,
           label='Step-CIELM', zorder=3)
    ax.bar(x_pos + width / 2,
           [results_without[v]['l2_mean'] for v in velocities], width,
           yerr=[results_without[v]['l2_std'] for v in velocities],
           capsize=4, color=C_ORANGE, edgecolor='#555', linewidth=0.8,
           label='CIELM (tanh only)', zorder=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'v={v}' for v in velocities], color=C_TEXT)
    ax.set_ylabel('Relative L$_2$ error', color=C_TEXT, fontsize=11)
    ax.set_title('Step-Neuron Ablation on the Periodic Square Wave',
                 color=C_TEXT, fontsize=14, fontfamily='serif')
    add_legend(ax)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"  Saved: {fname}")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  Section 6.4: Periodic advection with square-wave IC")
    print("=" * 70)

    base_config = {
        'N_tanh': 80, 'kappa': 500.0, 'lam': 1e-6,
        'n_ic': 500, 'n_eval': 1000,
    }

    velocities = [1, 5, 10, 20]
    snap_times = [0.0, T_FINAL]
    results_with, results_without = {}, {}

    for v in velocities:
        all_l2_s, all_l2_n = [], []
        for seed in range(10):
            cfg = {**base_config, 'seed': seed}
            res_s = cielm_periodic(cfg, v, snap_times, use_steps=True)
            res_n = cielm_periodic(cfg, v, snap_times, use_steps=False)
            all_l2_s.append(res_s['snapshots'][f"t={T_FINAL:.2f}"]['l2_error'])
            all_l2_n.append(res_n['snapshots'][f"t={T_FINAL:.2f}"]['l2_error'])

        results_with[v] = {
            'l2_mean': float(np.mean(all_l2_s)),
            'l2_std':  float(np.std(all_l2_s)),
            'l2_all':  all_l2_s,
        }
        results_without[v] = {
            'l2_mean': float(np.mean(all_l2_n)),
            'l2_std':  float(np.std(all_l2_n)),
            'l2_all':  all_l2_n,
        }
        print(f"  v={v}: step L2 = {results_with[v]['l2_mean']:.4f} "
              f"+- {results_with[v]['l2_std']:.4f} | "
              f"tanh-only L2 = {results_without[v]['l2_mean']:.4f} "
              f"+- {results_without[v]['l2_std']:.4f}")

    # Snapshots at v = 5 and v = 20
    cfg_show = {**base_config, 'seed': 7}
    for v_show, fname in [(5, 'fig04_square_wave_v5.png'),
                          (20, 'fig04_square_wave_v20.png')]:
        res = cielm_periodic(cfg_show, v_show,
                             [0.0, 0.25, 0.50, 1.00], use_steps=True)
        plot_snapshots(res['x_eval'], res['snapshots'],
                       f'Periodic Square-Wave Advection, v = {v_show}',
                       fig_path(fname))

    plot_ablation(results_with, results_without, fig_path('fig04_step_ablation.png'))

    out = result_path('results_04_square_wave.json')
    with open(out, 'w') as f:
        json.dump({
            'step_cielm':   {str(v): r for v, r in results_with.items()},
            'cielm_tanh_only': {str(v): r for v, r in results_without.items()},
        }, f, indent=2)
    print(f"  Saved: {out}")


if __name__ == '__main__':
    main()
