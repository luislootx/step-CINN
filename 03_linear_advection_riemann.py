"""
Section 6.3 -- Linear Advection: Riemann Initial Condition
==========================================================
Step-CIELM on the canonical Riemann problem for constant-velocity
linear advection.  A single step neuron placed at the initial
discontinuity captures the jump exactly; the tanh block fits the
(zero) smooth background.

PDE:   u_t + v * u_x = 0,        v = 1
IC:    u(x, 0) = u_L  if x < L/2,   u_R  otherwise    (u_L = 5, u_R = 1)
BC:    u(0, t) = u_L,   u(L, t) = u_R
Exact: u(x, t) = u_0(x - v * t)         (profile translates at speed v)

Domain: [0, L] x [0, T_final],   L = 2,   T_final = 0.8.

Method: fit the IC on an extended domain [-v*T, L] so that the
characteristic coordinate xi = x - v*t remains within the fitted range
for all t in [0, T_final].  Evaluating the basis at xi gives the
solution at any time without evolution.

Artifacts produced
------------------
figures/fig03_riemann_snapshots.png    four-time snapshot panel
figures/fig03_riemann_comparison.png   bar chart: CIELM vs CINN vs PINN vs NN
results/results_03_riemann.json        10-seed statistics + reference numbers
"""

import json
import time
import numpy as np
import matplotlib.pyplot as plt

from _core import (
    generate_tanh_weights, hidden_matrix, solve_ridge, compute_errors,
    style_ax, add_legend, fig_path, result_path,
    C_BLUE, C_RED, C_GREEN, C_ORANGE, C_CYAN, C_TEXT, BG,
)


# ----------------------------------------------------------------------------
# Problem setup (matches Braga-Neto 2023, Section 4.1.1)
# ----------------------------------------------------------------------------

L = 2.0
T_FINAL = 0.8
V = 1.0
U_L = 5.0
U_R = 1.0
X_DISC = L / 2


def riemann_ic(x):
    return np.where(x < X_DISC, U_L, U_R)


def exact_solution(x, t):
    return riemann_ic(x - V * t)


# ----------------------------------------------------------------------------
# Step-CIELM solver
# ----------------------------------------------------------------------------

def stepcielm_riemann(config, snap_times):
    """Fit IC on an extended domain so xi = x - v*t stays in range; then
    evaluate at xi for each requested snapshot time."""
    N_tanh = config['N_tanh']
    kappa = config['kappa']
    seed = config['seed']
    positions = [X_DISC]

    W_tanh, b_tanh = generate_tanh_weights(N_tanh, seed,
                                           scale=2.5, domain_scale=L)

    xi_min = -V * T_FINAL - 0.2
    xi_max = L + 0.2
    x_ic = np.linspace(xi_min, xi_max, config['n_ic'])
    y_ic = riemann_ic(x_ic)
    H_ic = hidden_matrix(x_ic, W_tanh, b_tanh, positions, kappa)
    beta = solve_ridge(H_ic, y_ic, config['lam'])
    ic_rmse = float(np.sqrt(np.mean((H_ic @ beta - y_ic) ** 2)))

    x_eval = np.linspace(0, L, config['n_eval'])
    snapshots = {}
    t0 = time.time()
    for ts in snap_times:
        xi = x_eval - V * ts
        H = hidden_matrix(xi, W_tanh, b_tanh, positions, kappa)
        u_pred = H @ beta
        u_ref = exact_solution(x_eval, ts)
        l1, l2 = compute_errors(u_pred, u_ref)
        snapshots[f"t={ts:.2f}"] = {
            't': float(ts),
            'u_pred': u_pred, 'u_ref': u_ref,
            'step_position': float(X_DISC + V * ts),
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
        ax.axvline(s['step_position'], color=C_ORANGE, linewidth=1.5,
                   linestyle=':', alpha=0.7, zorder=2)
        ax.set_title(f"t = {s['t']:.2f}   L$_2$ = {s['l2_error']:.4f}",
                     color=C_TEXT, fontsize=11, fontfamily='serif')
        ax.set_xlabel('x', color=C_TEXT, fontsize=10)
        if i == 0:
            ax.set_ylabel('u(x,t)', color=C_TEXT, fontsize=10)
        add_legend(ax, loc='upper right')
    fig.suptitle(title, color=C_TEXT, fontsize=14, fontfamily='serif', y=1.02)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"  Saved: {fname}")
    plt.close(fig)


def plot_comparison(stats, fname):
    """Bar chart: Step-CIELM vs CINN vs PINN vs plain NN (L2 and wall time)."""
    methods = ['Step-CIELM', 'CINN', 'PINN', 'NN']
    l2_means = [stats['l2_mean'], 0.0550, 0.0619, 0.3120]
    l2_stds  = [stats['l2_std'],  0.0265, 0.0275, 0.1066]
    times    = [stats['time_mean'], 5.7, 10.2, 5.9]
    colors   = [C_GREEN, C_BLUE, C_CYAN, '#999999']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(BG)

    x_pos = np.arange(len(methods))
    ax = axes[0]
    style_ax(ax)
    bars = ax.bar(x_pos, l2_means, yerr=l2_stds, capsize=5,
                  color=colors, edgecolor='#555', linewidth=0.8, zorder=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=9, color=C_TEXT)
    ax.set_ylabel('Relative L$_2$ error', color=C_TEXT, fontsize=11)
    ax.set_title(r'$L_2$ Error at $t = T$ (10 seeds)',
                 color=C_TEXT, fontsize=13, fontfamily='serif')
    for bar, m, s in zip(bars, l2_means, l2_stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.01,
                f'{m:.4f}', ha='center', va='bottom', fontsize=8, color=C_TEXT)

    ax = axes[1]
    style_ax(ax)
    bars_t = ax.bar(x_pos, times, color=colors, edgecolor='#555',
                    linewidth=0.8, zorder=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=9, color=C_TEXT)
    ax.set_ylabel('Time (seconds)', color=C_TEXT, fontsize=11)
    ax.set_title('Wall-clock running time',
                 color=C_TEXT, fontsize=13, fontfamily='serif')
    for bar, t_val in zip(bars_t, times):
        label = f'{t_val:.3f}' if t_val < 0.1 else f'{t_val:.1f}'
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f'{label}s', ha='center', va='bottom', fontsize=8, color=C_TEXT)

    fig.suptitle('Linear Advection Riemann -- Method Comparison',
                 color=C_TEXT, fontsize=15, fontfamily='serif', y=1.02)
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
    print("  Section 6.3: Linear advection with Riemann IC")
    print("=" * 70)

    base_config = {
        'N_tanh': 80, 'kappa': 500.0, 'lam': 1e-6,
        'n_ic': 500, 'n_eval': 1000,
    }

    # 10-seed statistics
    l1s, l2s, times = [], [], []
    for seed in range(10):
        cfg = {**base_config, 'seed': seed}
        res = stepcielm_riemann(cfg, [T_FINAL])
        snap = res['snapshots'][f"t={T_FINAL:.2f}"]
        l1s.append(snap['l1_error'])
        l2s.append(snap['l2_error'])
        times.append(res['elapsed_s'])

    stats = {
        'l1_mean': float(np.mean(l1s)), 'l1_std': float(np.std(l1s)),
        'l2_mean': float(np.mean(l2s)), 'l2_std': float(np.std(l2s)),
        'time_mean': float(np.mean(times)), 'time_std': float(np.std(times)),
        'l1_all': l1s, 'l2_all': l2s, 'time_all': times,
    }
    print(f"\n  10-seed statistics:")
    print(f"    L1   = {stats['l1_mean']:.4f} +- {stats['l1_std']:.4f}")
    print(f"    L2   = {stats['l2_mean']:.4f} +- {stats['l2_std']:.4f}")
    print(f"    Time = {stats['time_mean']:.4f}s")
    print(f"  Reference (Braga-Neto 2023, Table 1):")
    print(f"    CINN L2 = 0.0550 +- 0.0265   time = 5.7s")
    print(f"    PINN L2 = 0.0619 +- 0.0275   time = 10.2s")
    print(f"    NN   L2 = 0.3120 +- 0.1066   time = 5.9s")

    # Representative snapshots
    cfg_show = {**base_config, 'seed': 7}
    res_show = stepcielm_riemann(cfg_show, [0.0, 0.08, 0.40, 0.73])
    plot_snapshots(res_show['x_eval'], res_show['snapshots'],
                   r'Linear Advection with Riemann IC, $v = 1$',
                   fig_path('fig03_riemann_snapshots.png'))

    plot_comparison(stats, fig_path('fig03_riemann_comparison.png'))

    out = result_path('results_03_riemann.json')
    with open(out, 'w') as f:
        json.dump({
            'stepcielm': stats,
            'reference_braga_neto_2023': {
                'CINN': {'l1': '0.0160+-0.0094', 'l2': '0.0550+-0.0265', 'time': '5.7s'},
                'PINN': {'l1': '0.0118+-0.0066', 'l2': '0.0619+-0.0275', 'time': '10.2s'},
                'NN':   {'l1': '0.1251+-0.0649', 'l2': '0.3120+-0.1066', 'time': '5.9s'},
            },
        }, f, indent=2)
    print(f"  Saved: {out}")


if __name__ == '__main__':
    main()
