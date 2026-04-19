"""
Section 6.1 -- Periodic Advection with Smooth IC
================================================
CIELM versus CINN (stiffness immunity) on the canonical periodic
advection benchmark.

PDE:   u_t + v * u_x = 0                    (linear constant-velocity advection)
BC:    u(0, t) = u(2*pi, t)                 (periodic)
IC:    u_0(x) = sin(x)
Exact: u(x, t) = sin((x - v*t) mod 2*pi)

Method: fit a random-fixed tanh ELM to the IC on [0, 2*pi] and evaluate
the same basis at the characteristic coordinate xi = (x - v*t) mod 2*pi.
The PDE is satisfied by construction -- no residual loss, no collocation,
no gradient descent.

Artifacts produced
------------------
figures/fig01_periodic_snapshots.png         snapshots at v = 30
figures/fig01_periodic_velocity_sweep.png    L2 vs v for CIELM, CINN, PINN
results/results_01_periodic.json             10-seed statistics
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


def sin_ic(x):
    return np.sin(x)


def exact_periodic_sin(x, t, v):
    xi = np.mod(x - v * t, L)
    return np.sin(xi)


# ----------------------------------------------------------------------------
# CIELM solver (periodic, smooth)
# ----------------------------------------------------------------------------

def cielm_periodic(config, v, snap_times):
    """CIELM for u_t + v*u_x = 0 with periodic BC and smooth IC.

    Fit an ELM on [0, 2*pi]; evaluate at xi = (x - v*t) mod 2*pi.
    """
    N_tanh = config['N_tanh']
    lam = config['lam']
    seed = config['seed']

    W_tanh, b_tanh = generate_tanh_weights(N_tanh, seed,
                                           scale=2.5, domain_scale=L)

    x_ic = np.linspace(0, L, config['n_ic'], endpoint=False)
    y_ic = sin_ic(x_ic)
    H_ic = hidden_matrix(x_ic, W_tanh, b_tanh)
    beta = solve_ridge(H_ic, y_ic, lam)
    ic_rmse = float(np.sqrt(np.mean((H_ic @ beta - y_ic) ** 2)))

    x_eval = np.linspace(0, L, config['n_eval'], endpoint=False)
    snapshots = {}

    t0 = time.time()
    for ts in snap_times:
        xi = np.mod(x_eval - v * ts, L)
        H_eval = hidden_matrix(xi, W_tanh, b_tanh)
        u_pred = H_eval @ beta
        u_ref = exact_periodic_sin(x_eval, ts, v)
        l1, l2 = compute_errors(u_pred, u_ref)
        snapshots[f"t={ts:.2f}"] = {
            't': float(ts),
            'u_pred': u_pred, 'u_ref': u_ref,
            'l1_error': l1, 'l2_error': l2,
        }
    elapsed = time.time() - t0

    return {
        'ic_rmse': ic_rmse,
        'elapsed_s': elapsed,
        'snapshots': snapshots,
        'x_eval': x_eval,
        'n_params': len(beta),
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
                zorder=4, label='CIELM')
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


def plot_velocity_sweep(results, cinn_ref, pinn_ref, title, fname):
    """Bar chart: L2 error vs velocity for CIELM, CINN, PINN."""
    velocities = sorted(results.keys())
    cielm_l2 = [results[v]['l2_final'] for v in velocities]
    cielm_std = [results[v]['l2_std'] for v in velocities]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    style_ax(ax)

    x_pos = np.arange(len(velocities))
    width = 0.25

    ax.bar(x_pos - width, cielm_l2, width, yerr=cielm_std, capsize=4,
           color=C_GREEN, edgecolor='#555', linewidth=0.8,
           label='CIELM', zorder=3)

    cinn_l2  = [cinn_ref[v][0] for v in velocities]
    cinn_std = [cinn_ref[v][1] for v in velocities]
    ax.bar(x_pos, cinn_l2, width, yerr=cinn_std, capsize=4,
           color=C_BLUE, edgecolor='#555', linewidth=0.8,
           label='CINN', zorder=3)

    pinn_l2  = [pinn_ref[v][0] for v in velocities]
    pinn_std = [pinn_ref[v][1] for v in velocities]
    ax.bar(x_pos + width, pinn_l2, width, yerr=pinn_std, capsize=4,
           color=C_ORANGE, edgecolor='#555', linewidth=0.8,
           label='PINN', zorder=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'v={v}' for v in velocities], color=C_TEXT)
    ax.set_ylabel('Relative L$_2$ error', color=C_TEXT, fontsize=11)
    ax.set_title(title, color=C_TEXT, fontsize=14, fontfamily='serif')
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
    print("  Section 6.1: Periodic advection with smooth IC")
    print("=" * 70)

    base_config = {
        'N_tanh': 80, 'lam': 1e-6,
        'n_ic': 500, 'n_eval': 1000,
    }

    velocities = [20, 30, 40, 50]
    snap_times = [0.0, T_FINAL]
    results = {}

    for v in velocities:
        all_l2 = []
        for seed in range(10):
            cfg = {**base_config, 'seed': seed}
            res = cielm_periodic(cfg, v, snap_times)
            all_l2.append(res['snapshots'][f"t={T_FINAL:.2f}"]['l2_error'])
        results[v] = {
            'l2_final': float(np.mean(all_l2)),
            'l2_std':   float(np.std(all_l2)),
            'l2_all':   all_l2,
        }
        print(f"  v={v}: L2 = {results[v]['l2_final']:.5f} "
              f"+- {results[v]['l2_std']:.5f}")

    # Snapshots at v = 30 for visualization
    cfg_show = {**base_config, 'seed': 7}
    res_show = cielm_periodic(cfg_show, v=30,
                              snap_times=[0.0, 0.25, 0.50, 1.00])
    plot_snapshots(res_show['x_eval'], res_show['snapshots'],
                   r'Periodic Advection at $v = 30$: CIELM vs. Exact Solution',
                   fig_path('fig01_periodic_snapshots.png'))

    # Reference numbers from Braga-Neto (2023), CINN paper, Table 3
    cinn_ref = {20: (0.0300, 0.0043), 30: (0.0579, 0.0095),
                40: (0.0852, 0.0169), 50: (0.5365, 0.1729)}
    pinn_ref = {20: (0.0347, 0.0056), 30: (0.1003, 0.0188),
                40: (0.4395, 0.1120), 50: (0.7797, 0.0242)}

    plot_velocity_sweep(
        results, cinn_ref, pinn_ref,
        r'Velocity Robustness: $L_2$ Error for CIELM, CINN, and PINN',
        fig_path('fig01_periodic_velocity_sweep.png'))

    out = result_path('results_01_periodic.json')
    with open(out, 'w') as f:
        json.dump({
            'cielm': results,
            'cinn_reference': {str(k): v for k, v in cinn_ref.items()},
            'pinn_reference': {str(k): v for k, v in pinn_ref.items()},
        }, f, indent=2)
    print(f"  Saved: {out}")


if __name__ == '__main__':
    main()
