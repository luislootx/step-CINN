"""
Section 6.7 (part b) -- Time-Varying Transport Velocity  v(t) = V cos(omega t)
===============================================================================
Linear advection with a periodic, time-varying velocity:

    u_t + V cos(omega t) u_x = 0.

Characteristic ODE: dx/dt = V cos(omega t) integrates to
x(t) = xi + (V / omega) sin(omega t).  With omega = 1 the transform
reduces to xi = x - V sin(t), and the solution advances for half a cycle,
retreats for the other half, and returns to its initial position at
t = 2*pi.  CIELM captures the full oscillation with a single IC fit.

Artifacts produced
------------------
figures/fig07b_vt_snapshots.png          five-phase snapshot panel
results/results_07b_vt.json              statistics at t = pi/2 and t = 2*pi
"""

import json
import time
import numpy as np
import matplotlib.pyplot as plt

from _core import (
    generate_tanh_weights, hidden_matrix, solve_ridge, compute_errors,
    style_ax, add_legend, fig_path, result_path,
    C_BLUE, C_RED, C_TEXT, BG,
)


# ----------------------------------------------------------------------------
# Problem setup
# ----------------------------------------------------------------------------

X_MIN, X_MAX = -1.0, 1.0
V_AMP = 1.0
T_MAX = 2.0 * np.pi


def G_of_t(t):
    """Integral of v = V cos(t) from 0 to t."""
    return V_AMP * np.sin(t)


def xi_transform(x, t):
    return x - G_of_t(t)


def extended_ic_range():
    return (X_MIN - V_AMP - 0.2, X_MAX + V_AMP + 0.2)


def smooth_gaussian_ic(x, center=0.0, width=0.3, amp=1.0):
    return amp * np.exp(-((x - center) / width) ** 2)


# ----------------------------------------------------------------------------
# CIELM solver
# ----------------------------------------------------------------------------

def cielm_time_varying(config, snap_times):
    N_tanh = config['N_tanh']
    lam = config['lam']
    seed = config['seed']

    ic_min, ic_max = extended_ic_range()
    span = ic_max - ic_min

    W_tanh, b_tanh = generate_tanh_weights(N_tanh, seed,
                                           scale=2.5, domain_scale=span)

    x_ic = np.linspace(ic_min, ic_max, config['n_ic'])
    y_ic = smooth_gaussian_ic(x_ic)
    H_ic = hidden_matrix(x_ic, W_tanh, b_tanh)
    beta = solve_ridge(H_ic, y_ic, lam)
    ic_rmse = float(np.sqrt(np.mean((H_ic @ beta - y_ic) ** 2)))

    x_eval = np.linspace(X_MIN, X_MAX, config['n_eval'])
    snapshots = {}
    t0 = time.time()
    for ts in snap_times:
        xi_eval = xi_transform(x_eval, ts)
        H_eval = hidden_matrix(xi_eval, W_tanh, b_tanh)
        u_pred = H_eval @ beta
        u_ref = smooth_gaussian_ic(x_eval - G_of_t(ts))
        _, l2 = compute_errors(u_pred, u_ref)
        snapshots[f"t={ts:.3f}"] = {
            't': float(ts),
            'u_pred': u_pred, 'u_ref': u_ref, 'l2_error': l2,
            'shift': float(G_of_t(ts)),
        }
    elapsed = time.time() - t0

    return {
        'ic_rmse': ic_rmse, 'elapsed_s': elapsed,
        'snapshots': snapshots, 'x_eval': x_eval,
    }


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------

def plot_snapshots(x_eval, snapshots, title, fname):
    keys = sorted(snapshots.keys(), key=lambda k: snapshots[k]['t'])
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.8), sharey=True)
    fig.patch.set_facecolor(BG)
    if n == 1:
        axes = [axes]
    for i, key in enumerate(keys):
        s = snapshots[key]
        ax = axes[i]
        style_ax(ax)
        ax.plot(x_eval, s['u_ref'],  color=C_BLUE, linewidth=2.5,
                label='Exact', zorder=3)
        ax.plot(x_eval, s['u_pred'], color=C_RED,  linewidth=2,
                linestyle='--', label='CIELM', zorder=4)
        ax.set_title(rf"t = {s['t']/np.pi:.2f}$\pi$   "
                     rf"L$_2$ = {s['l2_error']:.4f}",
                     color=C_TEXT, fontsize=10, fontfamily='serif')
        ax.set_xlabel('x', color=C_TEXT, fontsize=10)
        if i == 0:
            ax.set_ylabel('u(x,t)', color=C_TEXT, fontsize=10)
        ax.set_xlim(X_MIN, X_MAX)
        add_legend(ax, loc='best')
    fig.suptitle(title, color=C_TEXT, fontsize=13, fontfamily='serif', y=1.02)
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
    print("  Section 6.7b: Time-varying velocity v(t) = V cos(t)")
    print("=" * 70)

    base_config = {
        'N_tanh': 80, 'lam': 1e-6,
        'n_ic': 500, 'n_eval': 1000,
    }
    snap_times = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]

    l2_halfpi, l2_full = [], []
    for seed in range(10):
        cfg = {**base_config, 'seed': seed}
        r = cielm_time_varying(cfg, [np.pi / 2, 2 * np.pi])
        l2_halfpi.append(r['snapshots'][f"t={np.pi/2:.3f}"]['l2_error'])
        l2_full.append(r['snapshots'][f"t={2*np.pi:.3f}"]['l2_error'])

    stats = {
        't_pi_over_2': {'l2_mean': float(np.mean(l2_halfpi)),
                        'l2_std':  float(np.std(l2_halfpi))},
        't_2pi_full_cycle': {'l2_mean': float(np.mean(l2_full)),
                             'l2_std':  float(np.std(l2_full))},
    }
    print(f"\n  10-seed statistics:")
    print(f"    t = pi/2:  L2 = {stats['t_pi_over_2']['l2_mean']:.4e} "
          f"+- {stats['t_pi_over_2']['l2_std']:.4e}")
    print(f"    t = 2*pi:  L2 = {stats['t_2pi_full_cycle']['l2_mean']:.4e} "
          f"+- {stats['t_2pi_full_cycle']['l2_std']:.4e}")

    cfg_show = {**base_config, 'seed': 7}
    res_show = cielm_time_varying(cfg_show, snap_times)
    plot_snapshots(res_show['x_eval'], res_show['snapshots'],
                   r'Time-Varying Velocity $v(t) = V \cos t$: '
                   r'Oscillatory Transport',
                   fig_path('fig07b_vt_snapshots.png'))

    out = result_path('results_07b_vt.json')
    with open(out, 'w') as f:
        json.dump({
            'cielm': stats,
            'domain': {'x_min': X_MIN, 'x_max': X_MAX,
                       't_max_pi_mult': 2.0, 'V_amp': V_AMP,
                       'velocity': 'v(t) = V cos(t)',
                       'xi': 'xi = x - V sin(t)'},
        }, f, indent=2)
    print(f"  Saved: {out}")


if __name__ == '__main__':
    main()
