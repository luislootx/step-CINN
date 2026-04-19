"""
Section 6.7 (part a) -- Space-Varying Transport Velocity  v(x) = x
==================================================================
Linear advection with a position-dependent velocity field:

    u_t + x u_x = 0,     (x, t) in [0.5, 3.0] x [0, 0.5].

Method of characteristics: along dx/dt = x the solution is constant,
so the characteristic transform is

    xi(x, t) = G(x) - t,    G(x) = integral dx / v(x) = ln(x).

CIELM fits the IC in xi-space and evaluates at xi = ln(x) - t.
Analytic reference: u(x, t) = u_0(x e^{-t}).

Artifacts produced
------------------
figures/fig07a_vx_snapshots.png          three-time snapshot panel
results/results_07a_vx.json              10-seed statistics
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

X_MIN, X_MAX = 0.5, 3.0
T_MAX = 0.5


def v_field(x):
    return x


def G(x):
    return np.log(x)


def extended_ic_range(t_max):
    """xi = G(x) - t with x in [X_MIN, X_MAX] at t in [0, t_max] samples
    x * exp(-t) -> lower end X_MIN * exp(-t_max). Add a small buffer."""
    return (X_MIN * np.exp(-t_max) * 0.9, X_MAX * 1.05)


def smooth_gaussian_ic(x, center=1.3, width=0.4, amp=1.0):
    return amp * np.exp(-((x - center) / width) ** 2)


# ----------------------------------------------------------------------------
# CIELM solver (xi-space fit, characteristic evaluation)
# ----------------------------------------------------------------------------

def cielm_variable_v(config, snap_times):
    N_tanh = config['N_tanh']
    lam = config['lam']
    seed = config['seed']

    ic_min, ic_max = extended_ic_range(T_MAX)
    xi_min, xi_max = G(ic_min), G(ic_max)
    xi_span = xi_max - xi_min

    W_tanh, b_tanh = generate_tanh_weights(N_tanh, seed,
                                           scale=2.5, domain_scale=xi_span)

    x_ic = np.linspace(ic_min, ic_max, config['n_ic'])
    xi_ic = G(x_ic)
    y_ic = smooth_gaussian_ic(x_ic)
    H_ic = hidden_matrix(xi_ic, W_tanh, b_tanh)
    beta = solve_ridge(H_ic, y_ic, lam)
    ic_rmse = float(np.sqrt(np.mean((H_ic @ beta - y_ic) ** 2)))

    x_eval = np.linspace(X_MIN, X_MAX, config['n_eval'])
    snapshots = {}
    t0 = time.time()
    for ts in snap_times:
        xi_eval = G(x_eval) - ts
        H_eval = hidden_matrix(xi_eval, W_tanh, b_tanh)
        u_pred = H_eval @ beta
        u_ref = smooth_gaussian_ic(x_eval * np.exp(-ts))
        _, l2 = compute_errors(u_pred, u_ref)
        snapshots[f"t={ts:.2f}"] = {
            't': float(ts),
            'u_pred': u_pred, 'u_ref': u_ref, 'l2_error': l2,
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
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
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


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  Section 6.7a: Space-varying velocity v(x) = x")
    print("=" * 70)

    base_config = {
        'N_tanh': 80, 'lam': 1e-6,
        'n_ic': 500, 'n_eval': 1000,
    }
    snap_times = [0.0, 0.25, T_MAX]

    l2_final = []
    for seed in range(10):
        cfg = {**base_config, 'seed': seed}
        r = cielm_variable_v(cfg, [T_MAX])
        l2_final.append(r['snapshots'][f"t={T_MAX:.2f}"]['l2_error'])

    stats = {
        'l2_mean': float(np.mean(l2_final)),
        'l2_std':  float(np.std(l2_final)),
        'l2_all':  l2_final,
    }
    print(f"\n  10-seed statistics at t = {T_MAX}:")
    print(f"    L2 = {stats['l2_mean']:.5f} +- {stats['l2_std']:.5f}")

    cfg_show = {**base_config, 'seed': 7}
    res_show = cielm_variable_v(cfg_show, snap_times)
    plot_snapshots(res_show['x_eval'], res_show['snapshots'],
                   r'Space-Varying Velocity $v(x) = x$: Gaussian Transport',
                   fig_path('fig07a_vx_snapshots.png'))

    out = result_path('results_07a_vx.json')
    with open(out, 'w') as f:
        json.dump({
            'cielm': stats,
            'domain': {'x_min': X_MIN, 'x_max': X_MAX,
                       't_max': T_MAX, 'velocity': 'v(x) = x',
                       'xi': 'xi = ln(x) - t'},
        }, f, indent=2)
    print(f"  Saved: {out}")


if __name__ == '__main__':
    main()
