"""
Section 6.5 -- Linear Acoustics System
======================================
CIELM applied to a 2x2 hyperbolic system by diagonalisation into
characteristic variables (Riemann invariants).

PDEs:
    p_t + rho0 c0^2 v_x = 0
    v_t + (1 / rho0) p_x = 0

Matrix form: u_t + A u_x = 0 with A = [[0, rho0 c0^2], [1/rho0, 0]].
Eigenvalues  lambda = +- c0.
Riemann invariants w = R^{-1} u decouple into two independent transports:
    w1_t + c0 w1_x = 0     (right-going)
    w2_t - c0 w2_x = 0     (left-going)
Each invariant is solved by its own CIELM with its own characteristic
shift.  Pressure and velocity are recovered as p = rho0 c0 (w1 - w2),
v = w1 + w2.

IC (Gaussian pressure pulse, zero velocity):
    p_0(x) = exp(-a x^2),   v_0(x) = 0,   a = 10.

Artifacts produced
------------------
figures/fig05_acoustics_snapshots.png     two-row snapshot panel
results/results_05_acoustics.json         10-seed statistics
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

X_MIN, X_MAX = -1.5, 1.5
T_FINAL = 0.8
RHO0, C0 = 1.0, 1.0
Z0 = RHO0 * C0
GAUSS_A = 10.0


def gaussian_ic_p(x):
    return np.exp(-GAUSS_A * x ** 2)


def gaussian_ic_v(x):
    return np.zeros_like(x)


def gaussian_exact_p(x, t):
    return 0.5 * (np.exp(-GAUSS_A * (x - C0 * t) ** 2) +
                  np.exp(-GAUSS_A * (x + C0 * t) ** 2))


def gaussian_exact_v(x, t):
    return 1.0 / (2 * Z0) * (np.exp(-GAUSS_A * (x - C0 * t) ** 2) -
                             np.exp(-GAUSS_A * (x + C0 * t) ** 2))


# ----------------------------------------------------------------------------
# CIELM solver for the acoustics system
# ----------------------------------------------------------------------------

def cielm_acoustics(config, snap_times):
    """Two independent CIELMs, one per Riemann invariant."""
    N_tanh = config['N_tanh']
    lam = config['lam']
    seed = config['seed']
    span = X_MAX - X_MIN

    margin = C0 * T_FINAL + 0.3
    x_ic = np.linspace(X_MIN - margin, X_MAX + margin, config['n_ic'])

    p0 = gaussian_ic_p(x_ic)
    v0 = gaussian_ic_v(x_ic)
    w1_0 =  0.5 / Z0 * p0 + 0.5 * v0
    w2_0 = -0.5 / Z0 * p0 + 0.5 * v0

    W1, b1 = generate_tanh_weights(N_tanh, seed,       scale=2.5, domain_scale=span)
    W2, b2 = generate_tanh_weights(N_tanh, seed + 100, scale=2.5, domain_scale=span)

    H1 = hidden_matrix(x_ic, W1, b1)
    H2 = hidden_matrix(x_ic, W2, b2)
    beta1 = solve_ridge(H1, w1_0, lam)
    beta2 = solve_ridge(H2, w2_0, lam)

    ic_rmse_w1 = float(np.sqrt(np.mean((H1 @ beta1 - w1_0) ** 2)))
    ic_rmse_w2 = float(np.sqrt(np.mean((H2 @ beta2 - w2_0) ** 2)))

    x_eval = np.linspace(X_MIN, X_MAX, config['n_eval'])
    snapshots = {}

    t0 = time.time()
    for ts in snap_times:
        xi1 = x_eval - C0 * ts
        xi2 = x_eval + C0 * ts
        H1s = hidden_matrix(xi1, W1, b1)
        H2s = hidden_matrix(xi2, W2, b2)
        w1 = H1s @ beta1
        w2 = H2s @ beta2
        p_pred = Z0 * (w1 - w2)
        v_pred = w1 + w2
        p_ref = gaussian_exact_p(x_eval, ts)
        v_ref = gaussian_exact_v(x_eval, ts)

        _, p_l2 = compute_errors(p_pred, p_ref)
        _, v_l2 = compute_errors(v_pred, v_ref)

        snapshots[f"t={ts:.2f}"] = {
            't': float(ts),
            'p_pred': p_pred, 'p_ref': p_ref, 'p_l2': p_l2,
            'v_pred': v_pred, 'v_ref': v_ref, 'v_l2': v_l2,
        }
    elapsed = time.time() - t0

    return {
        'ic_rmse_w1': ic_rmse_w1, 'ic_rmse_w2': ic_rmse_w2,
        'elapsed_s': elapsed, 'snapshots': snapshots,
        'x_eval': x_eval,
    }


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------

def plot_acoustics(x_eval, snapshots, title, fname):
    keys = sorted(snapshots.keys(), key=lambda k: snapshots[k]['t'])
    n = len(keys)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 5), sharex=True)
    fig.patch.set_facecolor(BG)

    for i, key in enumerate(keys):
        s = snapshots[key]
        ax_p = axes[0, i]
        style_ax(ax_p)
        ax_p.plot(x_eval, s['p_ref'],  color=C_BLUE, linewidth=2.2, label='Exact')
        ax_p.plot(x_eval, s['p_pred'], color=C_RED,  linewidth=1.6,
                  linestyle='--', label='CIELM')
        ax_p.set_title(f"t = {s['t']:.2f}   p L$_2$ = {s['p_l2']:.4f}",
                       color=C_TEXT, fontsize=10, fontfamily='serif')
        if i == 0:
            ax_p.set_ylabel('p(x,t)', color=C_TEXT, fontsize=10)
        add_legend(ax_p, loc='best')

        ax_v = axes[1, i]
        style_ax(ax_v)
        ax_v.plot(x_eval, s['v_ref'],  color=C_BLUE, linewidth=2.2, label='Exact')
        ax_v.plot(x_eval, s['v_pred'], color=C_RED,  linewidth=1.6,
                  linestyle='--', label='CIELM')
        ax_v.set_title(f"v L$_2$ = {s['v_l2']:.4f}",
                       color=C_TEXT, fontsize=10, fontfamily='serif')
        ax_v.set_xlabel('x', color=C_TEXT, fontsize=10)
        if i == 0:
            ax_v.set_ylabel('v(x,t)', color=C_TEXT, fontsize=10)
        add_legend(ax_v, loc='best')

    fig.suptitle(title, color=C_TEXT, fontsize=13, fontfamily='serif', y=1.01)
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
    print("  Section 6.5: Linear acoustics system")
    print("=" * 70)

    base_config = {
        'N_tanh': 200, 'lam': 1e-8,
        'n_ic': 1000, 'n_eval': 1000,
    }

    p_l2s, v_l2s, times = [], [], []
    for seed in range(10):
        cfg = {**base_config, 'seed': seed}
        res = cielm_acoustics(cfg, [0.0, T_FINAL])
        final = res['snapshots'][f"t={T_FINAL:.2f}"]
        p_l2s.append(final['p_l2'])
        v_l2s.append(final['v_l2'])
        times.append(res['elapsed_s'])

    print(f"\n  CIELM (10 seeds):")
    print(f"    p L2 = {np.mean(p_l2s):.4f} +- {np.std(p_l2s):.4f}")
    print(f"    v L2 = {np.mean(v_l2s):.4f} +- {np.std(v_l2s):.4f}")
    print(f"    time = {np.mean(times):.4f}s")
    print(f"  Reference (Braga-Neto 2023, Table 4):")
    print(f"    CINN p L2 = 0.1267 +- 0.0534   (1000 ADAM iterations)")
    print(f"    PINN p L2 = 0.5209 +- 0.2774   (5000 ADAM iterations)")

    cfg_show = {**base_config, 'seed': 7}
    res_show = cielm_acoustics(cfg_show, [0.0, 0.05, 0.40, 0.81])
    plot_acoustics(res_show['x_eval'], res_show['snapshots'],
                   r'Linear Acoustics System: Gaussian Pressure Pulse',
                   fig_path('fig05_acoustics_snapshots.png'))

    out = result_path('results_05_acoustics.json')
    with open(out, 'w') as f:
        json.dump({
            'cielm': {
                'p_l2_mean': float(np.mean(p_l2s)),
                'p_l2_std':  float(np.std(p_l2s)),
                'v_l2_mean': float(np.mean(v_l2s)),
                'v_l2_std':  float(np.std(v_l2s)),
                'time_mean': float(np.mean(times)),
                'all_p_l2': p_l2s, 'all_v_l2': v_l2s,
            },
            'reference_braga_neto_2023': {
                'cinn_p_l2': '0.1267+-0.0534',
                'pinn_p_l2': '0.5209+-0.2774',
            },
        }, f, indent=2)
    print(f"  Saved: {out}")


if __name__ == '__main__':
    main()
