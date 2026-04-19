"""
Section 6.8 -- Two-Dimensional Linear Advection
===============================================
CIELM extended to two spatial dimensions.  The characteristic transform
is a vector shift along the constant velocity v = (v_x, v_y):

    xi = x - v * t.

PDE:
    u_t + v_x u_x + v_y u_y = 0,      (x, y) in [-1, 1]^2
IC:
    u_0(x, y) = exp(-alpha (x^2 + y^2))
Exact:
    u(x, y, t) = u_0(x - v_x t, y - v_y t)

Method: a single-hidden-layer ELM in 2D,

    u_hat(x, y) = sum_i beta_i tanh(w_x^i x + w_y^i y + b^i),

with random fixed (w_x, w_y, b) and beta from a single ridge solve.
Evaluating at time t is a coordinate shift (x, y) -> (x - v_x t, y - v_y t).

Artifacts produced
------------------
figures/fig08_2d_advection.png          three-row exact / CIELM / error maps
results/results_08_2d.json              10-seed statistics
"""

import json
import time
import numpy as np
import matplotlib.pyplot as plt

from _core import (
    generate_tanh_weights_2d, hidden_matrix_2d, solve_ridge, compute_errors,
    fig_path, result_path, C_TEXT, BG,
)


# ----------------------------------------------------------------------------
# Problem setup
# ----------------------------------------------------------------------------

X_MIN, X_MAX = -1.0, 1.0
Y_MIN, Y_MAX = -1.0, 1.0
V_X, V_Y = 1.0, 0.7
T_FINAL = 0.5
ALPHA = 10.0


def ic_gaussian_2d(x, y):
    return np.exp(-ALPHA * (x ** 2 + y ** 2))


def exact_2d(x, y, t):
    return ic_gaussian_2d(x - V_X * t, y - V_Y * t)


# ----------------------------------------------------------------------------
# CIELM solver for 2D advection
# ----------------------------------------------------------------------------

def cielm_2d(config, snap_times):
    N_tanh = config['N_tanh']
    lam = config['lam']
    seed = config['seed']
    span = max(X_MAX - X_MIN, Y_MAX - Y_MIN) + \
           2 * max(abs(V_X), abs(V_Y)) * T_FINAL + 1.0

    Wx, Wy, b_tanh = generate_tanh_weights_2d(N_tanh, seed,
                                              scale=2.5, domain_scale=span)

    margin = max(abs(V_X), abs(V_Y)) * T_FINAL + 0.3
    n_ic = config['n_ic_per_axis']
    x_ic = np.linspace(X_MIN - margin, X_MAX + margin, n_ic)
    y_ic = np.linspace(Y_MIN - margin, Y_MAX + margin, n_ic)
    Xg, Yg = np.meshgrid(x_ic, y_ic)
    x_flat, y_flat = Xg.ravel(), Yg.ravel()

    u0_flat = ic_gaussian_2d(x_flat, y_flat)
    H_ic = hidden_matrix_2d(x_flat, y_flat, Wx, Wy, b_tanh)
    beta = solve_ridge(H_ic, u0_flat, lam)
    ic_rmse = float(np.sqrt(np.mean((H_ic @ beta - u0_flat) ** 2)))

    n_eval = config['n_eval_per_axis']
    x_ev = np.linspace(X_MIN, X_MAX, n_eval)
    y_ev = np.linspace(Y_MIN, Y_MAX, n_eval)
    Xe, Ye = np.meshgrid(x_ev, y_ev)
    xe_flat, ye_flat = Xe.ravel(), Ye.ravel()

    snapshots = {}
    for ts in snap_times:
        xi_x = xe_flat - V_X * ts
        xi_y = ye_flat - V_Y * ts
        H_eval = hidden_matrix_2d(xi_x, xi_y, Wx, Wy, b_tanh)
        u_pred = (H_eval @ beta).reshape(n_eval, n_eval)
        u_ref  = exact_2d(Xe, Ye, ts)
        _, l2 = compute_errors(u_pred.ravel(), u_ref.ravel())
        snapshots[f"t={ts:.2f}"] = {
            't': float(ts),
            'u_pred': u_pred, 'u_ref': u_ref,
            'Xe': Xe, 'Ye': Ye, 'l2_error': l2,
        }
    return {'ic_rmse': ic_rmse, 'snapshots': snapshots, 'n_params': N_tanh}


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------

def plot_snapshots_2d(snapshots, title, fname):
    keys = sorted(snapshots.keys(), key=lambda k: snapshots[k]['t'])
    n = len(keys)
    fig, axes = plt.subplots(3, n, figsize=(3.5 * n, 9))
    fig.patch.set_facecolor(BG)
    if n == 1:
        axes = axes.reshape(3, 1)

    for i, key in enumerate(keys):
        s = snapshots[key]
        Xe, Ye = s['Xe'], s['Ye']
        u_pred, u_ref = s['u_pred'], s['u_ref']
        err = u_pred - u_ref
        vmax = max(u_ref.max(), u_pred.max())

        ax = axes[0, i]
        im = ax.pcolormesh(Xe, Ye, u_ref, cmap='viridis', vmin=0, vmax=vmax,
                           shading='auto')
        ax.set_aspect('equal')
        ax.set_title(f"Exact at t = {s['t']:.2f}", color=C_TEXT,
                     fontsize=10, fontfamily='serif')
        if i == 0:
            ax.set_ylabel('y', color=C_TEXT)
        plt.colorbar(im, ax=ax, shrink=0.8)

        ax = axes[1, i]
        im = ax.pcolormesh(Xe, Ye, u_pred, cmap='viridis',
                           vmin=0, vmax=vmax, shading='auto')
        ax.set_aspect('equal')
        ax.set_title(f"CIELM at t = {s['t']:.2f}", color=C_TEXT,
                     fontsize=10, fontfamily='serif')
        if i == 0:
            ax.set_ylabel('y', color=C_TEXT)
        plt.colorbar(im, ax=ax, shrink=0.8)

        ax = axes[2, i]
        err_max = max(1e-6, np.max(np.abs(err)))
        im = ax.pcolormesh(Xe, Ye, err, cmap='RdBu_r',
                           vmin=-err_max, vmax=err_max, shading='auto')
        ax.set_aspect('equal')
        ax.set_title(rf"Error   $L_2 = {s['l2_error']:.2e}$",
                     color=C_TEXT, fontsize=10, fontfamily='serif')
        ax.set_xlabel('x', color=C_TEXT)
        if i == 0:
            ax.set_ylabel('y', color=C_TEXT)
        plt.colorbar(im, ax=ax, shrink=0.8)

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
    print("  Section 6.8: Two-dimensional linear advection")
    print("=" * 70)
    print(f"  v = ({V_X}, {V_Y}),  T = {T_FINAL},  alpha = {ALPHA}")

    config = {
        'N_tanh': 1200,
        'lam': 1e-8,
        'n_ic_per_axis': 80,
        'n_eval_per_axis': 100,
        'seed': 7,
    }

    snap_times = [0.0, 0.25, T_FINAL]
    t0 = time.perf_counter()
    res = cielm_2d(config, snap_times)
    elapsed = time.perf_counter() - t0
    print(f"\n  Representative seed: IC fit RMSE = {res['ic_rmse']:.6f}")
    print(f"  Total solve time: {elapsed:.2f}s")
    for key in sorted(res['snapshots']):
        s = res['snapshots'][key]
        print(f"    {key}:  L2 = {s['l2_error']:.3e}")

    plot_snapshots_2d(res['snapshots'],
                      r'2D Linear Advection $v = (1.0, 0.7)$: '
                      r'Exact, CIELM, and Error',
                      fig_path('fig08_2d_advection.png'))

    # 10-seed statistics at the final time
    l2s = []
    for s in range(10):
        cfg = {**config, 'seed': s}
        r = cielm_2d(cfg, [T_FINAL])
        l2s.append(r['snapshots'][f"t={T_FINAL:.2f}"]['l2_error'])
    stats = {
        'l2_mean': float(np.mean(l2s)),
        'l2_std':  float(np.std(l2s)),
        'l2_all':  l2s,
    }
    print(f"\n  10-seed statistics at t = {T_FINAL}: "
          f"L2 = {stats['l2_mean']:.3e} +- {stats['l2_std']:.3e}")

    out = result_path('results_08_2d.json')
    with open(out, 'w') as f:
        json.dump({
            'cielm': stats,
            'config': {k: v for k, v in config.items() if k != 'seed'},
            'problem': {
                'x_range': [X_MIN, X_MAX],
                'y_range': [Y_MIN, Y_MAX],
                'velocity': [V_X, V_Y],
                'alpha': ALPHA,
                't_final': T_FINAL,
            },
        }, f, indent=2)
    print(f"  Saved: {out}")


if __name__ == '__main__':
    main()
