"""
Section 6.6 (part a) -- Inviscid Burgers: Riemann Shock & Rarefaction
=====================================================================
Two Riemann sub-problems for the inviscid Burgers equation:

    u_t + u u_x = 0

Shock case:        u_L = 1, u_R = 0       (Lax-admissible shock)
    Rankine-Hugoniot speed s = (u_L + u_R) / 2 = 0.5.
    Step-CIELM evaluates the single step neuron at xi = x - s*t.

Rarefaction case:  u_L = 0, u_R = 1       (expansion fan)
    Exact solution is a fan u = (x - x_disc) / t between the two
    characteristic rays.  Step-CIELM from the jump IC plus Picard
    iteration for xi = x - u_ELM(xi) t reproduces the fan.

Artifacts produced
------------------
figures/fig06a_burgers_shock.png           snapshots, shock case
figures/fig06a_burgers_rarefaction.png     snapshots, rarefaction case
results/results_06a_burgers_shock.json     numerical results, both cases
"""

import json
import time
import numpy as np
import matplotlib.pyplot as plt

from _core import (
    generate_tanh_weights, hidden_matrix, solve_ridge, compute_errors,
    picard_fixed_point,
    style_ax, add_legend, fig_path, result_path,
    C_BLUE, C_RED, C_ORANGE, C_TEXT, BG,
)


# ----------------------------------------------------------------------------
# Exact solutions
# ----------------------------------------------------------------------------

def burgers_shock_exact(x, t, u_L, u_R, x_disc):
    s = 0.5 * (u_L + u_R)
    return np.where(x < x_disc + s * t, u_L, u_R)


def burgers_rarefaction_exact(x, t, u_L, u_R, x_disc):
    if t < 1e-14:
        return np.where(x < x_disc, u_L, u_R)
    x_left  = x_disc + u_L * t
    x_right = x_disc + u_R * t
    return np.where(x < x_left, u_L,
             np.where(x > x_right, u_R, (x - x_disc) / t))


# ----------------------------------------------------------------------------
# Shock case: R-H characteristic shift
# ----------------------------------------------------------------------------

def run_shock(seed=7):
    print("\n" + "=" * 70)
    print("  Burgers shock (u_L = 1, u_R = 0)")
    print("=" * 70)

    u_L, u_R, x_disc = 1.0, 0.0, 0.0
    x_min, x_max = -1.0, 2.0
    snap_times = [0.0, 0.3, 0.8, 1.5]

    N_tanh, kappa, n_ic, n_eval, lam = 80, 500, 800, 1000, 1e-6
    s_rh = 0.5 * (u_L + u_R)

    W_tanh, b_tanh = generate_tanh_weights(N_tanh, seed, scale=2.5,
                                            domain_scale=(x_max - x_min))
    positions = [x_disc]

    margin = max(abs(u_L), abs(u_R)) * max(snap_times) + 0.5
    x_ic = np.linspace(x_min - margin, x_max + margin, n_ic)
    y_ic = np.where(x_ic < x_disc, u_L, u_R)
    H_ic = hidden_matrix(x_ic, W_tanh, b_tanh, positions, kappa)
    beta = solve_ridge(H_ic, y_ic, lam)
    ic_rmse = float(np.sqrt(np.mean((H_ic @ beta - y_ic) ** 2)))
    print(f"  IC fit RMSE: {ic_rmse:.6f}")

    x_eval = np.linspace(x_min, x_max, n_eval)
    snapshots = {}
    t0 = time.time()
    for ts in snap_times:
        xi = x_eval - s_rh * ts
        H = hidden_matrix(xi, W_tanh, b_tanh, positions, kappa)
        u_pred = H @ beta
        u_ref = burgers_shock_exact(x_eval, ts, u_L, u_R, x_disc)
        _, l2 = compute_errors(u_pred, u_ref)
        snapshots[f"t={ts:.2f}"] = {
            't': float(ts), 'u_pred': u_pred, 'u_ref': u_ref,
            'shock_pos': float(x_disc + s_rh * ts), 'l2_error': l2,
        }
        print(f"    t = {ts:.2f}: L2 = {l2:.4e}, shock at x = {x_disc + s_rh * ts:.3f}")
    elapsed = time.time() - t0

    plot_snapshots(x_eval, snapshots,
                   'Inviscid Burgers with Riemann Shock IC',
                   fig_path('fig06a_burgers_shock.png'),
                   show_shock=True)
    return {'ic_rmse': ic_rmse, 'elapsed_s': elapsed,
            'snapshots': {k: {kk: vv for kk, vv in v.items()
                              if kk not in ('u_pred', 'u_ref')}
                          for k, v in snapshots.items()}}


# ----------------------------------------------------------------------------
# Rarefaction case: Picard iteration
# ----------------------------------------------------------------------------

def run_rarefaction(seed=7):
    print("\n" + "=" * 70)
    print("  Burgers rarefaction (u_L = 0, u_R = 1)")
    print("=" * 70)

    u_L, u_R, x_disc = 0.0, 1.0, 0.0
    x_min, x_max = -1.0, 2.0
    snap_times = [0.0, 0.2, 0.5, 1.0]

    N_tanh, kappa, n_ic, n_eval, lam = 80, 500, 800, 1000, 1e-6

    W_tanh, b_tanh = generate_tanh_weights(N_tanh, seed, scale=2.5,
                                            domain_scale=(x_max - x_min))
    positions = [x_disc]

    margin = max(abs(u_L), abs(u_R)) * max(snap_times) + 0.5
    x_ic = np.linspace(x_min - margin, x_max + margin, n_ic)
    y_ic = np.where(x_ic < x_disc, u_L, u_R)
    H_ic = hidden_matrix(x_ic, W_tanh, b_tanh, positions, kappa)
    beta = solve_ridge(H_ic, y_ic, lam)
    ic_rmse = float(np.sqrt(np.mean((H_ic @ beta - y_ic) ** 2)))
    print(f"  IC fit RMSE: {ic_rmse:.6f}")

    x_eval = np.linspace(x_min, x_max, n_eval)
    snapshots = {}
    t0 = time.time()
    for ts in snap_times:
        u_pred, info = picard_fixed_point(x_eval, ts, W_tanh, b_tanh, beta,
                                          positions=positions, kappa=kappa)
        u_ref = burgers_rarefaction_exact(x_eval, ts, u_L, u_R, x_disc)
        _, l2 = compute_errors(u_pred, u_ref)
        snapshots[f"t={ts:.2f}"] = {
            't': float(ts), 'u_pred': u_pred, 'u_ref': u_ref,
            'l2_error': l2, 'iters': info['iters'],
        }
        print(f"    t = {ts:.2f}: L2 = {l2:.4e}, iters = {info['iters']}")
    elapsed = time.time() - t0

    plot_snapshots(x_eval, snapshots,
                   'Inviscid Burgers with Riemann Rarefaction IC',
                   fig_path('fig06a_burgers_rarefaction.png'),
                   show_shock=False)
    return {'ic_rmse': ic_rmse, 'elapsed_s': elapsed,
            'snapshots': {k: {kk: vv for kk, vv in v.items()
                              if kk not in ('u_pred', 'u_ref')}
                          for k, v in snapshots.items()}}


# ----------------------------------------------------------------------------
# Shared plot helper
# ----------------------------------------------------------------------------

def plot_snapshots(x_eval, snapshots, title, fname, show_shock=False):
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
                linestyle='--', label='Step-CIELM', zorder=4)
        if show_shock and s.get('shock_pos') is not None:
            ax.axvline(s['shock_pos'], color=C_ORANGE, linewidth=1.5,
                       linestyle=':', alpha=0.7, zorder=2,
                       label='Step position')
        l2_str = (f"{s['l2_error']:.1e}"
                  if s['l2_error'] < 1e-3 else f"{s['l2_error']:.4f}")
        ax.set_title(f"t = {s['t']:.2f}   L$_2$ = {l2_str}",
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
    print("  Section 6.6a: Burgers shock and rarefaction")
    print("=" * 70)

    r_shock = run_shock()
    r_rare  = run_rarefaction()

    out = result_path('results_06a_burgers_shock.json')
    with open(out, 'w') as f:
        json.dump({'shock': r_shock, 'rarefaction': r_rare}, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == '__main__':
    main()
