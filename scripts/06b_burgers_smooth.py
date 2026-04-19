"""
Section 6.6 (part b) -- Inviscid Burgers: Smooth Pre-Shock and Rarefaction
==========================================================================
CIELM extended to the inviscid Burgers equation

    u_t + u u_x = 0

by Picard fixed-point on the implicit characteristic relation

    xi = x - u(xi) * t.

Starting from xi^{(0)} = x, the iteration

    xi^{(k+1)} = x - u_ELM(xi^{(k)}) * t

converges by the Banach contraction theorem whenever

    |du_ELM / dxi| * t < 1     <=>    t < t_break = 1 / max |u_0'(x)|,

i.e. up to the shock-formation time.  Two smooth benchmarks:

Case A -- pre-shock compression, u_0(x) = -sin(pi x),  t_break = 1/pi.
Case B -- rarefaction,            u_0(x) = 0.5 + 0.4 tanh(x);  no shock.

Artifacts produced
------------------
figures/fig06b_burgers_preshock.png         snapshots, pre-shock case
figures/fig06b_burgers_rarefaction.png      snapshots, smooth rarefaction
results/results_06b_burgers_smooth.json     10-seed statistics
"""

import json
import numpy as np
import matplotlib.pyplot as plt

from _core import (
    generate_tanh_weights, hidden_matrix, solve_ridge, compute_errors,
    picard_fixed_point, burgers_char_exact,
    style_ax, add_legend, fig_path, result_path,
    C_BLUE, C_RED, C_TEXT, BG,
)


# ----------------------------------------------------------------------------
# Case A: pre-shock compression, u_0(x) = -sin(pi x)
# ----------------------------------------------------------------------------

def run_preshock():
    print("\n" + "=" * 70)
    print("  Burgers smooth pre-shock, u_0(x) = -sin(pi x)")
    print("=" * 70)

    x_min, x_max = -1.0, 1.0
    t_break = 1.0 / np.pi
    snap_times = [0.0, 0.10, 0.20, 0.30]
    seed = 7

    def ic(x):
        return -np.sin(np.pi * x)

    N_tanh, n_ic, n_eval, lam = 120, 1000, 1000, 1e-8
    margin = 1.0 * max(snap_times) + 0.5
    W_tanh, b_tanh = generate_tanh_weights(
        N_tanh, seed, scale=3.0, domain_scale=(x_max - x_min))

    x_ic = np.linspace(x_min - margin, x_max + margin, n_ic)
    y_ic = ic(x_ic)
    H_ic = hidden_matrix(x_ic, W_tanh, b_tanh)
    beta = solve_ridge(H_ic, y_ic, lam)

    x_eval = np.linspace(x_min, x_max, n_eval)
    snapshots = {}
    for ts in snap_times:
        u_pred, info = picard_fixed_point(x_eval, ts, W_tanh, b_tanh, beta)
        u_ref = burgers_char_exact(x_eval, ts, ic,
                                   xi_min=x_min - margin,
                                   xi_max=x_max + margin)
        _, l2 = compute_errors(u_pred, u_ref)
        snapshots[f"t={ts:.2f}"] = {
            't': float(ts), 'u_pred': u_pred, 'u_ref': u_ref,
            'l2_error': l2, 'iters': info['iters'],
        }
        print(f"  t = {ts:.2f}   L2 = {l2:.2e}   iters = {info['iters']}")

    plot_burgers(x_eval, snapshots,
                 rf'Inviscid Burgers with Smooth Pre-Shock IC '
                 rf'$u_0 = -\sin(\pi x)$  '
                 rf'($t_{{\rm break}} = 1/\pi \approx {t_break:.3f}$)',
                 fig_path('fig06b_burgers_preshock.png'))

    # 10-seed statistics at t = 0.20
    all_l2 = []
    for s in range(10):
        W, b = generate_tanh_weights(N_tanh, s, scale=3.0,
                                     domain_scale=(x_max - x_min + 2 * margin))
        H = hidden_matrix(x_ic, W, b)
        bta = solve_ridge(H, y_ic, lam)
        u_pred, _ = picard_fixed_point(x_eval, 0.2, W, b, bta)
        u_ref = burgers_char_exact(x_eval, 0.2, ic,
                                   xi_min=x_min - margin,
                                   xi_max=x_max + margin)
        _, l2 = compute_errors(u_pred, u_ref)
        all_l2.append(l2)
    print(f"  10-seed at t = 0.20: L2 = {np.mean(all_l2):.2e} "
          f"+- {np.std(all_l2):.2e}")
    return {'l2_mean': float(np.mean(all_l2)),
            'l2_std':  float(np.std(all_l2))}


# ----------------------------------------------------------------------------
# Case B: smooth rarefaction, u_0(x) = 0.5 + 0.4 tanh(x)
# ----------------------------------------------------------------------------

def run_rarefaction():
    print("\n" + "=" * 70)
    print("  Burgers smooth rarefaction, u_0(x) = 0.5 + 0.4 tanh(x)")
    print("=" * 70)

    x_min, x_max = -3.0, 5.0
    snap_times = [0.0, 0.5, 1.0, 2.0]
    seed = 7

    def ic(x):
        return 0.5 + 0.4 * np.tanh(x)

    N_tanh, n_ic, n_eval, lam = 120, 1200, 1000, 1e-8
    margin = 0.9 * max(snap_times) + 1.0
    W_tanh, b_tanh = generate_tanh_weights(N_tanh, seed, scale=2.5,
                                            domain_scale=(x_max - x_min))

    x_ic = np.linspace(x_min - margin, x_max + margin, n_ic)
    y_ic = ic(x_ic)
    H_ic = hidden_matrix(x_ic, W_tanh, b_tanh)
    beta = solve_ridge(H_ic, y_ic, lam)

    x_eval = np.linspace(x_min, x_max, n_eval)
    snapshots = {}
    for ts in snap_times:
        u_pred, info = picard_fixed_point(x_eval, ts, W_tanh, b_tanh, beta)
        u_ref = burgers_char_exact(x_eval, ts, ic,
                                   xi_min=x_min - margin,
                                   xi_max=x_max + margin,
                                   n_char=20000)
        _, l2 = compute_errors(u_pred, u_ref)
        snapshots[f"t={ts:.2f}"] = {
            't': float(ts), 'u_pred': u_pred, 'u_ref': u_ref,
            'l2_error': l2, 'iters': info['iters'],
        }
        print(f"  t = {ts:.2f}   L2 = {l2:.2e}   iters = {info['iters']}")

    plot_burgers(x_eval, snapshots,
                 r'Inviscid Burgers: Smooth Rarefaction (no shock forms), '
                 r'$u_0 = 0.5 + 0.4\,\tanh(x)$',
                 fig_path('fig06b_burgers_rarefaction.png'))

    # 10-seed statistics at t = 2.0
    all_l2 = []
    for s in range(10):
        W, b = generate_tanh_weights(N_tanh, s, scale=2.5,
                                     domain_scale=(x_max - x_min))
        H = hidden_matrix(x_ic, W, b)
        bta = solve_ridge(H, y_ic, lam)
        u_pred, _ = picard_fixed_point(x_eval, 2.0, W, b, bta)
        u_ref = burgers_char_exact(x_eval, 2.0, ic,
                                   xi_min=x_min - margin,
                                   xi_max=x_max + margin,
                                   n_char=20000)
        _, l2 = compute_errors(u_pred, u_ref)
        all_l2.append(l2)
    print(f"  10-seed at t = 2.0: L2 = {np.mean(all_l2):.2e} "
          f"+- {np.std(all_l2):.2e}")
    return {'l2_mean': float(np.mean(all_l2)),
            'l2_std':  float(np.std(all_l2))}


# ----------------------------------------------------------------------------
# Shared plot helper
# ----------------------------------------------------------------------------

def plot_burgers(x_eval, snapshots, title, fname):
    keys = sorted(snapshots.keys(), key=lambda k: snapshots[k]['t'])
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4), sharey=True)
    fig.patch.set_facecolor(BG)
    if n == 1:
        axes = [axes]
    for i, key in enumerate(keys):
        s = snapshots[key]
        ax = axes[i]
        style_ax(ax)
        ax.plot(x_eval, s['u_ref'],  color=C_BLUE, linewidth=2.5,
                label='Exact', zorder=3)
        ax.plot(x_eval, s['u_pred'], color=C_RED,  linewidth=1.8,
                linestyle='--', label='CIELM (Picard)', zorder=4)
        ax.set_title(rf"t = {s['t']:.2f}   L$_2$ = {s['l2_error']:.2e}",
                     color=C_TEXT, fontsize=10, fontfamily='serif')
        ax.set_xlabel('x', color=C_TEXT, fontsize=10)
        if i == 0:
            ax.set_ylabel('u(x, t)', color=C_TEXT, fontsize=10)
        add_legend(ax, loc='best')
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
    print("  Section 6.6b: Burgers smooth (pre-shock and rarefaction)")
    print("=" * 70)

    r_pre  = run_preshock()
    r_rare = run_rarefaction()

    out = result_path('results_06b_burgers_smooth.json')
    with open(out, 'w') as f:
        json.dump({
            'smooth_preshock_t0.20':    r_pre,
            'smooth_rarefaction_t2.00': r_rare,
        }, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == '__main__':
    main()
