"""
Section 6.6 (part c) -- Unified Pre/Post-Shock Burgers via Newton Continuation
==============================================================================
A single algorithm that handles both smooth and post-shock regimes of the
inviscid Burgers equation

    u_t + u u_x = 0,   u_0(x) = -sin(pi x),   x in [-1, 1].

The characteristic relation g(xi) = xi + u_ELM(xi) t - x = 0 is solved by
Newton's method with continuation.  Two independent marches -- one from
the left boundary ascending in x, one from the right boundary descending
in x -- are performed.

If both marches agree, the solution is smooth.  Where they disagree, a
shock has formed: the shock position is the midpoint of the disagreement
region, and the entropy solution picks the left march on x < x_shock and
the right march on x > x_shock.

A Cole-Hopf reference (nu = 0.001) serves as ground truth post-shock.

Artifacts produced
------------------
figures/fig06c_burgers_unified.png          15-panel pre/post-shock snapshots
results/results_06c_burgers_unified.json    numerical results
"""

import json
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from _core import fig_path, result_path


# ----------------------------------------------------------------------------
# ELM primitives (specialised local copies with explicit bias column)
# ----------------------------------------------------------------------------

def generate_tanh_weights(n, seed=7, scale=2.5, ds=1.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, n), rng.uniform(-scale * ds, scale * ds, n)


def fit_elm(x, y, W, b, lam=1e-8):
    H = np.hstack([np.tanh(np.outer(x, W) + b), np.ones((len(x), 1))])
    beta_full = np.linalg.solve(H.T @ H + lam * np.eye(H.shape[1]), H.T @ y)
    return beta_full[:-1], beta_full[-1]


def elm_eval(x, W, b, beta, bias):
    return np.tanh(np.outer(x, W) + b) @ beta + bias


# ----------------------------------------------------------------------------
# Newton continuation
# ----------------------------------------------------------------------------

def newton_march(x_sorted, t, W, b, beta, bias, max_iter=50, tol=1e-13):
    """Solve g(xi) = xi + u_ELM(xi) t - x = 0 by Newton with continuation.
    x_sorted must be monotonic; returns (u_values, xi_values) in that order.
    """
    N = len(x_sorted)
    xi_arr = np.empty(N)
    u_arr = np.empty(N)
    beta_W = beta * W
    xi_curr = float(x_sorted[0])

    for i in range(N):
        x_target = float(x_sorted[i])
        for _ in range(max_iter):
            z = W * xi_curr + b
            tanh_z = np.tanh(z)
            u_val = float(np.dot(beta, tanh_z)) + bias
            sech2 = 1.0 - tanh_z * tanh_z
            du_val = float(np.dot(beta_W, sech2))

            g = xi_curr + u_val * t - x_target
            gp = 1.0 + du_val * t
            if abs(gp) < 1e-15:
                gp = 1e-15
            step = g / gp
            xi_curr -= step
            if abs(step) < tol:
                break
        xi_arr[i] = xi_curr
        u_arr[i] = float(np.dot(beta, np.tanh(W * xi_curr + b))) + bias
    return u_arr, xi_arr


def unified_cielm(x, t, W, b, beta, bias, shock_tol=0.01):
    """Single-pass solver combining two Newton marches and entropy selection.

    Returns (u_pred, shock_info) where shock_info is None (smooth case)
    or a dict with x_shock, u_L, u_R, and the jump magnitude.
    """
    if t < 1e-14:
        return elm_eval(x, W, b, beta, bias), None

    u_left, xi_left = newton_march(x, t, W, b, beta, bias)
    u_right_rev, xi_right_rev = newton_march(x[::-1], t, W, b, beta, bias)
    u_right = u_right_rev[::-1]
    xi_right = xi_right_rev[::-1]

    diff = np.abs(u_left - u_right)
    max_diff = np.max(diff)

    if max_diff < shock_tol:
        u_pred = 0.5 * (u_left + u_right)
        return u_pred, None

    xi_diff = xi_left - xi_right
    gap_mask = np.abs(xi_diff) > 0.01
    gap_indices = np.where(gap_mask)[0]
    if len(gap_indices) == 0:
        i_shock = int(np.argmax(diff))
    else:
        i_shock = gap_indices[len(gap_indices) // 2]
    x_shock = float(x[i_shock])
    if len(gap_indices) > 1:
        weights = np.abs(xi_diff[gap_indices])
        x_shock = float(np.average(x[gap_indices], weights=weights))

    u_pred = np.where(x < x_shock, u_left, u_right)
    left_edge = gap_indices[0]
    right_edge = gap_indices[-1]
    u_L = float(u_left[max(0, left_edge - 1)])
    u_R = float(u_right[min(len(x) - 1, right_edge + 1)])
    return u_pred, {'x_shock': x_shock, 'u_L': u_L, 'u_R': u_R,
                    'jump': abs(u_L - u_R)}


# ----------------------------------------------------------------------------
# Reference: Cole-Hopf (small-viscosity entropy solution)
# ----------------------------------------------------------------------------

def exact_colehopf(x, t, nu=0.001):
    if t < 1e-14:
        return -np.sin(np.pi * x)
    nq = max(8000, int(12000 / np.sqrt(nu)))
    xi = np.linspace(-3, 3, nq)
    dxi = xi[1] - xi[0]
    Phi = (1 - np.cos(np.pi * xi)) / np.pi
    d = x[:, None] - xi[None, :]
    e = -(d ** 2) / (4 * nu * t) + Phi[None, :] / (2 * nu)
    e -= e.max(axis=1, keepdims=True)
    w = np.exp(e)
    return np.sum((d / t) * w, axis=1) * dxi / (np.sum(w, axis=1) * dxi)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  Section 6.6c: Unified CIELM with automatic shock detection")
    print("=" * 70)

    x_min, x_max = -1.0, 1.0
    t_break = 1.0 / np.pi
    nu = 0.001

    N_tanh = 200
    W, b = generate_tanh_weights(N_tanh, seed=7, scale=3.5, ds=2.5)
    x_ic = np.linspace(-3.0, 3.0, 2000)
    y_ic = -np.sin(np.pi * x_ic)
    beta, bias = fit_elm(x_ic, y_ic, W, b, lam=1e-10)

    rmse = np.sqrt(np.mean((elm_eval(x_ic, W, b, beta, bias) - y_ic) ** 2))
    print(f"  IC: {N_tanh} neurons, RMSE = {rmse:.2e}")
    print(f"  t_break = {t_break:.4f}")

    x_eval = np.linspace(x_min, x_max, 500)
    test_times = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                  t_break, 0.35, 0.40, 0.50, 0.70, 1.0, 1.5, 2.0]

    print(f"\n  {'t':>7s} | {'L2':>10s} | {'shock':>11s} | {'jump':>6s}")
    print("  " + "-" * 50)

    results = []
    for t in test_times:
        u_ref = exact_colehopf(x_eval, t, nu)
        norm = max(np.linalg.norm(u_ref), 1e-12)

        t0 = time.time()
        u_pred, shock = unified_cielm(x_eval, t, W, b, beta, bias)
        elapsed = time.time() - t0

        l2 = float(np.linalg.norm(u_pred - u_ref) / norm)
        if shock is not None:
            shock_str = f"x = {shock['x_shock']:+.3f}"
            jump_str = f"{shock['jump']:.3f}"
        else:
            shock_str = "smooth"
            jump_str = "-"
        print(f"  {t:7.4f} | {l2:10.2e} | {shock_str:>11s} | {jump_str:>6s}  "
              f"({elapsed:.3f}s)")

        results.append({'t': float(t), 'l2': l2, 'shock': shock,
                        'u_pred': u_pred, 'u_ref': u_ref,
                        'elapsed_s': elapsed})

    plot_unified(x_eval, results, fig_path('fig06c_burgers_unified.png'))

    json_results = []
    for r in results:
        d = {'t': r['t'], 'l2': r['l2'], 'elapsed_s': r['elapsed_s']}
        if r['shock'] is not None:
            d['shock'] = {k: float(v) for k, v in r['shock'].items()}
        else:
            d['shock'] = None
        json_results.append(d)

    out = result_path('results_06c_burgers_unified.json')
    with open(out, 'w') as f:
        json.dump({'N_tanh': N_tanh, 'nu_reference': nu,
                   't_break': float(t_break),
                   'snapshots': json_results}, f, indent=2)
    print(f"\n  Saved: {out}")


def plot_unified(x_eval, results, fname):
    fig, axes = plt.subplots(3, 5, figsize=(22, 12), facecolor='white')
    axes = axes.flatten()

    C_EX = '#2166ac'
    C_PR = '#d6604d'
    C_SH = '#e08214'

    for i, r in enumerate(results):
        if i >= len(axes):
            break
        ax = axes[i]
        ax.set_facecolor('#fafafa')
        ax.grid(True, color='#e0e0e0', lw=0.5, zorder=0)
        for sp in ax.spines.values():
            sp.set_edgecolor('#aaa')
        ax.plot(x_eval, r['u_ref'],  color=C_EX, lw=2.5, zorder=3,
                label='Cole-Hopf')
        ax.plot(x_eval, r['u_pred'], color=C_PR, lw=1.8, ls='--', zorder=4,
                label=f'CIELM (L$_2$ = {r["l2"]:.1e})')
        if r['shock'] is not None:
            xs = r['shock']['x_shock']
            ax.axvline(xs, color=C_SH, lw=1.5, ls=':', alpha=0.6)
            ax.plot(xs, 0, 's', color=C_SH, ms=6, zorder=6)
            ax.set_title(f't = {r["t"]:.4f}    shock @ {xs:+.3f}    '
                         f'jump = {r["shock"]["jump"]:.2f}', fontsize=8)
        else:
            ax.set_title(f't = {r["t"]:.4f}    [smooth]    '
                         f'L$_2$ = {r["l2"]:.1e}', fontsize=8)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.4, 1.4)
        ax.legend(fontsize=6, loc='upper right')
        ax.tick_params(labelsize=7)

    fig.suptitle('Unified CIELM: Automatic shock detection '
                 '(single algorithm, pre and post shock)',
                 fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(fname, dpi=150, facecolor='white')
    plt.close(fig)
    print(f"  Saved: {fname}")


if __name__ == '__main__':
    main()
