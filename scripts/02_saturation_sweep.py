"""
Section 6.2 -- Saturation of the Smooth Basis on Discontinuous Problems
=======================================================================
Sweeps the tanh basis size N_tanh in {80, 160, 320, 640, 1280} on four
Riemann / jump benchmarks and compares against Step-CIELM at a fixed
N_tanh = 80.  The result motivates the step-neuron augmentation of
Step-CIELM: pure CIELM saturates while Step-CIELM beats it at 16x
fewer basis functions.

Benchmarks
----------
1. Linear advection Riemann      (u_t + u_x = 0)
2. Periodic square-wave advection (periodic BC, square-wave IC)
3. Linear acoustics Riemann       (2x2 system, pressure jump)
4. Burgers Riemann shock          (Rankine-Hugoniot tracking)

Artifacts produced
------------------
figures/fig02_saturation_sweep.png     4-panel L2 vs N_tanh
results/results_02_saturation.json     numerical results, all benchmarks
"""

import json
import time
import numpy as np
import matplotlib.pyplot as plt

from _core import (
    generate_tanh_weights, hidden_matrix, solve_ridge, compute_errors,
    style_ax, add_legend, fig_path, result_path,
    C_BLUE, C_RED, C_GREEN, C_ORANGE, C_PURPLE, C_TEXT, BG,
)


# ----------------------------------------------------------------------------
# Benchmark 1: linear advection Riemann
# ----------------------------------------------------------------------------

def bench_lin_advection_riemann(N_tanh, seeds, positions=()):
    """u_t + u_x = 0 on [0, 2], IC u = 5 if x < 1 else 1, T = 0.8."""
    L, T, v = 2.0, 0.8, 1.0
    u_L, u_R, x_disc = 5.0, 1.0, L / 2
    l2s = []
    for seed in seeds:
        W, b = generate_tanh_weights(N_tanh, seed, scale=2.5, domain_scale=L)
        margin = v * T + 0.2
        x_ic = np.linspace(-margin, L + 0.2, 1000)
        y_ic = np.where(x_ic < x_disc, u_L, u_R)
        H_ic = hidden_matrix(x_ic, W, b, positions=positions, kappa=500.0)
        beta = solve_ridge(H_ic, y_ic, 1e-6)

        x_eval = np.linspace(0, L, 1000)
        xi = x_eval - v * T
        H_eval = hidden_matrix(xi, W, b, positions=positions, kappa=500.0)
        u_pred = H_eval @ beta
        u_ref = np.where(x_eval - v * T < x_disc, u_L, u_R)
        _, l2 = compute_errors(u_pred, u_ref)
        l2s.append(l2)
    return l2s


# ----------------------------------------------------------------------------
# Benchmark 2: periodic square wave
# ----------------------------------------------------------------------------

def bench_square_wave(N_tanh, seeds, positions=()):
    L, T, v = 2 * np.pi, 1.0, 5.0
    disc = [np.pi / 2, 3 * np.pi / 2]

    def ic(x):
        return np.where((x > disc[0]) & (x < disc[1]), 1.0, 0.0)

    l2s = []
    for seed in seeds:
        W, b = generate_tanh_weights(N_tanh, seed, scale=2.5, domain_scale=L)
        x_ic = np.linspace(0, L, 1000, endpoint=False)
        y_ic = ic(x_ic)
        H_ic = hidden_matrix(x_ic, W, b, positions=positions, kappa=500.0)
        beta = solve_ridge(H_ic, y_ic, 1e-6)
        x_eval = np.linspace(0, L, 1000, endpoint=False)
        xi = np.mod(x_eval - v * T, L)
        H_eval = hidden_matrix(xi, W, b, positions=positions, kappa=500.0)
        u_pred = H_eval @ beta
        u_ref = ic(xi)
        _, l2 = compute_errors(u_pred, u_ref)
        l2s.append(l2)
    return l2s


# ----------------------------------------------------------------------------
# Benchmark 3: linear acoustics Riemann (pressure jump)
# ----------------------------------------------------------------------------

def bench_acoustics_riemann(N_tanh, seeds, positions=()):
    X_MIN, X_MAX, T = -1.5, 1.5, 0.25
    rho0, c0 = 1.0, 1.0
    Z0 = rho0 * c0
    p_L, p_R = 2.0, 0.0
    span = X_MAX - X_MIN

    def p_exact(x, t):
        left  = np.where(x + c0 * t < 0, p_L, p_R)
        right = np.where(x - c0 * t < 0, p_L, p_R)
        return 0.5 * (left + right)

    l2s = []
    for seed in seeds:
        W1, b1 = generate_tanh_weights(N_tanh, seed,       scale=2.5, domain_scale=span)
        W2, b2 = generate_tanh_weights(N_tanh, seed + 100, scale=2.5, domain_scale=span)
        margin = c0 * T + 0.3
        x_ic = np.linspace(X_MIN - margin, X_MAX + margin, 1000)
        p0 = np.where(x_ic < 0, p_L, p_R)
        w1_0 =  0.5 / Z0 * p0
        w2_0 = -0.5 / Z0 * p0

        H1 = hidden_matrix(x_ic, W1, b1, positions=positions, kappa=500.0)
        H2 = hidden_matrix(x_ic, W2, b2, positions=positions, kappa=500.0)
        beta1 = solve_ridge(H1, w1_0, 1e-8)
        beta2 = solve_ridge(H2, w2_0, 1e-8)

        x_eval = np.linspace(X_MIN, X_MAX, 1000)
        H1s = hidden_matrix(x_eval - c0 * T, W1, b1, positions=positions, kappa=500.0)
        H2s = hidden_matrix(x_eval + c0 * T, W2, b2, positions=positions, kappa=500.0)
        p_pred = Z0 * (H1s @ beta1 - H2s @ beta2)
        _, l2 = compute_errors(p_pred, p_exact(x_eval, T))
        l2s.append(l2)
    return l2s


# ----------------------------------------------------------------------------
# Benchmark 4: Burgers Riemann shock (R-H tracking)
# ----------------------------------------------------------------------------

def bench_burgers_shock(N_tanh, seeds, positions=()):
    u_L, u_R, x_disc = 1.0, 0.0, 0.0
    X_MIN, X_MAX = -1.0, 2.0
    T = 1.5
    s_rh = 0.5 * (u_L + u_R)
    span = X_MAX - X_MIN

    l2s = []
    for seed in seeds:
        W, b = generate_tanh_weights(N_tanh, seed, scale=2.5, domain_scale=span)
        margin = max(abs(u_L), abs(u_R)) * T + 0.5
        x_ic = np.linspace(X_MIN - margin, X_MAX + margin, 1200)
        y_ic = np.where(x_ic < x_disc, u_L, u_R)
        H_ic = hidden_matrix(x_ic, W, b, positions=positions, kappa=500.0)
        beta = solve_ridge(H_ic, y_ic, 1e-6)

        x_eval = np.linspace(X_MIN, X_MAX, 1000)
        xi = x_eval - s_rh * T
        H_eval = hidden_matrix(xi, W, b, positions=positions, kappa=500.0)
        u_pred = H_eval @ beta
        u_ref = np.where(x_eval < x_disc + s_rh * T, u_L, u_R)
        _, l2 = compute_errors(u_pred, u_ref)
        l2s.append(l2)
    return l2s


# ----------------------------------------------------------------------------
# Main sweep
# ----------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  Section 6.2: Saturation of the smooth basis")
    print("=" * 70)

    N_values = [80, 160, 320, 640, 1280]
    seeds = list(range(10))

    benchmarks = [
        ('Linear advection Riemann', bench_lin_advection_riemann, [1.0]),
        ('Periodic square wave',     bench_square_wave,           [np.pi / 2, 3 * np.pi / 2]),
        ('Acoustics Riemann (p)',    bench_acoustics_riemann,     [0.0]),
        ('Burgers Riemann shock',    bench_burgers_shock,         [0.0]),
    ]

    all_results = {}

    for name, fn, step_pos in benchmarks:
        print(f"\n--- {name} ---")
        cielm_only = {}
        for N in N_values:
            t0 = time.time()
            l2s = fn(N, seeds, positions=())
            cielm_only[N] = {
                'l2_mean': float(np.mean(l2s)),
                'l2_std':  float(np.std(l2s)),
                'time_s':  time.time() - t0,
            }
            print(f"  CIELM       N={N:4d}: L2 = {cielm_only[N]['l2_mean']:.3e} "
                  f"+- {cielm_only[N]['l2_std']:.3e}  ({cielm_only[N]['time_s']:.1f}s)")

        l2s_step = fn(80, seeds, positions=step_pos)
        step_ref = {
            'l2_mean': float(np.mean(l2s_step)),
            'l2_std':  float(np.std(l2s_step)),
        }
        print(f"  Step-CIELM  N=  80: L2 = {step_ref['l2_mean']:.3e} "
              f"+- {step_ref['l2_std']:.3e}")

        all_results[name] = {'cielm_only': cielm_only, 'step_ref_N80': step_ref}

    # Plot: four panels, one per benchmark
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=False)
    fig.patch.set_facecolor(BG)
    colors = [C_BLUE, C_GREEN, C_PURPLE, C_RED]

    for i, (name, _, _) in enumerate(benchmarks):
        ax = axes[i]
        style_ax(ax)
        d = all_results[name]
        l2_means = [d['cielm_only'][N]['l2_mean'] for N in N_values]
        l2_stds  = [d['cielm_only'][N]['l2_std']  for N in N_values]
        ax.errorbar(N_values, l2_means, yerr=l2_stds,
                    color=colors[i], linewidth=2, marker='s', markersize=7,
                    capsize=4, label='CIELM')
        ax.axhline(d['step_ref_N80']['l2_mean'], color=C_ORANGE,
                   linestyle='--', linewidth=1.5,
                   label='Step-CIELM (N = 80)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$N_{\rm tanh}$', color=C_TEXT, fontsize=11)
        if i == 0:
            ax.set_ylabel(r'Relative $L_2$ error', color=C_TEXT, fontsize=11)
        ax.set_title(name, color=C_TEXT, fontsize=11, fontfamily='serif')
        add_legend(ax, loc='best')

    fig.suptitle(r'Saturation of CIELM on Discontinuous Benchmarks',
                 color=C_TEXT, fontsize=14, fontfamily='serif', y=1.02)
    plt.tight_layout()
    fname = fig_path('fig02_saturation_sweep.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"\n  Saved: {fname}")
    plt.close(fig)

    out = result_path('results_02_saturation.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved: {out}")


if __name__ == '__main__':
    main()
