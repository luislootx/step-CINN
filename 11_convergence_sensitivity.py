"""
Section 6.11 -- Convergence and Sensitivity Analysis
=====================================================
Three complementary numerical studies on the smooth Burgers benchmark
u_0(x) = -sin(pi x) on [-1, 1]:

1. Convergence with N_tanh.  Relative L2 error vs basis size across
   three PDE classes (periodic advection, linear acoustics, smooth
   Burgers) as N_tanh grows geometrically in {10, 20, 40, 80, 160, 320}.

2. Sensitivity to kappa.  Relative L2 error for the Burgers Riemann
   shock versus the step-neuron steepness kappa in
   {50, 100, 200, 500, 1000, 2000}.

3. Numerical stability.  Condition number of the ridge normal
   equations (H^T H + lam I) as N_tanh grows, plus the Banach
   contraction diagnostic for the nonlinear Picard iteration near the
   shock-formation time.

Artifacts produced
------------------
figures/fig11a_convergence_N.png         L2 vs N_tanh, three PDE classes
figures/fig11b_kappa_sensitivity.png     L2 vs kappa, Burgers shock
figures/fig11c_stability.png             condition number + Banach
results/results_11_convergence.json      numerical results
"""

import json
import numpy as np
import matplotlib.pyplot as plt

from _core import (
    generate_tanh_weights, hidden_matrix, solve_ridge, compute_errors,
    picard_fixed_point, burgers_char_exact,
    style_ax, add_legend, fig_path, result_path,
    C_BLUE, C_RED, C_GREEN, C_ORANGE, C_PURPLE, C_TEXT, BG,
)


# ----------------------------------------------------------------------------
# Shared Burgers setup
# ----------------------------------------------------------------------------

X_MIN, X_MAX = -1.0, 1.0
T_BREAK = 1.0 / np.pi
MARGIN = 1.5


def ic_burgers(x):
    return -np.sin(np.pi * x)


def fit_burgers_ic(N_tanh, seed=7, n_ic=1000, lam=1e-8):
    W_tanh, b_tanh = generate_tanh_weights(N_tanh, seed, scale=3.0,
                                           domain_scale=(X_MAX - X_MIN))
    x_ic = np.linspace(X_MIN - MARGIN, X_MAX + MARGIN, n_ic)
    y_ic = ic_burgers(x_ic)
    H_ic = hidden_matrix(x_ic, W_tanh, b_tanh)
    beta = solve_ridge(H_ic, y_ic, lam)
    return W_tanh, b_tanh, beta, H_ic


# ----------------------------------------------------------------------------
# Part 1: convergence with N_tanh across PDE classes
# ----------------------------------------------------------------------------

def run_convergence_N():
    print("\n" + "=" * 70)
    print("  Part 1: L2 vs N_tanh across PDE classes")
    print("=" * 70)

    N_values = [10, 20, 40, 80, 160, 320]

    # Case 1: periodic advection
    L = 2 * np.pi
    v = 30.0
    T = 1.0
    x_eval_per = np.linspace(0, L, 1000, endpoint=False)

    def ic_sin(x):
        return np.sin(x)

    means_per, stds_per = [], []
    for N in N_values:
        l2s = []
        for s in range(10):
            W, b = generate_tanh_weights(N, s, scale=2.5, domain_scale=L)
            x_ic = np.linspace(0, L, 500, endpoint=False)
            H_ic = hidden_matrix(x_ic, W, b)
            beta = solve_ridge(H_ic, ic_sin(x_ic), 1e-6)
            xi = np.mod(x_eval_per - v * T, L)
            u_pred = hidden_matrix(xi, W, b) @ beta
            u_ref = ic_sin(xi)
            _, l2 = compute_errors(u_pred, u_ref)
            l2s.append(l2)
        means_per.append(float(np.mean(l2s)))
        stds_per.append(float(np.std(l2s)))
        print(f"  Periodic     N = {N:3d}: L2 = {means_per[-1]:.3e}")

    # Case 2: linear acoustics
    rho0, c0 = 1.0, 1.0
    Z0 = rho0 * c0
    a = 10.0
    Xmin_a, Xmax_a, T_a = -1.5, 1.5, 0.8
    span_a = Xmax_a - Xmin_a

    def ref_p(x, t):
        return 0.5 * (np.exp(-a * (x - c0 * t) ** 2) +
                      np.exp(-a * (x + c0 * t) ** 2))

    x_eval_a = np.linspace(Xmin_a, Xmax_a, 1000)
    margin_a = c0 * T_a + 0.3
    x_ic = np.linspace(Xmin_a - margin_a, Xmax_a + margin_a, 500)
    p0 = np.exp(-a * x_ic ** 2)
    w1_0 =  0.5 / Z0 * p0
    w2_0 = -0.5 / Z0 * p0

    means_a, stds_a = [], []
    for N in N_values:
        l2s = []
        for s in range(10):
            W1, b1 = generate_tanh_weights(N, s, scale=2.5, domain_scale=span_a)
            W2, b2 = generate_tanh_weights(N, s + 100, scale=2.5, domain_scale=span_a)
            H1 = hidden_matrix(x_ic, W1, b1)
            H2 = hidden_matrix(x_ic, W2, b2)
            beta1 = solve_ridge(H1, w1_0, 1e-8)
            beta2 = solve_ridge(H2, w2_0, 1e-8)
            H1s = hidden_matrix(x_eval_a - c0 * T_a, W1, b1)
            H2s = hidden_matrix(x_eval_a + c0 * T_a, W2, b2)
            p_pred = Z0 * (H1s @ beta1 - H2s @ beta2)
            _, l2 = compute_errors(p_pred, ref_p(x_eval_a, T_a))
            l2s.append(l2)
        means_a.append(float(np.mean(l2s)))
        stds_a.append(float(np.std(l2s)))
        print(f"  Acoustics    N = {N:3d}: L2 = {means_a[-1]:.3e}")

    # Case 3: smooth Burgers
    x_eval_b = np.linspace(X_MIN, X_MAX, 500)
    means_b, stds_b = [], []
    for N in N_values:
        l2s = []
        for s in range(10):
            W, b, beta, _ = fit_burgers_ic(N, seed=s)
            u_pred, _ = picard_fixed_point(x_eval_b, 0.20, W, b, beta)
            u_ref = burgers_char_exact(x_eval_b, 0.20, ic_burgers,
                                       xi_min=X_MIN - MARGIN,
                                       xi_max=X_MAX + MARGIN)
            _, l2 = compute_errors(u_pred, u_ref)
            l2s.append(l2)
        means_b.append(float(np.mean(l2s)))
        stds_b.append(float(np.std(l2s)))
        print(f"  Burgers      N = {N:3d}: L2 = {means_b[-1]:.3e}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor(BG)
    panels = [
        (axes[0], means_per, stds_per, C_BLUE,
         r'Periodic advection $\sin(x)$, $v = 30$'),
        (axes[1], means_a, stds_a, C_GREEN,
         r'Linear acoustics, Gaussian pulse'),
        (axes[2], means_b, stds_b, C_RED,
         r'Nonlinear Burgers, smooth pre-shock'),
    ]
    for ax, m, s, col, title in panels:
        style_ax(ax)
        ax.errorbar(N_values, m, yerr=s, color=col, linewidth=2,
                    marker='s', markersize=7, capsize=4)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$N_{\rm tanh}$', color=C_TEXT, fontsize=11)
        ax.set_ylabel(r'Relative $L_2$ error', color=C_TEXT, fontsize=11)
        ax.set_title(title, color=C_TEXT, fontsize=11, fontfamily='serif')

    fig.suptitle(r'Convergence of CIELM across PDE classes',
                 color=C_TEXT, fontsize=14, fontfamily='serif', y=1.02)
    plt.tight_layout()
    fname = fig_path('fig11a_convergence_N.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"  Saved: {fname}")
    plt.close(fig)

    return {'N_values': N_values,
            'periodic':  {'mean': means_per, 'std': stds_per},
            'acoustics': {'mean': means_a,   'std': stds_a},
            'burgers':   {'mean': means_b,   'std': stds_b}}


# ----------------------------------------------------------------------------
# Part 2: sensitivity to kappa on the Burgers Riemann shock
# ----------------------------------------------------------------------------

def run_kappa_sensitivity():
    print("\n" + "=" * 70)
    print("  Part 2: sensitivity to kappa on the Burgers Riemann shock")
    print("=" * 70)

    u_L, u_R, x_disc = 1.0, 0.0, 0.0
    x_min, x_max = -1.0, 2.0
    T = 1.0
    s_rh = 0.5 * (u_L + u_R)
    span = x_max - x_min
    kappa_values = [50, 100, 200, 500, 1000, 2000]
    N_tanh = 80
    seeds = list(range(10))

    l2_means, l2_stds = [], []
    for kappa in kappa_values:
        l2s = []
        for seed in seeds:
            W, b = generate_tanh_weights(N_tanh, seed, scale=2.5,
                                         domain_scale=span)
            positions = [x_disc]
            margin = max(abs(u_L), abs(u_R)) * T + 0.5
            x_ic = np.linspace(x_min - margin, x_max + margin, 1000)
            y_ic = np.where(x_ic < x_disc, u_L, u_R)
            H_ic = hidden_matrix(x_ic, W, b, positions, kappa)
            beta = solve_ridge(H_ic, y_ic, 1e-6)
            x_eval = np.linspace(x_min, x_max, 1000)
            xi = x_eval - s_rh * T
            u_pred = hidden_matrix(xi, W, b, positions, kappa) @ beta
            u_ref = np.where(x_eval < x_disc + s_rh * T, u_L, u_R)
            _, l2 = compute_errors(u_pred, u_ref)
            l2s.append(l2)
        l2_means.append(float(np.mean(l2s)))
        l2_stds.append(float(np.std(l2s)))
        print(f"  kappa = {kappa:5d}: L2 = {l2_means[-1]:.3e} "
              f"+- {l2_stds[-1]:.3e}")

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG)
    style_ax(ax)
    ax.errorbar(kappa_values, l2_means, yerr=l2_stds,
                color=C_PURPLE, linewidth=2, marker='^', markersize=8,
                capsize=4, label=r'Step-CIELM, Burgers Riemann shock')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\kappa$', color=C_TEXT, fontsize=11)
    ax.set_ylabel(r'Relative $L_2$ error', color=C_TEXT, fontsize=11)
    ax.set_title(r'Sensitivity of Step-CIELM to Step-Neuron Steepness $\kappa$',
                 color=C_TEXT, fontsize=13, fontfamily='serif')
    add_legend(ax)
    plt.tight_layout()
    fname = fig_path('fig11b_kappa_sensitivity.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"  Saved: {fname}")
    plt.close(fig)

    return {'kappa_values': kappa_values,
            'l2_mean': l2_means, 'l2_std': l2_stds}


# ----------------------------------------------------------------------------
# Part 3: numerical stability (conditioning + Banach)
# ----------------------------------------------------------------------------

def run_stability():
    print("\n" + "=" * 70)
    print("  Part 3: numerical stability")
    print("=" * 70)

    # 3a) Condition number of (H^T H + lam I)
    lam = 1e-6
    N_values = [10, 20, 40, 80, 160, 320]
    conds = []
    for N in N_values:
        W, b = generate_tanh_weights(N, seed=7, scale=3.0,
                                     domain_scale=(X_MAX - X_MIN))
        x_ic = np.linspace(X_MIN - MARGIN, X_MAX + MARGIN, 500)
        H = hidden_matrix(x_ic, W, b)
        A = H.T @ H + lam * np.eye(N)
        c = float(np.linalg.cond(A))
        conds.append(c)
        print(f"  N = {N:3d}:   cond(H^T H + lam I) = {c:.3e}")

    # 3b) Banach contraction diagnostic (smooth Burgers)
    W, b, beta, _ = fit_burgers_ic(N_tanh=120)
    x_eval = np.linspace(X_MIN, X_MAX, 500)
    t_values = np.linspace(0.01, 0.40, 40)
    iters_all, l2_all = [], []
    for t in t_values:
        u_pred, info = picard_fixed_point(x_eval, t, W, b, beta)
        u_ref = burgers_char_exact(x_eval, t, ic_burgers,
                                   xi_min=X_MIN - MARGIN,
                                   xi_max=X_MAX + MARGIN)
        _, l2 = compute_errors(u_pred, u_ref)
        iters_all.append(info['iters'])
        l2_all.append(l2)
    rho = np.pi * t_values

    # Combined plot: 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.patch.set_facecolor(BG)

    ax = axes[0]
    style_ax(ax)
    ax.loglog(N_values, conds, color=C_PURPLE, linewidth=2, marker='^',
              markersize=8, label=r'$\mathrm{cond}(H^\top H + \lambda I)$')
    ax.set_xlabel(r'$N_{\rm tanh}$', color=C_TEXT, fontsize=11)
    ax.set_ylabel('Condition number', color=C_TEXT, fontsize=11)
    ax.set_title(r'Ridge normal-equation conditioning',
                 color=C_TEXT, fontsize=11, fontfamily='serif')
    add_legend(ax)

    ax = axes[1]
    style_ax(ax)
    ax.plot(t_values, iters_all, color=C_RED, linewidth=2,
            marker='o', markersize=3)
    ax.axvline(T_BREAK, color=C_ORANGE, linestyle='--', linewidth=1.5,
               label=rf'$t_{{\rm break}} = 1/\pi = {T_BREAK:.3f}$')
    ax.set_xlabel('t', color=C_TEXT)
    ax.set_ylabel('Iterations to converge', color=C_TEXT)
    ax.set_title('Picard iterations (smooth Burgers)',
                 color=C_TEXT, fontsize=11, fontfamily='serif')
    add_legend(ax)

    ax = axes[2]
    style_ax(ax)
    ax.plot(t_values, rho, color=C_GREEN, linewidth=2,
            label=r'$\pi t$ (theoretical)')
    ax.axhline(1.0, color=C_RED, linestyle='--', linewidth=1.5,
               label='Contraction limit')
    ax.axvline(T_BREAK, color=C_ORANGE, linestyle='--', linewidth=1.5,
               label=r'$t_{\rm break}$')
    ax.set_xlabel('t', color=C_TEXT)
    ax.set_ylabel('Contraction factor', color=C_TEXT)
    ax.set_title('Banach contraction condition',
                 color=C_TEXT, fontsize=11, fontfamily='serif')
    add_legend(ax, loc='upper left')

    fig.suptitle(r'Numerical Stability Diagnostics',
                 color=C_TEXT, fontsize=13, fontfamily='serif', y=1.02)
    plt.tight_layout()
    fname = fig_path('fig11c_stability.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"  Saved: {fname}")
    plt.close(fig)

    return {
        'condition_number': {'N_values': N_values, 'conds': conds},
        'banach': {'t_values': t_values.tolist(),
                   'iters': iters_all, 'l2_errors': l2_all},
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  Section 6.11: Convergence and sensitivity analysis")
    print("=" * 70)

    results = {
        'convergence_N':      run_convergence_N(),
        'kappa_sensitivity':  run_kappa_sensitivity(),
        'stability':          run_stability(),
    }

    out = result_path('results_11_convergence.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == '__main__':
    main()
