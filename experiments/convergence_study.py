"""
Convergence and sensitivity studies for Step-CINN paper.

Study 1: L2 error vs N_tanh (number of tanh neurons)
  - Linear advection (Riemann IC, v=1)
  - Nonlinear Burgers (smooth IC, t=0.20)

Study 2: L2 error vs kappa (step steepness)
  - Linear advection (Riemann IC, v=1)

Generates figures for the paper.
"""
import numpy as np
import matplotlib.pyplot as plt
import os, time

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Plotting style (matching paper) ──────────────────────────────────────────
C_BLUE   = '#2166ac'
C_RED    = '#d6604d'
C_GREEN  = '#1b7837'
C_ORANGE = '#e08214'
C_PURPLE = '#7b3294'
C_TEXT   = '#333333'
BG       = '#ffffff'
BG_AX    = '#fafafa'
GRID     = '#e0e0e0'
SPINE    = '#aaaaaa'

def style_ax(ax):
    ax.set_facecolor(BG_AX)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.tick_params(colors=C_TEXT, labelsize=10)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE)

def add_legend(ax, **kw):
    ax.legend(fontsize=9, framealpha=0.9, edgecolor=SPINE, **kw)


# ── Network building blocks ─────────────────────────────────────────────────
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def generate_tanh_weights(n_tanh, seed, scale, domain_scale):
    rng = np.random.default_rng(seed)
    W = rng.uniform(-scale, scale, size=n_tanh)
    b = rng.uniform(-scale * domain_scale, scale * domain_scale, size=n_tanh)
    return W, b

def hidden_matrix(x, W_tanh, b_tanh, positions, kappa):
    z_tanh = np.outer(x, W_tanh) + b_tanh
    H_tanh = np.tanh(z_tanh)
    if len(positions) > 0:
        z_step = kappa * (x.reshape(-1, 1) - np.array(positions).reshape(1, -1))
        H_step = sigmoid(z_step)
        return np.hstack([H_tanh, H_step])
    return H_tanh

def solve_ridge(H, y, lam=1e-6):
    n = H.shape[1]
    A = H.T @ H + lam * np.eye(n)
    return np.linalg.solve(A, H.T @ y)

def elm_predict(x, W_tanh, b_tanh, positions, kappa, beta):
    H = hidden_matrix(x, W_tanh, b_tanh, positions, kappa)
    return H @ beta


# ── Exact solutions ──────────────────────────────────────────────────────────
def riemann_ic(x, x_disc=1.0, u_L=5.0, u_R=1.0):
    return np.where(x < x_disc, u_L, u_R)

def burgers_smooth_exact(x_eval, t, ic_func, xi_min, xi_max, n_char=20000):
    if t < 1e-14:
        return ic_func(x_eval)
    xi_fine = np.linspace(xi_min, xi_max, n_char)
    u_fine = ic_func(xi_fine)
    x_fine = xi_fine + u_fine * t
    return np.interp(x_eval, x_fine, u_fine)

def cielm_fixed_point(x_eval, t, W_tanh, b_tanh, positions, kappa, beta,
                      max_iter=500, tol=1e-12):
    if t < 1e-14:
        return elm_predict(x_eval, W_tanh, b_tanh, positions, kappa, beta)
    xi = x_eval.copy()
    for _ in range(max_iter):
        u_at_xi = elm_predict(xi, W_tanh, b_tanh, positions, kappa, beta)
        xi_new = x_eval - u_at_xi * t
        res = float(np.max(np.abs(xi_new - xi)))
        xi = xi_new
        if res < tol:
            break
    return elm_predict(xi, W_tanh, b_tanh, positions, kappa, beta)


# ═════════════════════════════════════════════════════════════════════════════
# Study 1: Convergence — L2 error vs N_tanh
# ═════════════════════════════════════════════════════════════════════════════
def study_convergence_n_tanh():
    print("=" * 70)
    print("Study 1: L2 error vs N_tanh")
    print("=" * 70)

    n_tanh_values = [10, 20, 40, 80, 160, 320]
    n_seeds = 10
    kappa = 500
    lam = 1e-6
    scale = 2.5

    # ── (a) Linear advection: Riemann IC ─────────────────────────────────
    print("\n  (a) Linear advection, Riemann IC, v=1")
    L_adv = 2.0
    x_disc = 1.0
    v = 1.0
    T = 0.8
    n_ic = 1200
    n_eval = 201
    margin = v * T + 0.2
    x_ic = np.linspace(-margin, L_adv + margin, n_ic)
    y_ic = riemann_ic(x_ic, x_disc=x_disc)

    x_eval_adv = np.linspace(0, L_adv, n_eval)
    t_eval_adv = np.linspace(0, T, n_eval)
    X, TT = np.meshgrid(x_eval_adv, t_eval_adv)
    U_exact_adv = riemann_ic(X - v * TT, x_disc=x_disc)

    adv_means = []
    adv_stds = []
    for n_tanh in n_tanh_values:
        l2_seeds = []
        for seed in range(n_seeds):
            W, b = generate_tanh_weights(n_tanh, seed, scale, L_adv)
            positions = [x_disc]
            H = hidden_matrix(x_ic, W, b, positions, kappa)
            beta = solve_ridge(H, y_ic, lam)
            # Evaluate on full grid
            errs = []
            for j, t in enumerate(t_eval_adv):
                xi = x_eval_adv - v * t
                u_pred = elm_predict(xi, W, b, positions, kappa, beta)
                u_exact = U_exact_adv[j]
                # Avoid edge effects
                mask = (x_eval_adv > 0.05) & (x_eval_adv < L_adv - 0.05)
                norm_ref = max(np.linalg.norm(u_exact[mask]), 1e-12)
                errs.append(np.linalg.norm(u_pred[mask] - u_exact[mask]) / norm_ref)
            l2_seeds.append(np.mean(errs))
        adv_means.append(np.mean(l2_seeds))
        adv_stds.append(np.std(l2_seeds))
        print(f"    N_tanh={n_tanh:4d}: L2 = {adv_means[-1]:.6f} +/- {adv_stds[-1]:.6f}")

    # ── (b) Nonlinear Burgers: smooth IC, t=0.20 ────────────────────────
    print("\n  (b) Burgers smooth, u0=-sin(pi*x), t=0.20")
    x_min_b, x_max_b = -1.0, 1.0
    t_burgers = 0.20

    def ic_burgers(x):
        return -np.sin(np.pi * x)

    margin_b = 1.0 * t_burgers + 0.5
    n_ic_b = 1200
    n_eval_b = 1000
    x_ic_b = np.linspace(x_min_b - margin_b, x_max_b + margin_b, n_ic_b)
    y_ic_b = ic_burgers(x_ic_b)
    x_eval_b = np.linspace(x_min_b, x_max_b, n_eval_b)
    u_exact_b = burgers_smooth_exact(x_eval_b, t_burgers, ic_burgers,
                                      xi_min=x_min_b - margin_b,
                                      xi_max=x_max_b + margin_b)

    burg_means = []
    burg_stds = []
    for n_tanh in n_tanh_values:
        l2_seeds = []
        for seed in range(n_seeds):
            W, b = generate_tanh_weights(n_tanh, seed, scale,
                                          (x_max_b - x_min_b + 2 * margin_b))
            positions = []
            H = hidden_matrix(x_ic_b, W, b, positions, kappa)
            beta = solve_ridge(H, y_ic_b, lam)
            u_pred = cielm_fixed_point(x_eval_b, t_burgers, W, b,
                                        positions, kappa, beta)
            norm_ref = max(np.linalg.norm(u_exact_b), 1e-12)
            l2 = np.linalg.norm(u_pred - u_exact_b) / norm_ref
            l2_seeds.append(l2)
        burg_means.append(np.mean(l2_seeds))
        burg_stds.append(np.std(l2_seeds))
        print(f"    N_tanh={n_tanh:4d}: L2 = {burg_means[-1]:.2e} +/- {burg_stds[-1]:.2e}")

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor(BG)

    # Advection
    style_ax(ax1)
    ax1.errorbar(n_tanh_values, adv_means, yerr=adv_stds,
                 color=C_BLUE, marker='o', linewidth=2, capsize=4, zorder=3,
                 label='ELM-CINN + step')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$N_{\mathrm{tanh}}$', color=C_TEXT, fontsize=12)
    ax1.set_ylabel(r'$L_2$ error', color=C_TEXT, fontsize=12)
    ax1.set_title('Linear advection (Riemann IC)', color=C_TEXT, fontsize=12,
                  fontfamily='serif')
    ax1.set_xticks(n_tanh_values)
    ax1.set_xticklabels([str(n) for n in n_tanh_values])
    add_legend(ax1, loc='upper right')

    # Burgers
    style_ax(ax2)
    ax2.errorbar(n_tanh_values, burg_means, yerr=burg_stds,
                 color=C_RED, marker='s', linewidth=2, capsize=4, zorder=3,
                 label='ELM-CINN (fixed-point)')
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$N_{\mathrm{tanh}}$', color=C_TEXT, fontsize=12)
    ax2.set_ylabel(r'$L_2$ error', color=C_TEXT, fontsize=12)
    ax2.set_title(r'Burgers smooth ($t = 0.20$)', color=C_TEXT, fontsize=12,
                  fontfamily='serif')
    ax2.set_xticks(n_tanh_values)
    ax2.set_xticklabels([str(n) for n in n_tanh_values])
    add_legend(ax2, loc='upper right')

    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, 'convergence_n_tanh.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"\n  Saved: {fname}")
    plt.close(fig)

    return dict(n_tanh=n_tanh_values,
                adv_means=adv_means, adv_stds=adv_stds,
                burg_means=burg_means, burg_stds=burg_stds)


# ═════════════════════════════════════════════════════════════════════════════
# Study 2: Sensitivity to kappa (step steepness)
# ═════════════════════════════════════════════════════════════════════════════
def study_kappa_sensitivity():
    print("\n" + "=" * 70)
    print("Study 2: L2 error vs kappa (step steepness)")
    print("=" * 70)

    kappa_values = [10, 50, 100, 200, 500, 1000, 2000, 5000]
    n_tanh = 80
    seed = 7
    scale = 2.5
    lam = 1e-6
    n_seeds = 10

    L_adv = 2.0
    x_disc = 1.0
    v = 1.0
    T = 0.8
    n_ic = 1200
    n_eval = 201
    margin = v * T + 0.2
    x_ic = np.linspace(-margin, L_adv + margin, n_ic)
    y_ic = riemann_ic(x_ic, x_disc=x_disc)

    x_eval = np.linspace(0, L_adv, n_eval)
    t_eval = np.linspace(0, T, n_eval)
    X, TT = np.meshgrid(x_eval, t_eval)
    U_exact = riemann_ic(X - v * TT, x_disc=x_disc)

    kap_means = []
    kap_stds = []
    for kappa in kappa_values:
        l2_seeds = []
        for s in range(n_seeds):
            W, b = generate_tanh_weights(n_tanh, s, scale, L_adv)
            positions = [x_disc]
            H = hidden_matrix(x_ic, W, b, positions, kappa)
            beta = solve_ridge(H, y_ic, lam)
            errs = []
            for j, t in enumerate(t_eval):
                xi = x_eval - v * t
                u_pred = elm_predict(xi, W, b, positions, kappa, beta)
                u_exact = U_exact[j]
                mask = (x_eval > 0.05) & (x_eval < L_adv - 0.05)
                norm_ref = max(np.linalg.norm(u_exact[mask]), 1e-12)
                errs.append(np.linalg.norm(u_pred[mask] - u_exact[mask]) / norm_ref)
            l2_seeds.append(np.mean(errs))
        kap_means.append(np.mean(l2_seeds))
        kap_stds.append(np.std(l2_seeds))
        print(f"    kappa={kappa:5d}: L2 = {kap_means[-1]:.6f} +/- {kap_stds[-1]:.6f}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    fig.patch.set_facecolor(BG)
    style_ax(ax)
    ax.errorbar(kappa_values, kap_means, yerr=kap_stds,
                color=C_GREEN, marker='D', linewidth=2, capsize=4, zorder=3,
                label='ELM-CINN + step')
    ax.axvline(500, color=C_ORANGE, linestyle='--', linewidth=1.5, alpha=0.7,
               label=r'$\kappa = 500$ (default)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\kappa$ (step steepness)', color=C_TEXT, fontsize=12)
    ax.set_ylabel(r'$L_2$ error', color=C_TEXT, fontsize=12)
    ax.set_title('Sensitivity to step steepness (linear advection)',
                 color=C_TEXT, fontsize=12, fontfamily='serif')
    add_legend(ax, loc='upper right')

    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, 'sensitivity_kappa.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"\n  Saved: {fname}")
    plt.close(fig)

    return dict(kappa=kappa_values, means=kap_means, stds=kap_stds)


# ═════════════════════════════════════════════════════════════════════════════
# Study 3: Condition number of H^T H + lambda I vs N_tanh
# ═════════════════════════════════════════════════════════════════════════════
def study_condition_number():
    print("\n" + "=" * 70)
    print("Study 3: Condition number vs N_tanh")
    print("=" * 70)

    n_tanh_values = [10, 20, 40, 80, 160, 320]
    kappa = 500
    lam = 1e-6
    scale = 2.5
    seed = 7

    x_min_b, x_max_b = -1.0, 1.0
    margin_b = 0.7
    n_ic = 1200
    x_ic = np.linspace(x_min_b - margin_b, x_max_b + margin_b, n_ic)

    conds = []
    for n_tanh in n_tanh_values:
        W, b = generate_tanh_weights(n_tanh, seed, scale,
                                      (x_max_b - x_min_b + 2 * margin_b))
        H = hidden_matrix(x_ic, W, b, [], kappa)
        A = H.T @ H + lam * np.eye(H.shape[1])
        cond = np.linalg.cond(A)
        conds.append(cond)
        print(f"    N_tanh={n_tanh:4d}: cond(H^T H + lambda I) = {cond:.2e}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    fig.patch.set_facecolor(BG)
    style_ax(ax)
    ax.plot(n_tanh_values, conds, color=C_PURPLE, marker='^', linewidth=2,
            zorder=3, label=r'$\mathrm{cond}(H^\top H + \lambda I)$')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel(r'$N_{\mathrm{tanh}}$', color=C_TEXT, fontsize=12)
    ax.set_ylabel('Condition number', color=C_TEXT, fontsize=12)
    ax.set_title(r'Numerical stability ($\lambda = 10^{-6}$)',
                 color=C_TEXT, fontsize=12, fontfamily='serif')
    ax.set_xticks(n_tanh_values)
    ax.set_xticklabels([str(n) for n in n_tanh_values])
    add_legend(ax, loc='upper left')

    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, 'condition_number.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"\n  Saved: {fname}")
    plt.close(fig)

    return dict(n_tanh=n_tanh_values, conds=conds)


if __name__ == '__main__':
    print("CONVERGENCE AND SENSITIVITY STUDIES FOR STEP-CINN PAPER")
    print("=" * 70)

    r1 = study_convergence_n_tanh()
    r2 = study_kappa_sensitivity()
    r3 = study_condition_number()

    print("\n" + "=" * 70)
    print("ALL STUDIES COMPLETE")
    print("=" * 70)
