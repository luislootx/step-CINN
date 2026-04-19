"""
CIELM core primitives: ELM basis, ridge readout, plotting, I/O helpers.

Shared across all experiment scripts in this directory.  Keeps each
script focused on the PDE physics rather than boilerplate.

Notation follows the paper:
    N_tanh   number of random-fixed tanh hidden units
    kappa    steepness of sigmoid step neurons (Step-CIELM)
    beta     output-layer weights (analytical ridge solve)
    xi       characteristic coordinate
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# 1. ELM primitives
# ============================================================================

def generate_tanh_weights(N_tanh, seed=7, scale=2.5, domain_scale=1.0):
    """Random fixed input-layer weights for the tanh ELM basis.

    Parameters
    ----------
    N_tanh : int
        Number of tanh hidden units.
    seed : int
        RNG seed (the input-layer weights are frozen for the entire run).
    scale : float
        Support radius of the slope parameter W.
    domain_scale : float
        Support radius of the bias parameter b.  Should roughly match the
        physical domain span so that tanh zero-crossings populate the
        input range.
    """
    rng = np.random.default_rng(seed)
    W = rng.uniform(-scale, scale, size=N_tanh)
    b = rng.uniform(-scale * domain_scale, scale * domain_scale, size=N_tanh)
    return W, b


def sigmoid(z):
    """Numerically stable sigmoid, used for step-neuron activations."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def hidden_matrix(x, W_tanh, b_tanh, positions=(), kappa=0.0):
    """Hidden-layer matrix H(x) for CIELM (and Step-CIELM if positions given).

    Returns an (N, N_tanh + K) matrix where K = len(positions).  When
    positions is empty, H(x) is the pure tanh block used by CIELM.
    """
    z_tanh = np.outer(x, W_tanh) + b_tanh
    H_tanh = np.tanh(z_tanh)
    if len(positions) > 0:
        z_step = kappa * (x.reshape(-1, 1) - np.asarray(positions).reshape(1, -1))
        H_step = sigmoid(z_step)
        return np.hstack([H_tanh, H_step])
    return H_tanh


def hidden_matrix_2d(x, y, Wx, Wy, b):
    """H(x, y) with entries tanh(W_x x + W_y y + b), shape (N, N_tanh)."""
    z = np.outer(x, Wx) + np.outer(y, Wy) + b
    return np.tanh(z)


def generate_tanh_weights_2d(N_tanh, seed, scale=2.5, domain_scale=2.0):
    """Two-dimensional variant: each neuron has independent (w_x, w_y, b)."""
    rng = np.random.default_rng(seed)
    Wx = rng.uniform(-scale, scale, size=N_tanh)
    Wy = rng.uniform(-scale, scale, size=N_tanh)
    b = rng.uniform(-scale * domain_scale, scale * domain_scale, size=N_tanh)
    return Wx, Wy, b


def solve_ridge(H, y, lam=1e-6):
    """Ridge (Tikhonov) solve for the output weights: the single trainable
    stage of CIELM.

        beta = (H^T H + lam I)^{-1} H^T y
    """
    n = H.shape[1]
    A = H.T @ H + lam * np.eye(n)
    return np.linalg.solve(A, H.T @ y)


def compute_errors(u_pred, u_ref):
    """Return (L1_rel, L2_rel) errors against a reference solution."""
    norm_ref = max(np.linalg.norm(u_ref), 1e-12)
    l2 = float(np.linalg.norm(u_pred - u_ref) / norm_ref)
    mean_abs_ref = max(np.mean(np.abs(u_ref)), 1e-12)
    l1 = float(np.mean(np.abs(u_pred - u_ref)) / mean_abs_ref)
    return l1, l2


# ============================================================================
# 2. Nonlinear characteristic solvers (Burgers and friends)
# ============================================================================

def picard_fixed_point(x_eval, t, W_tanh, b_tanh, beta,
                       positions=(), kappa=0.0,
                       max_iter=500, tol=1e-12):
    """Picard iteration for xi = x - u_ELM(xi) * t (inviscid Burgers).

    Returns (u_pred, info_dict).  Converges while |du_ELM/dxi| * t < 1
    (Banach contraction, i.e. before shock formation).
    """
    if t < 1e-14:
        u0 = hidden_matrix(x_eval, W_tanh, b_tanh, positions, kappa) @ beta
        return u0, {'iters': 0, 'residual': 0.0, 'converged': True}

    xi = x_eval.copy()
    residuals = []
    for k in range(max_iter):
        H = hidden_matrix(xi, W_tanh, b_tanh, positions, kappa)
        u_xi = H @ beta
        xi_new = x_eval - u_xi * t
        res = float(np.max(np.abs(xi_new - xi)))
        residuals.append(res)
        xi = xi_new
        if res < tol:
            break

    u_pred = hidden_matrix(xi, W_tanh, b_tanh, positions, kappa) @ beta
    return u_pred, {
        'iters': k + 1,
        'residual': residuals[-1] if residuals else 0.0,
        'converged': residuals[-1] < tol if residuals else True,
    }


def anderson_fixed_point(x_eval, t, W_tanh, b_tanh, beta,
                         positions=(), kappa=0.0,
                         max_iter=1000, tol=1e-12, m=5):
    """Anderson-accelerated fixed-point iteration (depth m)."""
    if t < 1e-14:
        u0 = hidden_matrix(x_eval, W_tanh, b_tanh, positions, kappa) @ beta
        return u0, {'iters': 0, 'converged': True}

    N = len(x_eval)
    xi = x_eval.copy()
    Xi_hist, G_hist, residuals = [], [], []

    for k in range(max_iter):
        H = hidden_matrix(xi, W_tanh, b_tanh, positions, kappa)
        u_xi = H @ beta
        g = x_eval - u_xi * t
        f = g - xi
        res = float(np.max(np.abs(f)))
        residuals.append(res)
        if res < tol:
            xi = g
            break

        Xi_hist.append(xi.copy())
        G_hist.append(g.copy())
        if len(Xi_hist) > m + 1:
            Xi_hist.pop(0)
            G_hist.pop(0)

        mk = len(Xi_hist) - 1
        if mk < 1:
            xi = g
            continue

        F_curr = G_hist[-1] - Xi_hist[-1]
        dF = np.zeros((N, mk))
        for j in range(mk):
            dF[:, j] = F_curr - (G_hist[j] - Xi_hist[j])
        try:
            alpha, _, _, _ = np.linalg.lstsq(dF, F_curr, rcond=None)
        except np.linalg.LinAlgError:
            xi = g
            continue
        xi_new = (1.0 - np.sum(alpha)) * G_hist[-1]
        for j in range(mk):
            xi_new += alpha[j] * G_hist[j]
        xi = xi_new

    u_pred = hidden_matrix(xi, W_tanh, b_tanh, positions, kappa) @ beta
    return u_pred, {
        'iters': k + 1,
        'converged': residuals[-1] < tol if residuals else True,
    }


def burgers_char_exact(x_eval, t, ic_func,
                       xi_min=-3.0, xi_max=3.0, n_char=10000):
    """Reference solution for smooth inviscid Burgers by forward-tracing
    characteristics from a dense xi grid.  Valid only while the map
    xi -> x = xi + u_0(xi) t is monotone (pre-shock).
    """
    if t < 1e-14:
        return ic_func(x_eval)
    xi_fine = np.linspace(xi_min, xi_max, n_char)
    u_fine = ic_func(xi_fine)
    x_fine = xi_fine + u_fine * t
    return np.interp(x_eval, x_fine, u_fine)


# ============================================================================
# 3. Plot styling (light academic palette, matches the paper)
# ============================================================================

C_BLUE   = '#2166ac'   # exact / reference solution
C_RED    = '#d6604d'   # CIELM prediction
C_GREEN  = '#1b7837'   # CIELM in comparison charts
C_ORANGE = '#e08214'   # step position / auxiliary marker
C_PURPLE = '#7b3294'   # numerical diagnostics
C_CYAN   = '#0571b0'   # PINN reference
C_TEXT   = '#333333'
BG       = '#ffffff'
BG_AX    = '#fafafa'
GRID     = '#e0e0e0'
SPINE    = '#aaaaaa'


def style_ax(ax):
    """Apply the light-academic style to a matplotlib Axes."""
    ax.set_facecolor(BG_AX)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.tick_params(colors=C_TEXT, labelsize=10)
    for s in ax.spines.values():
        s.set_edgecolor(SPINE)


def add_legend(ax, **kw):
    """Standard legend style (small font, light frame)."""
    ax.legend(fontsize=8, framealpha=0.9, edgecolor=SPINE, **kw)


# ============================================================================
# 4. File-system helpers (figures/ and results/ next to the scripts)
# ============================================================================

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')


def fig_path(name):
    """Absolute path for a figure file under paper/scripts/figures/."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    return os.path.join(FIGURES_DIR, name)


def result_path(name):
    """Absolute path for a JSON result file under paper/scripts/results/."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, name)
