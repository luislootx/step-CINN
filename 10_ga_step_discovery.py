"""
Section 6.10 -- Step Discovery from Data for a PDE with Unknown IC
==================================================================
The initial condition is not given analytically; only a set of samples
(x_i, u_0(x_i) + noise) is observed.  A genetic algorithm discovers the
number and positions of the discontinuities, after which Step-CIELM
propagates the solution forward by the characteristic shift.

PDE:   u_t + v u_x = 0,        v = 1.5,      x in [0, 10]
IC:    piecewise-smooth with three unknown jumps at {2.0, 5.0, 7.5}.

Three data regimes are tested and compared against two non-evolutionary
baselines:

    1. Clean data   (N = 500, no noise)
    2. Noisy data   (N = 500, sigma = 0.05 and sigma = 0.10)
    3. Sparse data  (N =  50, no noise)

    Oracle      -- step positions placed manually at the true locations
    Tanh only   -- Step-CIELM with zero step neurons

Artifacts produced
------------------
figures/fig10_ga_pde_discovery.png        six-panel figure
results/results_10_ga_pde.json            positions, IC + PDE errors
"""

import json
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize as sp_minimize

from _core import (
    generate_tanh_weights, hidden_matrix, solve_ridge,
    style_ax, add_legend, fig_path, result_path,
    C_BLUE, C_RED, C_GREEN, C_ORANGE, C_PURPLE, C_TEXT, BG,
)


# ----------------------------------------------------------------------------
# Problem setup
# ----------------------------------------------------------------------------

L = 10.0
T_FINAL = 1.0
V_CONST = 1.5
TRUE_DISC = [2.0, 5.0, 7.5]


def true_ic(x):
    u = np.zeros_like(x, dtype=float)
    for i, xi in enumerate(x):
        if xi < 2.0:
            u[i] = 1.0 + 0.3 * np.sin(2 * np.pi * xi / 2.0)
        elif xi < 5.0:
            u[i] = 3.0 + 0.4 * np.cos(np.pi * (xi - 2.0) / 3.0)
        elif xi < 7.5:
            u[i] = 1.5
        else:
            u[i] = 3.5 + 0.5 * np.exp(-2.0 * (xi - 8.5) ** 2)
    return u


def exact_solution(x, t):
    return true_ic(x - V_CONST * t)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def l2_relative(u_ref, u_pred):
    return float(np.linalg.norm(u_pred - u_ref) /
                 max(np.linalg.norm(u_ref), 1e-12))


def predict(x, W_tanh, b_tanh, positions, kappa, beta):
    return hidden_matrix(x, W_tanh, b_tanh, positions, kappa) @ beta


def cielm_evaluate(x_eval, t, W_tanh, b_tanh, positions, kappa, beta):
    """Characteristic-shift evaluation at time t."""
    xi = x_eval - V_CONST * t
    return predict(xi, W_tanh, b_tanh, positions, kappa, beta)


# ----------------------------------------------------------------------------
# GA for step discovery
# ----------------------------------------------------------------------------

_H_TANH_CACHE = {}


def get_tanh_hidden(x, W_tanh, b_tanh):
    key = (id(x), id(W_tanh))
    if key not in _H_TANH_CACHE:
        _H_TANH_CACHE[key] = hidden_matrix(x, W_tanh, b_tanh)
    return _H_TANH_CACHE[key]


def step_hidden(x, positions, kappa=500.0):
    if len(positions) == 0:
        return np.empty((len(x), 0))
    positions = np.asarray(positions)
    z = kappa * (x.reshape(-1, 1) - positions.reshape(1, -1))
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def evaluate_individual(positions, x_train, y_train, x_val, y_val,
                        W_tanh, b_tanh, kappa, lam, parsimony):
    H_tanh = get_tanh_hidden(x_train, W_tanh, b_tanh)
    H_step = step_hidden(x_train, positions, kappa)
    H = np.hstack([H_tanh, H_step])
    beta = solve_ridge(H, y_train, lam)

    H_tanh_v = get_tanh_hidden(x_val, W_tanh, b_tanh)
    H_step_v = step_hidden(x_val, positions, kappa)
    y_pred = np.hstack([H_tanh_v, H_step_v]) @ beta

    val_rmse = rmse(y_val, y_pred)
    return val_rmse + parsimony * len(positions), val_rmse, beta


class Individual:
    def __init__(self, positions):
        self.positions = sorted(positions)
        self.fitness = np.inf
        self.val_rmse = np.inf
        self.beta = None

    def copy(self):
        ind = Individual(list(self.positions))
        ind.fitness = self.fitness
        ind.val_rmse = self.val_rmse
        ind.beta = self.beta
        return ind


def run_ga(x_train, y_train, x_val, y_val, W_tanh, b_tanh, config,
           verbose=True):
    rng = np.random.default_rng(config['seed_ga'])
    x_lo, x_hi = float(x_train.min()), float(x_train.max())

    pop = []
    for _ in range(config['pop_size']):
        K = rng.integers(1, config['max_steps'] + 1)
        positions = sorted(rng.uniform(x_lo + 0.3, x_hi - 0.3,
                                       size=K).tolist())
        pop.append(Individual(positions))

    best_ever = None
    history = []

    for gen in range(config['n_gen']):
        for ind in pop:
            ind.fitness, ind.val_rmse, ind.beta = evaluate_individual(
                ind.positions, x_train, y_train, x_val, y_val,
                W_tanh, b_tanh, config['kappa'], config['lam'],
                config['parsimony'])
        pop.sort(key=lambda ind: ind.fitness)
        if best_ever is None or pop[0].fitness < best_ever.fitness:
            best_ever = pop[0].copy()
        history.append({
            'gen': gen,
            'best_fitness': float(pop[0].fitness),
            'best_val_rmse': float(pop[0].val_rmse),
            'best_n_steps': len(pop[0].positions),
        })

        if verbose and (gen % 10 == 0 or gen == config['n_gen'] - 1):
            pos_str = ', '.join(f'{p:.2f}' for p in pop[0].positions)
            print(f"  Gen {gen:3d}  |  RMSE = {pop[0].val_rmse:.6f}  "
                  f"K = {len(pop[0].positions)} [{pos_str}]")

        elite = config['elite_count']
        new_pop = [ind.copy() for ind in pop[:elite]]
        while len(new_pop) < config['pop_size']:
            idx_a = rng.choice(len(pop), size=3, replace=False)
            p1 = pop[min(idx_a, key=lambda i: pop[i].fitness)].copy()
            idx_b = rng.choice(len(pop), size=3, replace=False)
            p2 = pop[min(idx_b, key=lambda i: pop[i].fitness)].copy()
            all_pos = sorted(set(p1.positions + p2.positions))
            if len(all_pos) > 0:
                n = rng.integers(1, min(len(all_pos),
                                        config['max_steps']) + 1)
                chosen = sorted(rng.choice(all_pos, size=n,
                                           replace=False).tolist())
            else:
                chosen = [rng.uniform(x_lo + 0.3, x_hi - 0.3)]
            child = Individual(chosen)
            positions = list(child.positions)
            for i in range(len(positions)):
                if rng.random() < config['mutation_rate']:
                    positions[i] += rng.normal(0, 0.3)
                    positions[i] = np.clip(positions[i], x_lo + 0.1,
                                           x_hi - 0.1)
            if rng.random() < 0.10 and len(positions) < config['max_steps']:
                positions.append(rng.uniform(x_lo + 0.3, x_hi - 0.3))
            if rng.random() < 0.10 and len(positions) > 1:
                positions.pop(rng.integers(0, len(positions)))
            positions = sorted(positions)
            merged = [positions[0]]
            for p in positions[1:]:
                if abs(p - merged[-1]) < 0.3:
                    merged[-1] = (merged[-1] + p) / 2
                else:
                    merged.append(p)
            new_pop.append(Individual(merged))
        pop = new_pop

    # Nelder-Mead refinement
    if len(best_ever.positions) > 0:
        def nm_obj(pos_vec):
            positions = sorted(pos_vec.tolist())
            f, _, _ = evaluate_individual(
                positions, x_train, y_train, x_val, y_val,
                W_tanh, b_tanh, config['kappa'], config['lam'], 0.0)
            return f
        res = sp_minimize(nm_obj, np.array(best_ever.positions),
                          method='Nelder-Mead',
                          options={'maxfev': 500, 'xatol': 1e-4,
                                   'fatol': 1e-6})
        refined = sorted(res.x.tolist())
        f_ref, rmse_ref, beta_ref = evaluate_individual(
            refined, x_train, y_train, x_val, y_val,
            W_tanh, b_tanh, config['kappa'], config['lam'],
            config['parsimony'])
        if f_ref < best_ever.fitness:
            best_ever.positions = refined
            best_ever.fitness = f_ref
            best_ever.val_rmse = rmse_ref
            best_ever.beta = beta_ref
        if verbose:
            print(f"  NM refined: RMSE = {best_ever.val_rmse:.6f}  "
                  f"pos = {[f'{p:.3f}' for p in best_ever.positions]}")

    return best_ever, history


# ----------------------------------------------------------------------------
# Experiment runners
# ----------------------------------------------------------------------------

def run_setting(name, x_data, y_data, ga_config, verbose=True):
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  Setting: {name}")
        print(f"  Data points: {len(x_data)}, "
              f"noise std: {np.std(y_data - true_ic(x_data)):.4f}")
        print(f"{'=' * 70}")

    x_val = np.linspace(0, L, 2000)
    y_val = true_ic(x_val)

    margin = V_CONST * T_FINAL + 0.5
    x_ic_ext = np.linspace(-margin, L + 0.5, 1000)
    y_ic_ext = true_ic(x_ic_ext)

    W_tanh, b_tanh = generate_tanh_weights(
        ga_config['N_tanh'], ga_config['seed_tanh'],
        scale=2.5, domain_scale=L + margin)

    _H_TANH_CACHE.clear()

    t0 = time.time()
    best, history = run_ga(x_data, y_data, x_val, y_val,
                           W_tanh, b_tanh, ga_config, verbose=verbose)
    ga_time = time.time() - t0

    H_ext = hidden_matrix(x_ic_ext, W_tanh, b_tanh,
                          best.positions, ga_config['kappa'])
    beta_ext = solve_ridge(H_ext, y_ic_ext, ga_config['lam'])

    y_ic_pred = predict(x_val, W_tanh, b_tanh, best.positions,
                        ga_config['kappa'], beta_ext)
    ic_rmse_val = rmse(y_val, y_ic_pred)
    ic_l2 = l2_relative(y_val, y_ic_pred)

    snap_times = [0.0, 0.25, 0.5, 1.0]
    pde = {}
    for t in snap_times:
        u_pred = cielm_evaluate(x_val, t, W_tanh, b_tanh,
                                best.positions, ga_config['kappa'],
                                beta_ext)
        u_exact = exact_solution(x_val, t)
        pde[t] = {
            'l2': l2_relative(u_exact, u_pred),
            'rmse': rmse(u_exact, u_pred),
            'u_pred': u_pred, 'u_exact': u_exact,
        }

    pos_errors = []
    for true_p in TRUE_DISC:
        if len(best.positions) > 0:
            pos_errors.append(min(abs(p - true_p) for p in best.positions))
        else:
            pos_errors.append(float('inf'))

    result = {
        'setting': name,
        'n_data': len(x_data),
        'K_found': len(best.positions),
        'K_true': len(TRUE_DISC),
        'positions_found': list(best.positions),
        'positions_true': TRUE_DISC,
        'position_errors': pos_errors,
        'max_pos_error': max(pos_errors) if pos_errors else float('inf'),
        'ic_rmse': ic_rmse_val,
        'ic_l2': ic_l2,
        'pde_l2':  {str(t): pde[t]['l2']  for t in snap_times},
        'pde_rmse': {str(t): pde[t]['rmse'] for t in snap_times},
        'ga_time_s': ga_time,
    }

    if verbose:
        print(f"\n  K found: {result['K_found']}  (true: {result['K_true']})")
        print(f"  Positions found: "
              f"{[f'{p:.3f}' for p in best.positions]}")
        print(f"  Positions true:  {TRUE_DISC}")
        print(f"  IC RMSE: {ic_rmse_val:.6f}   IC L2: {ic_l2:.6f}")
        for t in snap_times:
            print(f"  PDE L2 (t = {t:.2f}): {pde[t]['l2']:.6f}")

    return result, pde, x_val, beta_ext, W_tanh, b_tanh


def run_oracle(verbose=True):
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  Oracle (manual step placement)")
        print(f"{'=' * 70}")
    margin = V_CONST * T_FINAL + 0.5
    x_ic = np.linspace(-margin, L + 0.5, 1000)
    y_ic = true_ic(x_ic)
    x_val = np.linspace(0, L, 2000)
    y_val = true_ic(x_val)
    W_tanh, b_tanh = generate_tanh_weights(80, 7, scale=2.5,
                                           domain_scale=L + margin)
    H = hidden_matrix(x_ic, W_tanh, b_tanh, TRUE_DISC, 500.0)
    beta = solve_ridge(H, y_ic, 1e-6)
    y_pred = predict(x_val, W_tanh, b_tanh, TRUE_DISC, 500.0, beta)
    ic_rmse_val = rmse(y_val, y_pred)
    ic_l2 = l2_relative(y_val, y_pred)

    snap_times = [0.0, 0.25, 0.5, 1.0]
    pde = {}
    for t in snap_times:
        u_pred = cielm_evaluate(x_val, t, W_tanh, b_tanh, TRUE_DISC,
                                500.0, beta)
        u_exact = exact_solution(x_val, t)
        pde[t] = {'l2': l2_relative(u_exact, u_pred),
                  'rmse': rmse(u_exact, u_pred),
                  'u_pred': u_pred, 'u_exact': u_exact}
    if verbose:
        print(f"  IC RMSE: {ic_rmse_val:.6f}   IC L2: {ic_l2:.6f}")
    return {'ic_rmse': ic_rmse_val, 'ic_l2': ic_l2,
            'pde_l2': {str(t): pde[t]['l2'] for t in snap_times}}, \
           pde, x_val, beta, W_tanh, b_tanh


def run_tanh_only(verbose=True):
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  Tanh-only baseline (no step neurons)")
        print(f"{'=' * 70}")
    margin = V_CONST * T_FINAL + 0.5
    x_ic = np.linspace(-margin, L + 0.5, 1000)
    y_ic = true_ic(x_ic)
    x_val = np.linspace(0, L, 2000)
    y_val = true_ic(x_val)
    W_tanh, b_tanh = generate_tanh_weights(80, 7, scale=2.5,
                                           domain_scale=L + margin)
    H = hidden_matrix(x_ic, W_tanh, b_tanh, [], 500.0)
    beta = solve_ridge(H, y_ic, 1e-6)
    y_pred = predict(x_val, W_tanh, b_tanh, [], 500.0, beta)
    ic_rmse_val = rmse(y_val, y_pred)
    ic_l2 = l2_relative(y_val, y_pred)

    snap_times = [0.0, 0.25, 0.5, 1.0]
    pde = {}
    for t in snap_times:
        u_pred = cielm_evaluate(x_val, t, W_tanh, b_tanh, [], 500.0, beta)
        u_exact = exact_solution(x_val, t)
        pde[t] = {'l2': l2_relative(u_exact, u_pred),
                  'rmse': rmse(u_exact, u_pred),
                  'u_pred': u_pred, 'u_exact': u_exact}
    if verbose:
        print(f"  IC RMSE: {ic_rmse_val:.6f}   IC L2: {ic_l2:.6f}")
    return {'ic_rmse': ic_rmse_val, 'ic_l2': ic_l2,
            'pde_l2': {str(t): pde[t]['l2'] for t in snap_times}}, \
           pde, x_val, beta, W_tanh, b_tanh


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------

def plot_results(all_pde, all_results, all_extra, x_val, fname):
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.patch.set_facecolor(BG)
    fig.suptitle("GA step discovery for a PDE with unknown IC",
                 fontsize=14, fontweight='bold', color=C_TEXT)

    snap_t = 0.5

    # Panel (0, 0): IC fit, clean data
    ax = axes[0, 0]
    style_ax(ax)
    y_true = true_ic(x_val)
    ax.plot(x_val, y_true, color=C_BLUE, lw=2, label='True IC')
    if 'clean' in all_extra:
        xv, beta, wt, bt, pos = all_extra['clean']
        y_pred = predict(x_val, wt, bt, pos, 500.0, beta)
        ax.plot(x_val, y_pred, '--', color=C_RED, lw=1.5,
                label=f'GA (K = {len(pos)})')
        for p in pos:
            ax.axvline(p, color=C_RED, ls=':', alpha=0.5, lw=1)
    for p in TRUE_DISC:
        ax.axvline(p, color=C_BLUE, ls='--', alpha=0.3, lw=1)
    ax.set_title('IC fit: clean data (N = 500)')
    ax.set_xlabel('x'); ax.set_ylabel('u'); add_legend(ax)

    # Panel (0, 1): PDE at t = 0.5
    ax = axes[0, 1]
    style_ax(ax)
    u_exact = exact_solution(x_val, snap_t)
    ax.plot(x_val, u_exact, color=C_BLUE, lw=2, label='Exact')
    if 'oracle' in all_pde and snap_t in all_pde['oracle']:
        ax.plot(x_val, all_pde['oracle'][snap_t]['u_pred'], '--',
                color=C_GREEN, lw=1.5,
                label=f"Oracle (L$_2$ = {all_pde['oracle'][snap_t]['l2']:.4f})")
    if 'clean' in all_pde and snap_t in all_pde['clean']:
        ax.plot(x_val, all_pde['clean'][snap_t]['u_pred'], '--',
                color=C_RED, lw=1.5,
                label=f"GA clean (L$_2$ = {all_pde['clean'][snap_t]['l2']:.4f})")
    if 'tanh_only' in all_pde and snap_t in all_pde['tanh_only']:
        ax.plot(x_val, all_pde['tanh_only'][snap_t]['u_pred'], ':',
                color=C_PURPLE, lw=1.5,
                label=f"Tanh only (L$_2$ = {all_pde['tanh_only'][snap_t]['l2']:.4f})")
    ax.set_title(f'PDE solution at t = {snap_t}')
    ax.set_xlabel('x'); add_legend(ax)

    # Panel (1, 0) and (1, 1): noisy IC fits
    key_n05 = 'noisy_sigma_0.05'
    key_n10 = 'noisy_sigma_0.10'
    x_500 = np.linspace(0, L, 500)
    rng = np.random.default_rng(42)

    for ax, key, sigma in [(axes[1, 0], key_n05, 0.05),
                           (axes[1, 1], key_n10, 0.10)]:
        style_ax(ax)
        ax.plot(x_val, y_true, color=C_BLUE, lw=2, label='True IC')
        if key in all_extra:
            xv, beta, wt, bt, pos = all_extra[key]
            y_pred = predict(x_val, wt, bt, pos, 500.0, beta)
            ax.plot(x_val, y_pred, '--', color=C_RED, lw=1.5,
                    label=f'GA noisy (K = {len(pos)})')
            for p in pos:
                ax.axvline(p, color=C_RED, ls=':', alpha=0.5, lw=1)
        y_n = true_ic(x_500) + rng.normal(0, sigma, size=500)
        ax.scatter(x_500[::5], y_n[::5], s=8, color='gray', alpha=0.4,
                   label='Noisy data', zorder=1)
        ax.set_title(f'IC fit: noisy data (sigma = {sigma})')
        ax.set_xlabel('x'); ax.set_ylabel('u'); add_legend(ax)

    # Panel (2, 0): sparse data
    ax = axes[2, 0]
    style_ax(ax)
    ax.plot(x_val, y_true, color=C_BLUE, lw=2, label='True IC')
    if 'sparse' in all_extra:
        xv, beta, wt, bt, pos = all_extra['sparse']
        y_pred = predict(x_val, wt, bt, pos, 500.0, beta)
        ax.plot(x_val, y_pred, '--', color=C_RED, lw=1.5,
                label=f'GA sparse (K = {len(pos)})')
        for p in pos:
            ax.axvline(p, color=C_RED, ls=':', alpha=0.5, lw=1)
    x_sp = np.linspace(0, L, 50)
    ax.scatter(x_sp, true_ic(x_sp), s=25, color='gray', alpha=0.6,
               label='Sparse data (N = 50)', zorder=3)
    ax.set_title('IC fit: sparse data (N = 50)')
    ax.set_xlabel('x'); ax.set_ylabel('u'); add_legend(ax)

    # Panel (2, 1): PDE error vs time
    ax = axes[2, 1]
    style_ax(ax)
    times = [0.0, 0.25, 0.5, 1.0]
    labels_keys = [
        ('Oracle', 'oracle', C_GREEN, '-'),
        ('GA clean', 'clean', C_RED, '--'),
        ('GA noisy 0.05', key_n05, C_ORANGE, '--'),
        ('GA noisy 0.10', key_n10, '#b2182b', '--'),
        ('GA sparse', 'sparse', C_PURPLE, '--'),
        ('Tanh only', 'tanh_only', '#555555', ':'),
    ]
    for lbl, key, color, ls in labels_keys:
        if key in all_results:
            res = all_results[key]
            l2s = [res['pde_l2'].get(str(t)) for t in times]
            if all(v is not None for v in l2s):
                ax.plot(times, l2s, ls, color=color, lw=1.5, marker='o',
                        markersize=4, label=lbl)
    ax.set_xlabel('Time t'); ax.set_ylabel('Relative L$_2$ error')
    ax.set_title('PDE solution quality over time'); ax.set_yscale('log')
    add_legend(ax)

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
    print("  Section 6.10: GA step discovery for a PDE with unknown IC")
    print("=" * 70)

    np.random.seed(42)

    ga_config = {
        'N_tanh': 80, 'kappa': 500.0, 'lam': 1e-6,
        'seed_tanh': 7, 'seed_ga': 42,
        'pop_size': 50, 'n_gen': 80,
        'elite_count': 5, 'mutation_rate': 0.3,
        'max_steps': 8, 'parsimony': 0.002,
    }

    rng = np.random.default_rng(42)

    x_clean = np.linspace(0, L, 500)
    y_clean = true_ic(x_clean)

    x_sparse = np.linspace(0, L, 50)
    y_sparse = true_ic(x_sparse)

    y_noisy_005 = y_clean + rng.normal(0, 0.05, size=len(y_clean))
    y_noisy_010 = y_clean + rng.normal(0, 0.10, size=len(y_clean))

    oracle_res, oracle_pde, x_val, *_ = run_oracle()
    tanh_res,   tanh_pde,   *_        = run_tanh_only()

    settings = [
        ('clean',              x_clean,   y_clean,      'Clean (N = 500)'),
        ('noisy_sigma_0.05',   x_clean,   y_noisy_005,  'Noisy sigma = 0.05 (N = 500)'),
        ('noisy_sigma_0.10',   x_clean,   y_noisy_010,  'Noisy sigma = 0.10 (N = 500)'),
        ('sparse',             x_sparse,  y_sparse,     'Sparse (N = 50)'),
    ]

    all_results = {'oracle': oracle_res, 'tanh_only': tanh_res}
    all_pde = {'oracle': oracle_pde, 'tanh_only': tanh_pde}
    all_extra = {}

    for key, x_data, y_data, label in settings:
        _H_TANH_CACHE.clear()
        ga_config['seed_ga'] = 42
        res, pde, xv, beta, wt, bt = run_setting(label, x_data, y_data,
                                                 ga_config)
        all_results[key] = res
        all_pde[key] = pde
        all_extra[key] = (xv, beta, wt, bt, res['positions_found'])

    # Summary table
    print("\n" + "=" * 90)
    print("  SUMMARY")
    print("=" * 90)
    header = (f"  {'Setting':<30} {'K':>3} {'max|err|':>10} "
              f"{'IC RMSE':>10} {'PDE L2 t=0.5':>13} {'PDE L2 t=1.0':>13}")
    print(header)
    print("-" * 90)

    def _row(name, res, K=None):
        k = K if K is not None else res.get('K_found',
                                            len(res.get('positions_found', [])))
        mpe = res.get('max_pos_error', 0.0)
        mpe_str = f"{mpe:.4f}" if mpe < 100 else '---'
        p05 = res['pde_l2'].get('0.5', 0)
        p10 = res['pde_l2'].get('1.0', 0)
        print(f"  {name:<30} {k:>3} {mpe_str:>10} "
              f"{res['ic_rmse']:>10.6f} {p05:>13.6f} {p10:>13.6f}")

    _row('Oracle (manual)', oracle_res, K=3)
    _row('Tanh only (no steps)', tanh_res, K=0)
    for key, _, _, label in settings:
        _row(label, all_results[key])

    # JSON dump
    save = {}
    for k, v in all_results.items():
        save[k] = {kk: vv for kk, vv in v.items() if kk != 'ga_history'}
    out = result_path('results_10_ga_pde.json')
    with open(out, 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\n  Saved: {out}")

    plot_results(all_pde, all_results, all_extra, x_val,
                 fig_path('fig10_ga_pde_discovery.png'))


if __name__ == '__main__':
    main()
