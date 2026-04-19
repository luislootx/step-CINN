"""
Section 6.9 -- Regression with Discontinuities
==============================================
A stand-alone piecewise regression problem that motivates the Step-CIELM
architecture: a target function with a smooth spline background and
a small number of sharp jumps at unknown locations.

Target:
    h(x) = s(x) + g(x),   x in [-5, 5]
    s(x) = piecewise constant with jumps at x in {-3, -1, 1, 3}
    g(x) = natural cubic spline through random knots (fixed seed)

Architecture: N_tanh random-fixed tanh neurons + K sigmoid step neurons.
Output weights solved analytically by ridge regression.

Step positions are discovered by a genetic algorithm that optimises the
validation RMSE plus a mild parsimony penalty on K, and a final
Nelder-Mead pass refines the best candidate.

Artifacts produced
------------------
figures/fig09_regression.png             prediction, step contributions,
                                         residual, and GA convergence
results/results_09_regression.json       discovered positions and errors
"""

import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from _core import (
    solve_ridge, style_ax, add_legend, fig_path, result_path,
    C_BLUE, C_RED, C_GREEN, C_ORANGE, C_PURPLE, C_TEXT, BG,
)


# ----------------------------------------------------------------------------
# Target function
# ----------------------------------------------------------------------------

SEGMENTS = [
    (-5, -3, -2),
    (-3, -1,  1),
    (-1,  1, -1),
    ( 1,  3,  3),
    ( 3,  5,  0),
]
X_MIN, X_MAX = -5.0, 5.0
SEED_DATA = 42
DENSITY = 20
TRUE_POSITIONS = [-3.0, -1.0, 1.0, 3.0]


def step_part(x_arr):
    out = np.empty_like(x_arr, dtype=float)
    for i, xi in enumerate(x_arr):
        for j, (x0, x1, y_val) in enumerate(SEGMENTS):
            last = (j == len(SEGMENTS) - 1)
            if xi >= x0 and (xi <= x1 if last else xi < x1):
                out[i] = y_val
                break
        else:
            out[i] = np.nan
    return out


def make_spline(seed=SEED_DATA, n_knots=8, amplitude=1.5):
    rng = np.random.default_rng(seed)
    kx = np.linspace(X_MIN, X_MAX, n_knots)
    ky = rng.uniform(-amplitude, amplitude, size=n_knots)
    return CubicSpline(kx, ky, bc_type='natural')


def target_fn(x_arr, cs):
    return step_part(x_arr) + cs(x_arr)


# ----------------------------------------------------------------------------
# ELM building blocks
# ----------------------------------------------------------------------------

def tanh_hidden(x, N_tanh, seed=7, scale=2.5):
    rng = np.random.default_rng(seed)
    W = rng.uniform(-scale, scale, size=(N_tanh, 1))
    b = rng.uniform(-scale * (X_MAX - X_MIN) / 2,
                    scale * (X_MAX - X_MIN) / 2, size=N_tanh)
    z = x.reshape(-1, 1) @ W.T + b
    return np.tanh(z)


def step_hidden(x, positions, kappa=500.0):
    if len(positions) == 0:
        return np.empty((len(x), 0))
    positions = np.asarray(positions)
    z = kappa * (x.reshape(-1, 1) - positions.reshape(1, -1))
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


_H_TANH_CACHE = {}


def get_tanh_hidden(x, N_tanh, seed):
    key = (id(x), N_tanh, seed)
    if key not in _H_TANH_CACHE:
        _H_TANH_CACHE[key] = tanh_hidden(x, N_tanh, seed)
    return _H_TANH_CACHE[key]


def evaluate_individual(positions, x_train, y_train, x_val, y_val,
                        N_tanh, kappa, lam, seed_tanh, parsimony):
    H_tanh = get_tanh_hidden(x_train, N_tanh, seed_tanh)
    H_step = step_hidden(x_train, positions, kappa)
    H = np.hstack([H_tanh, H_step])
    beta = solve_ridge(H, y_train, lam)

    H_tanh_v = get_tanh_hidden(x_val, N_tanh, seed_tanh)
    H_step_v = step_hidden(x_val, positions, kappa)
    H_v = np.hstack([H_tanh_v, H_step_v])
    y_pred = H_v @ beta

    val_rmse = rmse(y_val, y_pred)
    fitness = val_rmse + parsimony * len(positions)
    return fitness, val_rmse, beta


# ----------------------------------------------------------------------------
# Genetic algorithm
# ----------------------------------------------------------------------------

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


def _merge_close(positions, min_dist=0.2):
    if len(positions) <= 1:
        return positions
    merged = [positions[0]]
    for p in positions[1:]:
        if abs(p - merged[-1]) < min_dist:
            merged[-1] = (merged[-1] + p) / 2
        else:
            merged.append(p)
    return merged


def run_ga(x_train, y_train, x_val, y_val, config):
    rng = np.random.default_rng(config['seed_ga'])

    pop = []
    for _ in range(config['pop_size']):
        K = rng.integers(1, config['max_steps'] + 1)
        positions = sorted(rng.uniform(X_MIN + 0.5, X_MAX - 0.5,
                                       size=K).tolist())
        pop.append(Individual(positions))

    best_ever = None
    history = []

    for gen in range(config['n_gen']):
        for ind in pop:
            ind.fitness, ind.val_rmse, ind.beta = evaluate_individual(
                ind.positions, x_train, y_train, x_val, y_val,
                config['N_tanh'], config['kappa'], config['lam'],
                config['seed_tanh'], config['parsimony'])

        pop.sort(key=lambda ind: ind.fitness)
        if best_ever is None or pop[0].fitness < best_ever.fitness:
            best_ever = pop[0].copy()

        history.append({
            'gen': gen,
            'best_fitness': float(pop[0].fitness),
            'best_val_rmse': float(pop[0].val_rmse),
            'best_n_steps': len(pop[0].positions),
            'best_positions': list(pop[0].positions),
            'mean_fitness': float(np.mean([ind.fitness for ind in pop])),
        })

        if gen % 5 == 0 or gen == config['n_gen'] - 1:
            pos_str = ', '.join(f'{p:.2f}' for p in pop[0].positions)
            print(f"  Gen {gen:3d}  |  best RMSE = {pop[0].val_rmse:.6f}  "
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
                n = rng.integers(1, min(len(all_pos), 8) + 1)
                chosen = sorted(rng.choice(all_pos, size=n,
                                           replace=False).tolist())
            else:
                chosen = [rng.uniform(X_MIN + 0.5, X_MAX - 0.5)]
            child = Individual(chosen)

            positions = list(child.positions)
            for i in range(len(positions)):
                if rng.random() < config['mutation_rate']:
                    positions[i] += rng.normal(0, 0.3)
                    positions[i] = np.clip(positions[i], X_MIN + 0.1,
                                           X_MAX - 0.1)
            if rng.random() < 0.10 and len(positions) < config['max_steps']:
                positions.append(rng.uniform(X_MIN + 0.5, X_MAX - 0.5))
            if rng.random() < 0.10 and len(positions) > 1:
                positions.pop(rng.integers(0, len(positions)))
            positions = _merge_close(sorted(positions))
            new_pop.append(Individual(positions))

        pop = new_pop

    # Nelder-Mead refinement on the best individual
    if len(best_ever.positions) > 0:
        def nm_obj(pos_vec):
            positions = sorted(pos_vec.tolist())
            f, _, _ = evaluate_individual(
                positions, x_train, y_train, x_val, y_val,
                config['N_tanh'], config['kappa'], config['lam'],
                config['seed_tanh'], 0.0)
            return f
        res = minimize(nm_obj, np.array(best_ever.positions),
                       method='Nelder-Mead',
                       options={'maxfev': config.get('nm_maxfev', 500),
                                'xatol': 1e-4, 'fatol': 1e-6})
        refined = sorted(res.x.tolist())
        f_ref, rmse_ref, beta_ref = evaluate_individual(
            refined, x_train, y_train, x_val, y_val,
            config['N_tanh'], config['kappa'], config['lam'],
            config['seed_tanh'], config['parsimony'])
        if f_ref < best_ever.fitness:
            best_ever.positions = refined
            best_ever.fitness = f_ref
            best_ever.val_rmse = rmse_ref
            best_ever.beta = beta_ref

    return best_ever, history


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------

def plot_result(x_train, y_train, x_val, y_val, y_pred,
                best, history, config, fname):
    N_tanh = config['N_tanh']
    fig, axes = plt.subplots(4, 1, figsize=(12, 14),
                             gridspec_kw={'hspace': 0.35,
                                          'height_ratios': [3, 2, 1.5, 1.5]})
    fig.patch.set_facecolor(BG)
    for ax in axes:
        style_ax(ax)

    # Panel 1: prediction versus target
    ax = axes[0]
    ax.plot(x_val, y_val, color=C_BLUE, linewidth=2.5, zorder=3,
            label='Target h(x)')
    ax.plot(x_val, y_pred, color=C_GREEN, linewidth=2, linestyle='--',
            zorder=4,
            label=f'Step-CIELM ({len(best.positions)} steps, '
                  f'RMSE = {best.val_rmse:.4f})')
    ax.scatter(x_train, y_train, color=C_BLUE, s=12, zorder=5,
               alpha=0.4, edgecolors='none')
    for xp in best.positions:
        ax.axvline(xp, color=C_ORANGE, linewidth=1.3, linestyle='--',
                   alpha=0.7, zorder=2)
    for xp in TRUE_POSITIONS:
        ax.axvline(xp, color=C_PURPLE, linewidth=1, linestyle=':',
                   alpha=0.35, zorder=2)
    ax.set_ylabel('h(x)', color=C_TEXT, fontsize=11)
    ax.set_title(f'Step-CIELM regression: {len(best.positions)} '
                 f'interpretable steps found (true count = 4)',
                 color=C_TEXT, fontsize=13, fontfamily='serif', pad=10)
    add_legend(ax, loc='upper left')

    # Panel 2: individual step contributions
    ax = axes[1]
    colors_step = [C_RED, C_ORANGE, C_GREEN, C_BLUE, C_PURPLE]
    total = np.zeros_like(x_val)
    for i, xp in enumerate(best.positions):
        w = best.beta[N_tanh + i]
        h_i = step_hidden(x_val, [xp], config['kappa']).ravel() * w
        total += h_i
        c = colors_step[i % len(colors_step)]
        ax.plot(x_val, h_i, color=c, linewidth=1.4, alpha=0.8,
                label=f'Step {i+1}: x = {xp:.2f}, w = {w:.2f}')
    ax.plot(x_val, total, color=C_TEXT, linewidth=2, zorder=3,
            label='Total step contribution')
    ax.set_ylabel('step contribution', color=C_TEXT, fontsize=11)
    ax.set_title('Individual step-neuron contributions',
                 color=C_TEXT, fontsize=12, fontfamily='serif', pad=8)
    add_legend(ax, loc='upper left')

    # Panel 3: residual
    ax = axes[2]
    residual = y_val - y_pred
    ax.fill_between(x_val, residual, 0, color=C_RED, alpha=0.15, zorder=2)
    ax.plot(x_val, residual, color=C_RED, linewidth=1.4, zorder=3,
            label=f'Residual (max = {np.max(np.abs(residual)):.3f})')
    ax.set_ylabel('residual', color=C_TEXT, fontsize=11)
    ax.set_xlabel('x', color=C_TEXT, fontsize=11)
    add_legend(ax, loc='upper left')

    # Panel 4: GA convergence
    ax = axes[3]
    gens = [h['gen'] for h in history]
    best_rmses = [h['best_val_rmse'] for h in history]
    mean_fits = [h['mean_fitness'] for h in history]
    ax.plot(gens, best_rmses, color=C_GREEN, linewidth=2,
            label='Best validation RMSE')
    ax.plot(gens, mean_fits, color=C_RED, linewidth=1.3, alpha=0.7,
            label='Mean fitness')
    ax.set_xlabel('Generation', color=C_TEXT, fontsize=11)
    ax.set_ylabel('RMSE / fitness', color=C_TEXT, fontsize=11)
    ax.set_title('GA convergence',
                 color=C_TEXT, fontsize=12, fontfamily='serif', pad=8)
    add_legend(ax, loc='upper right')

    plt.savefig(fname, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"  Saved: {fname}")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  Section 6.9: Regression with discontinuities")
    print("=" * 70)

    t_start = time.time()
    cs = make_spline()
    n_train = max(2, int(round(DENSITY * (X_MAX - X_MIN))))
    x_train = np.linspace(X_MIN, X_MAX, n_train)
    y_train = target_fn(x_train, cs)

    x_val = np.linspace(X_MIN, X_MAX, 2000)
    y_val = target_fn(x_val, cs)

    print(f"  Training: {n_train} pts  |  validation: {len(x_val)} pts")

    config = {
        'N_tanh': 80,
        'kappa': 500.0,
        'lam': 1e-6,
        'seed_tanh': 7,
        'seed_ga': 42,
        'pop_size': 50,
        'n_gen': 80,
        'elite_count': 5,
        'mutation_rate': 0.3,
        'max_steps': 8,
        'parsimony': 0.002,
        'nm_maxfev': 500,
    }

    best, history = run_ga(x_train, y_train, x_val, y_val, config)
    elapsed = time.time() - t_start

    print("\n" + "=" * 70)
    print("  Results")
    print("=" * 70)
    print(f"  Steps found:   {len(best.positions)}")
    print(f"  Positions:     {[f'{p:.4f}' for p in best.positions]}")
    print(f"  True positions: {TRUE_POSITIONS}")
    print(f"  Validation RMSE: {best.val_rmse:.6f}")
    print(f"  Total time:    {elapsed:.1f}s")

    # Dense prediction for plotting
    H_tanh_d = get_tanh_hidden(x_val, config['N_tanh'], config['seed_tanh'])
    H_step_d = step_hidden(x_val, best.positions, config['kappa'])
    y_pred = np.hstack([H_tanh_d, H_step_d]) @ best.beta

    # Training error
    H_tanh_t = get_tanh_hidden(x_train, config['N_tanh'], config['seed_tanh'])
    H_step_t = step_hidden(x_train, best.positions, config['kappa'])
    y_pred_train = np.hstack([H_tanh_t, H_step_t]) @ best.beta
    train_rmse = rmse(y_train, y_pred_train)
    print(f"  Train RMSE:    {train_rmse:.6f}")

    plot_result(x_train, y_train, x_val, y_val, y_pred,
                best, history, config,
                fig_path('fig09_regression.png'))

    step_weights = best.beta[config['N_tanh']:].tolist()
    out = result_path('results_09_regression.json')
    with open(out, 'w') as f:
        json.dump({
            'config': config,
            'best_positions':  list(best.positions),
            'best_val_rmse':   float(best.val_rmse),
            'best_train_rmse': float(train_rmse),
            'best_fitness':    float(best.fitness),
            'n_steps_found':   len(best.positions),
            'step_weights':    step_weights,
            'true_positions':  TRUE_POSITIONS,
            'elapsed_s':       elapsed,
        }, f, indent=2)
    print(f"  Saved: {out}")


if __name__ == '__main__':
    main()
