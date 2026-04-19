# Characteristics-Informed Extreme Learning Machines

Reference implementation for the paper

> **Characteristics-Informed Extreme Learning Machines for Smooth and
> Discontinuous Hyperbolic PDEs** -- Luis Loo and Ulisses Braga-Neto, 2026.

CIELM is a mesh-free solver for hyperbolic conservation laws that
combines the method of characteristics with a random-fixed tanh basis
trained analytically by ridge regression. Step-CIELM adds a small number
of sigmoid step neurons so that piecewise-smooth solutions with shocks,
Riemann jumps, or discontinuous initial data are represented exactly.
No gradient-descent loop, no PDE residual loss, no collocation points.

## Citation

```bibtex
@article{loo2026cielm,
  title  = {Characteristics-Informed Extreme Learning Machines
            for Smooth and Discontinuous Hyperbolic PDEs},
  author = {Loo, Luis and Braga-Neto, Ulisses},
  year   = {2026}
}
```

## Quick start

```bash
pip install -r requirements.txt
python 01_periodic_advection_smooth.py
```

Each numbered script is self-contained and writes its figures to
`figures/` and its numerical results (JSON) to `results/`. To reproduce
every experiment in order, run `python run_all.py`.

## Scripts and paper sections

| Script | Paper section | Experiment |
|---|---|---|
| `01_periodic_advection_smooth.py`  | 6.1  | Periodic advection, smooth IC (velocity robustness) |
| `02_saturation_sweep.py`           | 6.2  | Saturation of the smooth basis on discontinuous problems |
| `03_linear_advection_riemann.py`   | 6.3  | Linear advection with Riemann IC |
| `04_periodic_square_wave.py`       | 6.4  | Periodic advection with a square-wave IC (step ablation) |
| `05_linear_acoustics.py`           | 6.5  | Linear acoustics system (2 x 2 hyperbolic) |
| `06a_burgers_shock.py`             | 6.6  | Inviscid Burgers: Riemann shock and rarefaction |
| `06b_burgers_smooth.py`            | 6.6  | Inviscid Burgers: smooth pre-shock and rarefaction |
| `06c_burgers_unified.py`           | 6.6  | Unified pre- and post-shock Burgers via Newton continuation |
| `07a_variable_velocity_x.py`       | 6.7  | Space-varying transport velocity v(x) = x |
| `07b_variable_velocity_t.py`       | 6.7  | Time-varying transport velocity v(t) = cos(omega t) |
| `08_two_d_advection.py`            | 6.8  | Two-dimensional linear advection |
| `09_regression_discontinuities.py` | 6.9  | Regression with unknown discontinuities (GA) |
| `10_ga_step_discovery.py`          | 6.10 | Step discovery from data for a PDE with unknown IC |
| `11_convergence_sensitivity.py`    | 6.11 | Convergence with N, sensitivity to kappa, numerical stability |

`_core.py` contains the shared ELM primitives (random tanh weights,
ridge solve, Picard and Anderson fixed-point iteration, 2D ELM basis,
plot styling, I/O helpers). `run_all.py` runs every experiment
end-to-end.

## Requirements

NumPy, SciPy, and Matplotlib (see `requirements.txt`). No GPU or deep
learning framework is required; every experiment runs on a laptop CPU.

## License

MIT.

## Affiliation

Department of Electrical and Computer Engineering, Texas A&M University.
