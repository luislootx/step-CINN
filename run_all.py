"""
Reproduce every experiment in Section 6 of the paper end to end.

Usage:
    python run_all.py

Runs each numbered script in paper order; writes all figures to
paper/scripts/figures/ and all JSON results to paper/scripts/results/.
"""

import importlib
import os
import sys
import time


SCRIPTS = [
    ('01_periodic_advection_smooth',     'Section 6.1'),
    ('02_saturation_sweep',              'Section 6.2'),
    ('03_linear_advection_riemann',      'Section 6.3'),
    ('04_periodic_square_wave',          'Section 6.4'),
    ('05_linear_acoustics',              'Section 6.5'),
    ('06a_burgers_shock',                'Section 6.6a'),
    ('06b_burgers_smooth',               'Section 6.6b'),
    ('06c_burgers_unified',              'Section 6.6c'),
    ('07a_variable_velocity_x',          'Section 6.7a'),
    ('07b_variable_velocity_t',          'Section 6.7b'),
    ('08_two_d_advection',               'Section 6.8'),
    ('09_regression_discontinuities',    'Section 6.9'),
    ('10_ga_step_discovery',             'Section 6.10'),
    ('11_convergence_sensitivity',       'Section 6.11'),
]


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    total_t0 = time.time()
    for module_name, section in SCRIPTS:
        t0 = time.time()
        print("\n" + "#" * 74)
        print(f"# {section}  --  {module_name}")
        print("#" * 74)
        mod = importlib.import_module(module_name)
        mod.main()
        print(f"\n  [{module_name}] finished in {time.time() - t0:.1f}s")
    print("\n" + "=" * 74)
    print(f"  All experiments completed in {time.time() - total_t0:.1f}s")
    print("=" * 74)


if __name__ == '__main__':
    main()
