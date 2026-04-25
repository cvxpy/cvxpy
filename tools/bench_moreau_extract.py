"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Compare Moreau on identity-pattern SOC problems with and without
ExtractIdentityCones.

The problem builder uses ``cp.SOC(t_i, X_i)`` on bare variables so the
A_block for each SOC is identity — that's the case where extraction
fires and routes the cone onto the Moreau primal variable.

Usage::

    uv run python tools/bench_moreau_extract.py
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np

import cvxpy as cp
from cvxpy.reductions.solvers.conic_solvers import moreau_conif


def build_socp(n: int, m: int, d: int, seed: int = 0) -> cp.Problem:
    """SOCP whose SOCs are over bare variables (extraction-eligible).

    Layout:
        x : R^n           — control variable
        T : R^{m, d}      — one SOC vector per cone
        ||T_i||_2 ≤ ⟨a_i, x⟩  for i = 1..m
        Σ_j x_j = 1
    Objective: minimize sum of t_norms (= upper bounds) + linear.

    Concretely we add slack variable ``up_i`` per cone equal to
    ⟨a_i, x⟩, then write ``cp.SOC(up_i, T_i)`` so the SOC's args are
    bare variables — that's what ExtractIdentityCones recognises.
    """
    rng = np.random.default_rng(seed)
    x = cp.Variable(n)
    T = cp.Variable((m, d - 1))
    up = cp.Variable(m)
    a_rows = rng.standard_normal((m, n)) / np.sqrt(n)

    cons = [up == a_rows @ x + 1.0]
    for i in range(m):
        cons.append(cp.SOC(up[i], T[i, :]))
    cons.append(cp.sum(x) == 1)

    obj = cp.Minimize(cp.sum(up) + 0.01 * cp.sum(x))
    return cp.Problem(obj, cons)


@contextmanager
def extraction_enabled(flag: bool):
    """Force MOREAU.x_cone_kinds() on/off.

    When False, ExtractIdentityCones is skipped — the problem flows
    through the slack-side primal Moreau path (same as old Moreau).
    """
    if flag:
        yield
        return
    with patch.object(
        moreau_conif.MOREAU, 'x_cone_kinds', return_value=frozenset(),
    ):
        yield


def solve_and_time(prob: cp.Problem) -> dict:
    t0 = time.perf_counter()
    try:
        val = prob.solve(solver=cp.MOREAU)
    except Exception as e:
        return {
            'val': None, 'status': f'ERROR: {type(e).__name__}',
            'wall': time.perf_counter() - t0,
            'solve_time': float('nan'), 'setup_time': float('nan'),
            'num_iters': -1, 'error': str(e)[:120],
        }
    wall = time.perf_counter() - t0
    stats = prob.solver_stats
    return {
        'val': val,
        'status': prob.status,
        'wall': wall,
        'solve_time': getattr(stats, 'solve_time', float('nan')),
        'setup_time': getattr(stats, 'setup_time', float('nan')),
        'num_iters': getattr(stats, 'num_iters', -1),
    }


def run(n: int, m: int, d: int, seed: int = 0) -> None:
    print(f"\n=== n={n}  m={m}  d={d}  (slack rows ≈ {m * d + n + 1}) ===")
    results = {}
    for label, use_extract in [('slack', False), ('extract', True)]:
        prob = build_socp(n, m, d, seed)
        with extraction_enabled(use_extract):
            results[label] = solve_and_time(prob)
        r = results[label]
        obj = f"{r['val']:+.6f}" if r['val'] is not None else "  n/a   "
        print(
            f"  {label:8s}  status={str(r['status']):<20s}  "
            f"obj={obj}  "
            f"wall={r['wall']:.3f}s  "
            f"solve={r['solve_time']:.3f}s  "
            f"setup={r['setup_time']:.3f}s  "
            f"iters={r['num_iters']}"
        )
    rs, re = results['slack'], results['extract']
    if 'error' in rs:
        print(f"  slack error: {rs['error']}")
    if 'error' in re:
        print(f"  extract error: {re['error']}")
    if rs['val'] is not None and re['val'] is not None:
        gap = abs(rs['val'] - re['val']) / max(1.0, abs(rs['val']))
        print(f"  obj agreement (relative): {gap:.2e}")
        if rs['solve_time'] > 0 and re['solve_time'] > 0:
            speedup = rs['solve_time'] / re['solve_time']
            print(f"  extract solve_time speedup: {speedup:.2f}x")


def main() -> None:
    configs = [
        # n, m, d
        (50, 20, 10),
        (100, 50, 20),
        (200, 100, 30),
        (400, 200, 50),
        (800, 300, 80),
    ]
    print("Moreau extraction benchmark — bare-variable SOCPs")
    print("Each row: slack-side cones vs ExtractIdentityCones (x_cones).")
    for n, m, d in configs:
        run(n, m, d)


if __name__ == '__main__':
    main()
