#!/usr/bin/env python3
"""
Run a catalog of SOCP examples through CVXPY (CUOPT or another conic solver).

Not part of the pytest suite. Reference objectives/primals are from CLARABEL.

Usage (from repo root, with cvxpy installed editable)::

    python scripts/run_cuopt_socp_examples.py --list
    python scripts/run_cuopt_socp_examples.py --solver CUOPT
    python scripts/run_cuopt_socp_examples.py --solver CLARABEL --names socp_0 socp_6
    python scripts/run_cuopt_socp_examples.py --solver CUOPT --check-matrix
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

import cvxpy as cp
import cvxpy.settings as s

# ---------------------------------------------------------------------------
# Problem builders: each returns (Problem, expected_objective, {var: expected})
# Reference values from CLARABEL unless noted.
# ---------------------------------------------------------------------------


def build_socp_0() -> tuple[cp.Problem, float, dict]:
    x = cp.Variable(2)
    prob = cp.Problem(cp.Minimize(cp.norm(x, 2) + 1), [x == 0])
    return prob, 1.0, {x: np.array([0.0, 0.0])}


def build_socp_1() -> tuple[cp.Problem, float, dict]:
    x = cp.Variable(3)
    y = cp.Variable()
    prob = cp.Problem(
        cp.Minimize(3 * x[0] + 2 * x[1] + x[2]),
        [cp.SOC(y, x), x[0] + x[1] + 3 * x[2] >= 1.0, y <= 5],
    )
    return prob, -13.548638904065102, {
        x: np.array([-3.87462186, -2.12978823, 2.33480343]),
        y: 5.0,
    }


def build_socp_2() -> tuple[cp.Problem, float, dict]:
    x = cp.Variable(2)
    expr = cp.reshape(x[0] + 2 * x[1], (1, 1), order="F")
    prob = cp.Problem(
        cp.Minimize(-4 * x[0] - 5 * x[1]),
        [2 * x[0] + x[1] <= 3, cp.SOC(cp.Constant([3]), expr), x >= 0],
    )
    return prob, -9.0, {x: np.array([1.0, 1.0])}


def build_socp_3(axis: int) -> tuple[cp.Problem, float, dict]:
    x = cp.Variable(2)
    c = np.array([-1.0, 2.0])
    root2 = np.sqrt(2)
    u = np.array([[1 / root2, -1 / root2], [1 / root2, 1 / root2]])
    mat1 = np.diag([root2, 1 / root2]) @ u.T
    mat2 = np.diag([1.0, 1.0])
    mat3 = np.diag([0.2, 1.8])
    X = cp.vstack([mat1 @ x, mat2 @ x, mat3 @ x])
    t = cp.Constant(np.ones(3))
    if axis == 0:
        con = cp.SOC(t, X.T, axis=0)
    else:
        con = cp.SOC(t, X, axis=1)
    prob = cp.Problem(cp.Minimize(c @ x), [con])
    return prob, -1.932105, {x: np.array([0.83666003, -0.54772256])}


def build_socp_4_two_cones() -> tuple[cp.Problem, float, dict]:
    x1, x2 = cp.Variable(2), cp.Variable(2)
    t1, t2 = cp.Variable(), cp.Variable()
    prob = cp.Problem(
        cp.Minimize(t1 + t2),
        [cp.SOC(t1, x1), x1 == 0, cp.SOC(t2, x2), x2 == np.array([3.0, 4.0])],
    )
    return prob, 5.0, {x1: np.zeros(2), x2: np.array([3.0, 4.0]), t1: 0.0, t2: 5.0}


def build_socp_5_scalar_tail() -> tuple[cp.Problem, float, dict]:
    x, t = cp.Variable(), cp.Variable()
    prob = cp.Problem(cp.Minimize(t), [cp.SOC(t, x), x >= 1])
    return prob, 1.0, {x: 1.0, t: 1.0}


def build_socp_6_robust_doc() -> tuple[cp.Problem, float, dict]:
    m, n, p, n_i = 3, 10, 5, 5
    np.random.seed(2)
    f = np.random.randn(n)
    mats_a, vecs_b, vecs_c, vecs_d = [], [], [], []
    x0 = np.random.randn(n)
    for i in range(m):
        mats_a.append(np.random.randn(n_i, n))
        vecs_b.append(np.random.randn(n_i))
        vecs_c.append(np.random.randn(n))
        vecs_d.append(
            np.linalg.norm(mats_a[-1] @ x0 + vecs_b[-1], 2) - float(vecs_c[-1] @ x0)
        )
    F = np.random.randn(p, n)
    g = F @ x0
    x = cp.Variable(n)
    soc = [
        cp.SOC(vecs_c[i].T @ x + vecs_d[i], mats_a[i] @ x + vecs_b[i])
        for i in range(m)
    ]
    prob = cp.Problem(cp.Minimize(f @ x), soc + [F @ x == g])
    expect_x = np.array([
        0.77864295, 2.25261785, 0.51309622, -0.47992225, 1.14644749,
        -1.38867005, 0.16794051, 1.62024586, 0.18099077, 0.40936688,
    ])
    return prob, -5.71851610530738, {x: expect_x}


def build_socp_7_quad_soc() -> tuple[cp.Problem, float, dict]:
    np.random.seed(1)
    n, m, p = 4, 3, 2
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    G = np.random.randn(p, n)
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), [cp.norm(G @ x, 2) <= 1])
    return prob, 0.0, {
        x: np.array([-0.10117552, 0.54229312, 0.39084271, -0.35426027]),
    }


def build_socp_8_portfolio() -> tuple[cp.Problem, float, dict]:
    np.random.seed(2)
    n = 4
    mu = np.abs(np.random.randn(n))
    raw = np.random.randn(n, n)
    sigma = raw.T @ raw + 0.1 * np.eye(n)
    G = np.linalg.cholesky(sigma)
    w = cp.Variable(n)
    risk = cp.Variable()
    prob = cp.Problem(
        cp.Minimize(risk),
        [cp.sum(w) == 1, mu @ w >= 0.1, cp.SOC(risk, G @ w)],
    )
    expect_w = np.array([0.01198571, 0.11103543, 0.95538377, -0.07840492])
    return prob, 0.32928260070781257, {w: expect_w, risk: 0.32928260070781257}


def build_socp_9_epigraph_norm() -> tuple[cp.Problem, float, dict]:
    np.random.seed(3)
    G = np.random.randn(3, 2)
    h = np.random.randn(3)
    x = cp.Variable(2)
    t = cp.Variable()
    prob = cp.Problem(cp.Minimize(t), [cp.SOC(t, G @ x - h)])
    return prob, 0.03713864693064098, {
        x: np.array([-0.12932582, 0.32680947]),
        t: 0.03713864693064098,
    }


def build_socp_10_multi_small_cones() -> tuple[cp.Problem, float, dict]:
    x = cp.Variable(3)
    prob = cp.Problem(
        cp.Minimize(cp.sum(x)),
        [cp.SOC(1, x[0:1]), cp.SOC(2, x[1:2]), cp.norm(x[2:3], 2) <= 3],
    )
    return prob, -6.0, {x: np.array([-1.0, -2.0, -3.0])}


def build_socp_11_eq_plus_soc() -> tuple[cp.Problem, float, dict]:
    x = cp.Variable(5)
    prob = cp.Problem(cp.Minimize(cp.sum(x)), [x[0] == 1, cp.norm(x, 2) <= 2])
    expect_x = np.array([1.0, -0.8660254, -0.8660254, -0.8660254, -0.8660254])
    return prob, -2.46410161804137, {x: expect_x}


def build_lorentz_min_x0() -> tuple[cp.Problem, float, dict]:
    x0 = cp.Variable(nonneg=True)
    x1 = cp.Variable(nonneg=True)
    x2 = cp.Variable()
    prob = cp.Problem(
        cp.Minimize(x0),
        [x1 == 1, cp.norm(cp.hstack([x1, x2])) <= x0],
    )
    return prob, 1.0, {x0: 1.0, x1: 1.0, x2: 0.0}


@dataclass
class SocpExample:
    name: str
    description: str
    build: Callable[[], tuple[cp.Problem, float, dict]]
    use_quad_obj: bool = False
    tags: list[str] = field(default_factory=list)


CATALOG: list[SocpExample] = [
    SocpExample("socp_0", "min ||x||_2 + 1, x = 0", build_socp_0),
    SocpExample(
        "socp_1",
        "linear obj, ||x|| <= y, two linear inequalities",
        build_socp_1,
        tags=["stress"],
    ),
    SocpExample("socp_2", "LP embedded in SOC form", build_socp_2),
    SocpExample("socp_3ax0", "three SOCs, axis=0", lambda: build_socp_3(0)),
    SocpExample("socp_3ax1", "three SOCs, axis=1", lambda: build_socp_3(1), tags=["stress"]),
    SocpExample("socp_4", "two independent SOCs", build_socp_4_two_cones, tags=["multi-cone"]),
    SocpExample("socp_5", "smallest Lorentz cone (t, scalar x)", build_socp_5_scalar_tail),
    SocpExample(
        "socp_6",
        "robust LP / 3 SOCs + equalities (n=10, seed=2)",
        build_socp_6_robust_doc,
        tags=["multi-cone", "large"],
    ),
    SocpExample(
        "socp_7",
        "sum_squares objective + norm SOC",
        build_socp_7_quad_soc,
        use_quad_obj=True,
        tags=["quadratic-objective"],
    ),
    SocpExample("socp_8", "min risk portfolio (Cholesky SOC)", build_socp_8_portfolio),
    SocpExample("socp_9", "min t, ||Gx-h|| <= t", build_socp_9_epigraph_norm),
    SocpExample(
        "socp_10",
        "three small SOCs on x[0], x[1], x[2]",
        build_socp_10_multi_small_cones,
        tags=["multi-cone"],
    ),
    SocpExample("socp_11", "equality + one SOC on full x", build_socp_11_eq_plus_soc),
    SocpExample(
        "lorentz_min_x0",
        "cuOpt Lorentz smoke: min x0, x1=1, ||(x1,x2)|| <= x0",
        build_lorentz_min_x0,
    ),
]


def check_matrix_stuffing(prob: cp.Problem) -> str:
    from cvxpy.reductions.solvers.conic_solvers.cuopt_conif import CUOPT

    data, _, _ = prob.get_problem_data(solver=cp.CUOPT)
    Acsr = data[s.A].tocsr()
    leq_end = data[s.DIMS].zero + data[s.DIMS].nonneg
    soc_dims = list(data[s.DIMS].soc)
    n_orig = data[s.C].shape[0]
    n_soc = sum(soc_dims)
    A_work, lb, ub, qc_cones = CUOPT._build_working_problem(
        Acsr, data[s.B], data[s.DIMS].zero, leq_end, soc_dims, n_orig
    )
    return (
        f"A_work {A_work.shape}, bounds {len(lb)}, "
        f"qc_cones {len(qc_cones)} (soc_dims={soc_dims})"
    )


def run_example(
    ex: SocpExample,
    solver: str,
    solver_kwargs: dict,
    places: int,
    check_matrix: bool,
) -> bool:
    prob, expect_obj, expect_vars = ex.build()
    kwargs = dict(solver_kwargs)
    if ex.use_quad_obj:
        kwargs["use_quad_obj"] = True

    t0 = time.perf_counter()
    try:
        prob.solve(solver=solver, **kwargs)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"FAIL {ex.name}: {exc} ({elapsed:.2f}s)")
        return False
    elapsed = time.perf_counter() - t0

    ok = prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    obj_err = abs(prob.value - expect_obj) if prob.value is not None else np.inf
    if ok and obj_err > 10 ** (-places):
        ok = False

    prim_err = 0.0
    for var, expected in expect_vars.items():
        if var.value is None:
            ok = False
            continue
        prim_err = max(prim_err, float(np.max(np.abs(var.value - np.asarray(expected)))))

    if ok and prim_err > 10 ** (-places):
        ok = False

    status = "PASS" if ok else "FAIL"
    print(
        f"{status} {ex.name}: status={prob.status} "
        f"obj={prob.value} (expect {expect_obj:.6g}) "
        f"max_prim_err={prim_err:.2e} time={elapsed:.2f}s"
    )
    if check_matrix and prob.status not in {cp.INFEASIBLE, cp.UNBOUNDED}:
        try:
            print(f"       matrix: {check_matrix_stuffing(prob)}")
        except Exception as exc:
            print(f"       matrix check failed: {exc}")

    if not ok and prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        for var, expected in expect_vars.items():
            print(f"       {var}: {var.value} (expect {expected})")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", default="CUOPT")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--names", nargs="*", help="Subset to run (default: all)")
    parser.add_argument("--places", type=int, default=3, help="Digits for ref comparison")
    parser.add_argument("--solver-method", default="Barrier")
    parser.add_argument("--presolve", type=int, default=0)
    parser.add_argument(
        "--check-matrix",
        action="store_true",
        help="Print CUOPT A_work / QCMATRIX lift shapes after each solve",
    )
    parser.add_argument("--tag", help="Only run examples with this tag")
    args = parser.parse_args()

    if args.list:
        for ex in CATALOG:
            tags = f" [{', '.join(ex.tags)}]" if ex.tags else ""
            quad = " [quad_obj]" if ex.use_quad_obj else ""
            print(f"{ex.name:16} {ex.description}{tags}{quad}")
        return 0

    names = args.names or [ex.name for ex in CATALOG]
    by_name = {ex.name: ex for ex in CATALOG}

    solver_kwargs: dict = {}
    if args.solver.upper() == "CUOPT":
        solver_kwargs = {
            "solver_method": args.solver_method,
            "presolve": args.presolve,
        }

    selected = []
    for name in names:
        if name not in by_name:
            print(f"Unknown example {name!r}; use --list", file=sys.stderr)
            return 1
        ex = by_name[name]
        if args.tag and args.tag not in ex.tags:
            continue
        selected.append(ex)

    if args.tag and not selected:
        print(f"No examples with tag {args.tag!r}", file=sys.stderr)
        return 1

    print(f"Solver={args.solver} kwargs={solver_kwargs} n={len(selected)}")
    passed = 0
    for ex in selected:
        if run_example(ex, args.solver, solver_kwargs, args.places, args.check_matrix):
            passed += 1

    print(f"\n{passed}/{len(selected)} passed")
    return 0 if passed == len(selected) else 1


if __name__ == "__main__":
    sys.exit(main())
