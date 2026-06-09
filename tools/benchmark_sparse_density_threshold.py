"""
Benchmark dense versus sparse NLP diff-engine matmul dispatch.

By default, this script compares forced dense and forced sparse dispatch for
each matrix size and density. That is the direct data needed to choose
cvxpy.settings.SPARSE_DENSITY_THRESHOLD: sparse should only be used for the
density/size region where it beats dense end to end.

The script also has a threshold-sweep mode for checking the behavior of
particular candidate threshold values.

Example
-------
    uv run python tools/benchmark_sparse_density_threshold.py --solve-with IPOPT

    uv run python tools/benchmark_sparse_density_threshold.py \
        --sizes 100 300 600 --densities 0.01 0.025 0.05 0.10 \
        --repeats 5 --solve-with IPOPT

    uv run python tools/benchmark_sparse_density_threshold.py --mode threshold --n 400
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

import cvxpy as cp
import cvxpy.settings as settings
from cvxpy.reductions.dnlp2smooth.dnlp2smooth import Dnlp2Smooth
from cvxpy.reductions.solvers.nlp_solvers.diff_engine import C_problem

DEFAULT_DENSITIES = (0.005, 0.01, 0.025, 0.05, 0.075, 0.10, 0.20)
DEFAULT_SIZES = (40, 120, 240)
DEFAULT_THRESHOLDS = (0.0, 0.01, 0.025, 0.05, 0.075, 0.10, 0.20, 1.0)
FORCE_DENSE_THRESHOLD = 0.0
FORCE_SPARSE_THRESHOLD = 1.0


@dataclass(frozen=True)
class TrialResult:
    n: int
    density: float
    threshold: float
    route: str
    construct_seconds: float
    sparsity_seconds: float
    hessian_nnz: int
    solve_seconds: float | None = None
    status: str | None = None


@dataclass(frozen=True)
class ComparisonResult:
    n: int
    density: float
    dense_construct_seconds: float
    sparse_construct_seconds: float
    dense_sparsity_seconds: float
    sparse_sparsity_seconds: float
    dense_hessian_nnz: int
    sparse_hessian_nnz: int
    dense_solve_seconds: float | None
    sparse_solve_seconds: float | None
    dense_status: str | None
    sparse_status: str | None


def _exact_density_matrix(n: int, density: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    size = n * n
    nnz = min(size, max(1, round(size * density)))
    indices = rng.choice(size, size=nnz, replace=False)
    data = rng.standard_normal(nnz)
    matrix = np.zeros(size)
    matrix[indices] = data
    return matrix.reshape((n, n))


def _build_problem(A: np.ndarray) -> cp.Problem:
    n = A.shape[1]
    x = cp.Variable(n)
    x.value = np.zeros(n)
    b = np.ones(A.shape[0])
    return cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))


def _time_once(A: np.ndarray, solver: str | None) -> \
        tuple[float, float, int, float | None, str | None]:
    problem = _build_problem(A)
    canonicalized_problem, _ = Dnlp2Smooth().apply(problem)

    start = time.perf_counter()
    c_problem = C_problem(canonicalized_problem, verbose=False)
    construct_seconds = time.perf_counter() - start

    start = time.perf_counter()
    c_problem.init_jacobian_coo()
    c_problem.init_hessian_coo_lower_tri()
    sparsity_seconds = time.perf_counter() - start

    hess_rows, _ = c_problem.get_problem_hessian_sparsity_coo()
    hessian_nnz = len(hess_rows)

    solve_seconds = None
    status = None
    if solver is not None:
        solve_problem = _build_problem(A)
        solver_opts = {"print_level": 0, "sb": "yes"} if solver == "IPOPT" else {}
        start = time.perf_counter()
        solve_problem.solve(solver=solver, nlp=True, verbose=False, **solver_opts)
        solve_seconds = time.perf_counter() - start
        status = solve_problem.status

    return construct_seconds, sparsity_seconds, hessian_nnz, solve_seconds, status


def _median_trial(A: np.ndarray, threshold: float,
                  repeats: int, solver: str | None) -> TrialResult:
    n = A.shape[0]
    density = float(np.count_nonzero(A) / A.size)
    original_threshold = settings.SPARSE_DENSITY_THRESHOLD
    settings.SPARSE_DENSITY_THRESHOLD = threshold
    try:
        timings = [_time_once(A, solver) for _ in range(repeats)]
    finally:
        settings.SPARSE_DENSITY_THRESHOLD = original_threshold

    construct = statistics.median(timing[0] for timing in timings)
    sparsity = statistics.median(timing[1] for timing in timings)
    hessian_nnz = round(statistics.median(timing[2] for timing in timings))
    solve_seconds = None
    status = None
    if solver is not None:
        solve_seconds = statistics.median(
                timing[3] for timing in timings if timing[3] is not None)
        statuses = {timing[4] for timing in timings}
        status = statuses.pop() if len(statuses) == 1 else "mixed"
    route = "sparse" if density < threshold else "dense"
    return TrialResult(
        n, density, threshold, route, construct, sparsity, hessian_nnz, solve_seconds, status
    )


def _comparison_trial(A: np.ndarray, repeats: int, solver: str | None) -> ComparisonResult:
    dense = _median_trial(A, FORCE_DENSE_THRESHOLD, repeats, solver)
    sparse = _median_trial(A, FORCE_SPARSE_THRESHOLD, repeats, solver)
    return ComparisonResult(
        n=A.shape[0],
        density=dense.density,
        dense_construct_seconds=dense.construct_seconds,
        sparse_construct_seconds=sparse.construct_seconds,
        dense_sparsity_seconds=dense.sparsity_seconds,
        sparse_sparsity_seconds=sparse.sparsity_seconds,
        dense_hessian_nnz=dense.hessian_nnz,
        sparse_hessian_nnz=sparse.hessian_nnz,
        dense_solve_seconds=dense.solve_seconds,
        sparse_solve_seconds=sparse.solve_seconds,
        dense_status=dense.status,
        sparse_status=sparse.status,
    )


def _format_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    if not rows:
        return ""
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]
    table = ["  ".join(header.ljust(widths[i]) for i, header in enumerate(headers))]
    table.append("  ".join("-" * width for width in widths))
    table.extend("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows)
    return "\n".join(table)


def _format_threshold_table(results: Iterable[TrialResult]) -> str:
    rows = [
        (
            str(result.n),
            f"{result.density:.4f}",
            f"{result.threshold:.4f}",
            result.route,
            f"{1000 * result.construct_seconds:.2f}",
            f"{1000 * result.sparsity_seconds:.2f}",
            str(result.hessian_nnz),
            "" if result.solve_seconds is None
                else f"{1000 * result.solve_seconds:.2f}",
            "" if result.status is None else result.status,
        )
        for result in results
    ]
    headers = (
        "n",
        "matrix_density",
        "threshold",
        "route",
        "construct_ms",
        "sparsity_ms",
        "hessian_nnz",
        "solve_ms",
        "status",
    )
    return _format_table(headers, rows)


def _format_comparison_table(results: Iterable[ComparisonResult]) -> str:
    rows = []
    for result in results:
        speedup = ""
        if result.dense_solve_seconds is not None and result.sparse_solve_seconds is not None:
            speedup = f"{result.dense_solve_seconds / result.sparse_solve_seconds:.2f}x"
        rows.append(
            (
                str(result.n),
                f"{result.density:.4f}",
                f"{1000 * result.dense_construct_seconds:.2f}",
                f"{1000 * result.sparse_construct_seconds:.2f}",
                f"{1000 * result.dense_sparsity_seconds:.2f}",
                f"{1000 * result.sparse_sparsity_seconds:.2f}",
                str(result.dense_hessian_nnz),
                str(result.sparse_hessian_nnz),
                "" if result.dense_solve_seconds is None
                    else f"{1000 * result.dense_solve_seconds:.2f}",
                "" if result.sparse_solve_seconds is None
                    else f"{1000 * result.sparse_solve_seconds:.2f}",
                speedup,
                "" if result.dense_status is None else result.dense_status,
                "" if result.sparse_status is None else result.sparse_status,
            )
        )
    headers = (
        "n",
        "density",
        "dense_construct_ms",
        "sparse_construct_ms",
        "dense_sparsity_ms",
        "sparse_sparsity_ms",
        "dense_hess_nnz",
        "sparse_hess_nnz",
        "dense_solve_ms",
        "sparse_solve_ms",
        "dense/sparse",
        "dense_status",
        "sparse_status",
    )
    return _format_table(headers, rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark dense and sparse NLP diff-engine matmul dispatch."
    )
    parser.add_argument(
        "--mode",
        choices=("compare", "threshold"),
        default="compare",
        help="compare forces dense and sparse for each case; threshold sweeps candidate values.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        help="Matrix dimensions for compare mode.",
    )
    parser.add_argument("--n", type=int, default=120, help="Matrix dimension for the n by n case.")
    parser.add_argument(
        "--densities",
        type=float,
        nargs="+",
        default=DEFAULT_DENSITIES,
        help="Actual dense-constant nonzero densities to test.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
        help="SPARSE_DENSITY_THRESHOLD values to sweep.",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Repetitions per case.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for matrix patterns.")
    parser.add_argument(
        "--solve-with",
        choices=("IPOPT", "KNITRO", "UNO", "COPT"),
        help="Also time full end-to-end NLP solves with the selected solver.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.solve_with is not None and args.solve_with not in cp.installed_solvers():
        installed = ", ".join(cp.installed_solvers())
        raise SystemExit(f"{args.solve_with} is not installed. Installed solvers: {installed}")

    if args.mode == "compare":
        results = []
        for size_index, n in enumerate(args.sizes):
            for density_index, density in enumerate(args.densities):
                seed = args.seed + 1000 * size_index + density_index
                matrix = _exact_density_matrix(n, density, seed)
                results.append(_comparison_trial(matrix, args.repeats, args.solve_with))
        print(_format_comparison_table(results))
    else:
        matrices = [
            _exact_density_matrix(args.n, density, args.seed + index)
            for index, density in enumerate(args.densities)
        ]
        results = []
        for matrix in matrices:
            for threshold in args.thresholds:
                results.append(_median_trial(matrix, threshold, args.repeats, args.solve_with))
        print(_format_threshold_table(results))


if __name__ == "__main__":
    main()
