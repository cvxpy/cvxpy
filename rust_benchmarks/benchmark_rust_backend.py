#!/usr/bin/env python
# ruff: noqa
"""
Comprehensive benchmark for CVXPY's Rust canonicalization backend.

This script compares the performance of different canonicalization backends:
- RUST: The new Rust backend (via cvxpy_rust)
- SCIPY: Pure Python backend using scipy.sparse
- CPP: The C++ backend (if available)

IMPORTANT: Each benchmark creates a fresh Problem instance to avoid caching effects.
"""

import gc
import statistics
import time
from typing import Callable

import numpy as np
import scipy.sparse as sp

import cvxpy as cp


def time_canonicalization(
    problem_factory: Callable[[], cp.Problem],
    backend: str,
    warmup: int = 1,
    iterations: int = 5,
) -> dict:
    """
    Time the canonicalization phase of a problem.

    Args:
        problem_factory: Factory function that creates a fresh Problem instance
        backend: Backend name ('RUST', 'SCIPY', 'NUMPY', 'CPP')
        warmup: Number of warmup iterations
        iterations: Number of timed iterations

    Returns:
        Dictionary with timing statistics
    """
    times = []

    # Warmup runs
    for _ in range(warmup):
        prob = problem_factory()
        try:
            prob.get_problem_data(cp.CLARABEL, canon_backend=backend)
        except Exception as e:
            return {"error": str(e), "times": [], "backend": backend}
        del prob
        gc.collect()

    # Timed runs
    for _ in range(iterations):
        prob = problem_factory()
        gc.collect()

        start = time.perf_counter()
        try:
            prob.get_problem_data(cp.CLARABEL, canon_backend=backend)
        except Exception as e:
            return {"error": str(e), "times": [], "backend": backend}
        end = time.perf_counter()

        times.append(end - start)
        del prob
        gc.collect()

    return {
        "backend": backend,
        "times": times,
        "mean": statistics.mean(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0,
        "min": min(times),
        "max": max(times),
    }


def time_direct_build_matrix(
    linop_factory: Callable,
    backend_class,
    backend_kwargs: dict,
    warmup: int = 1,
    iterations: int = 5,
) -> dict:
    """
    Time just the build_matrix call directly for more accurate benchmarking.
    """
    times = []

    # Get linops once for warmup (they should be identical each time)
    for _ in range(warmup):
        lin_ops = linop_factory()
        backend = backend_class(**backend_kwargs)
        try:
            backend.build_matrix(lin_ops)
        except Exception as e:
            return {"error": str(e), "times": []}
        gc.collect()

    for _ in range(iterations):
        lin_ops = linop_factory()
        backend = backend_class(**backend_kwargs)
        gc.collect()

        start = time.perf_counter()
        try:
            backend.build_matrix(lin_ops)
        except Exception as e:
            return {"error": str(e), "times": []}
        end = time.perf_counter()

        times.append(end - start)
        gc.collect()

    return {
        "times": times,
        "mean": statistics.mean(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0,
        "min": min(times),
        "max": max(times),
    }


# =============================================================================
# Problem Factories - Each creates a fresh problem instance
# =============================================================================

def make_dense_qp(n: int) -> Callable[[], cp.Problem]:
    """Dense quadratic program with n variables."""
    def factory():
        x = cp.Variable(n)
        Q = np.random.randn(n, n)
        Q = Q @ Q.T  # Make positive semidefinite
        c = np.random.randn(n)
        A = np.random.randn(n // 2, n)
        b = np.random.randn(n // 2)

        obj = cp.Minimize(0.5 * cp.quad_form(x, Q) + c @ x)
        constraints = [A @ x <= b, x >= -1, x <= 1]
        return cp.Problem(obj, constraints)
    return factory


def make_sparse_lp(n: int, m: int, density: float = 0.01) -> Callable[[], cp.Problem]:
    """Sparse linear program with n variables and m constraints."""
    def factory():
        x = cp.Variable(n)
        c = np.random.randn(n)
        A = sp.random(m, n, density=density, format='csc')
        b = np.random.randn(m)

        obj = cp.Minimize(c @ x)
        constraints = [A @ x <= b]
        return cp.Problem(obj, constraints)
    return factory


def make_lasso(n: int, m: int) -> Callable[[], cp.Problem]:
    """LASSO problem with n features and m samples."""
    def factory():
        x = cp.Variable(n)
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        lambd = 0.1

        obj = cp.Minimize(0.5 * cp.sum_squares(A @ x - b) + lambd * cp.norm(x, 1))
        return cp.Problem(obj)
    return factory


def make_svm(n: int, m: int) -> Callable[[], cp.Problem]:
    """SVM problem with n features and m samples."""
    def factory():
        w = cp.Variable(n)
        b = cp.Variable()
        xi = cp.Variable(m)

        X = np.random.randn(m, n)
        y = np.sign(np.random.randn(m))
        C = 1.0

        obj = cp.Minimize(0.5 * cp.sum_squares(w) + C * cp.sum(xi))
        constraints = [
            cp.multiply(y, X @ w + b) >= 1 - xi,
            xi >= 0
        ]
        return cp.Problem(obj, constraints)
    return factory


def make_portfolio_optimization(n: int) -> Callable[[], cp.Problem]:
    """Portfolio optimization with n assets."""
    def factory():
        w = cp.Variable(n)
        mu = np.random.randn(n) * 0.1
        Sigma = np.random.randn(n, n)
        Sigma = Sigma @ Sigma.T / n
        gamma = 1.0

        obj = cp.Maximize(mu @ w - gamma * cp.quad_form(w, Sigma))
        constraints = [cp.sum(w) == 1, w >= 0]
        return cp.Problem(obj, constraints)
    return factory


def make_matrix_completion(n: int, rank: int = 10, frac_observed: float = 0.3) -> Callable[[], cp.Problem]:
    """Matrix completion with n x n matrix."""
    def factory():
        X = cp.Variable((n, n))

        # Create observed entries mask
        mask = np.random.rand(n, n) < frac_observed
        observed_i, observed_j = np.where(mask)

        # True low-rank matrix
        U = np.random.randn(n, rank)
        V = np.random.randn(rank, n)
        M_true = U @ V
        observed_vals = M_true[mask]

        # Objective: minimize nuclear norm
        obj = cp.Minimize(cp.normNuc(X))

        # Constraints: match observed entries
        constraints = [X[observed_i, observed_j] == observed_vals]
        return cp.Problem(obj, constraints)
    return factory


def make_sdp(n: int) -> Callable[[], cp.Problem]:
    """Semidefinite program with n x n matrix variable."""
    def factory():
        X = cp.Variable((n, n), symmetric=True)
        C = np.random.randn(n, n)
        C = C + C.T
        A = np.random.randn(n, n)
        A = A + A.T

        obj = cp.Minimize(cp.trace(C @ X))
        constraints = [X >> 0, cp.trace(A @ X) == 1]
        return cp.Problem(obj, constraints)
    return factory


def make_many_small_constraints(n_vars: int, n_constraints: int) -> Callable[[], cp.Problem]:
    """Problem with many small constraints to test parallel processing."""
    def factory():
        x = cp.Variable(n_vars)
        constraints = []
        for i in range(n_constraints):
            a = np.random.randn(n_vars)
            constraints.append(a @ x <= np.random.randn())
        obj = cp.Minimize(cp.sum(x))
        return cp.Problem(obj, constraints)
    return factory


def make_nested_operations(depth: int, width: int) -> Callable[[], cp.Problem]:
    """Problem with deeply nested expression trees."""
    def factory():
        x = cp.Variable(width)
        expr = x
        for _ in range(depth):
            expr = expr + x
            expr = 2 * expr
            expr = expr.T @ np.eye(width) @ expr  # This creates a scalar
            expr = cp.reshape(expr, (1,))
            # Now reshape to match original width to continue nesting
            expr_broadcast = cp.hstack([expr] * width)
            expr = cp.Variable(width) + expr_broadcast  # Need new var each level
        obj = cp.Minimize(cp.sum(expr))
        # Actually, let's simplify:
        x = cp.Variable(width)
        A = np.random.randn(width, width)
        expr = x
        for _ in range(depth):
            expr = A @ expr
            expr = expr + x
        obj = cp.Minimize(cp.sum_squares(expr))
        return cp.Problem(obj)
    return factory


def make_kronecker_problem(n: int, m: int) -> Callable[[], cp.Problem]:
    """Problem involving Kronecker products."""
    def factory():
        X = cp.Variable((n, m))
        A = np.random.randn(n, n)
        B = np.random.randn(m, m)
        C = np.random.randn(n, m)

        obj = cp.Minimize(cp.sum_squares(cp.kron(A, B) @ cp.vec(X) - cp.vec(C)))
        return cp.Problem(obj)
    return factory


def make_convolution_problem(signal_len: int, kernel_len: int) -> Callable[[], cp.Problem]:
    """Problem involving convolution."""
    def factory():
        x = cp.Variable(signal_len)
        kernel = np.random.randn(kernel_len)
        target = np.random.randn(signal_len + kernel_len - 1)

        # conv requires the first arg to be constant
        obj = cp.Minimize(cp.sum_squares(cp.conv(kernel, x) - target))
        return cp.Problem(obj)
    return factory


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_benchmark(name: str, problem_factory: Callable, backends: list, iterations: int = 5):
    """Run a benchmark for a given problem across backends."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {name}")
    print(f"{'='*60}")

    results = {}
    for backend in backends:
        result = time_canonicalization(problem_factory, backend, warmup=1, iterations=iterations)
        results[backend] = result

        if "error" in result:
            print(f"  {backend:8s}: ERROR - {result['error']}")
        else:
            print(f"  {backend:8s}: mean={result['mean']*1000:8.2f}ms, std={result['std']*1000:6.2f}ms, min={result['min']*1000:8.2f}ms")

    # Calculate speedups
    if "RUST" in results and "SCIPY" in results and "error" not in results["RUST"] and "error" not in results["SCIPY"]:
        speedup = results["SCIPY"]["mean"] / results["RUST"]["mean"]
        print(f"  Rust speedup over SciPy: {speedup:.2f}x")

    if "RUST" in results and "CPP" in results and "error" not in results["RUST"] and "error" not in results["CPP"]:
        speedup = results["CPP"]["mean"] / results["RUST"]["mean"]
        print(f"  Rust speedup over C++: {speedup:.2f}x")

    return results


def main():
    print("CVXPY Rust Backend Benchmark Suite")
    print("=" * 60)

    # Check available backends
    backends = ["RUST", "SCIPY"]
    try:
        import cvxpy_rust
        print("✓ Rust backend available")
    except ImportError:
        print("✗ Rust backend not available")
        backends.remove("RUST")

    try:
        from cvxpy.cvxcore.python import cppbackend
        print("✓ C++ backend available")
        backends.append("CPP")
    except ImportError:
        print("✗ C++ backend not available")

    print(f"\nBenchmarking backends: {backends}")
    print("Using solver: CLARABEL")

    all_results = {}

    # Small problems
    print("\n" + "="*60)
    print("SMALL PROBLEMS")
    print("="*60)

    all_results["dense_qp_small"] = run_benchmark(
        "Dense QP (n=50)", make_dense_qp(50), backends
    )

    all_results["sparse_lp_small"] = run_benchmark(
        "Sparse LP (n=100, m=50)", make_sparse_lp(100, 50), backends
    )

    all_results["lasso_small"] = run_benchmark(
        "LASSO (n=50, m=100)", make_lasso(50, 100), backends
    )

    # Medium problems
    print("\n" + "="*60)
    print("MEDIUM PROBLEMS")
    print("="*60)

    all_results["dense_qp_medium"] = run_benchmark(
        "Dense QP (n=200)", make_dense_qp(200), backends
    )

    all_results["sparse_lp_medium"] = run_benchmark(
        "Sparse LP (n=1000, m=500)", make_sparse_lp(1000, 500), backends
    )

    all_results["lasso_medium"] = run_benchmark(
        "LASSO (n=200, m=500)", make_lasso(200, 500), backends
    )

    all_results["svm_medium"] = run_benchmark(
        "SVM (n=100, m=500)", make_svm(100, 500), backends
    )

    all_results["portfolio_medium"] = run_benchmark(
        "Portfolio (n=100)", make_portfolio_optimization(100), backends
    )

    # Large problems
    print("\n" + "="*60)
    print("LARGE PROBLEMS")
    print("="*60)

    all_results["dense_qp_large"] = run_benchmark(
        "Dense QP (n=500)", make_dense_qp(500), backends, iterations=3
    )

    all_results["sparse_lp_large"] = run_benchmark(
        "Sparse LP (n=5000, m=2000)", make_sparse_lp(5000, 2000), backends, iterations=3
    )

    all_results["lasso_large"] = run_benchmark(
        "LASSO (n=500, m=2000)", make_lasso(500, 2000), backends, iterations=3
    )

    # Stress tests for parallelization
    print("\n" + "="*60)
    print("PARALLELIZATION STRESS TESTS")
    print("="*60)

    all_results["many_constraints_100"] = run_benchmark(
        "Many constraints (n=50, m=100)", make_many_small_constraints(50, 100), backends
    )

    all_results["many_constraints_1000"] = run_benchmark(
        "Many constraints (n=50, m=1000)", make_many_small_constraints(50, 1000), backends, iterations=3
    )

    all_results["many_constraints_5000"] = run_benchmark(
        "Many constraints (n=50, m=5000)", make_many_small_constraints(50, 5000), backends, iterations=3
    )

    # Specialized operations
    print("\n" + "="*60)
    print("SPECIALIZED OPERATIONS")
    print("="*60)

    all_results["sdp_small"] = run_benchmark(
        "SDP (n=20)", make_sdp(20), backends
    )

    all_results["sdp_medium"] = run_benchmark(
        "SDP (n=50)", make_sdp(50), backends, iterations=3
    )

    all_results["convolution"] = run_benchmark(
        "Convolution (signal=1000, kernel=50)", make_convolution_problem(1000, 50), backends
    )

    # Nested operations
    print("\n" + "="*60)
    print("NESTED EXPRESSION TREES")
    print("="*60)

    all_results["nested_shallow"] = run_benchmark(
        "Nested (depth=5, width=100)", make_nested_operations(5, 100), backends
    )

    all_results["nested_deep"] = run_benchmark(
        "Nested (depth=20, width=50)", make_nested_operations(20, 50), backends
    )

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    rust_wins = 0
    scipy_wins = 0
    cpp_wins = 0

    for name, results in all_results.items():
        if "RUST" in results and "SCIPY" in results:
            if "error" not in results["RUST"] and "error" not in results["SCIPY"]:
                if results["RUST"]["mean"] < results["SCIPY"]["mean"]:
                    rust_wins += 1
                else:
                    scipy_wins += 1

    print(f"\nRust faster than SciPy: {rust_wins} / {rust_wins + scipy_wins} benchmarks")

    # Calculate average speedup
    speedups = []
    for name, results in all_results.items():
        if "RUST" in results and "SCIPY" in results:
            if "error" not in results["RUST"] and "error" not in results["SCIPY"]:
                speedups.append(results["SCIPY"]["mean"] / results["RUST"]["mean"])

    if speedups:
        print(f"Average Rust speedup over SciPy: {statistics.mean(speedups):.2f}x")
        print(f"Min speedup: {min(speedups):.2f}x, Max speedup: {max(speedups):.2f}x")

    return all_results


if __name__ == "__main__":
    results = main()
