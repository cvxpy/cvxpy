#!/usr/bin/env python
# ruff: noqa
"""
Detailed profiling of the Rust backend to identify performance bottlenecks.

This script focuses on isolating different aspects of canonicalization:
1. Python-to-Rust data transfer (LinOp extraction)
2. Core Rust processing
3. Rust-to-Python data transfer (result conversion)
"""

import cProfile
import gc
import io
import pstats
import time
from typing import Callable

import numpy as np
import scipy.sparse as sp

import cvxpy as cp
from cvxpy.lin_ops.canon_backend import (
    RustCanonBackend,
    SciPyCanonBackend,
)


def profile_problem(problem_factory: Callable[[], cp.Problem], name: str, backend: str = "RUST"):
    """Profile a problem's canonicalization with detailed timing breakdown."""
    print(f"\n{'='*60}")
    print(f"Profiling: {name} (Backend: {backend})")
    print(f"{'='*60}")

    prob = problem_factory()

    # Profile with cProfile
    profiler = cProfile.Profile()
    profiler.enable()

    prob.get_problem_data(cp.CLARABEL, canon_backend=backend)

    profiler.disable()

    # Get stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())

    return profiler


def compare_build_matrix_timing(linops_factory, backend_kwargs: dict, n_iterations: int = 10):
    """
    Compare build_matrix timing between Rust and SciPy backends.
    This isolates just the matrix building step.
    """
    print("\n" + "="*60)
    print("Direct build_matrix comparison")
    print("="*60)

    # Time SciPy backend
    scipy_times = []
    for _ in range(n_iterations):
        lin_ops = linops_factory()
        backend = SciPyCanonBackend(**backend_kwargs)
        gc.collect()

        start = time.perf_counter()
        backend.build_matrix(lin_ops)
        end = time.perf_counter()
        scipy_times.append(end - start)

    # Time Rust backend
    rust_times = []
    for _ in range(n_iterations):
        lin_ops = linops_factory()
        backend = RustCanonBackend(**backend_kwargs)
        gc.collect()

        start = time.perf_counter()
        backend.build_matrix(lin_ops)
        end = time.perf_counter()
        rust_times.append(end - start)

    print(f"SciPy: mean={np.mean(scipy_times)*1000:.2f}ms, min={np.min(scipy_times)*1000:.2f}ms")
    print(f"Rust:  mean={np.mean(rust_times)*1000:.2f}ms, min={np.min(rust_times)*1000:.2f}ms")
    print(f"Speedup: {np.mean(scipy_times)/np.mean(rust_times):.2f}x")


def time_linop_extraction():
    """
    Measure the time spent extracting LinOp trees from Python to Rust.
    This helps identify if the FFI boundary is a bottleneck.
    """
    print("\n" + "="*60)
    print("LinOp Extraction Timing (Python -> Rust FFI)")
    print("="*60)

    import cvxpy_rust

    # Create a problem to get linops
    n = 500
    x = cp.Variable(n)
    A = np.random.randn(n, n)
    b = np.random.randn(n)

    prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))

    # Get problem data to trigger canonicalization and get linops
    data, _, _ = prob.get_problem_data(cp.CLARABEL, canon_backend="SCIPY")

    # Now get the raw linops
    from cvxpy.reductions.chain import Chain
    from cvxpy.reductions.cone_matrix_stuffing import ConeMatrixStuffing

    # Get the cone program
    chain = Chain(reductions=[ConeMatrixStuffing()])
    cone_prog, _ = chain.apply(prob)

    # Get linops from constraints
    all_linops = []
    for constr in cone_prog.constraints:
        for lin_op in constr.canonical_form[0]:
            all_linops.append(lin_op)

    print(f"Number of LinOps: {len(all_linops)}")

    # Time just the Rust function call overhead (with minimal data)
    id_to_col = {0: 0}
    param_to_size = {}
    param_to_col = {}
    param_size_plus_one = 1
    var_length = n

    # Create minimal linops for timing
    from cvxpy.lin_ops.lin_op import NO_OP, LinOp

    minimal_linops = [LinOp(NO_OP, (1,), [], None) for _ in range(100)]

    times = []
    for _ in range(100):
        gc.collect()
        start = time.perf_counter()
        try:
            cvxpy_rust.build_matrix(
                minimal_linops, param_size_plus_one, id_to_col,
                param_to_size, param_to_col, var_length
            )
        except Exception:
            pass
        end = time.perf_counter()
        times.append(end - start)

    print(f"100 NO_OP linops: mean={np.mean(times)*1000:.3f}ms, min={np.min(times)*1000:.3f}ms")


def profile_operation_breakdown():
    """
    Profile which operations take the most time in canonicalization.
    """
    print("\n" + "="*60)
    print("Operation Breakdown Analysis")
    print("="*60)


    # Test different operation patterns

    # 1. Dense matrix multiplication (common in QP)
    print("\n1. Dense matrix multiplication:")
    n = 200
    x = cp.Variable(n)
    A = np.random.randn(n, n)
    expr = A @ x

    prob = cp.Problem(cp.Minimize(cp.sum_squares(expr)))

    times = []
    for _ in range(5):
        prob_fresh = cp.Problem(cp.Minimize(cp.sum_squares(A @ cp.Variable(n))))
        gc.collect()
        start = time.perf_counter()
        prob_fresh.get_problem_data(cp.CLARABEL, canon_backend="RUST")
        end = time.perf_counter()
        times.append(end - start)
    print(f"  RUST: mean={np.mean(times)*1000:.2f}ms")

    times = []
    for _ in range(5):
        prob_fresh = cp.Problem(cp.Minimize(cp.sum_squares(A @ cp.Variable(n))))
        gc.collect()
        start = time.perf_counter()
        prob_fresh.get_problem_data(cp.CLARABEL, canon_backend="SCIPY")
        end = time.perf_counter()
        times.append(end - start)
    print(f"  SCIPY: mean={np.mean(times)*1000:.2f}ms")

    # 2. Sparse matrix multiplication
    print("\n2. Sparse matrix multiplication:")
    A_sparse = sp.random(n, n, density=0.1, format='csc')

    times = []
    for _ in range(5):
        prob_fresh = cp.Problem(cp.Minimize(cp.sum_squares(A_sparse @ cp.Variable(n))))
        gc.collect()
        start = time.perf_counter()
        prob_fresh.get_problem_data(cp.CLARABEL, canon_backend="RUST")
        end = time.perf_counter()
        times.append(end - start)
    print(f"  RUST: mean={np.mean(times)*1000:.2f}ms")

    times = []
    for _ in range(5):
        prob_fresh = cp.Problem(cp.Minimize(cp.sum_squares(A_sparse @ cp.Variable(n))))
        gc.collect()
        start = time.perf_counter()
        prob_fresh.get_problem_data(cp.CLARABEL, canon_backend="SCIPY")
        end = time.perf_counter()
        times.append(end - start)
    print(f"  SCIPY: mean={np.mean(times)*1000:.2f}ms")

    # 3. Many small variable expressions
    print("\n3. Many small expressions (stacking):")
    n_exprs = 100

    times = []
    for _ in range(5):
        x = cp.Variable(10)
        exprs = [np.random.randn(10) @ x for _ in range(n_exprs)]
        prob_fresh = cp.Problem(cp.Minimize(cp.sum(cp.hstack(exprs))))
        gc.collect()
        start = time.perf_counter()
        prob_fresh.get_problem_data(cp.CLARABEL, canon_backend="RUST")
        end = time.perf_counter()
        times.append(end - start)
    print(f"  RUST: mean={np.mean(times)*1000:.2f}ms")

    times = []
    for _ in range(5):
        x = cp.Variable(10)
        exprs = [np.random.randn(10) @ x for _ in range(n_exprs)]
        prob_fresh = cp.Problem(cp.Minimize(cp.sum(cp.hstack(exprs))))
        gc.collect()
        start = time.perf_counter()
        prob_fresh.get_problem_data(cp.CLARABEL, canon_backend="SCIPY")
        end = time.perf_counter()
        times.append(end - start)
    print(f"  SCIPY: mean={np.mean(times)*1000:.2f}ms")

    # 4. Index/slicing operations
    print("\n4. Index/slicing operations:")

    times = []
    for _ in range(5):
        X = cp.Variable((50, 50))
        expr = X[10:40, 10:40]
        prob_fresh = cp.Problem(cp.Minimize(cp.sum_squares(expr)))
        gc.collect()
        start = time.perf_counter()
        prob_fresh.get_problem_data(cp.CLARABEL, canon_backend="RUST")
        end = time.perf_counter()
        times.append(end - start)
    print(f"  RUST: mean={np.mean(times)*1000:.2f}ms")

    times = []
    for _ in range(5):
        X = cp.Variable((50, 50))
        expr = X[10:40, 10:40]
        prob_fresh = cp.Problem(cp.Minimize(cp.sum_squares(expr)))
        gc.collect()
        start = time.perf_counter()
        prob_fresh.get_problem_data(cp.CLARABEL, canon_backend="SCIPY")
        end = time.perf_counter()
        times.append(end - start)
    print(f"  SCIPY: mean={np.mean(times)*1000:.2f}ms")


def profile_constraint_count_scaling():
    """
    Profile how performance scales with the number of constraints.
    This helps identify parallelization benefits.
    """
    print("\n" + "="*60)
    print("Constraint Count Scaling")
    print("="*60)

    constraint_counts = [10, 50, 100, 500, 1000, 2000]
    n_vars = 50

    print(f"\n{'Constraints':<12} {'RUST (ms)':<12} {'SCIPY (ms)':<12} {'Speedup':<10}")
    print("-" * 50)

    for n_constrs in constraint_counts:
        # Rust timing
        rust_times = []
        for _ in range(3):
            x = cp.Variable(n_vars)
            constraints = [np.random.randn(n_vars) @ x <= np.random.randn() for _ in range(n_constrs)]
            prob = cp.Problem(cp.Minimize(cp.sum(x)), constraints)
            gc.collect()
            start = time.perf_counter()
            prob.get_problem_data(cp.CLARABEL, canon_backend="RUST")
            end = time.perf_counter()
            rust_times.append(end - start)

        # SciPy timing
        scipy_times = []
        for _ in range(3):
            x = cp.Variable(n_vars)
            constraints = [np.random.randn(n_vars) @ x <= np.random.randn() for _ in range(n_constrs)]
            prob = cp.Problem(cp.Minimize(cp.sum(x)), constraints)
            gc.collect()
            start = time.perf_counter()
            prob.get_problem_data(cp.CLARABEL, canon_backend="SCIPY")
            end = time.perf_counter()
            scipy_times.append(end - start)

        rust_mean = np.mean(rust_times) * 1000
        scipy_mean = np.mean(scipy_times) * 1000
        speedup = scipy_mean / rust_mean

        print(f"{n_constrs:<12} {rust_mean:<12.2f} {scipy_mean:<12.2f} {speedup:<10.2f}x")


def profile_variable_size_scaling():
    """
    Profile how performance scales with variable size.
    """
    print("\n" + "="*60)
    print("Variable Size Scaling")
    print("="*60)

    var_sizes = [10, 50, 100, 500, 1000, 2000]
    n_constrs = 10

    print(f"\n{'Var Size':<12} {'RUST (ms)':<12} {'SCIPY (ms)':<12} {'Speedup':<10}")
    print("-" * 50)

    for n_vars in var_sizes:
        # Rust timing
        rust_times = []
        for _ in range(3):
            x = cp.Variable(n_vars)
            A = np.random.randn(n_constrs, n_vars)
            b = np.random.randn(n_constrs)
            prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [A @ x <= b])
            gc.collect()
            start = time.perf_counter()
            prob.get_problem_data(cp.CLARABEL, canon_backend="RUST")
            end = time.perf_counter()
            rust_times.append(end - start)

        # SciPy timing
        scipy_times = []
        for _ in range(3):
            x = cp.Variable(n_vars)
            A = np.random.randn(n_constrs, n_vars)
            b = np.random.randn(n_constrs)
            prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [A @ x <= b])
            gc.collect()
            start = time.perf_counter()
            prob.get_problem_data(cp.CLARABEL, canon_backend="SCIPY")
            end = time.perf_counter()
            scipy_times.append(end - start)

        rust_mean = np.mean(rust_times) * 1000
        scipy_mean = np.mean(scipy_times) * 1000
        speedup = scipy_mean / rust_mean

        print(f"{n_vars:<12} {rust_mean:<12.2f} {scipy_mean:<12.2f} {speedup:<10.2f}x")


def profile_dense_const_overhead():
    """
    Profile the overhead of dense constant handling.
    This is a suspected bottleneck based on initial results.
    """
    print("\n" + "="*60)
    print("Dense Constant Overhead Analysis")
    print("="*60)

    # Test with increasing dense matrix sizes
    sizes = [10, 50, 100, 200, 500]

    print(f"\n{'Matrix Size':<12} {'RUST (ms)':<12} {'SCIPY (ms)':<12} {'Speedup':<10}")
    print("-" * 50)

    for n in sizes:
        A = np.random.randn(n, n)

        # Rust timing
        rust_times = []
        for _ in range(5):
            x = cp.Variable(n)
            prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x)))
            gc.collect()
            start = time.perf_counter()
            prob.get_problem_data(cp.CLARABEL, canon_backend="RUST")
            end = time.perf_counter()
            rust_times.append(end - start)

        # SciPy timing
        scipy_times = []
        for _ in range(5):
            x = cp.Variable(n)
            prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x)))
            gc.collect()
            start = time.perf_counter()
            prob.get_problem_data(cp.CLARABEL, canon_backend="SCIPY")
            end = time.perf_counter()
            scipy_times.append(end - start)

        rust_mean = np.mean(rust_times) * 1000
        scipy_mean = np.mean(scipy_times) * 1000
        speedup = scipy_mean / rust_mean

        print(f"{n}x{n:<10} {rust_mean:<12.2f} {scipy_mean:<12.2f} {speedup:<10.2f}x")


def main():
    print("CVXPY Rust Backend Profiling Suite")
    print("=" * 60)

    # Profile different aspects
    profile_constraint_count_scaling()
    profile_variable_size_scaling()
    profile_dense_const_overhead()
    profile_operation_breakdown()

    # Detailed cProfile for a representative problem
    print("\n" + "="*60)
    print("DETAILED CPROFILE: Medium LASSO Problem")
    print("="*60)

    def make_lasso():
        n, m = 200, 500
        x = cp.Variable(n)
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        return cp.Problem(cp.Minimize(0.5 * cp.sum_squares(A @ x - b) + 0.1 * cp.norm(x, 1)))

    print("\n--- RUST Backend ---")
    profile_problem(make_lasso, "LASSO (n=200, m=500)", backend="RUST")

    print("\n--- SCIPY Backend ---")
    profile_problem(make_lasso, "LASSO (n=200, m=500)", backend="SCIPY")


if __name__ == "__main__":
    main()
