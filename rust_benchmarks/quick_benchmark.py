#!/usr/bin/env python
# ruff: noqa
"""Quick benchmark comparing Rust and SciPy backends."""

import gc
import time

import numpy as np

import cvxpy as cp


def time_problem(problem_factory, backend, iterations=5):
    """Time canonicalization for a problem."""
    times = []
    for _ in range(iterations):
        prob = problem_factory()
        gc.collect()
        start = time.perf_counter()
        prob.get_problem_data(cp.CLARABEL, canon_backend=backend)
        end = time.perf_counter()
        times.append(end - start)
        del prob
    return np.mean(times) * 1000, np.std(times) * 1000

def benchmark(name, factory, iterations=5):
    """Run benchmark for a problem."""
    print(f"\n{name}:")
    try:
        rust_mean, rust_std = time_problem(factory, "RUST", iterations)
        scipy_mean, scipy_std = time_problem(factory, "SCIPY", iterations)
        speedup = scipy_mean / rust_mean
        print(f"  RUST:  {rust_mean:8.2f} ± {rust_std:5.2f} ms")
        print(f"  SCIPY: {scipy_mean:8.2f} ± {scipy_std:5.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        return speedup
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

def main():
    print("=" * 60)
    print("Quick CVXPY Rust Backend Benchmark")
    print("=" * 60)

    speedups = []

    # Small problems
    print("\n--- Small Problems ---")

    s = benchmark("Dense QP (n=50)", lambda: (
        x := cp.Variable(50),
        cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, np.eye(50)) + np.ones(50) @ x),
                   [x >= -1, x <= 1])
    )[1])
    if s: speedups.append(s)

    s = benchmark("LASSO (n=50, m=100)", lambda: (
        x := cp.Variable(50),
        cp.Problem(cp.Minimize(0.5 * cp.sum_squares(np.random.randn(100, 50) @ x - np.random.randn(100))
                              + 0.1 * cp.norm(x, 1)))
    )[1])
    if s: speedups.append(s)

    # Medium problems
    print("\n--- Medium Problems ---")

    s = benchmark("Dense QP (n=200)", lambda: (
        x := cp.Variable(200),
        cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, np.eye(200)) + np.ones(200) @ x),
                   [x >= -1, x <= 1])
    )[1])
    if s: speedups.append(s)

    s = benchmark("LASSO (n=200, m=500)", lambda: (
        x := cp.Variable(200),
        cp.Problem(cp.Minimize(0.5 * cp.sum_squares(np.random.randn(500, 200) @ x - np.random.randn(500))
                              + 0.1 * cp.norm(x, 1)))
    )[1])
    if s: speedups.append(s)

    # Parallelization stress tests
    print("\n--- Parallelization Stress Tests ---")

    s = benchmark("Many constraints (n=50, m=100)", lambda: (
        x := cp.Variable(50),
        cp.Problem(cp.Minimize(cp.sum(x)),
                   [np.random.randn(50) @ x <= np.random.randn() for _ in range(100)])
    )[1])
    if s: speedups.append(s)

    s = benchmark("Many constraints (n=50, m=500)", lambda: (
        x := cp.Variable(50),
        cp.Problem(cp.Minimize(cp.sum(x)),
                   [np.random.randn(50) @ x <= np.random.randn() for _ in range(500)])
    )[1], iterations=3)
    if s: speedups.append(s)

    s = benchmark("Many constraints (n=50, m=1000)", lambda: (
        x := cp.Variable(50),
        cp.Problem(cp.Minimize(cp.sum(x)),
                   [np.random.randn(50) @ x <= np.random.randn() for _ in range(1000)])
    )[1], iterations=3)
    if s: speedups.append(s)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if speedups:
        print(f"Average speedup: {np.mean(speedups):.2f}x")
        print(f"Min speedup: {min(speedups):.2f}x")
        print(f"Max speedup: {max(speedups):.2f}x")
        print(f"Rust faster: {sum(1 for s in speedups if s > 1)} / {len(speedups)} benchmarks")

if __name__ == "__main__":
    main()
