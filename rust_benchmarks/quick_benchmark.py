#!/usr/bin/env python
"""Quick benchmark comparing CVXPY canonicalization backends.

This benchmark uses TRUE COLD START methodology - each measurement runs in a
fresh Python subprocess to accurately measure single-problem performance,
which is the typical user experience.

The in-process "warm" benchmark is also available for comparison, but cold
start is the primary metric since most users solve one problem per script.
"""

import argparse
import subprocess
import sys
import textwrap

import numpy as np


def run_cold_start_benchmark(problem_code: str, backend: str, samples: int = 10) -> list[float]:
    """Run benchmark in fresh Python processes (true cold start)."""
    # Strip and dedent the problem code to avoid indentation issues
    problem_code = textwrap.dedent(problem_code).strip()
    code = f"""import time, gc, numpy as np, cvxpy as cp
np.random.seed(42)
{problem_code}
gc.collect()
start = time.perf_counter()
prob.get_problem_data(cp.CLARABEL, canon_backend='{backend}')
print((time.perf_counter() - start) * 1000)
"""

    times = []
    for _ in range(samples):
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Subprocess failed: {result.stderr}")
        times.append(float(result.stdout.strip()))
    return times


def benchmark_cold(name: str, problem_code: str, backends: list[str], samples: int = 10):
    """Run cold start benchmark for a problem across backends."""
    print(f"\n{name}:")

    results = {}
    for backend in backends:
        try:
            times = run_cold_start_benchmark(problem_code, backend, samples)
            results[backend] = times
            arr = np.array(times)
            print(f"  {backend:5}: {np.mean(arr):6.1f} ± {np.std(arr):4.1f} ms  "
                  f"(median: {np.median(arr):5.1f}, range: {np.min(arr):.0f}-{np.max(arr):.0f})")
        except Exception as e:
            print(f"  {backend:5}: ERROR - {e}")

    # Show speedups vs SCIPY
    if "SCIPY" in results and len(results) > 1:
        scipy_mean = np.mean(results["SCIPY"])
        print("  ---")
        for backend, times in results.items():
            if backend != "SCIPY":
                speedup = scipy_mean / np.mean(times)
                print(f"  {backend} vs SCIPY: {speedup:.2f}x")

    return results


# Problem definitions (code strings for subprocess execution)
PROBLEMS = {
    "LASSO (n=50, m=100)": """
A = np.random.randn(100, 50)
b = np.random.randn(100)
x = cp.Variable(50)
prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(A @ x - b) + 0.1 * cp.norm(x, 1)))
""",
    "LASSO (n=200, m=500)": """
A = np.random.randn(500, 200)
b = np.random.randn(500)
x = cp.Variable(200)
prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(A @ x - b) + 0.1 * cp.norm(x, 1)))
""",
    "Dense QP (n=50)": """
x = cp.Variable(50)
prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, np.eye(50)) + np.ones(50) @ x),
                  [x >= -1, x <= 1])
""",
    "Dense QP (n=200)": """
x = cp.Variable(200)
prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, np.eye(200)) + np.ones(200) @ x),
                  [x >= -1, x <= 1])
""",
    "Many constraints (n=50, m=100)": """
x = cp.Variable(50)
constraints = [np.random.randn(50) @ x <= np.random.randn() for _ in range(100)]
prob = cp.Problem(cp.Minimize(cp.sum(x)), constraints)
""",
    "Many constraints (n=50, m=500)": """
x = cp.Variable(50)
constraints = [np.random.randn(50) @ x <= np.random.randn() for _ in range(500)]
prob = cp.Problem(cp.Minimize(cp.sum(x)), constraints)
""",
}


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark CVXPY canonicalization backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python quick_benchmark.py                    # Run all benchmarks
              python quick_benchmark.py --backends RUST SCIPY  # Compare only RUST and SCIPY
              python quick_benchmark.py --samples 20       # More samples for accuracy
              python quick_benchmark.py --problems LASSO   # Only LASSO problems
        """)
    )
    parser.add_argument(
        "--backends", nargs="+", default=["RUST", "SCIPY", "CPP"],
        help="Backends to benchmark (default: RUST SCIPY CPP)"
    )
    parser.add_argument(
        "--samples", type=int, default=10,
        help="Number of cold start samples per backend (default: 10)"
    )
    parser.add_argument(
        "--problems", nargs="+", default=None,
        help="Filter problems by substring (e.g., 'LASSO' or 'QP')"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("CVXPY Canonicalization Backend Benchmark (Cold Start)")
    print("=" * 70)
    print(f"Methodology: {args.samples} fresh Python processes per measurement")
    print(f"Backends: {', '.join(args.backends)}")

    # Filter problems if requested
    problems = PROBLEMS
    if args.problems:
        problems = {k: v for k, v in PROBLEMS.items()
                   if any(p.lower() in k.lower() for p in args.problems)}

    all_results = {}

    # Group problems by category
    categories = {
        "LASSO Problems": [k for k in problems if "LASSO" in k],
        "Dense QP": [k for k in problems if "QP" in k],
        "Many Constraints": [k for k in problems if "constraints" in k],
    }

    for category, problem_names in categories.items():
        if not problem_names:
            continue
        print(f"\n--- {category} ---")
        for name in problem_names:
            results = benchmark_cold(name, problems[name], args.backends, args.samples)
            all_results[name] = results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Build summary table
    print(f"\n{'Problem':<30} ", end="")
    for backend in args.backends:
        print(f"{backend:>10} ", end="")
    if "SCIPY" in args.backends:
        for backend in args.backends:
            if backend != "SCIPY":
                print(f"{backend + '/SCIPY':>12} ", end="")
    print()
    print("-" * (32 + 11 * len(args.backends) + 11 * (len(args.backends) - 1)))

    speedups = {b: [] for b in args.backends if b != "SCIPY"}

    for name, results in all_results.items():
        print(f"{name:<30} ", end="")
        for backend in args.backends:
            if backend in results:
                print(f"{np.mean(results[backend]):>8.1f}ms ", end="")
            else:
                print(f"{'N/A':>10} ", end="")

        if "SCIPY" in results:
            scipy_mean = np.mean(results["SCIPY"])
            for backend in args.backends:
                if backend != "SCIPY" and backend in results:
                    spd = scipy_mean / np.mean(results[backend])
                    speedups[backend].append(spd)
                    print(f"{spd:>10.2f}x ", end="")
        print()

    # Overall stats
    if speedups:
        print()
        for backend, spds in speedups.items():
            if spds:
                print(f"{backend} vs SCIPY: avg {np.mean(spds):.2f}x, "
                      f"min {min(spds):.2f}x, max {max(spds):.2f}x, "
                      f"wins {sum(1 for s in spds if s > 1)}/{len(spds)}")


if __name__ == "__main__":
    main()
