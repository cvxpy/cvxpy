"""
DPP (Disciplined Parameterized Programming) Canonicalization Benchmarks

Comprehensive benchmarks for DPP canonicalization performance, focusing on:
1. Large parameter matrices (100K-1M parameters)
2. Cold path (first canonicalization) vs warm path (re-solve)
3. Comparison of current SCIPY backend vs CompactTensor approach
4. Various problem structures (LP, QP)

Usage:
    python -m cvxpy.experimental.dpp_benchmarks [--quick] [--profile] [--compact]

    --quick: Run only small/medium configurations
    --profile: Enable cProfile output for bottleneck analysis
    --compact: Run CompactTensor vs Stacked comparison
    --all: Run everything
"""

import argparse
import time
from dataclasses import dataclass

import numpy as np

import cvxpy as cp

# Optional imports for detailed profiling
try:
    import cProfile
    import pstats
    from io import StringIO
    HAS_PROFILER = True
except ImportError:
    HAS_PROFILER = False

try:
    import tracemalloc
    HAS_TRACEMALLOC = True
except ImportError:
    HAS_TRACEMALLOC = False


# =============================================================================
# Backend Comparison Utilities
# =============================================================================

def benchmark_single_config(m: int, k: int, n: int, n_runs: int = 3) -> tuple[float, float, float]:
    """
    Benchmark SCIPY vs COO backend for a parametrized matmul problem.

    Creates a problem: A_param @ x <= b where A_param is (m, k) parameter matrix.

    Args:
        m: Number of constraints (rows of A)
        k: Number of variables (cols of A)
        n: Not used (kept for legacy interface, originally was output cols)
        n_runs: Number of runs for timing

    Returns:
        (scipy_time_ms, coo_time_ms, max_diff) where times are cold path times
        and max_diff is the max absolute difference in problem data.
    """
    np.random.seed(42)

    # Create parametrized problem
    x = cp.Variable(k)
    A_param = cp.Parameter((m, k))
    b = np.random.randn(m) + 10
    c = np.random.randn(k)

    prob = cp.Problem(
        cp.Minimize(c @ x),
        [A_param @ x <= b, x >= 0]
    )

    A_param.value = np.random.randn(m, k)

    # SCIPY backend timing
    scipy_times = []
    for _ in range(n_runs):
        prob._cache = type(prob._cache)()
        A_param.value = np.random.randn(m, k)
        start = time.perf_counter()
        scipy_data = prob.get_problem_data(cp.SCIPY, canon_backend='SCIPY')
        elapsed = (time.perf_counter() - start) * 1000
        scipy_times.append(elapsed)
    scipy_time = np.median(scipy_times)

    # COO backend timing
    coo_times = []
    for _ in range(n_runs):
        prob._cache = type(prob._cache)()
        A_param.value = np.random.randn(m, k)
        start = time.perf_counter()
        coo_data = prob.get_problem_data(cp.SCIPY, canon_backend='COO')
        elapsed = (time.perf_counter() - start) * 1000
        coo_times.append(elapsed)
    coo_time = np.median(coo_times)

    # Compare results
    max_diff = 0.0
    try:
        scipy_A = scipy_data[0]['A']
        coo_A = coo_data[0]['A']
        if hasattr(scipy_A, 'toarray'):
            scipy_A = scipy_A.toarray()
        if hasattr(coo_A, 'toarray'):
            coo_A = coo_A.toarray()
        max_diff = np.max(np.abs(scipy_A - coo_A))
    except Exception:
        pass

    return float(scipy_time), float(coo_time), float(max_diff)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    name: str
    n_vars: int
    n_constraints: int
    problem_type: str = "LP"

    @property
    def param_size(self) -> int:
        return self.n_vars * self.n_constraints

    def __str__(self):
        return f"{self.name} ({self.param_size:,} params)"


# Quick configs for fast testing
QUICK_CONFIGS = [
    BenchmarkConfig("tiny", 50, 50),
    BenchmarkConfig("small", 100, 100),
    BenchmarkConfig("medium", 100, 500),
]

# Full configs including large problems
FULL_CONFIGS = [
    BenchmarkConfig("tiny", 50, 50),
    BenchmarkConfig("small", 100, 100),
    BenchmarkConfig("medium", 100, 500),
    BenchmarkConfig("large", 200, 500),
    BenchmarkConfig("xlarge", 500, 500),
    BenchmarkConfig("xxlarge", 1000, 500),
    BenchmarkConfig("huge", 1000, 1000),
]

# Scaling analysis configs
SCALING_CONFIGS = [
    BenchmarkConfig(f"scale_{n}x{n}", n, n)
    for n in [50, 100, 150, 200, 300, 400, 500, 700, 1000]
]


# =============================================================================
# Problem Generators
# =============================================================================

def create_param_lp(n_vars: int, n_constraints: int):
    """
    Create LP with parametrized constraint matrix: A_param @ x <= b

    This is the most common DPP pattern and the main bottleneck case.
    Pattern: mul (parametrized matmul)
    """
    x = cp.Variable(n_vars)
    A_param = cp.Parameter((n_constraints, n_vars))
    b_param = cp.Parameter(n_constraints)
    c = np.random.randn(n_vars)

    prob = cp.Problem(
        cp.Minimize(c @ x),
        [A_param @ x <= b_param, x >= 0]
    )

    def init_params():
        A_param.value = np.random.randn(n_constraints, n_vars)
        b_param.value = np.random.randn(n_constraints) + 10

    return prob, init_params


def create_param_qp(n_vars: int, n_constraints: int):
    """
    Create QP with parametrized constraint matrix.
    Pattern: mul + quad_form
    """
    x = cp.Variable(n_vars)
    A_param = cp.Parameter((n_constraints, n_vars))
    b_param = cp.Parameter(n_constraints)
    P = np.eye(n_vars)  # Simple quadratic term
    c = np.random.randn(n_vars)

    prob = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(x, P) + c @ x),
        [A_param @ x <= b_param, x >= -10, x <= 10]
    )

    def init_params():
        A_param.value = np.random.randn(n_constraints, n_vars)
        b_param.value = np.random.randn(n_constraints) + 10

    return prob, init_params


def create_elementwise_param(n_vars: int, n_constraints: int):
    """
    Create LP with element-wise parameter scaling: diag(d) @ A @ x

    Pattern: mul_elem (element-wise multiplication with parameter)
    """
    x = cp.Variable(n_vars)
    d_param = cp.Parameter(n_constraints, nonneg=True)  # Scaling factors
    A = np.random.randn(n_constraints, n_vars)
    b = np.random.randn(n_constraints) + 10
    c = np.random.randn(n_vars)

    # Element-wise scaling of constraint rows
    prob = cp.Problem(
        cp.Minimize(c @ x),
        [cp.multiply(d_param.reshape((n_constraints, 1)), A @ x) <= b, x >= 0]
    )

    def init_params():
        d_param.value = np.abs(np.random.randn(n_constraints)) + 0.1

    return prob, init_params


def create_multi_param_lp(n_vars: int, n_constraints: int):
    """
    Create LP with multiple parameter matrices.

    Pattern: Multiple mul operations with different parameters
    """
    x = cp.Variable(n_vars)
    A1_param = cp.Parameter((n_constraints // 2, n_vars))
    A2_param = cp.Parameter((n_constraints // 2, n_vars))
    b1_param = cp.Parameter(n_constraints // 2)
    b2_param = cp.Parameter(n_constraints // 2)
    c = np.random.randn(n_vars)

    prob = cp.Problem(
        cp.Minimize(c @ x),
        [
            A1_param @ x <= b1_param,
            A2_param @ x <= b2_param,
            x >= 0
        ]
    )

    def init_params():
        A1_param.value = np.random.randn(n_constraints // 2, n_vars)
        A2_param.value = np.random.randn(n_constraints // 2, n_vars)
        b1_param.value = np.random.randn(n_constraints // 2) + 10
        b2_param.value = np.random.randn(n_constraints // 2) + 10

    return prob, init_params


def create_sum_param(n_vars: int, n_constraints: int):
    """
    Create problem with sum over parametrized expression.

    Pattern: sum_entries with parameter
    """
    x = cp.Variable(n_vars)
    A_param = cp.Parameter((n_constraints, n_vars))
    b = np.random.randn(n_constraints) + 10
    c_param = cp.Parameter(n_vars)

    prob = cp.Problem(
        cp.Minimize(cp.sum(cp.multiply(c_param, x))),  # Parametrized objective
        [A_param @ x <= b, x >= 0]
    )

    def init_params():
        A_param.value = np.random.randn(n_constraints, n_vars)
        c_param.value = np.random.randn(n_vars)

    return prob, init_params


def create_reshape_param(n_vars: int, n_constraints: int):
    """
    Create problem with reshape of parametrized expression.

    Pattern: reshape with parameter
    """
    # Make n_vars a perfect square for reshape
    side = int(np.sqrt(n_vars))
    n_vars = side * side

    X = cp.Variable((side, side))
    A_param = cp.Parameter((n_constraints, n_vars))
    b = np.random.randn(n_constraints) + 10
    c = np.random.randn(side, side)

    prob = cp.Problem(
        cp.Minimize(cp.sum(cp.multiply(c, X))),
        [A_param @ cp.reshape(X, (n_vars,)) <= b, X >= 0]
    )

    def init_params():
        A_param.value = np.random.randn(n_constraints, n_vars)

    return prob, init_params


def create_vstack_param(n_vars: int, n_constraints: int):
    """
    Create problem with vstack of parametrized constraints.

    Pattern: vstack/concatenate with parameters
    """
    x = cp.Variable(n_vars)
    n_blocks = 4
    block_size = n_constraints // n_blocks
    total_constraints = block_size * n_blocks  # Ensure exact division

    A_params = [cp.Parameter((block_size, n_vars)) for _ in range(n_blocks)]
    b = np.random.randn(total_constraints) + 10
    c = np.random.randn(n_vars)

    # Stack multiple parametrized constraints
    stacked_exprs = [cp.reshape(A @ x, (block_size, 1)) for A in A_params]
    stacked_A = cp.vstack(stacked_exprs)

    prob = cp.Problem(
        cp.Minimize(c @ x),
        [cp.reshape(stacked_A, (total_constraints,)) <= b, x >= 0]
    )

    def init_params():
        for A in A_params:
            A.value = np.random.randn(block_size, n_vars)

    return prob, init_params


def create_diag_param(n_vars: int, n_constraints: int):
    """
    Create problem with diagonal parameter matrix.

    Pattern: diag_vec with parameter
    """
    x = cp.Variable(n_vars)
    d_param = cp.Parameter(n_vars)  # Diagonal elements
    A = np.random.randn(n_constraints, n_vars)
    b = np.random.randn(n_constraints) + 10

    # D @ x where D = diag(d_param)
    prob = cp.Problem(
        cp.Minimize(cp.sum(cp.multiply(d_param, x))),
        [A @ x <= b, x >= 0]
    )

    def init_params():
        d_param.value = np.random.randn(n_vars)

    return prob, init_params


def create_nested_param(n_vars: int, n_constraints: int):
    """
    Create problem with nested parametrized operations.

    Pattern: A_param @ (B @ x + c) - nested mul
    """
    x = cp.Variable(n_vars)
    A_param = cp.Parameter((n_constraints, n_vars))
    B = np.random.randn(n_vars, n_vars)
    c_param = cp.Parameter(n_vars)
    b = np.random.randn(n_constraints) + 10

    prob = cp.Problem(
        cp.Minimize(cp.sum(x)),
        [A_param @ (B @ x + c_param) <= b, x >= -10, x <= 10]
    )

    def init_params():
        A_param.value = np.random.randn(n_constraints, n_vars)
        c_param.value = np.random.randn(n_vars)

    return prob, init_params


# Problem generator registry
PROBLEM_GENERATORS = {
    'param_lp': create_param_lp,
    'param_qp': create_param_qp,
    'elementwise': create_elementwise_param,
    'multi_param': create_multi_param_lp,
    'sum_param': create_sum_param,
    'reshape_param': create_reshape_param,
    'vstack_param': create_vstack_param,
    'diag_param': create_diag_param,
    'nested_param': create_nested_param,
}


# =============================================================================
# Benchmark Functions
# =============================================================================

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config: BenchmarkConfig
    cold_time_ms: float
    warm_time_ms: float
    peak_memory_mb: float = 0.0


def run_single_benchmark(config: BenchmarkConfig,
                         backend: str = 'SCIPY',
                         n_runs: int = 3) -> BenchmarkResult:
    """Run benchmark for a given configuration."""
    np.random.seed(42)

    # Create problem
    if config.problem_type == 'LP':
        prob, init_params = create_param_lp(config.n_vars, config.n_constraints)
        solver = cp.SCIPY
    else:
        prob, init_params = create_param_qp(config.n_vars, config.n_constraints)
        solver = cp.OSQP

    init_params()

    # Memory tracking
    peak_memory = 0.0
    if HAS_TRACEMALLOC:
        tracemalloc.start()

    # Cold path timing
    cold_times = []
    for _ in range(n_runs):
        prob._cache = type(prob._cache)()  # Clear cache
        init_params()

        start = time.perf_counter()
        prob.get_problem_data(solver, canon_backend=backend)
        elapsed = (time.perf_counter() - start) * 1000
        cold_times.append(elapsed)

    cold_time = np.median(cold_times)

    # Warm path timing
    warm_times = []
    for _ in range(n_runs):
        init_params()
        start = time.perf_counter()
        prob.get_problem_data(solver, canon_backend=backend)
        elapsed = (time.perf_counter() - start) * 1000
        warm_times.append(elapsed)

    warm_time = np.median(warm_times)

    # Memory stats
    if HAS_TRACEMALLOC:
        _, peak = tracemalloc.get_traced_memory()
        peak_memory = peak / (1024 * 1024)
        tracemalloc.stop()

    return BenchmarkResult(
        config=config,
        cold_time_ms=cold_time,
        warm_time_ms=warm_time,
        peak_memory_mb=peak_memory
    )


def run_scaling_analysis(configs: list[BenchmarkConfig], backend: str = 'SCIPY'):
    """Run scaling analysis across configurations."""
    print("=" * 80)
    print(f"DPP Canonicalization Scaling Analysis (Backend: {backend})")
    print("=" * 80)
    print()

    print(f"{'Config':<15} {'Params':>12} {'Cold (ms)':>12} {'Warm (ms)':>12} {'Memory (MB)':>12}")
    print("-" * 65)

    results = []
    for config in configs:
        try:
            result = run_single_benchmark(config, backend=backend)
            results.append(result)
            print(f"{config.name:<15} {config.param_size:>12,} {result.cold_time_ms:>12.1f} "
                  f"{result.warm_time_ms:>12.1f} {result.peak_memory_mb:>12.1f}")
        except MemoryError:
            print(f"{config.name:<15} {config.param_size:>12,} {'MEMORY ERROR':>12}")
            break
        except Exception as e:
            print(f"{config.name:<15} {config.param_size:>12,} ERROR: {e}")

    print()

    # Fit scaling
    if len(results) >= 3:
        params = np.array([r.config.param_size for r in results])
        times = np.array([r.cold_time_ms for r in results])

        log_params = np.log(params)
        log_times = np.log(times)
        b, log_a = np.polyfit(log_params, log_times, 1)

        print(f"Scaling: time ≈ O(n^{b:.2f})")
        if b < 1.2:
            print("  → Near-linear scaling")
        elif b < 1.5:
            print("  → Slightly superlinear")
        else:
            print("  → Superlinear (potential bottleneck)")

    return results


def run_compact_comparison():
    """
    Compare SCIPY backend vs COO backend for parametrized matmul.
    """
    print("=" * 80)
    print("SCIPY Backend vs COO Backend (DPP Canonicalization)")
    print("=" * 80)
    print()
    print("Compares full canonicalization time for parametrized A @ x problems.")
    print("SCIPY uses stacked sparse matrices: O(param_size × rows)")
    print("COO uses 3D sparse COO format: O(nnz)")
    print()

    configs = [
        (50, 50, 50),       # 2,500 params
        (100, 100, 100),    # 10K params
        (100, 500, 100),    # 50K params
        (200, 500, 100),    # 100K params
        (500, 500, 100),    # 250K params
        (1000, 500, 100),   # 500K params
        (1000, 1000, 100),  # 1M params
    ]

    print(f"{'Params':>12} {'Shape':>12} {'SCIPY (ms)':>12} {'COO (ms)':>12} {'Speedup':>10}")
    print("-" * 65)

    for m, k, n in configs:
        param_size = m * k
        try:
            t_scipy, t_coo, diff = benchmark_single_config(m, k, n)
            speedup = t_scipy / t_coo if t_coo > 0 else float('inf')
            shape = f'{m}x{k}'
            line = (f"{param_size:>12,} {shape:>12} {t_scipy:>12.1f} "
                    f"{t_coo:>12.1f} {speedup:>8.1f}x")
            print(line)
            if diff > 1e-10:
                print(f"  WARNING: max diff = {diff:.2e}")
        except MemoryError:
            shape = f'{m}x{k}'
            print(f"{param_size:>12,} {shape:>12} MEMORY ERROR")
            break
        except Exception as e:
            shape = f'{m}x{k}'
            print(f"{param_size:>12,} {shape:>12} ERROR: {e}")

    print()
    print("SCIPY = SciPyCanonBackend (stacked sparse matrices)")
    print("COO = COOCanonBackend (3D sparse COO format)")


def run_detailed_profile(config: BenchmarkConfig, backend: str = 'SCIPY'):
    """Run detailed profiling."""
    if not HAS_PROFILER:
        print("cProfile not available")
        return

    print("=" * 80)
    print(f"Detailed Profile: {config}")
    print("=" * 80)
    print()

    np.random.seed(42)

    if config.problem_type == 'LP':
        prob, init_params = create_param_lp(config.n_vars, config.n_constraints)
        solver = cp.SCIPY
    else:
        prob, init_params = create_param_qp(config.n_vars, config.n_constraints)
        solver = cp.OSQP

    init_params()
    prob._cache = type(prob._cache)()

    pr = cProfile.Profile()
    pr.enable()
    prob.get_problem_data(solver, canon_backend=backend)
    pr.disable()

    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())


def run_end_to_end_comparison():
    """
    Compare end-to-end canonicalization times: SCIPY vs COO backend.
    """
    print("=" * 80)
    print("End-to-End Analysis: SCIPY vs COO Backend")
    print("=" * 80)
    print()

    configs = [
        BenchmarkConfig("small", 100, 100),
        BenchmarkConfig("medium", 100, 500),
        BenchmarkConfig("large", 200, 500),
        BenchmarkConfig("xlarge", 500, 500),
    ]

    hdr = (f"{'Config':<12} {'Params':>10} {'SCIPY (ms)':>12} "
           f"{'COO (ms)':>12} {'Speedup':>10} {'MaxDiff':>12}")
    print(hdr)
    print("-" * 75)

    for config in configs:
        try:
            # Full canonicalization comparison
            m, k, n = config.n_constraints, config.n_vars, config.n_vars
            t_scipy, t_coo, max_diff = benchmark_single_config(m, k, n)
            speedup = t_scipy / t_coo if t_coo > 0 else float('inf')

            print(f"{config.name:<12} {config.param_size:>10,} {t_scipy:>12.1f} "
                  f"{t_coo:>12.1f} {speedup:>9.1f}x {max_diff:>12.2e}")
        except Exception as e:
            print(f"{config.name:<12} ERROR: {e}")

    print()
    print("MaxDiff = max absolute difference in constraint matrix A")
    print("Speedup > 1 means COO is faster")


def run_pattern_benchmarks(n_vars: int = 100, n_constraints: int = 200):
    """
    Benchmark all problem patterns at a fixed size.
    """
    print("=" * 80)
    print(f"Problem Pattern Benchmarks ({n_vars} vars, {n_constraints} constraints)")
    print("=" * 80)
    print()

    print(f"{'Pattern':<20} {'Cold (ms)':>12} {'Warm (ms)':>12} {'Description'}")
    print("-" * 70)

    pattern_descriptions = {
        'param_lp': 'A_param @ x <= b',
        'param_qp': 'quad_form + A_param @ x',
        'elementwise': 'd_param * (A @ x)',
        'multi_param': 'A1 @ x, A2 @ x (2 params)',
        'sum_param': 'sum(c_param * x)',
        'reshape_param': 'A @ reshape(X)',
        'vstack_param': 'vstack([A1@x, A2@x, ...])',
        'diag_param': 'sum(d_param * x)',
        'nested_param': 'A_param @ (B @ x + c)',
    }

    # Map patterns to appropriate solvers
    pattern_solvers = {
        'param_qp': cp.OSQP,
    }

    results = {}
    for pattern_name, generator in PROBLEM_GENERATORS.items():
        try:
            prob, init_params = generator(n_vars, n_constraints)
            init_params()

            solver = pattern_solvers.get(pattern_name, cp.SCIPY)

            # Cold path
            prob._cache = type(prob._cache)()
            start = time.perf_counter()
            prob.get_problem_data(solver, canon_backend='SCIPY')
            cold_time = (time.perf_counter() - start) * 1000

            # Warm path
            init_params()
            start = time.perf_counter()
            prob.get_problem_data(solver, canon_backend='SCIPY')
            warm_time = (time.perf_counter() - start) * 1000

            desc = pattern_descriptions.get(pattern_name, '')
            print(f"{pattern_name:<20} {cold_time:>12.1f} {warm_time:>12.1f} {desc}")
            results[pattern_name] = (cold_time, warm_time)
        except Exception as e:
            print(f"{pattern_name:<20} ERROR: {str(e)[:40]}")

    return results


def run_pattern_scaling(pattern: str = 'param_lp'):
    """
    Run scaling analysis for a specific problem pattern.
    """
    if pattern not in PROBLEM_GENERATORS:
        print(f"Unknown pattern: {pattern}")
        print(f"Available: {list(PROBLEM_GENERATORS.keys())}")
        return

    print("=" * 80)
    print(f"Scaling Analysis: {pattern}")
    print("=" * 80)
    print()

    generator = PROBLEM_GENERATORS[pattern]

    configs = [
        (50, 50),
        (100, 100),
        (100, 500),
        (200, 500),
        (500, 500),
        (1000, 500),
    ]

    print(f"{'Size':>15} {'Params':>12} {'Cold (ms)':>12} {'Warm (ms)':>12}")
    print("-" * 55)

    for n_vars, n_constraints in configs:
        try:
            prob, init_params = generator(n_vars, n_constraints)
            init_params()

            # Cold path
            prob._cache = type(prob._cache)()
            start = time.perf_counter()
            prob.get_problem_data(cp.SCIPY, canon_backend='SCIPY')
            cold_time = (time.perf_counter() - start) * 1000

            # Warm path
            init_params()
            start = time.perf_counter()
            prob.get_problem_data(cp.SCIPY, canon_backend='SCIPY')
            warm_time = (time.perf_counter() - start) * 1000

            param_size = n_vars * n_constraints
            shape = f'{n_vars}x{n_constraints}'
            print(f"{shape:>15} {param_size:>12,} {cold_time:>12.1f} {warm_time:>12.1f}")
        except Exception as e:
            shape = f'{n_vars}x{n_constraints}'
            print(f"{shape:>15} ERROR: {str(e)[:30]}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='DPP Canonicalization Benchmarks')
    parser.add_argument('--quick', action='store_true', help='Quick benchmarks only')
    parser.add_argument('--scaling', action='store_true', help='Scaling analysis')
    parser.add_argument('--compact', action='store_true', help='CompactTensor comparison')
    parser.add_argument('--profile', action='store_true', help='Detailed profiling')
    parser.add_argument('--e2e', action='store_true', help='End-to-end comparison')
    parser.add_argument('--patterns', action='store_true', help='Benchmark all problem patterns')
    parser.add_argument('--pattern', type=str, help='Scaling for specific pattern')
    parser.add_argument('--all', action='store_true', help='Run all benchmarks')
    args = parser.parse_args()

    if args.all:
        args.scaling = True
        args.compact = True
        args.e2e = True
        args.patterns = True

    configs = QUICK_CONFIGS if args.quick else FULL_CONFIGS

    print()
    print("CVXPY DPP Canonicalization Benchmarks")
    print("=" * 80)
    print()

    # Main scaling analysis
    run_scaling_analysis(configs)

    if args.scaling:
        print()
        run_scaling_analysis(SCALING_CONFIGS)

    if args.patterns:
        print()
        run_pattern_benchmarks()

    if args.pattern:
        print()
        run_pattern_scaling(args.pattern)

    if args.compact:
        print()
        run_compact_comparison()

    if args.e2e:
        print()
        run_end_to_end_comparison()

    if args.profile:
        print()
        profile_config = configs[min(2, len(configs)-1)]
        run_detailed_profile(profile_config)


if __name__ == "__main__":
    main()
