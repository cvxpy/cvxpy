import time

import numpy as np

import cvxpy as cp


def max_constraint_violation(prob):
    """Compute max violation over all original constraints."""
    max_violation = 0.0

    for constraint in prob.constraints:
        violation = constraint.violation()
        max_violation = max(
            max_violation,
            float(np.max(np.asarray(violation))),
        )

    return max_violation


def benchmark_vectorized(n):
    """One constraint object: x >= 0."""
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [x >= 0])

    prob.solve(solver=cp.CLARABEL)

    start = time.perf_counter()
    violation = max_constraint_violation(prob)
    elapsed = time.perf_counter() - start

    return elapsed, violation, len(prob.constraints)


def benchmark_many_scalar(n):
    """Many constraint objects: x[i] >= 0 for each i."""
    x = cp.Variable(n)
    constraints = [x[i] >= 0 for i in range(n)]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), constraints)

    prob.solve(solver=cp.CLARABEL)

    start = time.perf_counter()
    violation = max_constraint_violation(prob)
    elapsed = time.perf_counter() - start

    return elapsed, violation, len(prob.constraints)


if __name__ == "__main__":
    sizes = [100, 1_000]

    for n in sizes:
        print(f"\n=== n = {n} ===")

        vec_time, vec_viol, vec_num_constraints = benchmark_vectorized(n)
        print("Vectorized:")
        print(f"  constraints: {vec_num_constraints}")
        print(f"  check time:  {vec_time:.6f} seconds")
        print(f"  violation:   {vec_viol:.2e}")

        scalar_time, scalar_viol, scalar_num_constraints = benchmark_many_scalar(n)
        print("Many scalar:")
        print(f"  constraints: {scalar_num_constraints}")
        print(f"  check time:  {scalar_time:.6f} seconds")
        print(f"  violation:   {scalar_viol:.2e}")

        print("Ratio:")
        print(f"  scalar/vectorized time: {scalar_time / vec_time:.1f}x")
