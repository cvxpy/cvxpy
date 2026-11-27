import numpy as np

import cvxpy as cp


def test_clarabel_warm_start_sparsity_change():
    """
    Test that Clarabel gracefully falls back to new_solver() if sparsity changes
    during a warm start, preventing a Data formatting error. (Issue #2800)
    """
    x = cp.Variable(2)
    p = cp.Parameter(2)

    # Using sum so dropping a parameter to 0 remains mathematically feasible
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [cp.sum(cp.multiply(p, x)) >= 1])

    # Solve with fully dense parameter
    p.value = np.array([1.0, 1.0])
    prob.solve(solver=cp.CLARABEL, warm_start=True)

    # Introduce a zero to change the sparsity pattern of the data passed to Clarabel
    p.value = np.array([1.0, 0.0])

    # With the fix in PR #3225, this will gracefully fall back to a new solver
    # instead of crashing with a "Data formatting error"
    prob.solve(solver=cp.CLARABEL, warm_start=True)

    assert prob.status == cp.OPTIMAL
