import numpy as np

import cvxpy as cp


def test_rsoc_basic():
    n = 3
    x = cp.Variable(n)
    y = cp.Variable()
    z = cp.Variable()

    constraints = [
        cp.RSOC(x, y, z),
        y == 1,
        z == 1
    ]

    prob = cp.Problem(cp.Minimize(cp.norm(x)), constraints)
    prob.solve()

    assert prob.status == cp.OPTIMAL
    assert np.allclose(x.value, np.zeros(n), atol=1e-6)


def test_rsoc_equivalence_to_quad_over_lin():
    n = 3
    x = cp.Variable(n)
    y = cp.Variable(nonneg=True)
    z = cp.Variable()

    constraints1 = [cp.RSOC(x, y, z), y == 1]
    constraints2 = [cp.quad_over_lin(x, y) <= z, y == 1]

    prob1 = cp.Problem(cp.Minimize(z), constraints1)
    prob2 = cp.Problem(cp.Minimize(z), constraints2)

    val1 = prob1.solve()
    val2 = prob2.solve()

    assert abs(val1 - val2) <= 1e-5
