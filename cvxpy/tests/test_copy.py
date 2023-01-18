from copy import deepcopy, copy
from dataclasses import dataclass

import pytest
import cvxpy as cp
import numpy as np


@dataclass
class MockMutableObject:
    """
    A mock mutable object for testing deepcopy.
    """
    x: cp.Variable
    P: list[list[int]]  # requires deepcopy



def test_valid_deepcopy_example():
    """
    Example where deepcopy is required.
    Even though for the CVXPY expression, deepcopy is not required, it is required for the
    mutable object.
    """
    P = [[1, 0], [0, 1]]
    x = cp.Variable(2)
    obj = MockMutableObject(x, P)

    problem1 = cp.Problem(cp.Minimize(cp.quad_form(obj.x, obj.P)), [obj.x == 1])
    problem1.solve()
    assert problem1.status == cp.OPTIMAL
    assert np.isclose(problem1.value, 2)

    # Deepcopy should work.
    obj_copy = deepcopy(obj)
    obj_copy.P[0][0] = 2

    problem2 = cp.Problem(cp.Minimize(cp.quad_form(obj_copy.x, obj_copy.P)), [obj_copy.x == 1])
    problem2.solve()
    assert problem2.status == cp.OPTIMAL
    assert np.isclose(problem2.value, 3)

    # Original problem should not be affected.
    problem1.solve()
    assert problem1.status == cp.OPTIMAL
    assert np.isclose(problem1.value, 2)


def test_deepcopy_same_identity():
    x = cp.Variable(nonneg=True, name="x")
    y = deepcopy(x)

    # Leafs keep their identity (id()), as well as their name and .id
    assert y.name() == "x"
    assert y.id == x.id
    assert id(y) == id(x)

    constraints = [x >= y + 1]

    copied_constraint = deepcopy(constraints[0])
    # Other expressions change their identity (id()), but their .id stays the same
    assert copied_constraint.id == constraints[0].id
    assert id(copied_constraint) != id(constraints[0])

    problem = cp.Problem(cp.Minimize(0), constraints)
    problem.solve()
    assert problem.status == cp.INFEASIBLE


@pytest.mark.parametrize("copy_func", [copy, deepcopy])
def test_copy_constraints(copy_func) -> None:
    """
    Test copy and deepcopy of constraints.
    """
    x = cp.Variable()
    y = cp.Variable()

    constraints = [
        x + y == 1,
        x - y >= 1
    ]
    constraints[0].atoms()
    constraints = copy_func(constraints)

    obj = cp.Minimize((x - y) ** 2)
    prob = cp.Problem(obj, constraints)
    prob.solve()
    assert prob.status == cp.OPTIMAL
    assert np.allclose(x.value, 1)
    assert np.allclose(y.value, 0)
