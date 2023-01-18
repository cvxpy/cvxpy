"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass

import numpy as np
import pytest

import cvxpy as cp


@dataclass
class MockMutableObject:
    """
    A mock mutable object for testing deepcopy.
    """
    x: cp.Variable
    P: list[list[int]]  # requires deepcopy


@pytest.mark.parametrize("copy_func", [copy, deepcopy])
def test_valid_deepcopy_example(copy_func):
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

    # Deepcopy should work
    obj_copy = copy_func(obj)
    obj_copy.P[0][0] = 2

    problem2 = cp.Problem(cp.Minimize(cp.quad_form(obj_copy.x, obj_copy.P)), [obj_copy.x == 1])
    problem2.solve()
    assert problem2.status == cp.OPTIMAL
    assert np.isclose(problem2.value, 3)

    problem1 = cp.Problem(cp.Minimize(cp.quad_form(obj.x, obj.P)), [obj.x == 1])
    problem1.solve()
    assert problem1.status == cp.OPTIMAL
    if copy_func == deepcopy:
        # Original problem not affected by deepcopy
        assert np.isclose(problem1.value, 2)
    else:
        # Original problem affected by shallow copy
        assert np.isclose(problem1.value, 3)


def test_deepcopy_same_identity():
    x = cp.Variable(nonneg=True, name="x")
    y = deepcopy(x)

    # Leafs keep their identity (id()), as well as their name and .id
    assert y.name() == "x"
    assert y.id == x.id
    assert id(y) == id(x)

    constraints = [x >= y + 1]

    copied_constraint = deepcopy(constraints[0])
    # Constraints change their identity (id()), but their .id stays the same
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
