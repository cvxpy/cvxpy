"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import copy

import numpy as np

import cvxpy as cp
from cvxpy.constraints import Equality


def test_leaf():
    a = cp.Variable()
    b = copy.copy(a)
    c = copy.deepcopy(a)

    assert a.id == b.id
    assert a.id != c.id

    assert id(a) == id(b)
    assert id(a) != id(c)


def test_constraint():
    x = cp.Variable()

    a = Equality(x, 0)
    b = copy.copy(a)
    c = copy.deepcopy(a)

    assert a.id == b.id
    assert a.id != c.id

    assert id(a) != id(b)
    assert id(a) != id(c)
    assert id(b) != id(c)


def test_expression():
    x = cp.Variable()

    a = x + 1
    b = copy.copy(a)
    c = copy.deepcopy(a)

    assert a.id != b.id
    assert a.id != c.id

    assert id(a) != id(b)
    assert id(a) != id(c)
    assert id(b) != id(c)


def test_problem():
    x = cp.Variable()
    y = cp.Variable()
    obj = cp.Minimize((x + y) ** 2)
    constraints = [x + y == 1]
    prob = cp.Problem(obj, constraints)
    prob_copy = copy.copy(prob)
    prob_deepcopy = copy.deepcopy(prob)

    assert id(prob) != id(prob_copy)
    assert id(prob) != id(prob_deepcopy)
    assert id(prob_copy) != id(prob_deepcopy)

    prob.solve()
    assert prob.status == cp.OPTIMAL

    prob_copy.solve()
    assert prob_copy.status == cp.OPTIMAL

    prob_deepcopy.solve()
    assert prob_deepcopy.status == cp.OPTIMAL


def test_constraints_in_problem():

    x = cp.Variable(name='x', nonneg=True)
    y = cp.Variable(name='y', nonneg=True)

    original_constraints = [
        x + y == 1
    ]
    shallow_constraints = copy.copy(original_constraints)

    obj = cp.Maximize((x + 2 * y))
    prob = cp.Problem(obj, shallow_constraints)
    prob.solve()
    assert prob.status == cp.OPTIMAL
    assert np.allclose(x.value, 0)
    assert np.allclose(y.value, 1)

    deep_constraints = copy.deepcopy(original_constraints)
    prob = cp.Problem(obj, deep_constraints)
    prob.solve()
    # The constraints refer to the original variables, so the problem is unbounded
    assert prob.status == cp.UNBOUNDED

    x_copied = deep_constraints[0].variables()[0]
    y_copied = deep_constraints[0].variables()[1]

    deep_obj = cp.Maximize((x_copied + 2 * y_copied))
    prob = cp.Problem(deep_obj, deep_constraints)
    prob.solve()
    # Can get back the same solution by using copied variables
    assert prob.status == cp.OPTIMAL
    assert np.allclose(x_copied.value, 0)
    assert np.allclose(y_copied.value, 1)
