"""
Copyright 2013 Steven Diamond, 2017 Akshay Agrawal

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

import numpy as np
import pytest

import cvxpy as cp
from cvxpy.problems.validation import (
    _constraint_violation_to_scalar,
    _max_constraint_violation,
)


def test_constraint_violation_to_scalar_for_equality_vector_residual():
    x = cp.Variable(2)
    constraint = x == np.array([0.0, 0.0])

    x.value = np.array([3.0, -4.0])

    np.testing.assert_allclose(constraint.violation(), np.array([3.0, 4.0]))
    assert _constraint_violation_to_scalar(constraint) == pytest.approx(5.0)


def test_constraint_violation_to_scalar_for_upper_bound_inequality():
    x = cp.Variable(2)
    constraint = x <= np.array([0.0, 0.0])

    x.value = np.array([3.0, -4.0])

    np.testing.assert_allclose(constraint.violation(), np.array([3.0, 0.0]))
    assert _constraint_violation_to_scalar(constraint) == pytest.approx(3.0)


def test_constraint_violation_to_scalar_for_lower_bound_inequality():
    x = cp.Variable(2)
    constraint = x >= np.array([0.0, 0.0])

    x.value = np.array([3.0, -4.0])

    np.testing.assert_allclose(constraint.violation(), np.array([0.0, 4.0]))
    assert _constraint_violation_to_scalar(constraint) == pytest.approx(4.0)


def test_max_constraint_violation_returns_largest_scalar_violation():
    x = cp.Variable(2)

    constraints = [
        x == np.array([0.0, 0.0]),
        x <= np.array([0.0, 0.0]),
        x >= np.array([0.0, 0.0]),
    ]

    x.value = np.array([3.0, -4.0])
    assert _max_constraint_violation(constraints) == pytest.approx(5.0)


def test_max_constraint_violation_empty_constraints():
    assert _max_constraint_violation([]) == 0.0


def test_constraint_violation_to_scalar_raises_when_value_missing():
    x = cp.Variable()
    constraint = x >= 0

    with pytest.raises(ValueError):
        _constraint_violation_to_scalar(constraint)
