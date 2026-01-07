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

Unit tests for the NumPy-style ``cp.stack`` helper. Each test focuses on a
single aspect of the API contract so failures clearly indicate which axis
handling or validation rule regressed.
"""

import numpy as np
import pytest

import cvxpy as cp


def test_stack_1d_axis0() -> None:
    """Two 1-D Parameters stack along axis 0 to form a 2x4 array."""
    a = cp.Parameter((4,))
    b = cp.Parameter((4,))
    y = cp.stack([a, b], axis=0)
    assert y.shape == (2, 4)


def test_stack_1d_axis_last_numeric_parity() -> None:
    """Scalar arrays stacked along the last axis match NumPy numerically."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    y = cp.stack([a, b], axis=-1)
    assert y.shape == (3, 2)
    expected = np.stack([a, b], axis=-1)
    assert np.allclose(y.value, expected)


def test_stack_2d_various_axes() -> None:
    """Validate axis normalization on 2-D operands for several positions."""
    a = cp.Parameter((3, 4))
    b = cp.Parameter((3, 4))
    assert cp.stack([a, b], axis=0).shape == (2, 3, 4)
    assert cp.stack([a, b], axis=1).shape == (3, 2, 4)
    assert cp.stack([a, b], axis=-1).shape == (3, 4, 2)


def test_stack_scalar_inputs() -> None:
    """Literal scalars auto-wrap into Constants and stack as a 1-D vector."""
    y = cp.stack([1, 2, 3], axis=0)
    assert y.shape == (3,)
    assert np.allclose(y.value, np.array([1, 2, 3]))


def test_stack_shape_mismatch_raises() -> None:
    """Inputs of different shapes trigger the NumPy-style ValueError."""
    a = cp.Parameter((3,))
    b = cp.Parameter((4,))
    with pytest.raises(ValueError):
        cp.stack([a, b], axis=0)


def test_stack_empty_list_raises() -> None:
    """An empty input sequence is rejected."""
    with pytest.raises(ValueError):
        cp.stack([], axis=0)


def test_stack_axis_bounds_check() -> None:
    """Axis validation mirrors NumPy bounds for the resulting ndim."""
    a = cp.Parameter((3,))
    # ndim=1 -> result_ndim=2; valid axes: -2, -1, 0, 1
    for good in (-2, -1, 0, 1):
        cp.stack([a, a], axis=good)
    for bad in (-3, 2, 3):
        with pytest.raises(ValueError):
            cp.stack([a, a], axis=bad)


def test_stack_non_int_axis_raises() -> None:
    """Non-integer axes raise a TypeError before shape checks run."""
    a = cp.Parameter((2,))
    with pytest.raises(TypeError):
        cp.stack([a, a], axis=0.5)


def test_stack_variables_shape_only() -> None:
    """Variables share shape metadata with the stacked expression."""
    x = cp.Variable((2, 3))
    y = cp.Variable((2, 3))
    z = cp.stack([x, y], axis=0)
    assert z.shape == (2, 2, 3)


def test_stack_canonicalization_resolves_equalities() -> None:
    """Canonicalization maps scalar variables onto the stacked vector."""
    x = cp.Variable()
    y = cp.Variable()
    z = cp.Variable(2)
    z_tilde = cp.stack([x, y])
    problem = cp.Problem(cp.Minimize(0), [z_tilde == z, x == 1, y == 2])
    problem.solve(solver=cp.SCS)
    assert problem.status == cp.OPTIMAL
    assert np.allclose(z.value, np.array([1.0, 2.0]))
