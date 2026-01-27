"""
Copyright 2013 Steven Diamond

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

import cvxpy as cp


class TestBounds:
    """Tests for bounds propagation through expression trees."""

    def test_variable_no_bounds(self) -> None:
        """Test unbounded variable returns (-inf, inf)."""
        x = cp.Variable(3)
        lb, ub = x.get_bounds()
        assert lb.shape == (3,)
        assert ub.shape == (3,)
        assert np.all(lb == -np.inf)
        assert np.all(ub == np.inf)

    def test_variable_with_bounds(self) -> None:
        """Test variable with explicit bounds."""
        x = cp.Variable(3, bounds=[0, 1])
        lb, ub = x.get_bounds()
        assert np.allclose(lb, 0)
        assert np.allclose(ub, 1)

    def test_variable_with_tuple_bounds(self) -> None:
        """Test variable with tuple bounds."""
        x = cp.Variable(3, bounds=(0, 1))
        lb, ub = x.get_bounds()
        assert np.allclose(lb, 0)
        assert np.allclose(ub, 1)

    def test_variable_nonneg(self) -> None:
        """Test nonneg variable has lower bound of 0."""
        x = cp.Variable(3, nonneg=True)
        lb, ub = x.get_bounds()
        assert np.allclose(lb, 0)
        assert np.all(ub == np.inf)

    def test_variable_nonpos(self) -> None:
        """Test nonpos variable has upper bound of 0."""
        x = cp.Variable(3, nonpos=True)
        lb, ub = x.get_bounds()
        assert np.all(lb == -np.inf)
        assert np.allclose(ub, 0)

    def test_variable_boolean(self) -> None:
        """Test boolean variable has bounds [0, 1]."""
        x = cp.Variable(3, boolean=True)
        lb, ub = x.get_bounds()
        assert np.allclose(lb, 0)
        assert np.allclose(ub, 1)

    def test_constant_bounds(self) -> None:
        """Test constant has exact bounds."""
        c = cp.Constant([1, 2, 3])
        lb, ub = c.get_bounds()
        assert np.allclose(lb, [1, 2, 3])
        assert np.allclose(ub, [1, 2, 3])

    def test_parameter_with_bounds(self) -> None:
        """Test parameter with explicit bounds."""
        p = cp.Parameter(3, bounds=[-1, 1])
        lb, ub = p.get_bounds()
        assert np.allclose(lb, -1)
        assert np.allclose(ub, 1)

    def test_negation(self) -> None:
        """Test bounds propagation through negation."""
        x = cp.Variable(3, bounds=[1, 2])
        expr = -x
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, -2)
        assert np.allclose(ub, -1)

    def test_addition(self) -> None:
        """Test bounds propagation through addition."""
        x = cp.Variable(3, bounds=[0, 1])
        y = cp.Variable(3, bounds=[2, 3])
        expr = x + y
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, 2)
        assert np.allclose(ub, 4)

    def test_scalar_multiplication(self) -> None:
        """Test bounds propagation through scalar multiplication."""
        x = cp.Variable(3, bounds=[1, 2])
        expr = 2 * x
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, 2)
        assert np.allclose(ub, 4)

        # Test negative scalar
        expr2 = -2 * x
        lb2, ub2 = expr2.get_bounds()
        assert np.allclose(lb2, -4)
        assert np.allclose(ub2, -2)

    def test_elementwise_multiplication(self) -> None:
        """Test bounds propagation through elementwise multiplication."""
        x = cp.Variable(3, bounds=[1, 2])
        y = cp.Variable(3, bounds=[3, 4])
        expr = cp.multiply(x, y)
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, 3)
        assert np.allclose(ub, 8)

    def test_multiply_spanning_zero(self) -> None:
        """Test multiplication with intervals spanning zero."""
        x = cp.Variable(3, bounds=[-1, 2])
        y = cp.Variable(3, bounds=[-3, 4])
        expr = cp.multiply(x, y)
        lb, ub = expr.get_bounds()
        # Products: (-1)*(-3)=3, (-1)*4=-4, 2*(-3)=-6, 2*4=8
        assert np.allclose(lb, -6)
        assert np.allclose(ub, 8)

    def test_sum(self) -> None:
        """Test bounds propagation through sum."""
        x = cp.Variable((2, 3), bounds=[1, 2])
        expr = cp.sum(x)
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, 6)  # 2*3*1 = 6
        assert np.allclose(ub, 12)  # 2*3*2 = 12

    def test_sum_axis(self) -> None:
        """Test bounds propagation through sum with axis."""
        x = cp.Variable((2, 3), bounds=[1, 2])
        expr = cp.sum(x, axis=0)
        lb, ub = expr.get_bounds()
        assert lb.shape == (3,)
        assert np.allclose(lb, 2)  # 2*1 = 2
        assert np.allclose(ub, 4)  # 2*2 = 4

    def test_abs_nonneg(self) -> None:
        """Test bounds for abs of non-negative expression."""
        x = cp.Variable(3, bounds=[1, 2])
        expr = cp.abs(x)
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, 1)
        assert np.allclose(ub, 2)

    def test_abs_nonpos(self) -> None:
        """Test bounds for abs of non-positive expression."""
        x = cp.Variable(3, bounds=[-3, -1])
        expr = cp.abs(x)
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, 1)
        assert np.allclose(ub, 3)

    def test_abs_spanning_zero(self) -> None:
        """Test bounds for abs when argument spans zero."""
        x = cp.Variable(3, bounds=[-2, 3])
        expr = cp.abs(x)
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, 0)
        assert np.allclose(ub, 3)

    def test_exp(self) -> None:
        """Test bounds propagation through exp."""
        x = cp.Variable(3, bounds=[0, 1])
        expr = cp.exp(x)
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, 1)
        assert np.allclose(ub, np.e)

    def test_log(self) -> None:
        """Test bounds propagation through log."""
        x = cp.Variable(3, bounds=[1, np.e])
        expr = cp.log(x)
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, 0)
        assert np.allclose(ub, 1)

    def test_power(self) -> None:
        """Test bounds propagation through power."""
        x = cp.Variable(3, bounds=[1, 2])
        expr = x ** 2
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, 1)
        assert np.allclose(ub, 4)

    def test_maximum(self) -> None:
        """Test bounds propagation through maximum."""
        x = cp.Variable(3, bounds=[1, 2])
        y = cp.Variable(3, bounds=[0, 3])
        expr = cp.maximum(x, y)
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, 1)
        assert np.allclose(ub, 3)

    def test_minimum(self) -> None:
        """Test bounds propagation through minimum."""
        x = cp.Variable(3, bounds=[1, 2])
        y = cp.Variable(3, bounds=[0, 3])
        expr = cp.minimum(x, y)
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, 0)
        assert np.allclose(ub, 2)

    def test_max_reduction(self) -> None:
        """Test bounds propagation through max reduction."""
        x = cp.Variable((2, 3), bounds=[1, 5])
        expr = cp.max(x)
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, 1)
        assert np.allclose(ub, 5)

    def test_min_reduction(self) -> None:
        """Test bounds propagation through min reduction."""
        x = cp.Variable((2, 3), bounds=[1, 5])
        expr = cp.min(x)
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, 1)
        assert np.allclose(ub, 5)

    def test_norm1(self) -> None:
        """Test bounds propagation through norm1."""
        x = cp.Variable(3, bounds=[1, 2])
        expr = cp.norm1(x)
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, 3)  # 3 * 1
        assert np.allclose(ub, 6)  # 3 * 2

    def test_norm1_spanning_zero(self) -> None:
        """Test bounds for norm1 when argument spans zero."""
        x = cp.Variable(3, bounds=[-2, 1])
        expr = cp.norm1(x)
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, 0)
        assert np.allclose(ub, 6)  # 3 * 2

    def test_norm_inf(self) -> None:
        """Test bounds propagation through norm_inf."""
        x = cp.Variable(3, bounds=[1, 2])
        expr = cp.norm_inf(x)
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, 1)
        assert np.allclose(ub, 2)

    def test_composed_expression(self) -> None:
        """Test bounds propagation through composed expressions."""
        x = cp.Variable(bounds=[0, 1])
        expr = 2 * x + 1
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, 1)
        assert np.allclose(ub, 3)

    def test_complex_expression(self) -> None:
        """Test bounds propagation through a more complex expression."""
        x = cp.Variable(3, bounds=[0, 1])
        y = cp.Variable(3, bounds=[1, 2])
        expr = cp.abs(x - y)
        lb, ub = expr.get_bounds()
        # x - y has bounds [-2, 0]
        # abs(x - y) has bounds [0, 2]
        assert np.allclose(lb, 0)
        assert np.allclose(ub, 2)

    def test_sign_refinement(self) -> None:
        """Test that sign information refines bounds."""
        x = cp.Variable(3, nonneg=True)  # lb = 0
        expr = cp.abs(x)  # abs of nonneg is nonneg, so still [0, inf]
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, 0)
        assert np.all(ub == np.inf)

    def test_matrix_variable_bounds(self) -> None:
        """Test bounds for matrix variables."""
        X = cp.Variable((2, 3), bounds=[0, 5])
        lb, ub = X.get_bounds()
        assert lb.shape == (2, 3)
        assert ub.shape == (2, 3)
        assert np.allclose(lb, 0)
        assert np.allclose(ub, 5)

    def test_division_positive(self) -> None:
        """Test bounds for division with positive divisor."""
        x = cp.Variable(3, bounds=[2, 4])
        y = cp.Variable(3, bounds=[1, 2])
        expr = x / y
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, 1)  # 2/2 = 1
        assert np.allclose(ub, 4)  # 4/1 = 4


class TestBoundsUtilities:
    """Tests for the bounds utility functions."""

    def test_unbounded(self) -> None:
        """Test unbounded function."""
        from cvxpy.utilities.bounds import unbounded
        lb, ub = unbounded((2, 3))
        assert lb.shape == (2, 3)
        assert ub.shape == (2, 3)
        assert np.all(lb == -np.inf)
        assert np.all(ub == np.inf)

    def test_add_bounds(self) -> None:
        """Test add_bounds function."""
        from cvxpy.utilities.bounds import add_bounds
        lb1 = np.array([1, 2])
        ub1 = np.array([3, 4])
        lb2 = np.array([5, 6])
        ub2 = np.array([7, 8])
        lb, ub = add_bounds(lb1, ub1, lb2, ub2)
        assert np.allclose(lb, [6, 8])
        assert np.allclose(ub, [10, 12])

    def test_neg_bounds(self) -> None:
        """Test neg_bounds function."""
        from cvxpy.utilities.bounds import neg_bounds
        lb = np.array([1, 2])
        ub = np.array([3, 4])
        new_lb, new_ub = neg_bounds(lb, ub)
        # neg_bounds returns (-ub, -lb)
        assert np.allclose(new_lb, [-3, -4])
        assert np.allclose(new_ub, [-1, -2])

    def test_mul_bounds(self) -> None:
        """Test mul_bounds function."""
        from cvxpy.utilities.bounds import mul_bounds
        # Positive intervals
        lb1 = np.array([1, 2])
        ub1 = np.array([3, 4])
        lb2 = np.array([5, 6])
        ub2 = np.array([7, 8])
        lb, ub = mul_bounds(lb1, ub1, lb2, ub2)
        assert np.allclose(lb, [5, 12])
        assert np.allclose(ub, [21, 32])

    def test_abs_bounds(self) -> None:
        """Test abs_bounds function."""
        from cvxpy.utilities.bounds import abs_bounds
        # Spanning zero
        lb = np.array([-2, 1])
        ub = np.array([3, 2])
        new_lb, new_ub = abs_bounds(lb, ub)
        assert np.allclose(new_lb, [0, 1])
        assert np.allclose(new_ub, [3, 2])

    def test_sum_bounds(self) -> None:
        """Test sum_bounds function."""
        from cvxpy.utilities.bounds import sum_bounds
        lb = np.array([[1, 2], [3, 4]])
        ub = np.array([[5, 6], [7, 8]])
        new_lb, new_ub = sum_bounds(lb, ub, axis=None)
        assert np.allclose(new_lb, 10)
        assert np.allclose(new_ub, 26)

    def test_max_reduction_bounds(self) -> None:
        """Test max_reduction_bounds function."""
        from cvxpy.utilities.bounds import max_reduction_bounds
        lb = np.array([[1, 2], [3, 4]])
        ub = np.array([[5, 6], [7, 8]])
        new_lb, new_ub = max_reduction_bounds(lb, ub, axis=None)
        assert np.allclose(new_lb, 4)
        assert np.allclose(new_ub, 8)

    def test_min_reduction_bounds(self) -> None:
        """Test min_reduction_bounds function."""
        from cvxpy.utilities.bounds import min_reduction_bounds
        lb = np.array([[1, 2], [3, 4]])
        ub = np.array([[5, 6], [7, 8]])
        new_lb, new_ub = min_reduction_bounds(lb, ub, axis=None)
        assert np.allclose(new_lb, 1)
        assert np.allclose(new_ub, 5)


class TestExtractBounds:
    """Tests for extract_lower_bounds / extract_upper_bounds in matrix_stuffing."""

    def test_nonneg_with_explicit_bounds(self) -> None:
        """Nonneg + explicit bounds should use the tighter lower bound."""
        from cvxpy.reductions.matrix_stuffing import (
            extract_lower_bounds,
            extract_upper_bounds,
        )

        x = cp.Variable(3, nonneg=True, bounds=[2, 5])
        variables = [x]
        var_size = x.size

        lb = extract_lower_bounds(variables, var_size)
        ub = extract_upper_bounds(variables, var_size)

        # The lower bound should be 2 (tighter than nonneg's 0).
        assert lb is not None
        assert np.allclose(lb, 2)
        # The upper bound should be 5.
        assert ub is not None
        assert np.allclose(ub, 5)

    def test_nonpos_with_explicit_bounds(self) -> None:
        """Nonpos + explicit bounds should use the tighter upper bound."""
        from cvxpy.reductions.matrix_stuffing import (
            extract_lower_bounds,
            extract_upper_bounds,
        )

        x = cp.Variable(3, nonpos=True, bounds=[-5, -2])
        variables = [x]
        var_size = x.size

        lb = extract_lower_bounds(variables, var_size)
        ub = extract_upper_bounds(variables, var_size)

        # The lower bound should be -5.
        assert lb is not None
        assert np.allclose(lb, -5)
        # The upper bound should be -2 (tighter than nonpos's 0).
        assert ub is not None
        assert np.allclose(ub, -2)
