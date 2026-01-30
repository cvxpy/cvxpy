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


class TestBoundsExpressionsMissing:
    """Tests for bounds propagation through expressions not covered above."""

    def test_reshape(self) -> None:
        """Test bounds propagation through reshape."""
        x = cp.Variable((2, 3), bounds=[1, 6])
        expr = cp.reshape(x, (3, 2))
        lb, ub = expr.get_bounds()
        assert lb.shape == (3, 2)
        assert ub.shape == (3, 2)
        assert np.allclose(lb, 1)
        assert np.allclose(ub, 6)

    def test_transpose(self) -> None:
        """Test bounds propagation through transpose."""
        x = cp.Variable((2, 3), bounds=[1, 6])
        expr = x.T
        lb, ub = expr.get_bounds()
        assert lb.shape == (3, 2)
        assert ub.shape == (3, 2)
        assert np.allclose(lb, 1)
        assert np.allclose(ub, 6)

    def test_indexing_single(self) -> None:
        """Test bounds propagation through single element indexing."""
        x = cp.Variable(5, bounds=[0, 10])
        expr = x[2]
        lb, ub = expr.get_bounds()
        assert lb.shape == ()
        assert ub.shape == ()
        assert np.allclose(lb, 0)
        assert np.allclose(ub, 10)

    def test_indexing_slice(self) -> None:
        """Test bounds propagation through slice indexing."""
        x = cp.Variable(5, bounds=[0, 10])
        expr = x[1:4]
        lb, ub = expr.get_bounds()
        assert lb.shape == (3,)
        assert ub.shape == (3,)
        assert np.allclose(lb, 0)
        assert np.allclose(ub, 10)

    def test_indexing_2d(self) -> None:
        """Test bounds propagation through 2D indexing."""
        x = cp.Variable((3, 4), bounds=[1, 5])
        expr = x[0:2, 1:3]
        lb, ub = expr.get_bounds()
        assert lb.shape == (2, 2)
        assert ub.shape == (2, 2)
        assert np.allclose(lb, 1)
        assert np.allclose(ub, 5)

    def test_matmul_constant_matrix(self) -> None:
        """Test bounds propagation through constant @ variable multiplication."""
        A = np.array([[1, 2, 3], [4, 5, 6]])  # Constant
        x = cp.Variable(3, bounds=[0, 1])
        expr = A @ x
        lb, ub = expr.get_bounds()
        assert lb.shape == (2,)
        assert ub.shape == (2,)
        # A is constant with positive entries, so lb = A @ lb_x, ub = A @ ub_x
        assert np.allclose(lb, 0)
        assert np.allclose(ub, [6, 15])  # [1+2+3, 4+5+6]

    def test_matmul_variable_intervals(self) -> None:
        """Test bounds for matmul with two variable intervals returns unbounded."""
        A = cp.Variable((2, 3), bounds=[0, 1])
        B = cp.Variable((3, 4), bounds=[0, 1])
        expr = A @ B
        lb, ub = expr.get_bounds()
        assert lb.shape == (2, 4)
        assert ub.shape == (2, 4)
        # Both are intervals, so matmul returns unbounded
        assert np.all(lb == -np.inf)
        assert np.all(ub == np.inf)

    def test_sqrt(self) -> None:
        """Test bounds propagation through sqrt."""
        x = cp.Variable(3, bounds=[1, 4])
        expr = cp.sqrt(x)
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, 1)
        assert np.allclose(ub, 2)

    def test_promote(self) -> None:
        """Test bounds propagation through promote (scalar to vector)."""
        x = cp.Variable(bounds=[2, 5])
        # Create a promoted expression by adding scalar to vector
        y = cp.Variable(3, bounds=[0, 0])
        expr = x + y  # This broadcasts x
        lb, ub = expr.get_bounds()
        assert lb.shape == (3,)
        assert np.allclose(lb, 2)
        assert np.allclose(ub, 5)


class TestBoundsUtilitiesMissing:
    """Tests for bounds utility functions not covered above."""

    def test_scalar_bounds(self) -> None:
        """Test scalar_bounds function."""
        from cvxpy.utilities.bounds import scalar_bounds
        lb, ub = scalar_bounds(-1.0, 2.0)
        assert lb.shape == ()
        assert ub.shape == ()
        assert np.allclose(lb, -1.0)
        assert np.allclose(ub, 2.0)

    def test_div_bounds_positive(self) -> None:
        """Test div_bounds with positive divisor."""
        from cvxpy.utilities.bounds import div_bounds
        lb1 = np.array([2, 4])
        ub1 = np.array([6, 8])
        lb2 = np.array([1, 2])
        ub2 = np.array([2, 4])
        lb, ub = div_bounds(lb1, ub1, lb2, ub2)
        # [2,6]/[1,2] = [1, 6], [4,8]/[2,4] = [1, 4]
        assert np.allclose(lb, [1, 1])
        assert np.allclose(ub, [6, 4])

    def test_div_bounds_spanning_zero(self) -> None:
        """Test div_bounds when divisor spans zero gives unbounded."""
        from cvxpy.utilities.bounds import div_bounds
        lb1 = np.array([1])
        ub1 = np.array([2])
        lb2 = np.array([-1])
        ub2 = np.array([1])
        lb, ub = div_bounds(lb1, ub1, lb2, ub2)
        assert np.all(lb == -np.inf)
        assert np.all(ub == np.inf)

    def test_scale_bounds_positive(self) -> None:
        """Test scale_bounds with positive scalar."""
        from cvxpy.utilities.bounds import scale_bounds
        lb = np.array([1, 2])
        ub = np.array([3, 4])
        new_lb, new_ub = scale_bounds(lb, ub, 2.0)
        assert np.allclose(new_lb, [2, 4])
        assert np.allclose(new_ub, [6, 8])

    def test_scale_bounds_negative(self) -> None:
        """Test scale_bounds with negative scalar (flips bounds)."""
        from cvxpy.utilities.bounds import scale_bounds
        lb = np.array([1, 2])
        ub = np.array([3, 4])
        new_lb, new_ub = scale_bounds(lb, ub, -2.0)
        assert np.allclose(new_lb, [-6, -8])
        assert np.allclose(new_ub, [-2, -4])

    def test_maximum_bounds(self) -> None:
        """Test maximum_bounds function."""
        from cvxpy.utilities.bounds import maximum_bounds
        bounds_list = [
            (np.array([1, 2]), np.array([3, 4])),
            (np.array([0, 5]), np.array([2, 6])),
        ]
        lb, ub = maximum_bounds(bounds_list)
        # max lower bound: max(1,0)=1, max(2,5)=5
        # max upper bound: max(3,2)=3, max(4,6)=6
        assert np.allclose(lb, [1, 5])
        assert np.allclose(ub, [3, 6])

    def test_minimum_bounds(self) -> None:
        """Test minimum_bounds function."""
        from cvxpy.utilities.bounds import minimum_bounds
        bounds_list = [
            (np.array([1, 2]), np.array([3, 4])),
            (np.array([0, 5]), np.array([2, 6])),
        ]
        lb, ub = minimum_bounds(bounds_list)
        # min lower bound: min(1,0)=0, min(2,5)=2
        # min upper bound: min(3,2)=2, min(4,6)=4
        assert np.allclose(lb, [0, 2])
        assert np.allclose(ub, [2, 4])

    def test_power_bounds_square(self) -> None:
        """Test power_bounds for squaring."""
        from cvxpy.utilities.bounds import power_bounds
        lb = np.array([2, -3])
        ub = np.array([4, -1])
        new_lb, new_ub = power_bounds(lb, ub, 2.0)
        # [2,4]^2 = [4,16], [-3,-1]^2 = [1,9]
        assert np.allclose(new_lb, [4, 1])
        assert np.allclose(new_ub, [16, 9])

    def test_power_bounds_square_spanning_zero(self) -> None:
        """Test power_bounds for squaring when spanning zero."""
        from cvxpy.utilities.bounds import power_bounds
        lb = np.array([-2])
        ub = np.array([3])
        new_lb, new_ub = power_bounds(lb, ub, 2.0)
        # [-2,3]^2: min is 0, max is 9
        assert np.allclose(new_lb, [0])
        assert np.allclose(new_ub, [9])

    def test_power_bounds_half(self) -> None:
        """Test power_bounds for square root (p=0.5)."""
        from cvxpy.utilities.bounds import power_bounds
        lb = np.array([1, 4])
        ub = np.array([4, 9])
        new_lb, new_ub = power_bounds(lb, ub, 0.5)
        assert np.allclose(new_lb, [1, 2])
        assert np.allclose(new_ub, [2, 3])

    def test_exp_bounds(self) -> None:
        """Test exp_bounds function."""
        from cvxpy.utilities.bounds import exp_bounds
        lb = np.array([0, 1])
        ub = np.array([1, 2])
        new_lb, new_ub = exp_bounds(lb, ub)
        assert np.allclose(new_lb, [1, np.e])
        assert np.allclose(new_ub, [np.e, np.e**2])

    def test_log_bounds(self) -> None:
        """Test log_bounds function."""
        from cvxpy.utilities.bounds import log_bounds
        lb = np.array([1, np.e])
        ub = np.array([np.e, np.e**2])
        new_lb, new_ub = log_bounds(lb, ub)
        assert np.allclose(new_lb, [0, 1])
        assert np.allclose(new_ub, [1, 2])

    def test_sqrt_bounds(self) -> None:
        """Test sqrt_bounds function."""
        from cvxpy.utilities.bounds import sqrt_bounds
        lb = np.array([1, 4])
        ub = np.array([4, 9])
        new_lb, new_ub = sqrt_bounds(lb, ub)
        assert np.allclose(new_lb, [1, 2])
        assert np.allclose(new_ub, [2, 3])

    def test_norm1_bounds(self) -> None:
        """Test norm1_bounds function."""
        from cvxpy.utilities.bounds import norm1_bounds
        lb = np.array([1, 2, 3])
        ub = np.array([2, 3, 4])
        new_lb, new_ub = norm1_bounds(lb, ub)
        assert np.allclose(new_lb, 6)  # 1+2+3
        assert np.allclose(new_ub, 9)  # 2+3+4

    def test_norm_inf_bounds(self) -> None:
        """Test norm_inf_bounds function."""
        from cvxpy.utilities.bounds import norm_inf_bounds
        lb = np.array([1, 2, 3])
        ub = np.array([2, 5, 4])
        new_lb, new_ub = norm_inf_bounds(lb, ub)
        assert np.allclose(new_lb, 3)  # max of lower bounds
        assert np.allclose(new_ub, 5)  # max of upper bounds

    def test_broadcast_bounds(self) -> None:
        """Test broadcast_bounds function."""
        from cvxpy.utilities.bounds import broadcast_bounds
        lb = np.array([1])
        ub = np.array([2])
        new_lb, new_ub = broadcast_bounds(lb, ub, (3, 4))
        assert new_lb.shape == (3, 4)
        assert new_ub.shape == (3, 4)
        assert np.allclose(new_lb, 1)
        assert np.allclose(new_ub, 2)

    def test_reshape_bounds(self) -> None:
        """Test reshape_bounds function."""
        from cvxpy.utilities.bounds import reshape_bounds
        lb = np.array([[1, 2], [3, 4]])
        ub = np.array([[5, 6], [7, 8]])
        new_lb, new_ub = reshape_bounds(lb, ub, (4,))
        assert new_lb.shape == (4,)
        assert new_ub.shape == (4,)
        # Fortran order reshape (CVXPY default)
        assert np.allclose(new_lb, [1, 3, 2, 4])
        assert np.allclose(new_ub, [5, 7, 6, 8])

    def test_transpose_bounds(self) -> None:
        """Test transpose_bounds function."""
        from cvxpy.utilities.bounds import transpose_bounds
        lb = np.array([[1, 2, 3], [4, 5, 6]])
        ub = np.array([[7, 8, 9], [10, 11, 12]])
        new_lb, new_ub = transpose_bounds(lb, ub)
        assert new_lb.shape == (3, 2)
        assert new_ub.shape == (3, 2)
        assert np.allclose(new_lb, [[1, 4], [2, 5], [3, 6]])
        assert np.allclose(new_ub, [[7, 10], [8, 11], [9, 12]])

    def test_index_bounds(self) -> None:
        """Test index_bounds function."""
        from cvxpy.utilities.bounds import index_bounds
        lb = np.array([1, 2, 3, 4, 5])
        ub = np.array([6, 7, 8, 9, 10])
        new_lb, new_ub = index_bounds(lb, ub, slice(1, 4))
        assert new_lb.shape == (3,)
        assert np.allclose(new_lb, [2, 3, 4])
        assert np.allclose(new_ub, [7, 8, 9])

    def test_matmul_bounds_constant_left(self) -> None:
        """Test matmul_bounds with constant left operand."""
        from cvxpy.utilities.bounds import matmul_bounds
        # Constant left operand (lb1 == ub1)
        A = np.array([[1, 2], [3, 4]])
        lb2 = np.array([[1, 1], [1, 1]])
        ub2 = np.array([[2, 2], [2, 2]])
        lb, ub = matmul_bounds(A, A, lb2, ub2)
        assert lb.shape == (2, 2)
        assert ub.shape == (2, 2)
        # A has all positive entries, so lb = A @ lb2, ub = A @ ub2
        assert np.allclose(lb, A @ lb2)
        assert np.allclose(ub, A @ ub2)

    def test_matmul_bounds_both_intervals(self) -> None:
        """Test matmul_bounds returns unbounded when both are intervals."""
        from cvxpy.utilities.bounds import matmul_bounds
        lb1 = np.array([[1, 2], [3, 4]])
        ub1 = np.array([[2, 3], [4, 5]])
        lb2 = np.array([[1, 1], [1, 1]])
        ub2 = np.array([[2, 2], [2, 2]])
        lb, ub = matmul_bounds(lb1, ub1, lb2, ub2)
        # Both operands are intervals, so returns unbounded
        assert np.all(lb == -np.inf)
        assert np.all(ub == np.inf)

    def test_refine_bounds_from_sign_nonneg(self) -> None:
        """Test refine_bounds_from_sign with nonneg."""
        from cvxpy.utilities.bounds import refine_bounds_from_sign
        lb = np.array([-1, -2])
        ub = np.array([3, 4])
        new_lb, new_ub = refine_bounds_from_sign(lb, ub, is_nonneg=True, is_nonpos=False)
        assert np.allclose(new_lb, [0, 0])
        assert np.allclose(new_ub, [3, 4])

    def test_refine_bounds_from_sign_nonpos(self) -> None:
        """Test refine_bounds_from_sign with nonpos."""
        from cvxpy.utilities.bounds import refine_bounds_from_sign
        lb = np.array([-3, -4])
        ub = np.array([1, 2])
        new_lb, new_ub = refine_bounds_from_sign(lb, ub, is_nonneg=False, is_nonpos=True)
        assert np.allclose(new_lb, [-3, -4])
        assert np.allclose(new_ub, [0, 0])


class TestBoundsMemoryOptimization:
    """Test memory optimization for bounds propagation."""

    def test_variable_uniform_bounds_use_broadcast(self) -> None:
        """Test that variables with uniform bounds use broadcast views."""
        # A variable with no explicit bounds should use broadcast views
        x = cp.Variable((100, 100))
        lb, ub = x.get_bounds()

        # Check values are correct
        assert np.all(lb == -np.inf)
        assert np.all(ub == np.inf)

        # Check that it's a broadcast view (strides should be 0)
        # A broadcast view has 0 strides in dimensions that were broadcast
        assert lb.strides == (0, 0)
        assert ub.strides == (0, 0)

    def test_nonneg_variable_uniform_bounds_use_broadcast(self) -> None:
        """Test that nonneg variables with uniform bounds use broadcast views."""
        x = cp.Variable((100, 100), nonneg=True)
        lb, ub = x.get_bounds()

        assert np.all(lb == 0)
        assert np.all(ub == np.inf)

        # Should use broadcast views (0 strides)
        assert lb.strides == (0, 0)
        assert ub.strides == (0, 0)

    def test_bounded_variable_scalar_uses_broadcast(self) -> None:
        """Test that variables with scalar bounds use broadcast views."""
        x = cp.Variable((100, 100), bounds=(-5, 10))
        lb, ub = x.get_bounds()

        assert np.all(lb == -5)
        assert np.all(ub == 10)

        # Scalar bounds should use broadcast views
        assert lb.strides == (0, 0)
        assert ub.strides == (0, 0)

    def test_array_bounds_do_not_use_broadcast(self) -> None:
        """Test that array bounds don't use broadcast (different values)."""
        lb_values = np.arange(6).reshape(2, 3)
        ub_values = np.arange(6, 12).reshape(2, 3)
        x = cp.Variable((2, 3), bounds=(lb_values, ub_values))
        lb, ub = x.get_bounds()

        assert np.allclose(lb, lb_values)
        assert np.allclose(ub, ub_values)

        # Array bounds should NOT have 0 strides (unless coincidentally)
        # For a (2, 3) float array in C order, strides are (24, 8)
        assert lb.strides != (0, 0)
        assert ub.strides != (0, 0)

    def test_boolean_variable_broadcast(self) -> None:
        """Test that boolean variables use broadcast views."""
        x = cp.Variable((100, 100), boolean=True)
        lb, ub = x.get_bounds()

        assert np.all(lb == 0)
        assert np.all(ub == 1)

        # Should use broadcast views
        assert lb.strides == (0, 0)
        assert ub.strides == (0, 0)
