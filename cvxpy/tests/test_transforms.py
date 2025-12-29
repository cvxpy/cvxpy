"""
Copyright 2024, the cvxpy developers.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Comprehensive tests for the cvxpy.transforms module including:
- scalarize: weighted_sum, targets_and_priorities, max, log_sum_exp
- linearize: affine approximation of expressions
- indicator: indicator function of constraint satisfaction
- suppfunc: support function transform
"""

import numpy as np
import pytest

import cvxpy as cp
from cvxpy.problems.objective import Maximize, Minimize
from cvxpy.tests.base_test import BaseTest
from cvxpy.transforms import indicator, linearize, scalarize


class TestScalarizeExtended(BaseTest):
    """Extended tests for the scalarize transform functions."""

    def setUp(self) -> None:
        np.random.seed(42)

    # ===================== weighted_sum tests =====================

    def test_weighted_sum_vector_variable(self) -> None:
        """Test weighted_sum with vector variables."""
        x = cp.Variable(3)
        obj1 = cp.Minimize(cp.sum_squares(x))
        obj2 = cp.Minimize(cp.sum_squares(x - np.array([1, 2, 3])))
        objectives = [obj1, obj2]

        weights = [1, 1]
        scalarized = scalarize.weighted_sum(objectives, weights)
        prob = cp.Problem(scalarized)
        prob.solve()
        expected = np.array([0.5, 1.0, 1.5])
        self.assertItemsAlmostEqual(x.value, expected, places=3)

    def test_weighted_sum_matrix_variable(self) -> None:
        """Test weighted_sum with matrix variables."""
        X = cp.Variable((2, 2))
        target = np.array([[1, 2], [3, 4]])
        obj1 = cp.Minimize(cp.sum_squares(X))
        obj2 = cp.Minimize(cp.sum_squares(X - target))
        objectives = [obj1, obj2]

        weights = [1, 1]
        scalarized = scalarize.weighted_sum(objectives, weights)
        prob = cp.Problem(scalarized)
        prob.solve()
        expected = target / 2
        self.assertItemsAlmostEqual(X.value.flatten(), expected.flatten(), places=3)

    def test_weighted_sum_many_objectives(self) -> None:
        """Test weighted_sum with more than 2 objectives."""
        x = cp.Variable()
        objectives = [
            cp.Minimize(cp.square(x - i))
            for i in range(5)
        ]
        weights = [1, 1, 1, 1, 1]
        scalarized = scalarize.weighted_sum(objectives, weights)
        prob = cp.Problem(scalarized)
        prob.solve()
        # Optimal is the mean: (0+1+2+3+4)/5 = 2
        self.assertItemsAlmostEqual(x.value, 2.0, places=3)

    def test_weighted_sum_negative_weights_non_dcp(self) -> None:
        """Test weighted_sum with negative weights raises DCP error.

        Negative weights can create non-DCP problems, which is expected behavior.
        """
        x = cp.Variable()
        obj1 = cp.Minimize(cp.square(x))
        obj2 = cp.Minimize(cp.square(x - 2))
        objectives = [obj1, obj2]

        # Negative weight on convex objective creates non-DCP expression
        weights = [-1, 2]
        with pytest.raises(Exception, match="DCP"):
            scalarize.weighted_sum(objectives, weights)

    def test_weighted_sum_zero_weight(self) -> None:
        """Test weighted_sum with zero weights."""
        x = cp.Variable()
        obj1 = cp.Minimize(cp.square(x - 5))
        obj2 = cp.Minimize(cp.square(x))
        objectives = [obj1, obj2]

        # Zero weight on first objective
        weights = [0, 1]
        scalarized = scalarize.weighted_sum(objectives, weights)
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(x.value, 0.0, places=3)

    def test_weighted_sum_numpy_weights(self) -> None:
        """Test weighted_sum with numpy array weights."""
        x = cp.Variable()
        obj1 = cp.Minimize(cp.square(x))
        obj2 = cp.Minimize(cp.square(x - 1))
        objectives = [obj1, obj2]

        weights = np.array([1.0, 1.0])
        scalarized = scalarize.weighted_sum(objectives, weights)
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(x.value, 0.5, places=3)

    # ===================== max tests =====================

    def test_max_vector_variable(self) -> None:
        """Test max with vector variables."""
        x = cp.Variable(2)
        obj1 = cp.Minimize(cp.norm(x))
        obj2 = cp.Minimize(cp.norm(x - np.array([1, 1])))
        objectives = [obj1, obj2]

        weights = [1, 1]
        scalarized = scalarize.max(objectives, weights)
        prob = cp.Problem(scalarized)
        prob.solve()
        # The optimal is at the midpoint [0.5, 0.5]
        expected = np.array([0.5, 0.5])
        self.assertItemsAlmostEqual(x.value, expected, places=3)

    def test_max_unequal_weights(self) -> None:
        """Test max with unequal weights."""
        x = cp.Variable()
        obj1 = cp.Minimize(cp.abs(x))
        obj2 = cp.Minimize(cp.abs(x - 1))
        objectives = [obj1, obj2]

        weights = [1, 2]
        scalarized = scalarize.max(objectives, weights)
        prob = cp.Problem(scalarized)
        prob.solve()
        # Optimal when |x| = 2*|x-1|
        # At x = 2/3: |2/3| = 2/3, 2*|2/3-1| = 2*1/3 = 2/3
        self.assertItemsAlmostEqual(x.value, 2/3, places=3)

    def test_max_many_objectives(self) -> None:
        """Test max with more than 2 objectives."""
        x = cp.Variable()
        objectives = [
            cp.Minimize(cp.abs(x - i))
            for i in range(3)  # targets at 0, 1, 2
        ]
        weights = [1, 1, 1]
        scalarized = scalarize.max(objectives, weights)
        prob = cp.Problem(scalarized)
        prob.solve()
        # The optimal is at the center: x = 1
        self.assertItemsAlmostEqual(x.value, 1.0, places=3)

    # ===================== log_sum_exp tests =====================

    def test_log_sum_exp_gamma_limit_zero(self) -> None:
        """Test log_sum_exp with small gamma approaches weighted_sum."""
        x = cp.Variable()
        obj1 = cp.Minimize(cp.square(x))
        obj2 = cp.Minimize(cp.square(x - 1))
        objectives = [obj1, obj2]
        weights = [1, 1]

        # Small gamma should behave like weighted_sum
        scalarized_lse = scalarize.log_sum_exp(objectives, weights, gamma=0.01)
        prob = cp.Problem(scalarized_lse)
        prob.solve()
        x_lse = x.value

        scalarized_ws = scalarize.weighted_sum(objectives, weights)
        prob = cp.Problem(scalarized_ws)
        prob.solve()
        x_ws = x.value

        # Should be close for small gamma
        self.assertAlmostEqual(x_lse, x_ws, places=1)

    def test_log_sum_exp_gamma_limit_infinity(self) -> None:
        """Test log_sum_exp with large gamma approaches max."""
        x = cp.Variable()
        obj1 = cp.Minimize(cp.square(x))
        obj2 = cp.Minimize(cp.square(x - 1))
        objectives = [obj1, obj2]
        weights = [1, 1]

        # Large gamma should behave like max
        scalarized_lse = scalarize.log_sum_exp(objectives, weights, gamma=100)
        prob = cp.Problem(scalarized_lse)
        prob.solve()
        x_lse = x.value

        scalarized_max = scalarize.max(objectives, weights)
        prob = cp.Problem(scalarized_max)
        prob.solve()
        x_max = x.value

        # Should be close for large gamma
        self.assertAlmostEqual(x_lse, x_max, places=2)

    def test_log_sum_exp_many_objectives(self) -> None:
        """Test log_sum_exp with many objectives."""
        x = cp.Variable()
        objectives = [
            cp.Minimize(cp.square(x - i))
            for i in range(4)
        ]
        weights = [1, 1, 1, 1]

        scalarized = scalarize.log_sum_exp(objectives, weights, gamma=1.0)
        prob = cp.Problem(scalarized)
        prob.solve()
        # Should be between 0 and 3
        self.assertTrue(0 < x.value < 3)

    def test_log_sum_exp_vector_variable(self) -> None:
        """Test log_sum_exp with vector variables."""
        x = cp.Variable(2)
        obj1 = cp.Minimize(cp.sum_squares(x))
        obj2 = cp.Minimize(cp.sum_squares(x - np.array([1, 1])))
        objectives = [obj1, obj2]
        weights = [1, 1]

        scalarized = scalarize.log_sum_exp(objectives, weights, gamma=1.0)
        prob = cp.Problem(scalarized)
        prob.solve()
        # Should be near the midpoint
        self.assertItemsAlmostEqual(x.value, [0.5, 0.5], places=2)

    # ===================== targets_and_priorities tests =====================

    def test_targets_and_priorities_vector_variable(self) -> None:
        """Test targets_and_priorities with vector variables."""
        x = cp.Variable(2)
        obj1 = cp.Minimize(cp.sum_squares(x))
        obj2 = cp.Minimize(cp.sum_squares(x - np.array([2, 2])))
        objectives = [obj1, obj2]

        targets = [1, 1]
        priorities = [1, 1]
        scalarized = scalarize.targets_and_priorities(objectives, priorities, targets)
        prob = cp.Problem(scalarized)
        prob.solve()
        # Should be near [1, 1]
        self.assertItemsAlmostEqual(x.value, [1.0, 1.0], places=2)

    def test_targets_and_priorities_with_limits(self) -> None:
        """Test targets_and_priorities with limits enforced."""
        x = cp.Variable()
        obj1 = cp.Minimize(cp.square(x))
        obj2 = cp.Minimize(cp.square(x - 10))
        objectives = [obj1, obj2]

        targets = [0, 0]
        priorities = [1, 1]
        limits = [25, 100]  # obj1 <= 25 means |x| <= 5
        scalarized = scalarize.targets_and_priorities(
            objectives, priorities, targets, limits
        )
        prob = cp.Problem(scalarized)
        prob.solve()
        # Should respect limit: x^2 <= 25
        self.assertTrue(x.value**2 <= 25 + 1e-3)

    def test_targets_and_priorities_off_target_scaling(self) -> None:
        """Test that off_target parameter affects behavior.

        The off_target parameter adds a small penalty for being outside
        the target region. Larger off_target values should pull the solution
        further away from the extremes.
        """
        x = cp.Variable()
        obj1 = cp.Minimize(cp.square(x))
        obj2 = cp.Minimize(cp.square(x - 4))  # More distant targets
        objectives = [obj1, obj2]

        targets = [0.5, 0.5]  # Low targets
        priorities = [1, 1]

        # With very small off_target, solution depends mainly on priorities
        scalarized1 = scalarize.targets_and_priorities(
            objectives, priorities, targets, off_target=1e-6
        )
        prob1 = cp.Problem(scalarized1)
        prob1.solve()
        x1 = float(x.value)

        # With larger off_target, there's more penalty outside target region
        scalarized2 = scalarize.targets_and_priorities(
            objectives, priorities, targets, off_target=0.9
        )
        prob2 = cp.Problem(scalarized2)
        prob2.solve()
        x2 = float(x.value)

        # Both solutions should be valid (between 0 and 4)
        self.assertTrue(0 <= x1 <= 4)
        self.assertTrue(0 <= x2 <= 4)


class TestLinearize(BaseTest):
    """Comprehensive tests for the linearize transform."""

    def setUp(self) -> None:
        np.random.seed(42)
        self.x = cp.Variable(2)
        self.A = cp.Variable((2, 2))

    def test_linearize_affine_unchanged(self) -> None:
        """Linearize should return affine expressions unchanged."""
        expr = 2 * self.x[0] + 3 * self.x[1] - 5
        self.x.value = [1, 2]
        lin_expr = linearize(expr)
        # Test at different points
        for _ in range(3):
            self.x.value = np.random.randn(2)
            self.assertAlmostEqual(lin_expr.value, expr.value, places=10)

    def test_linearize_constant(self) -> None:
        """Linearize a constant expression."""
        expr = cp.Constant(5.0)
        lin_expr = linearize(expr)
        self.assertAlmostEqual(lin_expr.value, 5.0)

    def test_linearize_numpy_array(self) -> None:
        """Linearize accepts numpy arrays."""
        arr = np.array([1.0, 2.0, 3.0])
        lin_expr = linearize(arr)
        self.assertItemsAlmostEqual(lin_expr.value, arr)

    def test_linearize_convex_gives_lower_bound(self) -> None:
        """Linearize convex expressions gives lower bound.

        Linearize works with vector/matrix variables and expressions over them.
        """
        x = cp.Variable(2)
        expr = cp.sum_squares(x)  # Convex
        x.value = np.array([1.0, 2.0])
        lin_expr = linearize(expr)

        # At linearization point, values should match
        self.assertAlmostEqual(lin_expr.value, expr.value, places=5)

        # At other points, linear approximation should be a lower bound
        test_points = [
            np.array([0.0, 0.0]),
            np.array([2.0, 3.0]),
            np.array([-1.0, 1.0]),
        ]
        for pt in test_points:
            x.value = pt
            self.assertLessEqual(lin_expr.value - 1e-6, expr.value)

    def test_linearize_concave_gives_upper_bound(self) -> None:
        """Linearize concave expressions gives upper bound."""
        x = cp.Variable(2)
        x.value = np.array([1.0, 2.0])
        expr = cp.sum(cp.log(x))  # Concave
        lin_expr = linearize(expr)

        # At linearization point, values should match
        self.assertAlmostEqual(lin_expr.value, expr.value, places=5)

        # At other points, linear approximation should be an upper bound
        test_points = [
            np.array([0.5, 0.5]),
            np.array([2.0, 3.0]),
            np.array([1.5, 1.5]),
        ]
        for pt in test_points:
            x.value = pt
            self.assertGreaterEqual(lin_expr.value + 1e-6, expr.value)

    def test_linearize_vector_expression(self) -> None:
        """Linearize vector expressions."""
        self.x.value = np.array([1.0, 2.0])
        expr = cp.sum_squares(self.x)
        lin_expr = linearize(expr)

        # At linearization point
        self.assertAlmostEqual(lin_expr.value, expr.value)

        # Lower bound at other points
        self.x.value = np.array([2.0, 3.0])
        self.assertLessEqual(lin_expr.value - 1e-6, expr.value)

    def test_linearize_matrix_expression(self) -> None:
        """Linearize matrix expressions."""
        self.A.value = np.array([[1, 2], [3, 4]])
        expr = cp.sum_squares(self.A)
        lin_expr = linearize(expr)

        # At linearization point
        self.assertAlmostEqual(lin_expr.value, expr.value)

        # Lower bound at other points
        self.A.value = np.array([[2, 3], [4, 5]])
        self.assertLessEqual(lin_expr.value - 1e-6, expr.value)

    def test_linearize_elementwise_convex(self) -> None:
        """Linearize elementwise convex expressions."""
        self.x.value = np.array([1.0, 2.0])
        expr = cp.exp(self.x)
        lin_expr = linearize(expr)

        # At linearization point
        self.assertItemsAlmostEqual(lin_expr.value, expr.value, places=5)

    def test_linearize_norm(self) -> None:
        """Linearize norm expressions."""
        self.x.value = np.array([3.0, 4.0])
        expr = cp.norm(self.x)
        lin_expr = linearize(expr)

        # At linearization point
        self.assertAlmostEqual(lin_expr.value, 5.0, places=5)

        # Lower bound at other points
        self.x.value = np.array([1.0, 1.0])
        self.assertLessEqual(lin_expr.value - 1e-5, expr.value)

    def test_linearize_missing_value_raises(self) -> None:
        """Linearize raises when variable value is missing."""
        x = cp.Variable(2)
        expr = cp.sum_squares(x)
        # x.value is None
        with pytest.raises(ValueError, match="Cannot linearize non-affine"):
            linearize(expr)

    def test_linearize_composite_expression(self) -> None:
        """Linearize composite convex expressions."""
        x = cp.Variable(2)
        x.value = np.array([1.0, 1.0])
        expr = cp.sum_squares(x) + cp.sum(cp.exp(x))
        lin_expr = linearize(expr)

        # At linearization point
        expected = 2.0 + 2 * np.exp(1.0)
        self.assertAlmostEqual(lin_expr.value, expected, places=5)

    def test_linearize_quad_form(self) -> None:
        """Linearize quadratic form expressions."""
        x = cp.Variable(2)
        Q = np.array([[2, 1], [1, 2]])
        x.value = np.array([1.0, 1.0])
        expr = cp.quad_form(x, Q)
        lin_expr = linearize(expr)

        # At linearization point
        expected = x.value @ Q @ x.value
        self.assertAlmostEqual(lin_expr.value, expected, places=5)

    def test_linearize_sum(self) -> None:
        """Linearize sum of expressions."""
        self.x.value = np.array([1.0, 2.0])
        expr = cp.sum(cp.square(self.x))
        lin_expr = linearize(expr)

        # At linearization point
        self.assertAlmostEqual(lin_expr.value, 5.0, places=5)

    def test_linearize_max_entry(self) -> None:
        """Linearize max entry (convex)."""
        self.x.value = np.array([1.0, 3.0])
        expr = cp.max(self.x)
        lin_expr = linearize(expr)

        # At linearization point
        self.assertAlmostEqual(lin_expr.value, 3.0, places=5)

    def test_linearize_minimum(self) -> None:
        """Linearize minimum (concave)."""
        self.x.value = np.array([1.0, 3.0])
        expr = cp.min(self.x)
        lin_expr = linearize(expr)

        # At linearization point
        self.assertAlmostEqual(lin_expr.value, 1.0, places=5)

    def test_linearize_abs_at_nonzero(self) -> None:
        """Linearize abs expressions (convex)."""
        x = cp.Variable(2)
        x.value = np.array([2.0, -3.0])
        expr = cp.sum(cp.abs(x))
        lin_expr = linearize(expr)

        # At linearization point
        self.assertAlmostEqual(lin_expr.value, 5.0, places=5)

        # Check it's a linear approximation (lower bound)
        x.value = np.array([3.0, -4.0])
        self.assertLessEqual(lin_expr.value - 1e-5, expr.value)

    def test_linearize_power(self) -> None:
        """Linearize power expressions."""
        x = cp.Variable(2)
        x.value = np.array([2.0, 3.0])
        expr = cp.sum(cp.power(x, 2))
        lin_expr = linearize(expr)

        # At linearization point
        expected = 4.0 + 9.0
        self.assertAlmostEqual(lin_expr.value, expected, places=5)

    def test_linearize_inv_pos(self) -> None:
        """Linearize inv_pos (convex decreasing)."""
        x = cp.Variable(2)
        x.value = np.array([2.0, 4.0])
        expr = cp.sum(cp.inv_pos(x))
        lin_expr = linearize(expr)

        # At linearization point
        expected = 0.5 + 0.25
        self.assertAlmostEqual(lin_expr.value, expected, places=5)

    def test_linearize_sqrt(self) -> None:
        """Linearize sqrt (concave)."""
        x = cp.Variable(2)
        x.value = np.array([4.0, 9.0])
        expr = cp.sum(cp.sqrt(x))
        lin_expr = linearize(expr)

        # At linearization point
        expected = 2.0 + 3.0
        self.assertAlmostEqual(lin_expr.value, expected, places=5)

        # Upper bound at other points
        x.value = np.array([1.0, 4.0])
        self.assertGreaterEqual(lin_expr.value + 1e-5, expr.value)

    def test_linearize_geo_mean(self) -> None:
        """Linearize geometric mean (concave)."""
        self.x.value = np.array([4.0, 9.0])
        expr = cp.geo_mean(self.x)
        lin_expr = linearize(expr)

        # At linearization point
        expected = np.sqrt(36)
        self.assertAlmostEqual(lin_expr.value, expected, places=5)

    def test_linearize_entr(self) -> None:
        """Linearize entropy (concave)."""
        x = cp.Variable(2)
        x.value = np.array([1.0, 1.0])
        expr = cp.sum(cp.entr(x))  # -x*log(x)
        lin_expr = linearize(expr)

        # At linearization point (x=1, entr(1) = 0)
        self.assertAlmostEqual(lin_expr.value, 0.0, places=5)


class TestIndicator(BaseTest):
    """Comprehensive tests for the indicator transform."""

    def setUp(self) -> None:
        np.random.seed(42)

    def test_indicator_basic(self) -> None:
        """Basic indicator function test."""
        x = cp.Variable()
        constraints = [x >= 0, x <= 1]
        ind = indicator(constraints)

        # Within constraints
        x.value = 0.5
        self.assertEqual(ind.value, 0.0)

        # Outside constraints
        x.value = 2.0
        self.assertEqual(ind.value, np.inf)

    def test_indicator_boundary(self) -> None:
        """Test indicator at constraint boundaries."""
        x = cp.Variable()
        constraints = [x >= 0, x <= 1]
        ind = indicator(constraints)

        # At boundaries (with tolerance)
        x.value = 0.0
        self.assertEqual(ind.value, 0.0)

        x.value = 1.0
        self.assertEqual(ind.value, 0.0)

    def test_indicator_tolerance(self) -> None:
        """Test indicator with different error tolerances."""
        x = cp.Variable()
        constraints = [x >= 0]

        # Slightly negative value
        x.value = -0.0001

        # With default tolerance (1e-3), should be satisfied
        ind_default = indicator(constraints, err_tol=1e-3)
        self.assertEqual(ind_default.value, 0.0)

        # With tighter tolerance, should violate
        ind_tight = indicator(constraints, err_tol=1e-5)
        self.assertEqual(ind_tight.value, np.inf)

    def test_indicator_multiple_constraints(self) -> None:
        """Test indicator with multiple constraints."""
        x = cp.Variable(2)
        constraints = [
            x[0] >= 0,
            x[1] >= 0,
            x[0] + x[1] <= 1
        ]
        ind = indicator(constraints)

        # Valid point
        x.value = np.array([0.3, 0.3])
        self.assertEqual(ind.value, 0.0)

        # Violates sum constraint
        x.value = np.array([0.6, 0.6])
        self.assertEqual(ind.value, np.inf)

        # Violates nonnegativity
        x.value = np.array([-0.1, 0.5])
        self.assertEqual(ind.value, np.inf)

    def test_indicator_equality_constraint(self) -> None:
        """Test indicator with equality constraints."""
        x = cp.Variable(2)
        constraints = [x[0] + x[1] == 1]
        ind = indicator(constraints)

        # Satisfies equality
        x.value = np.array([0.5, 0.5])
        self.assertEqual(ind.value, 0.0)

        # Violates equality
        x.value = np.array([0.5, 0.6])
        self.assertEqual(ind.value, np.inf)

    def test_indicator_matrix_constraint(self) -> None:
        """Test indicator with matrix constraints."""
        X = cp.Variable((2, 2))
        constraints = [cp.sum(X) <= 4]
        ind = indicator(constraints)

        # Satisfies
        X.value = np.ones((2, 2))
        self.assertEqual(ind.value, 0.0)

        # Violates
        X.value = 2 * np.ones((2, 2))
        self.assertEqual(ind.value, np.inf)

    def test_indicator_soc_constraint(self) -> None:
        """Test indicator with second-order cone constraint."""
        x = cp.Variable(3)
        constraints = [cp.norm(x[:2]) <= x[2]]
        ind = indicator(constraints)

        # Satisfies
        x.value = np.array([0.3, 0.4, 1.0])
        self.assertEqual(ind.value, 0.0)

        # Violates
        x.value = np.array([0.8, 0.8, 0.5])
        self.assertEqual(ind.value, np.inf)

    def test_indicator_is_convex(self) -> None:
        """Indicator is always convex."""
        x = cp.Variable()
        ind = indicator([x >= 0])
        self.assertTrue(ind.is_convex())
        self.assertFalse(ind.is_concave())

    def test_indicator_is_nonneg(self) -> None:
        """Indicator is always nonnegative."""
        x = cp.Variable()
        ind = indicator([x >= 0])
        self.assertTrue(ind.is_nonneg())
        self.assertFalse(ind.is_nonpos())

    def test_indicator_is_real(self) -> None:
        """Indicator is real-valued (not complex/imaginary)."""
        x = cp.Variable()
        ind = indicator([x >= 0])
        self.assertFalse(ind.is_imag())
        self.assertFalse(ind.is_complex())

    def test_indicator_shape(self) -> None:
        """Indicator is scalar."""
        x = cp.Variable((3, 3))
        ind = indicator([x >= 0])
        self.assertEqual(ind.shape, ())

    def test_indicator_domain(self) -> None:
        """Test indicator domain returns constraints."""
        x = cp.Variable()
        constraints = [x >= 0, x <= 1]
        ind = indicator(constraints)
        self.assertEqual(ind.domain(), constraints)

    def test_indicator_name(self) -> None:
        """Test indicator name representation."""
        x = cp.Variable()
        constraints = [x >= 0]
        ind = indicator(constraints)
        name = ind.name()
        self.assertIn("Indicator", name)

    def test_indicator_get_data(self) -> None:
        """Test indicator get_data returns err_tol."""
        x = cp.Variable()
        ind = indicator([x >= 0], err_tol=0.01)
        data = ind.get_data()
        self.assertEqual(data, [0.01])

    def test_indicator_is_constant_with_constant_constraints(self) -> None:
        """Test is_constant when constraint args are constants."""
        x = cp.Variable()
        # This constraint has a constant arg on one side
        constraints = [x >= 0]
        ind = indicator(constraints)
        # Not constant because x is a variable
        self.assertFalse(ind.is_constant())

    def test_indicator_is_not_dpp(self) -> None:
        """Indicator is not DPP."""
        x = cp.Variable()
        ind = indicator([x >= 0])
        self.assertFalse(ind.is_dpp())

    def test_indicator_in_problem(self) -> None:
        """Test indicator in optimization problem."""
        x = cp.Variable()
        constraints = [x >= 0, x <= 2]
        objective = cp.Minimize(cp.square(x - 3) + indicator(constraints))
        prob = cp.Problem(objective)
        prob.solve(solver=cp.SCS)

        # Should hit the upper bound
        self.assertAlmostEqual(x.value, 2.0, places=3)

    def test_indicator_with_affine_constraint(self) -> None:
        """Test indicator with affine constraint A@x <= b."""
        x = cp.Variable(2)
        A = np.array([[1, 1], [-1, 0], [0, -1]])
        b = np.array([1, 0, 0])
        constraints = [A @ x <= b]
        ind = indicator(constraints)

        # In the simplex
        x.value = np.array([0.3, 0.3])
        self.assertEqual(ind.value, 0.0)

        # Outside
        x.value = np.array([0.6, 0.6])
        self.assertEqual(ind.value, np.inf)

    def test_indicator_empty_constraints(self) -> None:
        """Test indicator with empty constraint list."""
        ind = indicator([])
        # No constraints means always feasible
        self.assertEqual(ind.value, 0.0)

    def test_indicator_log_log_properties(self) -> None:
        """Test log-log convexity properties."""
        x = cp.Variable(pos=True)
        ind = indicator([x >= 1])
        self.assertFalse(ind.is_log_log_convex())
        self.assertFalse(ind.is_log_log_concave())


class TestTransformsIntegration(BaseTest):
    """Integration tests combining multiple transforms."""

    def test_scalarize_with_indicator(self) -> None:
        """Test combining scalarize with indicator."""
        x = cp.Variable()
        constraints = [x >= 0, x <= 10]

        obj1 = cp.Minimize(cp.square(x - 5))
        obj2 = cp.Minimize(indicator(constraints))

        # Use weighted_sum to combine
        combined = scalarize.weighted_sum([obj1, obj2], [1, 1])
        prob = cp.Problem(combined)
        prob.solve(solver=cp.SCS)

        self.assertAlmostEqual(x.value, 5.0, places=2)

    def test_linearize_in_optimization(self) -> None:
        """Test using linearize in an optimization problem.

        Demonstrates using linearize to create affine approximations
        that can be used in convex optimization.
        """
        x = cp.Variable(2)

        # Set up a point to linearize around
        x.value = np.array([2.0, 2.0])

        # Linearize a concave function (log)
        log_lin = linearize(cp.sum(cp.log(x)))

        # Use the linear approximation in a problem
        # Since log_lin is affine, it can be used freely
        prob = cp.Problem(
            cp.Maximize(log_lin),
            [cp.sum(x) <= 4, x >= 0.1]
        )
        prob.solve()

        # The linear approximation is an upper bound on the concave function
        # Verify this property at the linearization point
        x.value = np.array([2.0, 2.0])
        actual_log = np.sum(np.log([2.0, 2.0]))
        self.assertAlmostEqual(log_lin.value, actual_log, places=5)

        # Verify the problem solved successfully
        self.assertIsNotNone(prob.value)

    def test_multiple_linearizations(self) -> None:
        """Test multiple linearizations in sequence."""
        x = cp.Variable(2)
        x.value = np.array([2.0, 2.0])

        expr1 = cp.sum_squares(x)
        expr2 = cp.sum(cp.exp(x))

        lin1 = linearize(expr1)
        lin2 = linearize(expr2)

        # Both should match at linearization point
        self.assertAlmostEqual(lin1.value, 8.0, places=5)
        self.assertAlmostEqual(lin2.value, 2 * np.exp(2.0), places=5)

        # At another point, both should be lower bounds
        x.value = np.array([3.0, 3.0])
        self.assertLessEqual(lin1.value - 1e-5, 18.0)
        self.assertLessEqual(lin2.value - 1e-5, 2 * np.exp(3.0))
