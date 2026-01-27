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

Tests for zero-sized expressions.
"""
import unittest

import numpy as np
from numpy.testing import assert_array_equal

import cvxpy as cp
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.eliminate_zero_sized import EliminateZeroSized


class TestZeroSizedCreation(unittest.TestCase):
    """Test creation of zero-sized leaves."""

    def test_zero_sized_variable(self):
        x = Variable((2, 0))
        self.assertEqual(x.shape, (2, 0))
        self.assertEqual(x.size, 0)

        y = Variable((0,))
        self.assertEqual(y.shape, (0,))
        self.assertEqual(y.size, 0)

        z = Variable((0, 3))
        self.assertEqual(z.shape, (0, 3))
        self.assertEqual(z.size, 0)

    def test_zero_sized_parameter(self):
        p = cp.Parameter((0, 3))
        self.assertEqual(p.shape, (0, 3))
        self.assertEqual(p.size, 0)

    def test_zero_sized_constant(self):
        c = Constant(np.zeros((0, 3)))
        self.assertEqual(c.shape, (0, 3))
        self.assertEqual(c.size, 0)

    def test_zero_sized_constant_from_empty_array(self):
        c = Constant(np.array([]))
        self.assertEqual(c.shape, (0,))
        self.assertEqual(c.size, 0)


class TestEliminateZeroSizedReduction(unittest.TestCase):
    """Test the EliminateZeroSized reduction directly."""

    def test_no_zero_sized_passthrough(self):
        """Problems without zero-sized vars pass through unchanged."""
        x = Variable(3)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 1])
        reduction = EliminateZeroSized()
        new_prob, inverse_data = reduction.apply(prob)
        self.assertEqual(len(new_prob.constraints), 1)
        self.assertEqual(inverse_data, {})

    def test_zero_sized_constraint_dropped(self):
        """Zero-sized constraints are dropped (vacuously true)."""
        x = Variable(3)
        z = Variable((0,))
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 1, z >= 0])
        reduction = EliminateZeroSized()
        new_prob, inverse_data = reduction.apply(prob)
        # The zero-sized constraint z >= 0 should be dropped.
        self.assertEqual(len(new_prob.constraints), 1)
        self.assertIn(z.id, inverse_data)

    def test_zero_sized_var_replaced_in_objective(self):
        """Zero-sized variables in the objective are replaced with constants."""
        x = Variable(3)
        z = Variable((0,))
        prob = cp.Problem(cp.Minimize(cp.sum(x) + cp.sum(z)), [x >= 1])
        reduction = EliminateZeroSized()
        new_prob, inverse_data = reduction.apply(prob)
        # z should be replaced; problem should have no zero-sized variables.
        zero_vars_in_new = [v for v in new_prob.variables() if v.size == 0]
        self.assertEqual(len(zero_vars_in_new), 0)

    def test_invert_adds_empty_values(self):
        """Invert adds empty array values for eliminated variables."""
        z = Variable((0, 3))
        reduction = EliminateZeroSized()
        zero_vars = {z.id: z}

        # Create a mock solution.
        from cvxpy.reductions.solution import Solution
        sol = Solution("optimal", 0.0, {}, {}, {})
        result = reduction.invert(sol, zero_vars)
        self.assertIn(z.id, result.primal_vars)
        assert_array_equal(result.primal_vars[z.id], np.zeros((0, 3)))

    def test_bounded_variable_domain_preserved(self):
        """Bounded non-zero-sized variable orphaned by zero-sized elimination
        gets its domain constraints re-added to the reduced problem.

        When the only reference to a bounded variable is inside a zero-sized
        constraint, that constraint is dropped.  The variable disappears from
        the reduced problem, but its domain (bounds, nonneg, etc.) must be
        re-added so downstream reductions still see it.
        """
        x = Variable(nonneg=True)
        z = Variable((0,))
        # x only appears inside a zero-sized constraint.
        prob = cp.Problem(cp.Minimize(0), [z + x <= 0])
        reduction = EliminateZeroSized()
        new_prob, inverse_data = reduction.apply(prob)
        # x should NOT be eliminated — its domain constraints keep it alive.
        self.assertNotIn(x.id, inverse_data)
        new_var_ids = {v.id for v in new_prob.variables()}
        self.assertIn(x.id, new_var_ids)
        # The reduced problem should contain domain constraints for x.
        self.assertGreater(len(new_prob.constraints), 0)


class TestZeroSizedSolve(unittest.TestCase):
    """Test solving problems with zero-sized variables."""

    def test_solve_with_zero_sized_variable(self):
        """A problem with a zero-sized variable alongside normal ones solves."""
        y = Variable(3)
        z = Variable((0,))
        prob = cp.Problem(cp.Minimize(cp.sum(y) + cp.sum(z)), [y >= 1])
        prob.solve()
        self.assertIsNotNone(y.value)
        np.testing.assert_allclose(y.value, np.ones(3), atol=1e-5)

    def test_zero_sized_variable_gets_empty_value(self):
        """Zero-sized variables get empty array values after solve."""
        y = Variable(3)
        z = Variable((0,))
        prob = cp.Problem(cp.Minimize(cp.sum(y) + cp.sum(z)), [y >= 1])
        prob.solve()
        self.assertIsNotNone(z.value)
        self.assertEqual(z.value.shape, (0,))

    def test_mixed_problem_zero_constraint(self):
        """Normal objective with zero-sized constraints solves correctly."""
        y = Variable(3)
        z = Variable((0,))
        prob = cp.Problem(cp.Minimize(cp.sum(y)), [y >= 1, z >= 0])
        prob.solve()
        np.testing.assert_allclose(y.value, np.ones(3), atol=1e-5)

    def test_zero_sized_constraint_from_broadcast(self):
        """A variable broadcast into a zero-sized constraint still solves."""
        x = Variable()
        prob = cp.Problem(cp.Minimize(0), [np.array([]) + x <= 0])
        prob.solve()
        self.assertIsNotNone(x.value)

    def test_nonlinear_plus_empty_array(self):
        """A nonlinear expression broadcast with an empty array produces a zero-sized result."""
        x = Variable()
        expr = cp.log(x - 1) + cp.log(2 - x) + np.array([])
        self.assertEqual(expr.shape, (0,))
        self.assertEqual(expr.size, 0)
        prob = cp.Problem(cp.Maximize(cp.sum(expr)))
        prob.solve()
        self.assertEqual(prob.status, cp.OPTIMAL)
        self.assertIsNotNone(x.value)
        self.assertGreater(x.value, 1)
        self.assertLess(x.value, 2)

    def test_zero_sized_constraint_from_broadcast_bounded(self):
        """Bounded variable broadcast into a zero-sized constraint still solves."""
        x = Variable(nonneg=True, bounds=[2, 3])
        prob = cp.Problem(cp.Minimize(0), [np.array([]) + x <= 0])
        prob.solve()
        self.assertIsNotNone(x.value)
        self.assertGreaterEqual(x.value, 2)
        self.assertLessEqual(x.value, 3)


class TestZeroSizedShapes(unittest.TestCase):
    """Test expression shape propagation with zero-sized inputs."""

    def test_sum_zero_sized(self):
        x = Variable((0, 3))
        expr = cp.sum(x)
        self.assertEqual(expr.shape, ())

    def test_hstack_with_zero_sized(self):
        x = Variable((0,))
        y = Variable((3,))
        expr = cp.hstack([x, y])
        self.assertEqual(expr.shape, (3,))

    def test_vstack_with_zero_sized(self):
        x = Variable((0, 2))
        y = Variable((3, 2))
        expr = cp.vstack([x, y])
        self.assertEqual(expr.shape, (3, 2))

    def test_reshape_zero_sized(self):
        x = Variable((0, 3))
        expr = cp.reshape(x, (0, 3))
        self.assertEqual(expr.shape, (0, 3))


if __name__ == "__main__":
    unittest.main()
