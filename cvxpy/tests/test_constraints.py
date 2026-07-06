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
import pytest

import cvxpy as cp
from cvxpy.atoms.affine.reshape import reshape as reshape_atom
from cvxpy.constraints.power import PowCone3D, PowConeND
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest


class TestConstraints(BaseTest):
    """ Unit tests for the expression/expression module. """

    def setUp(self) -> None:
        self.a = Variable(name='a')
        self.b = Variable(name='b')

        self.x = Variable(2, name='x')
        self.y = Variable(3, name='y')
        self.z = Variable(2, name='z')

        self.A = Variable((2, 2), name='A')
        self.B = Variable((2, 2), name='B')
        self.C = Variable((3, 2), name='C')

    def test_boolean_violation(self):
        # https://github.com/cvxpy/cvxpy/issues/2900
        z = cp.Variable(1, boolean=True)
        for value in ([1], [1.0], [True], np.array([1]), np.array([1.0]), np.array([True])):
            with self.subTest(value=value):
                z.value = value

                constraint = z >= 0.6
                actual = constraint.violation()
                expected = np.array([0.0])
                np.testing.assert_array_equal(actual, expected, strict=True)

                constraint = 1 - z <= 0.6
                actual = constraint.violation()
                expected = np.array([0.0])
                np.testing.assert_array_equal(actual, expected, strict=True)

                constraint = z <= 0.6
                actual = constraint.violation()
                expected = np.array([0.4])
                np.testing.assert_array_equal(actual, expected, strict=True)

                constraint = 1 - z >= 0.6
                actual = constraint.violation()
                expected = np.array([0.6])
                np.testing.assert_array_equal(actual, expected, strict=True)

    def test_equality(self) -> None:
        """Test the Equality class.
        """
        constr = self.x == self.z
        self.assertEqual(constr.name(), "x == z")
        self.assertEqual(constr.shape, (2,))
        # Test value and dual_value.
        assert constr.dual_value is None
        with self.assertRaises(ValueError):
            constr.value()
        self.x.save_value(2)
        self.z.save_value(2)
        assert constr.value()
        self.x.save_value(3)
        assert not constr.value()

        self.x.value = np.array([2, 1])
        self.z.value = np.array([2, 2])
        assert not constr.value()
        self.assertItemsAlmostEqual(constr.violation(), [0, 1])
        self.assertItemsAlmostEqual(constr.residual, [0, 1])

        self.z.value = np.array([2, 1])
        assert constr.value()
        self.assertItemsAlmostEqual(constr.violation(), [0, 0])
        self.assertItemsAlmostEqual(constr.residual, [0, 0])

        # Incompatible dimensions
        with self.assertRaises(ValueError):
            (self.x == self.y)

        # Test copy with args=None
        copy = constr.copy()
        self.assertTrue(type(copy) is type(constr))
        # A new object is constructed, so copy.args == constr.args but copy.args
        # is not constr.args.
        self.assertEqual(copy.args, constr.args)
        self.assertFalse(copy.args is constr.args)
        # Test copy with new args
        copy = constr.copy(args=[self.A, self.B])
        self.assertTrue(type(copy) is type(constr))
        self.assertTrue(copy.args[0] is self.A)

    def test_inequality(self) -> None:
        """Test the Inequality class.
        """
        constr = self.x <= self.z
        self.assertEqual(constr.name(), "x <= z")
        self.assertEqual(constr.shape, (2,))
        # Test value and dual_value.
        assert constr.dual_value is None
        with self.assertRaises(ValueError):
            constr.value()
        self.x.save_value(1)
        self.z.save_value(2)
        assert constr.value()
        self.x.save_value(3)
        assert not constr.value()
        # self.assertItemsEqual(constr.variables().keys(), [self.x.id, self.z.id])

        self.x.value = np.array([2, 1])
        self.z.value = np.array([2, 0])
        assert not constr.value()
        self.assertItemsAlmostEqual(constr.violation(), [0, 1])
        self.assertItemsAlmostEqual(constr.residual, [0, 1])

        self.z.value = np.array([2, 2])
        assert constr.value()
        self.assertItemsAlmostEqual(constr.violation(), [0, 0])
        self.assertItemsAlmostEqual(constr.residual, [0, 0])

        # Incompatible dimensions
        with self.assertRaises(Exception):
            (self.x <= self.y)

        # Test copy with args=None
        copy = constr.copy()
        self.assertTrue(type(copy) is type(constr))
        # A new object is constructed, so copy.args == constr.args but copy.args
        # is not constr.args.
        self.assertEqual(copy.args, constr.args)
        self.assertFalse(copy.args is constr.args)
        # Test copy with new args
        copy = constr.copy(args=[self.A, self.B])
        self.assertTrue(type(copy) is type(constr))
        self.assertTrue(copy.args[0] is self.A)

    def test_psd_constraint(self) -> None:
        """Test the PSD constraint <<.
        """
        constr = self.A >> self.B
        self.assertEqual(constr.name(), "A + -B >> 0")
        self.assertEqual(constr.shape, (2, 2))
        # Test value and dual_value.
        assert constr.dual_value is None
        with self.assertRaises(ValueError):
            constr.value()
        self.A.save_value(np.array([[2, -1], [1, 2]]))
        self.B.save_value(np.array([[1, 0], [0, 1]]))
        assert constr.value()
        self.assertAlmostEqual(constr.violation(), 0)
        self.assertAlmostEqual(constr.residual, 0)

        self.B.save_value(np.array([[3, 0], [0, 3]]))
        assert not constr.value()
        self.assertAlmostEqual(constr.violation(), 1)
        self.assertAlmostEqual(constr.residual, 1)

        with self.assertRaises(Exception) as cm:
            (self.x >> 0)
        self.assertEqual(str(cm.exception), "Non-square matrix in positive definite constraint.")

        # Test copy with args=None
        copy = constr.copy()
        self.assertTrue(type(copy) is type(constr))
        # A new object is constructed, so copy.args == constr.args but copy.args
        # is not constr.args.
        self.assertEqual(copy.args, constr.args)
        self.assertFalse(copy.args is constr.args)
        # Test copy with new args
        copy = constr.copy(args=[self.B])
        self.assertTrue(type(copy) is type(constr))
        self.assertTrue(copy.args[0] is self.B)

    def test_nsd_constraint(self) -> None:
        """Test the PSD constraint <<.
        """
        constr = self.A << self.B
        self.assertEqual(constr.name(), "B + -A >> 0")
        self.assertEqual(constr.shape, (2, 2))
        # Test value and dual_value.
        assert constr.dual_value is None
        with self.assertRaises(ValueError):
            constr.value()
        self.B.save_value(np.array([[2, -1], [1, 2]]))
        self.A.save_value(np.array([[1, 0], [0, 1]]))
        assert constr.value()
        self.A.save_value(np.array([[3, 0], [0, 3]]))
        assert not constr.value()

        with self.assertRaises(Exception) as cm:
            self.x << 0
        self.assertEqual(str(cm.exception),
                         "Non-square matrix in positive definite constraint.")

    def test_geq(self) -> None:
        """Test the >= operator.
        """
        constr = self.z >= self.x
        self.assertEqual(constr.name(), "x <= z")
        self.assertEqual(constr.shape, (2,))

        # Incompatible dimensions
        with self.assertRaises(ValueError):
            (self.y >= self.x)

    # Test the SOC class.
    def test_soc_constraint(self) -> None:
        exp = self.x + self.z
        scalar_exp = self.a + self.b
        constr = SOC(scalar_exp, exp)
        self.assertEqual(constr.size, 3)

        # Test invalid dimensions.
        error_str = ("Argument dimensions (1,) and (1, 4), with axis=0, "
                     "are incompatible.")
        with self.assertRaises(Exception) as cm:
            SOC(Variable(1), Variable((1, 4)))
        self.assertEqual(str(cm.exception), error_str)

        # Test residual.
        # 1D
        n = 5
        x0 = np.arange(n)
        t0 = 2
        x = cp.Variable(n, value=x0)
        t = cp.Variable(value=t0)
        resid = SOC(t, x).residual
        assert resid.ndim == 0
        dist = cp.sum_squares(x - x0) + cp.square(t - t0)
        prob = cp.Problem(cp.Minimize(dist), [SOC(t, x)])
        prob.solve()
        self.assertAlmostEqual(np.sqrt(dist.value), resid)

        # 2D, axis = 0.
        n = 5
        k = 3
        x0 = np.arange(n * k).reshape((n, k))
        t0 = np.array([1, 2, 3])
        x = cp.Variable((n, k), value=x0)
        t = cp.Variable(k, value=t0)
        resid = SOC(t, x, axis=0).residual
        assert resid.shape == (k,)
        for i in range(k):
            dist = cp.sum_squares(x[:, i] - x0[:, i]) + cp.sum_squares(t[i] - t0[i])
            prob = cp.Problem(cp.Minimize(dist), [SOC(t[i], x[:, i])])
            prob.solve()
            self.assertAlmostEqual(np.sqrt(dist.value), resid[i])

        # 2D, axis = 1.
        n = 5
        k = 3
        x0 = np.arange(n * k).reshape((k, n))
        t0 = np.array([1, 2, 3])
        x = cp.Variable((k, n), value=x0)
        t = cp.Variable(k, value=t0)
        resid = SOC(t, x, axis=1).residual
        assert resid.shape == (k,)
        for i in range(k):
            dist = cp.sum_squares(x[i, :] - x0[i, :]) + cp.sum_squares(t[i] - t0[i])
            prob = cp.Problem(cp.Minimize(dist), [SOC(t[i], x[i, :])])
            prob.solve()
            self.assertAlmostEqual(np.sqrt(dist.value), resid[i])

        # Test all three cases:
        # 1. t >= ||x||
        # 2. -||x|| < t < ||x||
        # 3. t <= -||x||

        k, n = 3, 3
        x0 = np.ones((k, n))
        norms = np.linalg.norm(x0, ord=2)
        t0 = np.array([2, 0.5, -2]) * norms
        x = cp.Variable((k, n), value=x0)
        t = cp.Variable(k, value=t0)
        resid = SOC(t, x, axis=1).residual
        assert resid.shape == (k,)
        for i in range(k):
            dist = cp.sum_squares(x[i, :] - x0[i, :]) + cp.sum_squares(t[i] - t0[i])
            prob = cp.Problem(cp.Minimize(dist), [SOC(t[i], x[i, :])])
            prob.solve()
            self.assertAlmostEqual(np.sqrt(dist.value), resid[i], places=4)

    def test_soc_constraint_scalar(self) -> None:
        """Test SOC constraint with scalar X (issue #3054)."""
        # Test basic scalar SOC: min c s.t. |x| <= c
        c = cp.Variable()
        x = cp.Variable()
        constr = SOC(c, x)

        # Verify constraint properties
        self.assertEqual(constr.size, 2)  # 1 for t + 1 for scalar x
        self.assertEqual(constr.cone_sizes(), [2])
        self.assertEqual(constr.num_cones(), 1)

        # Solve problem: optimal is c=0, x=0
        prob = cp.Problem(cp.Minimize(c), [constr])
        prob.solve()
        self.assertAlmostEqual(c.value, 0, places=4)
        self.assertAlmostEqual(x.value, 0, places=4)

        # Test with x constrained: min c s.t. |x| <= c, x == 3
        c = cp.Variable()
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(c), [SOC(c, x), x == 3])
        prob.solve()
        self.assertAlmostEqual(c.value, 3, places=4)
        self.assertAlmostEqual(x.value, 3, places=4)

        # Test residual for scalar X
        x0, t0 = 2.0, 3.0  # Feasible: |x0| < t0
        x = cp.Variable(value=x0)
        t = cp.Variable(value=t0)
        resid = SOC(t, x).residual
        self.assertEqual(resid.ndim, 0)
        dist = cp.sum_squares(x - x0) + cp.square(t - t0)
        prob = cp.Problem(cp.Minimize(dist), [SOC(t, x)])
        prob.solve()
        self.assertAlmostEqual(np.sqrt(dist.value), resid, places=4)

    def test_pow3d_constraint(self) -> None:
        n = 3
        np.random.seed(0)
        alpha = 0.275
        x, y, z = Variable(n), Variable(n), Variable(n)
        con = PowCone3D(x, y, z, alpha)
        # check violation against feasible values
        x0, y0 = 0.1 + np.random.rand(n), 0.1 + np.random.rand(n)
        z0 = x0**alpha * y0**(1-alpha)
        z0[1] *= -1
        x.value, y.value, z.value = x0, y0, z0
        viol = con.residual
        self.assertLessEqual(viol, 1e-7)
        # check violation against infeasible values
        x1 = x0.copy()
        x1[0] *= -0.9
        x.value = x1
        viol = con.residual
        self.assertGreaterEqual(viol, 0.99*abs(x1[0]))
        # check invalid constraint data
        with self.assertRaises(ValueError):
            con = PowCone3D(x, y, z, 1.001)
        with self.assertRaises(ValueError):
            con = PowCone3D(x, y, z, -0.00001)

    def test_pow3d_scalar_alpha_constraint(self) -> None:
        """
        Simple test case with scalar AND vector `alpha`
        inputs to `PowCone3D`
        """""
        x_0 = cp.Variable(shape=(3,))
        x = cp.Variable(shape=(3,))
        cons = [cp.PowCone3D(x_0[0], x_0[1], x_0[2], 0.25),
                x <= -10]
        obj = cp.Minimize(cp.norm(x - x_0))
        prob = cp.Problem(obj, cons)
        prob.solve()
        self.assertAlmostEqual(prob.value, 17.320508075380552)

    def test_pownd_constraint(self) -> None:
        n = 4
        W, z = Variable(n), Variable()
        np.random.seed(0)
        alpha = 0.5 + np.random.rand(n)
        alpha /= np.sum(alpha)
        with self.assertRaises(ValueError):
            # entries don't sum to one
            con = PowConeND(W, z, alpha+0.01)
        with self.assertRaises(ValueError):
            # shapes don't match exactly
            con = PowConeND(W, z, alpha.reshape((n, 1)))
        with self.assertRaises(ValueError):
            # wrong axis
            con = PowConeND(reshape_atom(W, (n, 1), order='F'), z,
                            alpha.reshape((n, 1)),
                            axis=1)
        # Compute a violation
        con = PowConeND(W, z, alpha)
        W0 = 0.1 + np.random.rand(n)
        z0 = np.prod(np.power(W0, alpha))+0.05
        W.value, z.value = W0, z0
        viol = con.violation()
        self.assertGreaterEqual(viol, 0.01)
        self.assertLessEqual(viol, 0.06)

    def test_chained_constraints(self) -> None:
        """Tests that chaining constraints raises an error.
        """
        error_str = ("Cannot evaluate the truth value of a constraint or "
                     "chain constraints, e.g., 1 >= x >= 0.")
        with self.assertRaises(Exception) as cm:
            (self.z <= self.x <= 1)
        self.assertEqual(str(cm.exception), error_str)

        with self.assertRaises(Exception) as cm:
            (self.x == self.z == 1)
        self.assertEqual(str(cm.exception), error_str)

        with self.assertRaises(Exception) as cm:
            (self.z <= self.x).__bool__()
        self.assertEqual(str(cm.exception), error_str)

    def test_nonneg(self) -> None:
        """Solve a trivial NonNeg-constrained problem through
        the conic and QP code paths.
        """
        x = cp.Variable(3)
        c = np.arange(3)
        prob = cp.Problem(cp.Minimize(cp.sum(x)),
                          [cp.NonNeg(x - c)])
        prob.solve(solver=cp.CLARABEL)
        self.assertItemsAlmostEqual(x.value, c)
        prob.solve(solver=cp.OSQP)
        self.assertItemsAlmostEqual(x.value, c)

    def test_nonpos(self) -> None:
        """Tests the NonPos constraint for correctness with conic and
        QP code paths.
        """
        x = cp.Variable(3)
        c = np.arange(3)
        prob = cp.Problem(cp.Maximize(cp.sum(x)), [cp.NonPos(x - c)])
        prob.solve(solver=cp.CLARABEL)
        self.assertItemsAlmostEqual(x.value, c)
        prob.solve(solver=cp.OSQP)
        self.assertItemsAlmostEqual(x.value, c)

    def test_nonneg_dual(self) -> None:
        # Compute reference solution with an Inequality constraint.
        x = cp.Variable(3)
        c = np.arange(3)
        objective = cp.Minimize(cp.sum(x))
        prob = cp.Problem(objective, [c - x <= 0])
        prob.solve(solver=cp.CLARABEL)
        dual = prob.constraints[0].dual_value
        # reported dual variables are the same with NonNeg, even though
        # the convention for how they add to the Lagrangian differs by sign.
        prob = cp.Problem(objective, [cp.NonNeg(x - c)])
        prob.solve(solver=cp.CLARABEL)
        self.assertItemsAlmostEqual(prob.constraints[0].dual_value, dual)
        prob.solve(solver=cp.OSQP)
        self.assertItemsAlmostEqual(prob.constraints[0].dual_value, dual)

    def test_bound_properties(self) -> None:
        """Test basic bound properties."""
        assert cp.Variable(bounds=[1, None])._has_lower_bounds()
        assert not cp.Variable(bounds=[1, None])._has_upper_bounds()
        assert not cp.Variable(bounds=[None, 1])._has_lower_bounds()
        assert cp.Variable(bounds=[None, 1])._has_upper_bounds()
        assert cp.Variable(bounds=[1, 2])._has_lower_bounds()
        assert cp.Variable(bounds=[1, 2])._has_upper_bounds()

    def test_broadcasting(self) -> None:
        """Test interaction of constraints and broadcasting."""
        x = cp.Variable((3, 1))
        c = cp.Constant(np.ones(3))
        # Equality constraint.
        con = (x == c)
        assert con.shape == (3, 3)
        prob = cp.Problem(cp.Minimize(0), [con])
        prob.solve(solver=cp.CLARABEL)
        assert np.allclose(x.value, c.value)

        with pytest.raises(ValueError, match="The CPP backend cannot be used"):
            prob = cp.Problem(cp.Minimize(0), [con])
            prob.solve(solver=cp.CLARABEL, canon_backend=cp.CPP_CANON_BACKEND)

        # Test QP codepath.
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [con])
        prob.solve(solver=cp.OSQP)
        assert np.allclose(x.value, c.value)

        with pytest.raises(ValueError, match="The CPP backend cannot be used"):
            prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [con])
            prob.solve(solver=cp.OSQP, canon_backend=cp.CPP_CANON_BACKEND)

        # Inequality constraint.
        con = (x >= c)
        assert con.shape == (3, 3)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [con])
        prob.solve(solver=cp.CLARABEL)
        assert np.allclose(x.value, c.value)

        with pytest.raises(ValueError, match="The CPP backend cannot be used"):
            prob = cp.Problem(cp.Minimize(cp.sum(x)), [con])
            prob.solve(solver=cp.CLARABEL, canon_backend=cp.CPP_CANON_BACKEND)

        # TODO other constraints


    def test_bounds_attr(self) -> None:
        """Test that the bounds attribute for variables and parameters is set correctly.
        """
        # Test if bounds attribute generates correct bounds
        Q = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        x_1 = cp.Variable((3,), bounds=[np.array([1,2,3]), np.array([4,5,6])])
        x_2 = cp.Variable((2,), bounds=[1,2])
        x_3 = cp.Variable((2,), bounds=[np.array([4,5]), 6])
        x_4 = cp.Variable((3,), bounds=[1, np.array([2,3,4])])
        c_1 = np.array([1,1])
        c_2 = np.array([1,1,1])

        # Case 1: Check solution for lower and upper bound arrays
        # Check if bounds attribute is compatible with linear objective
        cp.Problem(cp.Minimize(x_1@c_2)).solve()
        self.assertItemsAlmostEqual(x_1.value, [1, 2, 3])
        # Check value validation and project.
        self.assertItemsAlmostEqual(x_1.project([0, 0, 0]), [1, 2, 3])
        with pytest.raises(ValueError, match="in bounds."):
            x_1.value = [0, 0, 0]

        # Try again using domain.
        z = cp.Variable(shape=x_1.shape, var_id=x_1.id)
        cp.Problem(cp.Minimize(z@c_2), x_1.domain).solve()
        self.assertItemsAlmostEqual(z.value, [1, 2, 3])

        # Check if bounds attribute is compatible with quadratic objective
        cp.Problem(cp.Minimize(cp.quad_form(x_1, Q) + c_2.T @ x_1)).solve()
        self.assertItemsAlmostEqual(x_1.value,[1,2,3])

        # Case 2: Check solution for scalar lower and upper bounds
        cp.Problem(cp.Minimize(x_2@c_1)).solve()
        self.assertItemsAlmostEqual(x_2.value, [1, 1])

        # Case 3: Check solution for lower bound array and scalar upper bound
        cp.Problem(cp.Maximize(x_3@c_1)).solve()
        self.assertItemsAlmostEqual(x_3.value, [6, 6])

        # Case 4: Check solution for scalar lower bound and upper bound array
        cp.Problem(cp.Maximize(x_4@c_2)).solve()
        self.assertItemsAlmostEqual(x_4.value, [2, 3, 4])

        # Check if bounds are a list of 2 items
        with pytest.raises(ValueError, match="Bounds should be a list of two items."):
            cp.Variable((2,), bounds=[np.array([0, 1, 2])])

        # Check for mismatch in dimensions of lower and upper bounds
        with pytest.raises(ValueError, match="with the same dimensions as the variable/parameter."):
            cp.Variable((2,), bounds=[np.array([1,2]), np.array([1,2,3])])
        with pytest.raises(ValueError, match="with the same dimensions as the variable/parameter."):
            cp.Variable((2,), bounds=[np.array([1,2,3,4]), 5])

        # Check that bounds attribute handles -inf and inf correctly
        x_5 = cp.Variable((2,), bounds=[-np.inf, np.array([1,2])])
        x_6 = cp.Variable((3,), bounds=[3, np.inf])
        x_7 = cp.Variable((2,),bounds=[-np.inf, np.inf])
        cp.Problem(cp.Maximize(x_5@c_1)).solve()
        self.assertItemsAlmostEqual(x_5.value, [1, 2])
        # Check value validation and project.
        self.assertItemsAlmostEqual(x_5.project([2, 1]), [1, 1])
        x_5.value = [0, 0]
        with pytest.raises(ValueError, match="in bounds."):
            x_5.value = [2, 1]
        cp.Problem(cp.Minimize(x_6@c_2)).solve()
        self.assertItemsAlmostEqual(x_6.value, [3,3,3])
        # Check that adding constraints are handled correctly for unbounded domain
        # (no error from solver)
        cp.Problem(cp.Minimize(x_7 @ c_1)).solve()
        self.assertIsNone(x_7.value)

        # Test mix of lower and upper bounds.
        lower_bounds = [None, -np.inf, 1]
        upper_bounds = [None, np.inf, 2]
        for lower, upper in zip(lower_bounds, upper_bounds):
            z = cp.Variable(bounds=[lower, upper])

            min_z = cp.Problem(cp.Minimize(z)).solve()
            if lower is None:
                lower = -np.inf
            assert np.isclose(min_z, lower)

            max_z = cp.Problem(cp.Maximize(z)).solve()
            if upper is None:
                upper = np.inf
            assert np.isclose(max_z, upper)

        with pytest.raises(ValueError,match="Invalid bounds: some upper "
                                            "bounds are less than corresponding lower bounds."):
            cp.Variable((2,), bounds=[np.array([2,3]), np.array([1,4])])

        with pytest.raises(ValueError, match="-np.inf is not feasible as an upper bound."):
            cp.Variable((2,), bounds=[None, -np.inf])
        with pytest.raises(ValueError, match="-np.inf is not feasible as an upper bound."):
            cp.Variable((2,), bounds=[-np.inf, np.array([1,-np.inf])])
        with pytest.raises(ValueError, match="-np.inf is not feasible as an upper bound."):
            cp.Variable((2,), bounds=[np.array([1, -np.inf]), np.array([2, -np.inf])])

        with pytest.raises(ValueError, match="np.inf is not feasible as a lower bound."):
            cp.Variable((2,), bounds=[np.inf, np.inf])
        with pytest.raises(ValueError, match="np.inf is not feasible as a lower bound."):
            cp.Variable((2,), bounds=[np.array([1,np.inf]), np.inf])
        with pytest.raises(ValueError, match="np.inf is not feasible as a lower bound."):
            cp.Variable((2,), bounds=[np.array([1, np.inf]), np.array([2, np.inf])])

        with pytest.raises(ValueError, match="np.nan is not feasible as lower or upper bound."):
            cp.Variable((2,), bounds=[np.nan, np.nan])
        with pytest.raises(ValueError, match="np.nan is not feasible as lower or upper bound."):
            cp.Variable((2,), bounds=[np.nan, np.array([1, np.nan])])
        with pytest.raises(ValueError, match="np.nan is not feasible as lower or upper bound."):
            cp.Variable((2,), bounds=[np.array([1, np.nan]), np.nan])
        with pytest.raises(ValueError, match="np.nan is not feasible as lower or upper bound."):
            cp.Variable((2,), bounds=[np.array([1, np.nan]), np.array([2, np.nan])])

    def test_psd_residual_complex(self) -> None:
        """PSD.residual should use .H (not .T) for complex Hermitian matrices."""
        X = cp.Variable((2, 2), hermitian=True)
        constr = X >> 0
        # A Hermitian PSD matrix
        X.value = np.array([[2, 1+1j], [1-1j, 2]])
        self.assertAlmostEqual(constr.residual, 0.0)
        # A non-PSD Hermitian matrix
        X.value = np.array([[1, 3+1j], [3-1j, 1]])
        self.assertTrue(constr.residual > 0)

    def test_finite_set_residual_none(self) -> None:
        """FiniteSet.residual should return None when variable has no value."""
        from cvxpy.constraints.finite_set import FiniteSet
        x = cp.Variable()
        constr = FiniteSet(x, [1, 2, 3])
        self.assertIsNone(constr.residual)

    def test_dual_cone_no_args(self) -> None:
        """_dual_cone() without args should work for PSD and SOC."""
        from cvxpy.constraints.psd import PSD
        # PSD
        X = cp.Variable((2, 2), symmetric=True)
        psd_constr = X >> 0
        prob = cp.Problem(cp.Minimize(cp.trace(X)), [psd_constr, cp.trace(X) >= 1])
        prob.solve(solver=cp.CLARABEL)
        self.assertEqual(prob.status, cp.OPTIMAL)
        dual_constr = psd_constr._dual_cone()
        self.assertIsInstance(dual_constr, PSD)

        # SOC
        x = cp.Variable(3)
        t = cp.Variable()
        soc_constr = SOC(t, x)
        prob = cp.Problem(cp.Minimize(t), [soc_constr, x == np.ones(3)])
        prob.solve(solver=cp.CLARABEL)
        self.assertEqual(prob.status, cp.OPTIMAL)
        dual_constr = soc_constr._dual_cone()
        self.assertIsInstance(dual_constr, SOC)

    def test_exponential_cone_metadata_duals_and_validation(self) -> None:
        x = cp.Variable(2)
        y = cp.Variable(2)
        z = cp.Variable(2)
        con = cp.constraints.ExpCone(x, y, z)

        self.assertTrue(str(con).startswith("ExpCone("))
        self.assertTrue(repr(con).startswith("ExpCone("))
        self.assertIsNone(con.residual)
        self.assertEqual(con.size, 6)
        self.assertEqual(con.num_cones(), 2)
        self.assertEqual(con.cone_sizes(), [3, 3])
        self.assertEqual(con.shape, (3, 2))
        self.assertTrue(con.is_dcp())
        self.assertTrue(con.is_dcp(dpp=True))
        self.assertFalse(con.is_dgp())
        self.assertTrue(con.is_dqcp())
        quad_approx = con.as_quad_approx(2, 3)
        self.assertIsInstance(quad_approx, cp.constraints.RelEntrConeQuad)
        self.assertIs(quad_approx.x, y)
        self.assertIs(quad_approx.y, z)
        self.assertEqual(quad_approx.m, 2)
        self.assertEqual(quad_approx.k, 3)
        x.value = np.array([1.0, 2.0])
        y.value = np.array([3.0, 4.0])
        z.value = np.array([5.0, 6.0])
        np.testing.assert_allclose(quad_approx.z.value, -x.value)

        dual = con._dual_cone()
        self.assertIsInstance(dual, cp.constraints.ExpCone)
        explicit_dual = con._dual_cone(x, y, z)
        self.assertIsInstance(explicit_dual, cp.constraints.ExpCone)
        np.testing.assert_allclose(explicit_dual.x.value, -y.value)
        np.testing.assert_allclose(explicit_dual.y.value, -x.value)
        np.testing.assert_allclose(explicit_dual.z.value, np.exp(1) * z.value)
        with self.assertRaises(AssertionError):
            con._dual_cone(cp.Variable(1), y, z)

        con.save_dual_value(np.arange(6))
        np.testing.assert_allclose(con.dual_variables[0].value, [0, 3])
        np.testing.assert_allclose(con.dual_variables[1].value, [1, 4])
        np.testing.assert_allclose(con.dual_variables[2].value, [2, 5])
        np.testing.assert_allclose(dual.x.value, -con.dual_variables[1].value)
        np.testing.assert_allclose(dual.y.value, -con.dual_variables[0].value)
        np.testing.assert_allclose(dual.z.value, np.exp(1) * con.dual_variables[2].value)

        with pytest.raises(ValueError, match="affine and real"):
            cp.constraints.ExpCone(cp.square(x), y, z)
        with pytest.raises(ValueError, match="same shapes"):
            cp.constraints.ExpCone(cp.Variable(1), y, z)

    def test_exp_cone_matrix_arg_duals(self) -> None:
        """Duals of an ExpCone with matrix args must match the flattened problem.

        ConeMatrixStuffing flattens matrix args in Fortran order;
        save_dual_value used to reshape the recovered duals in C order,
        permuting the dual entries whenever both dimensions exceed one.
        """
        m, k = 2, 3
        rng = np.random.default_rng(4)
        c = rng.uniform(0.5, 1, (m, k))
        x, y, z = cp.Variable((m, k)), cp.Variable((m, k)), cp.Variable((m, k))
        con = cp.constraints.ExpCone(x, y, z)
        objective = -cp.sum(cp.multiply(c, x)) + cp.sum(y) + cp.sum(z)
        cp.Problem(cp.Minimize(objective), [con]).solve(solver=cp.CLARABEL)

        xf, yf, zf = cp.Variable(m * k), cp.Variable(m * k), cp.Variable(m * k)
        con_f = cp.constraints.ExpCone(xf, yf, zf)
        objective_f = -c.flatten('F') @ xf + cp.sum(yf) + cp.sum(zf)
        cp.Problem(cp.Minimize(objective_f), [con_f]).solve(solver=cp.CLARABEL)

        for dv, dvf in zip(con.dual_value, con_f.dual_value):
            np.testing.assert_allclose(dv, dvf.reshape((m, k), order='F'), atol=1e-6)

    def test_pow_cone_3d_matrix_arg_duals(self) -> None:
        """Duals of a PowCone3D with matrix args must match the flattened problem.

        ConeMatrixStuffing restores PowCone3D matrix duals to a (3, m, k)
        array before save_dual_value runs; each coordinate block must then
        be reshaped in Fortran order.
        """
        m, k = 2, 3
        alpha = 0.4
        rng = np.random.default_rng(7)
        c = rng.uniform(0.5, 1, (m, k))
        x = cp.Variable((m, k))
        y = cp.Variable((m, k))
        z = cp.Variable((m, k))
        con = PowCone3D(x, y, z, alpha)
        objective = cp.sum(x) + cp.sum(y) - cp.sum(cp.multiply(c, z))
        cp.Problem(cp.Minimize(objective), [con]).solve(solver=cp.CLARABEL)

        xf = cp.Variable(m * k)
        yf = cp.Variable(m * k)
        zf = cp.Variable(m * k)
        con_f = PowCone3D(xf, yf, zf, alpha)
        objective_f = cp.sum(xf) + cp.sum(yf) - c.flatten('F') @ zf
        cp.Problem(cp.Minimize(objective_f), [con_f]).solve(solver=cp.CLARABEL)

        for dv, dvf in zip(con.dual_value, con_f.dual_value):
            np.testing.assert_allclose(dv, dvf.reshape((m, k), order='F'), atol=1e-6)

    def test_relative_entropy_quad_cones_metadata_and_validation(self) -> None:
        x = cp.Variable(2)
        y = cp.Variable(2)
        z = cp.Variable(2)
        con = cp.constraints.RelEntrConeQuad(x, y, z, 2, 3)

        self.assertEqual(con.get_data(), [2, 3, con.id])
        self.assertTrue(str(con).startswith("RelEntrConeQuad("))
        self.assertTrue(repr(con).startswith("RelEntrConeQuad("))
        self.assertIsNone(con.residual)
        self.assertEqual(con.size, 6)
        self.assertEqual(con.num_cones(), 2)
        self.assertEqual(con.cone_sizes(), [3, 3])
        self.assertTrue(con.is_dcp())
        self.assertTrue(con.is_dcp(dpp=True))
        self.assertFalse(con.is_dgp())
        self.assertTrue(con.is_dqcp())
        self.assertEqual(con.shape, (3, 2))
        self.assertIsNone(con.save_dual_value(None))

        with pytest.raises(ValueError, match="affine and real"):
            cp.constraints.RelEntrConeQuad(cp.square(x), y, z, 2, 3)
        with pytest.raises(ValueError, match="same shapes"):
            cp.constraints.RelEntrConeQuad(cp.Variable(1), y, z, 2, 3)

        X = cp.Variable((2, 2), symmetric=True)
        Y = cp.Variable((2, 2), symmetric=True)
        Z = cp.Variable((2, 2), symmetric=True)
        op_con = cp.constraints.OpRelEntrConeQuad(X, Y, Z, 2, 3)
        self.assertEqual(op_con.get_data(), [2, 3, op_con.id])
        self.assertTrue(str(op_con).startswith("OpRelEntrConeQuad("))
        self.assertTrue(repr(op_con).startswith("OpRelEntrConeQuad("))
        with self.assertRaises(NotImplementedError):
            op_con.residual
        self.assertEqual(op_con.size, 12)
        self.assertEqual(op_con.num_cones(), 4)
        self.assertEqual(op_con.cone_sizes(), [3, 3, 3, 3])
        self.assertTrue(op_con.is_dcp())
        self.assertTrue(op_con.is_dcp(dpp=True))
        self.assertFalse(op_con.is_dgp())
        self.assertTrue(op_con.is_dqcp())
        self.assertEqual(op_con.shape, (3, 2, 2))
        self.assertIsNone(op_con.save_dual_value(None))

        with pytest.raises(ValueError, match="same shapes"):
            cp.constraints.OpRelEntrConeQuad(cp.Variable((1, 1), symmetric=True), Y, Z, 2, 3)
