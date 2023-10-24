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
import torch

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
            con = PowConeND(reshape_atom(W, (n, 1)), z,
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
        prob.solve(solver=cp.ECOS)
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
        prob.solve(solver=cp.ECOS)
        self.assertItemsAlmostEqual(x.value, c)
        prob.solve(solver=cp.OSQP)
        self.assertItemsAlmostEqual(x.value, c)

    def test_nonneg_dual(self) -> None:
        # Compute reference solution with an Inequality constraint.
        x = cp.Variable(3)
        c = np.arange(3)
        objective = cp.Minimize(cp.sum(x))
        prob = cp.Problem(objective, [c - x <= 0])
        prob.solve(solver=cp.ECOS)
        dual = prob.constraints[0].dual_value
        # reported dual variables are the same with NonNeg, even though
        # the convention for how they add to the Lagrangian differs by sign.
        prob = cp.Problem(objective, [cp.NonNeg(x - c)])
        prob.solve(solver=cp.ECOS)
        self.assertItemsAlmostEqual(prob.constraints[0].dual_value, dual)
        prob.solve(solver=cp.OSQP)
        self.assertItemsAlmostEqual(prob.constraints[0].dual_value, dual)

    def test_constraint_torch_exp(self) -> None:
        #Tests generating a torch expression from a constraint
        m = 2
        n = 3
        s0 = np.array([-3, 2])
        s0 = np.maximum(s0, 0)
        x0 = np.ones(n)
        A = np.array([[1, -1, 2], [3, 1, -1]]) #2x3
        b = A @ x0 + s0 #[2,5]

        x = cp.Variable(n)
        z = cp.Variable(m)
        w = cp.Parameter(n)
        w.value=np.ones(n)
        X = cp.Variable((m,n))
        Y = cp.Parameter((m,n))

        constraint1 = (A @ x + z <= b) #Arbitrary constraint
        constraint2 = (z==0) #Equality
        constraint3 = (w@x <= 1) # <=
        constraint4 = (w@x >= 1) # >=
        constraint5 = 5*cp.norm(A@x) <= 1 #Unary operation
        constraint6 = X@Y.T >= 0
        constraint7 = X@Y.T <= 0
        tmp2kill = X@Y.T

        exp1, _ = constraint1.gen_torch_exp()
        exp2, _ = constraint2.gen_torch_exp()
        exp3, _ = constraint3.gen_torch_exp()
        exp4, _ = constraint4.gen_torch_exp()
        exp5, _ = constraint5.gen_torch_exp()
        exp6, _ = constraint6.gen_torch_exp()
        exp7, _ = constraint7.gen_torch_exp()

        x_test = np.array([1,2,3])
        z_test = np.zeros(m)
        w_test = np.array([-1,0,1])
        T1 = np.ones((m,n))
        T2 = np.ones((m,n))

        test1 = exp1(x_test, z_test)
        test2 = exp2(z_test)
        test3 = exp3(w_test, x_test)
        test4 = exp4(w_test, x_test)
        test5 = exp5(x_test)
        test6 = exp6(T1, T2)
        test7 = exp7(T1, T2)

        self.assertTrue(all(test1==torch.tensor([3, -3])))
        self.assertTrue(all(test2==np.array([0, 0])))
        self.assertTrue(all(test3==torch.tensor([1])))
        self.assertTrue(all(test4==torch.tensor([-1])))
        self.assertTrue(np.isclose(test5, 25.9258))
        self.assertTrue((test6==-n*np.ones((m,m))).all())
        self.assertTrue((test7==n*np.ones((m,m))).all())