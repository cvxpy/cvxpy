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
from cvxpy import Maximize, Minimize, Problem
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
from cvxpy.transforms import linearize
from cvxpy.transforms.partial_optimize import partial_optimize


class TestGrad(BaseTest):
    """ Unit tests for the grad module. """

    def setUp(self) -> None:
        self.a = Variable(name='a')

        self.x = Variable(2, name='x')
        self.y = Variable(2, name='y')

        self.A = Variable((2, 2), name='A')
        self.B = Variable((2, 2), name='B')
        self.C = Variable((3, 2), name='C')
        self.D = Variable((1, 2), name='D')

    def test_affine_prod(self) -> None:
        """Test gradient for affine_prod
        """
        expr = self.C @ self.A
        self.C.value = np.array([[1, -2], [3, 4], [-1, -3]])
        self.A.value = np.array([[3, 2], [-5, 1]])

        arr_val = np.array([[3, 0, 0, 2, 0, 0], [0, 3, 0, 0, 2, 0], [0, 0, 3, 0, 0, 2],
                            [-5, 0, 0, 1, 0, 0], [0, -5, 0, 0, 1, 0], [0, 0, -5, 0, 0, 1]])
        self.assertItemsAlmostEqual(expr.grad[self.C].toarray(), arr_val)
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(),
                                    np.array([[1, 3, -1, 0, 0, 0], [-2, 4, -3, 0, 0, 0],
                                              [0, 0, 0, 1, 3, -1], [0, 0, 0, -2, 4, -3]]))

    def test_pnorm(self) -> None:
        """Test gradient for pnorm
        """
        expr = cp.pnorm(self.x, 1)
        self.x.value = [-1, 0]
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), [-1, 0])

        self.x.value = [0, 10]
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), [0, 1])

        expr = cp.pnorm(self.x, 2)
        self.x.value = [-3, 4]
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), np.array([[-3.0/5], [4.0/5]]))

        expr = cp.pnorm(self.x, 0.5)
        self.x.value = [-1, 2]
        self.assertAlmostEqual(expr.grad[self.x], None)

        expr = cp.pnorm(self.x, 0.5)
        self.x.value = [0, 0]
        self.assertAlmostEqual(expr.grad[self.x], None)

        expr = cp.pnorm(self.x, 2)
        self.x.value = [0, 0]
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), [0, 0])

        expr = cp.pnorm(self.x[:, None], 2, axis=1)
        self.x.value = [1, 2]
        val = np.eye(2)
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), val)

        expr = cp.pnorm(self.A, 2)
        self.A.value = np.array([[2, -2], [2, 2]])
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), [0.5, 0.5, -0.5, 0.5])

        expr = cp.pnorm(self.A, 2, axis=0)
        self.A.value = np.array([[3, -3], [4, 4]])
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(),
                                    np.array([[0.6, 0], [0.8, 0], [0, -0.6], [0, 0.8]]))

        # Helper: normalize gradient to a flat ndarray (works for sparse/dense)
        def _flat(g):
            g_arr = g.toarray() if hasattr(g, "toarray") else np.asarray(g)
            return g_arr.ravel(order="F")

        # Regression test for issue #2896 - harmonic_mean gradient dtype=object error
        x = cp.Variable(2, name="x", value=[1.0, 1.0])
        expr = cp.harmonic_mean(x)
        grad = expr.grad[x]
        self.assertIsNotNone(grad)
        # Robust dtype check (float or complex, but not object)
        self.assertIn(getattr(grad, "dtype", np.asarray(grad).dtype).kind, ("f", "c"))
        # Check expected values: HM([1,1]) = 1, ∂HM/∂x_i = n*||x||_{-1}^2/x_i^2 = 0.5
        self.assertTrue(np.allclose(_flat(grad), [0.5, 0.5]))

        # Test gradient for negative p values (harmonic mean case p=-1)
        expr = cp.pnorm(self.x, -1)
        self.x.value = np.array([1.0, 2.0])
        grad = expr.grad[self.x]
        self.assertIsNotNone(grad)
        self.assertIn(getattr(grad, "dtype", np.asarray(grad).dtype).kind, ("f", "c"))

        # Test gradient sign correctness for p=3 with negative values
        expr = cp.pnorm(self.x, 3)
        self.x.value = np.array([-2.0, 3.0])
        grad = expr.grad[self.x]
        g = _flat(grad)
        # Sign checks
        self.assertLess(g[0], 0)
        self.assertGreater(g[1], 0)
        # Analytic check
        den = (np.linalg.norm(self.x.value, ord=3.0))**(3-1)
        expected = np.sign(self.x.value) * np.abs(self.x.value)**(3-1) / den
        self.assertTrue(np.allclose(g, expected))

        # Test p=-0.5 on column-shaped value (vector semantics)
        y = cp.Variable((3, 1), pos=True)
        y.value = np.array([[1.0], [2.0], [4.0]])
        expr = cp.pnorm(y, -0.5)
        grad = expr.grad[y]
        self.assertIsNotNone(grad)
        self.assertIn(getattr(grad, "dtype", np.asarray(grad).dtype).kind, ("f", "c"))

        expr = cp.pnorm(self.A, 2, axis=1)
        self.A.value = np.array([[3, -4], [4, 3]])
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(),
                                    np.array([[0.6, 0], [0, 0.8], [-0.8, 0], [0, 0.6]]))

        expr = cp.pnorm(self.A, 2, axis=1)
        self.A.value = np.array([[0, 0], [10, 0]])
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(),
                                    np.array([[0, 0], [0, 1], [0, 0], [0, 0]]))

        expr = cp.pnorm(self.A, 1, axis=1)
        self.A.value = np.array([[0, 0], [10, 0]])
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(),
                                    np.array([[0, 0], [0, 1], [0, 0], [0, 0]]))
        
        expr = cp.norm(self.D, 2, axis=0)
        self.D.value = np.array([[0, 10]])
        self.assertItemsAlmostEqual(expr.grad[self.D].toarray(), np.array([[0, 0], [0, 1]]))
        
        expr = cp.norm(self.D, 2, axis=1)
        self.D.value = np.array([[0, 10]])
        self.assertItemsAlmostEqual(expr.grad[self.D].toarray(), np.array([[0], [1]]))

        expr = cp.pnorm(self.A, 0.5)
        self.A.value = np.array([[3, -4], [4, 3]])
        self.assertAlmostEqual(expr.grad[self.A], None)

    def test_log_sum_exp(self) -> None:
        expr = cp.log_sum_exp(self.x)
        self.x.value = [0, 1]
        e = np.exp(1)
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), [1.0/(1+e), e/(1+e)])

        expr = cp.log_sum_exp(self.A)
        self.A.value = np.array([[0, 1], [-1, 0]])
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(),
                                    [1.0/(2+e+1.0/e), 1.0/e/(2+e+1.0/e),
                                     e/(2+e+1.0/e), 1.0/(2+e+1.0/e)])

        expr = cp.log_sum_exp(self.A, axis=0)
        self.A.value = np.array([[0, 1], [-1, 0]])
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(),
                                    np.transpose(np.array([[1.0/(1+1.0/e), 1.0/e/(1+1.0/e), 0, 0],
                                                           [0, 0, e/(1+e), 1.0/(1+e)]])))

    def test_geo_mean(self) -> None:
        """Test gradient for geo_mean
        """
        expr = cp.geo_mean(self.x)
        self.x.value = [1, 2]
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), [np.sqrt(2)/2, 1.0/2/np.sqrt(2)])

        self.x.value = [0, 2]
        self.assertAlmostEqual(expr.grad[self.x], None)

        expr = cp.geo_mean(self.x, [1, 0])
        self.x.value = [1, 2]
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), [1, 0])

        # No exception for single weight.
        self.x.value = [-1, 2]
        self.assertAlmostEqual(expr.grad[self.x], None)

    def test_lambda_max(self) -> None:
        """Test gradient for lambda_max
        """
        expr = cp.lambda_max(self.A)
        self.A.value = [[2, 0], [0, 1]]
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), [1, 0, 0, 0])

        self.A.value = [[1, 0], [0, 2]]
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), [0, 0, 0, 1])

        self.A.value = [[1, 0], [0, 1]]
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), [0, 0, 0, 1])

    def test_matrix_frac(self) -> None:
        """Test gradient for matrix_frac
        """
        expr = cp.matrix_frac(self.A, self.B)
        self.A.value = np.eye(2)
        self.B.value = np.eye(2)
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), [2, 0, 0, 2])
        self.assertItemsAlmostEqual(expr.grad[self.B].toarray(), [-1, 0, 0, -1])

        self.B.value = np.zeros((2, 2))
        self.assertAlmostEqual(expr.grad[self.A], None)
        self.assertAlmostEqual(expr.grad[self.B], None)

        expr = cp.matrix_frac(self.x[:, None], self.A)
        self.x.value = [2, 3]
        self.A.value = np.eye(2)
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), [4, 6])
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), [-4, -6, -6, -9])

        expr = cp.matrix_frac(self.x, self.A)
        self.x.value = [2, 3]
        self.A.value = np.eye(2)
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), [4, 6])
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), [-4, -6, -6, -9])

    def test_norm_nuc(self) -> None:
        """Test gradient for norm_nuc
        """
        expr = cp.normNuc(self.A)
        self.A.value = [[10, 4], [4, 30]]
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), [1, 0, 0, 1])

        # Failed for rectangular inputs.
        # https://github.com/cvxpy/cvxpy/issues/2364
        expr = cp.normNuc(self.C)
        self.C.value = np.array([[1, 0], [0, 0], [0, 2]])
        self.assertItemsAlmostEqual(expr.grad[self.C].toarray(), [1, 0, 0, 0, 0, 1])

    def test_log_det(self) -> None:
        """Test gradient for log_det
        """
        expr = cp.log_det(self.A)
        self.A.value = 2*np.eye(2)
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), 1.0/2*np.eye(2))

        mat = np.array([[1, 2], [3, 5]])
        self.A.value = mat.T.dot(mat)
        val = np.linalg.inv(self.A.value).T
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), val)

        self.A.value = np.zeros((2, 2))
        self.assertAlmostEqual(expr.grad[self.A], None)

        self.A.value = -np.array([[1, 2], [3, 4]])
        self.assertAlmostEqual(expr.grad[self.A], None)

        K = Variable((8, 8))
        expr = cp.log_det(K[[1, 2]][:, [1, 2]])
        K.value = np.eye(8)
        val = np.zeros((8, 8))
        val[[1, 2], [1, 2]] = 1
        self.assertItemsAlmostEqual(expr.grad[K].toarray(), val)

    def test_quad_over_lin(self) -> None:
        """Test gradient for quad_over_lin
        """
        expr = cp.quad_over_lin(self.x, self.a)
        self.x.value = [1, 2]
        self.a.value = 2
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), [1, 2])
        self.assertAlmostEqual(expr.grad[self.a], [-1.25])

        self.a.value = 0
        self.assertAlmostEqual(expr.grad[self.x], None)
        self.assertAlmostEqual(expr.grad[self.a], None)

        expr = cp.quad_over_lin(self.A, self.a)
        self.A.value = np.eye(2)
        self.a.value = 2
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), [1, 0, 0, 1])
        self.assertAlmostEqual(expr.grad[self.a], [-0.5])

        expr = cp.quad_over_lin(self.x, self.a) + cp.quad_over_lin(self.y, self.a)
        self.x.value = [1, 2]
        self.a.value = 2
        self.y.value = [1, 2]
        self.a.value = 2
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), [1, 2])
        self.assertItemsAlmostEqual(expr.grad[self.y].toarray(), [1, 2])
        self.assertAlmostEqual(expr.grad[self.a], [-2.5])

    def test_quad_form(self) -> None:
        """Test gradient for quad_form.
        """
        # Issue 1260
        n = 10
        np.random.seed(1)
        P = np.random.randn(n, n)
        P = P.T @ P
        q = np.random.randn(n)

        # define the optimization problem with the 2nd constraint as a quad_form constraint
        x = cp.Variable(n)
        prob = cp.Problem(cp.Maximize(q.T @ x - (1/2)*cp.quad_form(x, P)),
                          [cp.norm(x, 1) <= 1.0,
                           cp.quad_form(x, P) <= 10,   # quad form constraint
                           cp.abs(x) <= 0.01])
        prob.solve(solver=cp.SCS)

        # access quad_form.expr.grad without error
        prob.constraints[1].expr.grad

        # define the optimization problem with a two-dimensional decision variable
        x = cp.Variable((n, 1))
        prob = cp.Problem(
            cp.Maximize(q.T @ x - (1 / 2) * cp.quad_form(x, P)),
            [
                cp.norm(x, 1) <= 1.0,
                cp.quad_form(x, P) <= 10,  # quad form constraint
                cp.abs(x) <= 0.01,
            ],
        )
        prob.solve(solver=cp.SCS)

        # access quad_form.expr.grad without error
        prob.constraints[1].expr.grad

    def test_max(self) -> None:
        """Test gradient for max
        """
        expr = cp.max(self.x)
        self.x.value = [2, 1]
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), [1, 0])

        expr = cp.max(self.A)
        self.A.value = np.array([[1, 2], [4, 3]])
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), [0, 1, 0, 0])

        expr = cp.max(self.A, axis=0)
        self.A.value = np.array([[1, 2], [4, 3]])
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(),
                                    np.array([[0, 0], [1, 0], [0, 0], [0, 1]]))

        expr = cp.max(self.A, axis=1)
        self.A.value = np.array([[1, 2], [4, 3]])
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(),
                                    np.array([[0, 0], [0, 1], [1, 0], [0, 0]]))

    def test_sigma_max(self) -> None:
        """Test sigma_max.
        """
        expr = cp.sigma_max(self.A)
        self.A.value = [[1, 0], [0, 2]]
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), [0, 0, 0, 1])

        self.A.value = [[1, 0], [0, 1]]
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), [1, 0, 0, 0])

    def test_sum_largest(self) -> None:
        """Test sum_largest.
        """
        expr = cp.sum_largest(self.A, 2)

        self.A.value = [[4, 3], [2, 1]]
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), [1, 0, 1, 0])

        self.A.value = [[1, 2], [3, 0.5]]
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), [0, 1, 1, 0])

    def test_dotsort(self) -> None:
        """Test dotsort.
        """
        expr = cp.dotsort(self.A, [0.1, -2])

        self.A.value = [[4, 3], [2, 1]]
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), [0.1, 0, 0, -2])

        self.A.value = [[1, 2], [3, 0.5]]
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), [0, 0.1, 0, -2])

        # sum_largest tests:
        expr = cp.dotsort(self.A, [1, 1])
        self.A.value = [[4, 3], [2, 1]]
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), [1, 0, 1, 0])
        self.A.value = [[1, 2], [3, 0.5]]
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), [0, 1, 1, 0])

        # sum_smallest tests:
        expr = -cp.dotsort(self.A, [-1, -1])
        self.A.value = [[4, 3], [2, 1]]
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), [0, 1, 0, 1])
        self.A.value = [[1, 2], [3, 0.5]]
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), [1, 0, 0, 1])

    def test_abs(self) -> None:
        """Test abs.
        """
        expr = cp.abs(self.A)
        self.A.value = [[1, 2], [-1, 0]]
        val = np.zeros((4, 4)) + np.diag([1, 1, -1, 0])
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), val)

    def test_linearize(self) -> None:
        """Test linearize method.
        """
        # Affine.
        expr = (2*self.x - 5)[0]
        self.x.value = [1, 2]
        lin_expr = linearize(expr)
        self.x.value = [55, 22]
        self.assertAlmostEqual(lin_expr.value, expr.value)
        self.x.value = [-1, -5]
        self.assertAlmostEqual(lin_expr.value, expr.value)

        # Convex.
        expr = self.A**2 + 5

        with self.assertRaises(Exception) as cm:
            linearize(expr)
        self.assertEqual(str(cm.exception),
                         "Cannot linearize non-affine expression with missing variable values.")

        self.A.value = [[1, 2], [3, 4]]
        lin_expr = linearize(expr)
        manual = expr.value + 2*cp.reshape(
            cp.diag(cp.vec(self.A, order='F')).value @ cp.vec(self.A - self.A.value, order='F'), 
            (2, 2), 
            order='F'
        )
        self.assertItemsAlmostEqual(lin_expr.value, expr.value)
        self.A.value = [[-5, -5], [8.2, 4.4]]
        assert (lin_expr.value <= expr.value).all()
        self.assertItemsAlmostEqual(lin_expr.value, manual.value)

        # Concave.
        expr = cp.log(self.x)/2
        self.x.value = [1, 2]
        lin_expr = linearize(expr)
        manual = expr.value + cp.diag(0.5*self.x**-1).value @ (self.x - self.x.value)
        self.assertItemsAlmostEqual(lin_expr.value, expr.value)
        self.x.value = [3, 4.4]
        assert (lin_expr.value >= expr.value).all()
        self.assertItemsAlmostEqual(lin_expr.value, manual.value)

    def test_log(self) -> None:
        """Test gradient for log.
        """
        expr = cp.log(self.a)
        self.a.value = 2
        self.assertAlmostEqual(expr.grad[self.a], 1.0/2)

        self.a.value = 3
        self.assertAlmostEqual(expr.grad[self.a], 1.0/3)

        self.a.value = -1
        self.assertAlmostEqual(expr.grad[self.a], None)

        expr = cp.log(self.x)
        self.x.value = [3, 4]
        val = np.zeros((2, 2)) + np.diag([1/3, 1/4])
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), val)

        expr = cp.log(self.x)
        self.x.value = [-1e-9, 4]
        self.assertAlmostEqual(expr.grad[self.x], None)

        expr = cp.log(self.A)
        self.A.value = [[1, 2], [3, 4]]
        val = np.zeros((4, 4)) + np.diag([1, 1/2, 1/3, 1/4])
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), val)

    def test_log1p(self) -> None:
        """Test domain for log1p.
        """
        expr = cp.log1p(self.a)
        self.a.value = 2
        self.assertAlmostEqual(expr.grad[self.a], 1.0/3)

        self.a.value = 3
        self.assertAlmostEqual(expr.grad[self.a], 1.0/4)

        self.a.value = -1
        self.assertAlmostEqual(expr.grad[self.a], None)

        expr = cp.log1p(self.x)
        self.x.value = [3, 4]
        val = np.zeros((2, 2)) + np.diag([1/4, 1/5])
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), val)

        expr = cp.log1p(self.x)
        self.x.value = [-1e-9-1, 4]
        self.assertAlmostEqual(expr.grad[self.x], None)

        expr = cp.log1p(self.A)
        self.A.value = [[1, 2], [3, 4]]
        val = np.zeros((4, 4)) + np.diag([1/2, 1/3, 1/4, 1/5])
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), val)

    def test_entr(self) -> None:
        """Test domain for entr.
        """
        expr = cp.entr(self.a)
        self.a.value = 2
        self.assertAlmostEqual(expr.grad[self.a], -np.log(2) - 1)

        self.a.value = 3
        self.assertAlmostEqual(expr.grad[self.a], -(np.log(3) + 1))

        self.a.value = -1
        self.assertAlmostEqual(expr.grad[self.a], None)

        expr = cp.entr(self.x)
        self.x.value = [3, 4]
        val = np.zeros((2, 2)) + np.diag(-(np.log([3, 4]) + 1))
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), val)

        expr = cp.entr(self.x)
        self.x.value = [-1e-9, 4]
        self.assertAlmostEqual(expr.grad[self.x], None)

        expr = cp.entr(self.A)
        self.A.value = [[1, 2], [3, 4]]
        val = np.zeros((4, 4)) + np.diag(-(np.log([1, 2, 3, 4]) + 1))
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), val)

    def test_exp(self) -> None:
        """Test domain for exp.
        """
        expr = cp.exp(self.a)
        self.a.value = 2
        self.assertAlmostEqual(expr.grad[self.a], np.exp(2))

        self.a.value = 3
        self.assertAlmostEqual(expr.grad[self.a], np.exp(3))

        self.a.value = -1
        self.assertAlmostEqual(expr.grad[self.a], np.exp(-1))

        expr = cp.exp(self.x)
        self.x.value = [3, 4]
        val = np.zeros((2, 2)) + np.diag(np.exp([3, 4]))
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), val)

        expr = cp.exp(self.x)
        self.x.value = [-1e-9, 4]
        val = np.zeros((2, 2)) + np.diag(np.exp([-1e-9, 4]))
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), val)

        expr = cp.exp(self.A)
        self.A.value = [[1, 2], [3, 4]]
        val = np.zeros((4, 4)) + np.diag(np.exp([1, 2, 3, 4]))
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), val)

    def test_logistic(self) -> None:
        """Test domain for logistic.
        """
        expr = cp.logistic(self.a)
        self.a.value = 2
        self.assertAlmostEqual(expr.grad[self.a], np.exp(2)/(1+np.exp(2)))

        self.a.value = 3
        self.assertAlmostEqual(expr.grad[self.a], np.exp(3)/(1+np.exp(3)))

        self.a.value = -1
        self.assertAlmostEqual(expr.grad[self.a], np.exp(-1)/(1+np.exp(-1)))

        expr = cp.logistic(self.x)
        self.x.value = [3, 4]
        val = np.zeros((2, 2)) + np.diag(np.exp([3, 4])/(1+np.exp([3, 4])))
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), val)

        expr = cp.logistic(self.x)
        self.x.value = [-1e-9, 4]
        val = np.zeros((2, 2)) + np.diag(np.exp([-1e-9, 4])/(1+np.exp([-1e-9, 4])))
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), val)

        expr = cp.logistic(self.A)
        self.A.value = [[1, 2], [3, 4]]
        val = np.zeros((4, 4)) + np.diag(np.exp([1, 2, 3, 4])/(1+np.exp([1, 2, 3, 4])))
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), val)

    def test_huber(self) -> None:
        """Test domain for huber.
        """
        expr = cp.huber(self.a)
        self.a.value = 2
        self.assertAlmostEqual(expr.grad[self.a], 2)

        expr = cp.huber(self.a, M=2)
        self.a.value = 3
        self.assertAlmostEqual(expr.grad[self.a], 4)

        self.a.value = -1
        self.assertAlmostEqual(expr.grad[self.a], -2)

        expr = cp.huber(self.x)
        self.x.value = [3, 4]
        val = np.zeros((2, 2)) + np.diag([2, 2])
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), val)

        expr = cp.huber(self.x)
        self.x.value = [-1e-9, 4]
        val = np.zeros((2, 2)) + np.diag([0, 2])
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), val)

        expr = cp.huber(self.A, M=3)
        self.A.value = [[1, 2], [3, 4]]
        val = np.zeros((4, 4)) + np.diag([2, 4, 6, 6])
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), val)

    def test_kl_div(self) -> None:
        """Test domain for kl_div.
        """
        b = Variable()
        expr = cp.kl_div(self.a, b)
        self.a.value = 2
        b.value = 4
        self.assertAlmostEqual(expr.grad[self.a], np.log(2/4))
        self.assertAlmostEqual(expr.grad[b], 1 - (2/4))

        self.a.value = 3
        b.value = 0
        self.assertAlmostEqual(expr.grad[self.a], None)
        self.assertAlmostEqual(expr.grad[b], None)

        self.a.value = -1
        b.value = 2
        self.assertAlmostEqual(expr.grad[self.a], None)
        self.assertAlmostEqual(expr.grad[b], None)

        y = Variable(2)
        expr = cp.kl_div(self.x, y)
        self.x.value = [3, 4]
        y.value = [5, 8]
        val = np.zeros((2, 2)) + np.diag(np.log([3, 4]) - np.log([5, 8]))
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), val)
        val = np.zeros((2, 2)) + np.diag([1 - 3/5, 1 - 4/8])
        self.assertItemsAlmostEqual(expr.grad[y].toarray(), val)

        expr = cp.kl_div(self.x, y)
        self.x.value = [-1e-9, 4]
        y.value = [1, 2]
        self.assertAlmostEqual(expr.grad[self.x], None)
        self.assertAlmostEqual(expr.grad[y], None)

        expr = cp.kl_div(self.A, self.B)
        self.A.value = [[1, 2], [3, 4]]
        self.B.value = [[5, 1], [3.5, 2.3]]
        div = (self.A.value/self.B.value).ravel(order='F')
        val = np.zeros((4, 4)) + np.diag(np.log(div))
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), val)
        val = np.zeros((4, 4)) + np.diag(1 - div)
        self.assertItemsAlmostEqual(expr.grad[self.B].toarray(), val)

    def test_rel_entr(self) -> None:
        """Test domain for rel_entr.
        """
        b = Variable()
        expr = cp.rel_entr(self.a, b)
        self.a.value = 2
        b.value = 4
        self.assertAlmostEqual(expr.grad[self.a], np.log(2 / 4) + 1)
        self.assertAlmostEqual(expr.grad[b], - (2 / 4))

        self.a.value = 3
        b.value = 0
        self.assertAlmostEqual(expr.grad[self.a], None)
        self.assertAlmostEqual(expr.grad[b], None)

        self.a.value = -1
        b.value = 2
        self.assertAlmostEqual(expr.grad[self.a], None)
        self.assertAlmostEqual(expr.grad[b], None)

        y = Variable(2)
        expr = cp.rel_entr(self.x, y)
        self.x.value = [3, 4]
        y.value = [5, 8]
        val = np.zeros((2, 2)) + np.diag(np.log([3, 4]) - np.log([5, 8]) + 1)
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), val)
        val = np.zeros((2, 2)) + np.diag([- 3 / 5, - 4 / 8])
        self.assertItemsAlmostEqual(expr.grad[y].toarray(), val)

        expr = cp.rel_entr(self.x, y)
        self.x.value = [-1e-9, 4]
        y.value = [1, 2]
        self.assertAlmostEqual(expr.grad[self.x], None)
        self.assertAlmostEqual(expr.grad[y], None)

        expr = cp.rel_entr(self.A, self.B)
        self.A.value = [[1, 2], [3, 4]]
        self.B.value = [[5, 1], [3.5, 2.3]]
        div = (self.A.value / self.B.value).ravel(order='F')
        val = np.zeros((4, 4)) + np.diag(np.log(div) + 1)
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), val)
        val = np.zeros((4, 4)) + np.diag(- div)
        self.assertItemsAlmostEqual(expr.grad[self.B].toarray(), val)

    def test_maximum(self) -> None:
        """Test domain for maximum.
        """
        b = Variable()
        expr = cp.maximum(self.a, b)
        self.a.value = 2
        b.value = 4
        self.assertAlmostEqual(expr.grad[self.a], 0)
        self.assertAlmostEqual(expr.grad[b], 1)

        self.a.value = 3
        b.value = 0
        self.assertAlmostEqual(expr.grad[self.a], 1)
        self.assertAlmostEqual(expr.grad[b], 0)

        self.a.value = -1
        b.value = 2
        self.assertAlmostEqual(expr.grad[self.a], 0)
        self.assertAlmostEqual(expr.grad[b], 1)

        y = Variable(2)
        expr = cp.maximum(self.x, y)
        self.x.value = [3, 4]
        y.value = [5, -5]
        val = np.zeros((2, 2)) + np.diag([0, 1])
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), val)
        val = np.zeros((2, 2)) + np.diag([1, 0])
        self.assertItemsAlmostEqual(expr.grad[y].toarray(), val)

        expr = cp.maximum(self.x, y)
        self.x.value = [-1e-9, 4]
        y.value = [1, 4]
        val = np.zeros((2, 2)) + np.diag([0, 1])
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), val)
        val = np.zeros((2, 2)) + np.diag([1, 0])
        self.assertItemsAlmostEqual(expr.grad[y].toarray(), val)

        expr = cp.maximum(self.A, self.B)
        self.A.value = [[1, 2], [3, 4]]
        self.B.value = [[5, 1], [3, 2.3]]
        val = np.zeros((4, 4)) + np.diag([0, 1, 1, 1])
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), val)
        val = np.zeros((4, 4)) + np.diag([1, 0, 0, 0])
        self.assertItemsAlmostEqual(expr.grad[self.B].toarray(), val)

        # cummax
        expr = cp.cummax(self.x)
        self.x.value = [2, 1]
        val = np.zeros((2, 2))
        val[0, 0] = 1
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), val)

        expr = cp.cummax(self.x[:, None], axis=1)
        self.x.value = [2, 1]
        val = np.eye(2)
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), val)

    def test_minimum(self) -> None:
        """Test domain for minimum.
        """
        b = Variable()
        expr = cp.minimum(self.a, b)
        self.a.value = 2
        b.value = 4
        self.assertAlmostEqual(expr.grad[self.a], 1)
        self.assertAlmostEqual(expr.grad[b], 0)

        self.a.value = 3
        b.value = 0
        self.assertAlmostEqual(expr.grad[self.a], 0)
        self.assertAlmostEqual(expr.grad[b], 1)

        self.a.value = -1
        b.value = 2
        self.assertAlmostEqual(expr.grad[self.a], 1)
        self.assertAlmostEqual(expr.grad[b], 0)

        y = Variable(2)
        expr = cp.minimum(self.x, y)
        self.x.value = [3, 4]
        y.value = [5, -5]
        val = np.zeros((2, 2)) + np.diag([1, 0])
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), val)
        val = np.zeros((2, 2)) + np.diag([0, 1])
        self.assertItemsAlmostEqual(expr.grad[y].toarray(), val)

        expr = cp.minimum(self.x, y)
        self.x.value = [-1e-9, 4]
        y.value = [1, 4]
        val = np.zeros((2, 2)) + np.diag([1, 1])
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), val)
        val = np.zeros((2, 2)) + np.diag([0, 0])
        self.assertItemsAlmostEqual(expr.grad[y].toarray(), val)

        expr = cp.minimum(self.A, self.B)
        self.A.value = [[1, 2], [3, 4]]
        self.B.value = [[5, 1], [3, 2.3]]
        val = np.zeros((4, 4)) + np.diag([1, 0, 1, 0])
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), val)
        val = np.zeros((4, 4)) + np.diag([0, 1, 0, 1])
        self.assertItemsAlmostEqual(expr.grad[self.B].toarray(), val)

    def test_power(self) -> None:
        """Test grad for power.
        """
        expr = cp.sqrt(self.a)
        self.a.value = 2
        self.assertAlmostEqual(expr.grad[self.a], 0.5/np.sqrt(2))

        self.a.value = 3
        self.assertAlmostEqual(expr.grad[self.a], 0.5/np.sqrt(3))

        self.a.value = -1
        self.assertAlmostEqual(expr.grad[self.a], None)

        expr = (self.x)**3
        self.x.value = [3, 4]
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(),
                                    np.array([[27, 0], [0, 48]]))

        expr = (self.x)**3
        self.x.value = [-1e-9, 4]
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), np.array([[0, 0], [0, 48]]))

        expr = (self.A)**2
        self.A.value = [[1, -2], [3, 4]]
        val = np.zeros((4, 4)) + np.diag([2, -4, 6, 8])
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), val)

        # Constant.
        expr = (self.a)**0
        self.assertAlmostEqual(expr.grad[self.a], 0)

        expr = (self.x)**0
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), np.zeros((2, 2)))

    def test_partial_problem(self) -> None:
        """Test grad for partial minimization/maximization problems.
        """
        for obj in [Minimize((self.a)**-1), Maximize(cp.entr(self.a))]:
            prob = Problem(obj, [self.x + self.a >= [5, 8]])
            # Optimize over nothing.
            expr = partial_optimize(prob, dont_opt_vars=[self.x, self.a], solver=cp.CLARABEL)
            self.a.value = None
            self.x.value = None
            grad = expr.grad
            self.assertAlmostEqual(grad[self.a], None)
            self.assertAlmostEqual(grad[self.x], None)
            # Outside domain.
            self.a.value = 1.0
            self.x.value = [5, 5]
            grad = expr.grad
            self.assertAlmostEqual(grad[self.a], None)
            self.assertAlmostEqual(grad[self.x], None)

            self.a.value = 1
            self.x.value = [10, 10]
            grad = expr.grad
            self.assertAlmostEqual(grad[self.a], obj.args[0].grad[self.a])
            self.assertItemsAlmostEqual(grad[self.x].toarray(), [0, 0, 0, 0])

            # Optimize over x.
            expr = partial_optimize(prob, opt_vars=[self.x], solver=cp.CLARABEL)
            self.a.value = 1
            grad = expr.grad
            self.assertAlmostEqual(grad[self.a], obj.args[0].grad[self.a] + 0)

            # Optimize over a.
            fix_prob = Problem(obj, [self.x + self.a >= [5, 8], self.x == 0])
            fix_prob.solve(solver=cp.CLARABEL)
            dual_val = fix_prob.constraints[0].dual_variables[0].value
            expr = partial_optimize(prob, opt_vars=[self.a], solver=cp.CLARABEL)
            self.x.value = [0, 0]
            grad = expr.grad
            self.assertItemsAlmostEqual(grad[self.x].toarray(), dual_val)

            # Optimize over x and a.
            expr = partial_optimize(prob, opt_vars=[self.x, self.a], solver=cp.CLARABEL)
            grad = expr.grad
            self.assertAlmostEqual(grad, {})

    def test_affine(self) -> None:
        """Test grad for affine atoms.
        """
        expr = -self.a
        self.a.value = 2
        self.assertAlmostEqual(expr.grad[self.a], -1)

        expr = 2*self.a
        self.a.value = 2
        self.assertAlmostEqual(expr.grad[self.a], 2)

        expr = self.a/2
        self.a.value = 2
        self.assertAlmostEqual(expr.grad[self.a], 0.5)

        expr = -(self.x)
        self.x.value = [3, 4]
        val = np.zeros((2, 2)) - np.diag([1, 1])
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), val)

        expr = -(self.A)
        self.A.value = [[1, 2], [3, 4]]
        val = np.zeros((4, 4)) - np.diag([1, 1, 1, 1])
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), val)

        expr = self.A[0, 1]
        self.A.value = [[1, 2], [3, 4]]
        val = np.zeros((4, 1))
        val[2] = 1
        self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), val)

        z = Variable(3)
        expr = cp.hstack([self.x, z])
        self.x.value = [1, 2]
        z.value = [1, 2, 3]
        val = np.zeros((2, 5))
        val[:, 0:2] = np.eye(2)
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), val)

        val = np.zeros((3, 5))
        val[:, 2:] = np.eye(3)
        self.assertItemsAlmostEqual(expr.grad[z].toarray(), val)

        # cumsum
        expr = cp.cumsum(self.x)
        self.x.value = [1, 2]
        val = np.ones((2, 2))
        val[1, 0] = 0
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), val)

        expr = cp.cumsum(self.x[:, None], axis=1)
        self.x.value = [1, 2]
        val = np.eye(2)
        self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), val)

    def test_bilinear(self) -> None:
        """Test grad for bilinear expressions."""
        for n in [1, 2, 3]:
            with self.subTest(n=n):
                x_vals = np.arange(1, n + 1)
                y_vals = -np.arange(1, n + 1)
                x = cp.Variable(n, value=x_vals)
                y = cp.Variable(n, value=y_vals)

                # test bilinear expression x @ y
                # which has partial derivatives grad_x = y, grad_y = x
                expr = x @ y

                grad_x = expr.grad[x]
                grad_y = expr.grad[y]

                assert x.value is not None
                assert y.value is not None

                if n == 1:
                    assert grad_x.shape == ()
                    assert grad_y.shape == ()
                    self.assertAlmostEqual(grad_x, y.value)
                    self.assertAlmostEqual(grad_y, x.value)
                else:
                    assert grad_x.shape == (n, 1)
                    assert grad_y.shape == (n, 1)
                    self.assertItemsAlmostEqual(grad_x.toarray(), y.value.reshape(-1, 1))
                    self.assertItemsAlmostEqual(grad_y.toarray(), x.value.reshape(-1, 1))

    def test_matrix_product(self) -> None:
        """Test matrix-matrix product."""
        x_vals = np.array([[1, -1], [2, -2]])
        y_vals = np.array([[-1, 1], [-2, 2]])
        x = cp.Variable((2, 2), value=x_vals)
        y = cp.Variable((2, 2), value=y_vals)
        expr = x @ y
        grad_x = expr.grad[x]
        grad_y = expr.grad[y]
        assert x.value is not None
        assert y.value is not None
        
        # expected gradients are 4x4 Jacobian matrices for 2x2 matrix variables
        expected_grad_x = np.array([[-1.,  0.,  1.,  0.],
                                   [ 0., -1.,  0.,  1.],
                                   [-2.,  0.,  2.,  0.],
                                   [ 0., -2.,  0.,  2.]])
        expected_grad_y = np.array([[ 1.,  2.,  0.,  0.],
                                   [-1., -2.,  0.,  0.],
                                   [ 0.,  0.,  1.,  2.],
                                   [ 0.,  0., -1., -2.]])
        
        self.assertItemsAlmostEqual(grad_x.toarray(), expected_grad_x)
        self.assertItemsAlmostEqual(grad_y.toarray(), expected_grad_y)

