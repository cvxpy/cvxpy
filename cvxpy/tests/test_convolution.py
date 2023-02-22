"""
Copyright 2013 Steven Diamond, Eric Chu

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

import cvxpy as cvx
import cvxpy.problems.iterative as iterative
import cvxpy.settings as s
from cvxpy.lin_ops.tree_mat import prune_constants
from cvxpy.tests.base_test import BaseTest


class TestConvolution(BaseTest):
    """ Unit tests for convolution. """

    def test_1D_conv(self) -> None:
        """Test 1D convolution.
        """
        n = 3
        x = cvx.Variable(n)
        f = np.array([1, 2, 3])
        g = np.array([0, 1, 0.5])
        f_conv_g = np.array([0., 1., 2.5,  4., 1.5])
        with pytest.warns(DeprecationWarning, match="Use convolve"):
            expr = cvx.conv(f, g)
        assert expr.is_constant()
        self.assertEqual(expr.shape, (5,))
        self.assertEqual(expr.shape, expr.value.shape)
        self.assertItemsAlmostEqual(expr.value, f_conv_g)

        expr = cvx.conv(f, x)
        assert expr.is_affine()
        self.assertEqual(expr.shape, (5,))
        # Matrix stuffing.
        prob = cvx.Problem(cvx.Minimize(cvx.norm(expr, 1)),
                           [x == g])
        result = prob.solve(solver=cvx.SCS)
        self.assertAlmostEqual(result, sum(f_conv_g), places=3)
        self.assertItemsAlmostEqual(expr.value, f_conv_g)

        # Test other shape configurations.
        expr = cvx.conv(2, g)
        self.assertEqual(expr.shape, (3,))
        self.assertEqual(expr.shape, expr.value.shape)
        self.assertItemsAlmostEqual(expr.value, 2 * g)

        expr = cvx.conv(f, 2)
        self.assertEqual(expr.shape, (3,))
        self.assertEqual(expr.shape, expr.value.shape)
        self.assertItemsAlmostEqual(expr.value, 2 * f)

        expr = cvx.conv(f, g[:, None])
        self.assertEqual(expr.shape, (5, 1))
        self.assertEqual(expr.shape, expr.value.shape)
        self.assertItemsAlmostEqual(expr.value, f_conv_g)

        expr = cvx.conv(f[:, None], g)
        self.assertEqual(expr.shape, (5, 1))
        self.assertEqual(expr.shape, expr.value.shape)
        self.assertItemsAlmostEqual(expr.value, f_conv_g)

        expr = cvx.conv(f[:, None], g[:, None])
        self.assertEqual(expr.shape, (5, 1))
        self.assertEqual(expr.shape, expr.value.shape)
        self.assertItemsAlmostEqual(expr.value, f_conv_g)

    def test_convolve(self) -> None:
        """Test convolve.
        """
        n = 3
        x = cvx.Variable(n)
        f = np.array([1, 2, 3])
        g = np.array([0, 1, 0.5])
        f_conv_g = np.array([0., 1., 2.5,  4., 1.5])
        expr = cvx.convolve(f, g)
        assert expr.is_constant()
        self.assertEqual(expr.shape, (5,))
        self.assertEqual(expr.shape, expr.value.shape)
        self.assertItemsAlmostEqual(expr.value, f_conv_g)

        expr = cvx.convolve(f, x)
        assert expr.is_affine()
        self.assertEqual(expr.shape, (5,))
        # Matrix stuffing.
        prob = cvx.Problem(cvx.Minimize(cvx.norm(expr, 1)),
                           [x == g])
        result = prob.solve(solver=cvx.SCS)
        self.assertAlmostEqual(result, sum(f_conv_g), places=3)
        self.assertItemsAlmostEqual(expr.value, f_conv_g)

        # Test other shape configurations.
        expr = cvx.convolve(2, g)
        self.assertEqual(expr.shape, (3,))
        self.assertEqual(expr.shape, expr.value.shape)
        self.assertItemsAlmostEqual(expr.value, 2 * g)

        expr = cvx.convolve(f, 2)
        self.assertEqual(expr.shape, (3,))
        self.assertEqual(expr.shape, expr.value.shape)
        self.assertItemsAlmostEqual(expr.value, 2 * f)

        with pytest.raises(ValueError, match="must be scalar or 1D"):
            expr = cvx.convolve(f, g[:, None])

        with pytest.raises(ValueError, match="must be scalar or 1D"):
            expr = cvx.convolve(f[:, None], g)

        with pytest.raises(ValueError, match="must be scalar or 1D"):
            expr = cvx.convolve(f[:, None], g[:, None])

    def prob_mat_vs_mul_funcs(self, prob) -> None:
        data, dims = prob.get_problem_data(solver=cvx.SCS)
        A = data["A"]
        objective, constr_map, dims, solver = prob.canonicalize(cvx.SCS)

        all_ineq = constr_map[s.EQ] + constr_map[s.LEQ]
        var_offsets, var_sizes, x_length = prob._get_var_offsets(objective,
                                                                 all_ineq)
        constraints = constr_map[s.EQ] + constr_map[s.LEQ]
        constraints = prune_constants(constraints)
        Amul, ATmul = iterative.get_mul_funcs(constraints, dims,
                                              var_offsets, var_sizes,
                                              x_length)
        vec = np.array(range(1, x_length+1))
        # A @ vec
        result = np.zeros(A.shape[0])
        Amul(vec, result)
        self.assertItemsAlmostEqual(A @ vec, result)
        Amul(vec, result)
        self.assertItemsAlmostEqual(2*A @ vec, result)
        # A.T @ vec
        vec = np.array(range(A.shape[0]))
        result = np.zeros(A.shape[1])
        ATmul(vec, result)
        self.assertItemsAlmostEqual(A.T @ vec, result)
        ATmul(vec, result)
        self.assertItemsAlmostEqual(2*A.T @ vec, result)

    def mat_from_func(self, func, rows, cols):
        """Convert a multiplier function to a matrix.
        """
        test_vec = np.zeros(cols)
        result = np.zeros(rows)
        matrix = np.zeros((rows, cols))
        for i in range(cols):
            test_vec[i] = 1.0
            func(test_vec, result)
            matrix[:, i] = result
            test_vec *= 0
            result *= 0

        return matrix

    def test_conv_prob(self) -> None:
        """Test a problem with convolution.
        """
        N = 5
        # Test conv.
        y = np.random.randn(N, 1)
        h = np.random.randn(2, 1)
        x = cvx.Variable((N, 1))
        v = cvx.conv(h, x)
        obj = cvx.Minimize(cvx.sum(cvx.multiply(y, v[0:N])))
        prob = cvx.Problem(obj, [])
        prob.solve(solver=cvx.ECOS)
        assert prob.status is cvx.UNBOUNDED

        # Test convolve.
        y = np.random.randn(N)
        h = np.random.randn(2)
        x = cvx.Variable((N))
        v = cvx.convolve(h, x)
        obj = cvx.Minimize(cvx.sum(cvx.multiply(y, v[0:N])))
        prob = cvx.Problem(obj, [])
        prob.solve(solver=cvx.ECOS)
        assert prob.status is cvx.UNBOUNDED

    def test_0D_conv(self) -> None:
        """Convolution with 0D input.
        """
        for func in [cvx.conv, cvx.convolve]:
            x = cvx.Variable((1,))  # or cvx.Variable((1,1))
            problem = cvx.Problem(
                cvx.Minimize(
                    cvx.max(func(1., cvx.multiply(1., x)))
                ),
                [x >= 0]
            )
            problem.solve(cvx.ECOS)
            assert problem.status == cvx.OPTIMAL
