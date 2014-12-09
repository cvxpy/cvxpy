"""
Copyright 2013 Steven Diamond, Eric Chu

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

from cvxpy import *
import cvxpy.settings as s
from cvxpy.lin_ops.tree_mat import mul, tmul, prune_constants
import cvxpy.problems.iterative as iterative
from cvxpy.utilities import Curvature
from cvxpy.utilities import Sign
from base_test import BaseTest
import numpy as np

class TestConvolution(BaseTest):
    """ Unit tests for convolution. """

    def test_1D_conv(self):
        """Test 1D convolution.
        """
        n = 3
        x = Variable(n)
        f = [1, 2, 3]
        g = [0, 1, 0.5]
        f_conv_g = [ 0., 1., 2.5,  4., 1.5]
        expr = conv(f, g)
        assert expr.is_constant()
        self.assertEquals(expr.size, (5, 1))
        self.assertItemsAlmostEqual(expr.value, f_conv_g)

        expr = conv(f, x)
        assert expr.is_affine()
        self.assertEquals(expr.size, (5, 1))
        # Matrix stuffing.
        t = Variable()
        prob = Problem(Minimize(norm(expr, 1)),
            [x == g])
        result = prob.solve()
        self.assertAlmostEqual(result, sum(f_conv_g))
        self.assertItemsAlmostEqual(expr.value, f_conv_g)

        # # Expression trees.
        # prob = Problem(Minimize(norm(expr, 1)))
        # self.prob_mat_vs_mul_funcs(prob)
        # result = prob.solve(solver=SCS, expr_tree=True, verbose=True)
        # self.assertAlmostEqual(result, 0, places=1)

    def prob_mat_vs_mul_funcs(self, prob):
        data, dims = prob.get_problem_data(solver=SCS)
        A = data["A"]
        objective, constr_map, dims, solver = prob.canonicalize(SCS)

        all_ineq = constr_map[s.EQ] + constr_map[s.LEQ]
        var_offsets, var_sizes, x_length = prob._get_var_offsets(objective,
                                                                 all_ineq)
        opts = {}
        constraints = constr_map[s.EQ] + constr_map[s.LEQ]
        constraints = prune_constants(constraints)
        Amul, ATmul = iterative.get_mul_funcs(constraints, dims,
                                              var_offsets, var_sizes,
                                              x_length)
        vec = np.array(range(1, x_length+1))
        # A*vec
        result = np.zeros(A.shape[0])
        Amul(vec, result)
        mul_mat = self.mat_from_func(Amul, A.shape[0], A.shape[1])
        self.assertItemsAlmostEqual(A*vec, result)
        Amul(vec, result)
        self.assertItemsAlmostEqual(2*A*vec, result)
        # A.T*vec
        vec = np.array(range(A.shape[0]))
        result = np.zeros(A.shape[1])
        ATmul(vec, result)
        self.assertItemsAlmostEqual(A.T*vec, result)
        ATmul(vec, result)
        self.assertItemsAlmostEqual(2*A.T*vec, result)

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

    def test_conv_prob(self):
        """Test a problem with convolution.
        """
        import cvxpy as cvx
        import numpy as np
        N = 5
        y = np.asmatrix(np.random.randn(N, 1))
        h = np.asmatrix(np.random.randn(2, 1))
        x = cvx.Variable(N)
        v = cvx.conv(h, x)
        obj = cvx.Minimize(cvx.sum_entries(cvx.mul_elemwise(y,v[0:N])))
        print cvx.Problem(obj, []).solve()

    def test_mat_free(self):
        """Test matrix free convolution.
        """
        import numpy as np
        import random

        from math import pi, sqrt, exp

        def gauss(n=11,sigma=1):
            r = range(-int(n/2),int(n/2)+1)
            return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]

        np.random.seed(5)
        random.seed(5)
        DENSITY = 0.008
        n = 100
        x = Variable(n)
        # Create sparse signal.
        signal = np.zeros(n)
        nnz = 0
        for i in range(n):
            if random.random() < DENSITY:
                signal[i] = random.uniform(0, 100)
                nnz += 1

        # Gaussian kernel.
        m = 11
        kernel = gauss(m, m/10)

        # Noisy signal.
        std = 1
        noise = np.random.normal(scale=std, size=n+m-1)
        noisy_signal = conv(kernel, signal) #+ noise

        gamma = Parameter(sign="positive")
        fit = norm(conv(kernel, x) - noisy_signal, 2)
        A = np.random.randn(m+n-1, n)
        fit = norm(A*x - noisy_signal)
        regularization = norm(x, 1)
        constraints = [x >= 0]
        gamma.value = 0.06
        prob = Problem(Minimize(fit), constraints)
        result = prob.solve(solver=SCS_MAT_FREE, verbose=True,
            max_iters=2500, normalize=True)
        fit1 = fit.value
        result2 = prob.solve(solver=SCS, verbose=True, normalize=True)
        fit2 = fit.value
        self.assertAlmostEqual(fit1, fit2, places=3)
        assert False
