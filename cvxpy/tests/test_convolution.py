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

import cvxpy as cvx
import cvxpy.settings as s
from cvxpy.lin_ops.tree_mat import prune_constants
import cvxpy.problems.iterative as iterative
from cvxpy.tests.base_test import BaseTest
import numpy as np


class TestConvolution(BaseTest):
    """ Unit tests for convolution. """

    def test_1D_conv(self):
        """Test 1D convolution.
        """
        n = 3
        x = cvx.Variable(n)
        f = [1, 2, 3]
        g = [0, 1, 0.5]
        f_conv_g = [0., 1., 2.5,  4., 1.5]
        expr = cvx.conv(f, g)
        assert expr.is_constant()
        self.assertEqual(expr.shape, (5, 1))
        self.assertItemsAlmostEqual(expr.value, f_conv_g)

        expr = cvx.conv(f, x)
        assert expr.is_affine()
        self.assertEqual(expr.shape, (5, 1))
        # Matrix stuffing.
        prob = cvx.Problem(cvx.Minimize(cvx.norm(expr, 1)),
                           [x == g])
        result = prob.solve()
        self.assertAlmostEqual(result, sum(f_conv_g), places=3)
        self.assertItemsAlmostEqual(expr.value, f_conv_g)

        # # Expression trees.
        # prob = Problem(Minimize(norm(expr, 1)))
        # self.prob_mat_vs_mul_funcs(prob)
        # result = prob.solve(solver=SCS, expr_tree=True, verbose=True)
        # self.assertAlmostEqual(result, 0, places=1)

    def prob_mat_vs_mul_funcs(self, prob):
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
        # A*vec
        result = np.zeros(A.shape[0])
        Amul(vec, result)
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
        import numpy as np
        N = 5
        y = np.random.randn(N, 1)
        h = np.random.randn(2, 1)
        x = cvx.Variable(N)
        v = cvx.conv(h, x)
        obj = cvx.Minimize(cvx.sum(cvx.multiply(y, v[0:N])))
        print(cvx.Problem(obj, []).solve())
