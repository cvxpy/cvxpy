"""
Copyright 2013 Steven Diamond

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
from cvxpy.problems.solvers.utilities import SOLVERS
from cvxpy.problems.problem_data.sym_data import SymData
import numpy as np
import scipy.sparse as sp
import scipy.linalg as LA
import unittest
from cvxpy.tests.base_test import BaseTest

class test_tree_mat(BaseTest):
    """ Unit tests for the matrix ops with expression trees. """

    def test_mul(self):
        """Test the mul method.
        """
        n = 2
        ones = np.mat(np.ones((n, n)))
        # Multiplication
        x = Variable(n, n)
        A = np.matrix("1 2; 3 4")
        expr = (A*x).canonical_form[0]

        val_dict = {x.id: ones}

        result = mul(expr, val_dict)
        assert (result == A*ones).all()

        result_dict = tmul(expr, result)
        assert (result_dict[x.id] == A.T*A*ones).all()

        # Multiplication with promotion.
        t = Variable()
        A = np.matrix("1 2; 3 4")
        expr = (A*t).canonical_form[0]

        val_dict = {t.id: 2}

        result = mul(expr, val_dict)
        assert (result == A*2).all()

        result_dict = tmul(expr, result)
        total = 0
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                total += A[i, j]*result[i, j]
        assert (result_dict[t.id] == total)

        # Addition
        y = Variable(n, n)
        expr = (y + A*x).canonical_form[0]
        val_dict = {x.id: np.ones((n, n)),
                    y.id: np.ones((n, n))}

        result = mul(expr, val_dict)
        assert (result == A*ones + ones).all()

        result_dict = tmul(expr, result)
        assert (result_dict[y.id] == result).all()
        assert (result_dict[x.id] == A.T*result).all()

        val_dict = {x.id: A,
                    y.id: A}

        # Indexing
        expr = (x[:, 0] + y[:, 1]).canonical_form[0]
        result = mul(expr, val_dict)
        assert (result == A[:, 0] + A[:, 1]).all()

        result_dict = tmul(expr, result)
        mat = ones
        mat[:, 0] = result
        mat[:, 1] = 0
        assert (result_dict[x.id] == mat).all()

        # Negation
        val_dict = {x.id: A}
        expr = (-x).canonical_form[0]

        result = mul(expr, val_dict)
        assert (result == -A).all()

        result_dict = tmul(expr, result)
        assert (result_dict[x.id] == A).all()

        # Transpose
        expr = x.T.canonical_form[0]
        val_dict = {x.id: A}
        result = mul(expr, val_dict)
        assert (result == A.T).all()
        result_dict = tmul(expr, result)
        assert (result_dict[x.id] == A).all()

        # Convolution
        x = Variable(3)
        f = np.matrix(np.array([1, 2, 3])).T
        g = np.array([0, 1, 0.5])
        f_conv_g = np.array([ 0., 1., 2.5,  4., 1.5])
        expr = conv(f, x).canonical_form[0]
        val_dict = {x.id: g}
        result = mul(expr, val_dict)
        self.assertItemsAlmostEqual(result, f_conv_g)
        value = np.array(range(5))
        result_dict = tmul(expr, value)
        toep = LA.toeplitz(np.array([1,0,0]),
                           np.array([1, 2, 3, 0, 0]))
        x_val = toep.dot(value)
        self.assertItemsAlmostEqual(result_dict[x.id], x_val)

    def test_abs_mul(self):
        """Test the abs mul method.
        """
        n = 2
        ones = np.mat(np.ones((n, n)))
        # Multiplication
        x = Variable(n, n)
        A = np.matrix("-1 2; -3 4")
        abs_A = np.abs(A)
        expr = (A*x).canonical_form[0]

        val_dict = {x.id: ones}

        result = mul(expr, val_dict, True)
        assert (result == abs_A*ones).all()

        result_dict = tmul(expr, result, True)
        assert (result_dict[x.id] == abs_A.T*abs_A*ones).all()

        # Multiplication with promotion.
        t = Variable()
        A = np.matrix("1 -2; -3 -4")
        abs_A = np.abs(A)
        expr = (A*t).canonical_form[0]

        val_dict = {t.id: 2}

        result = mul(expr, val_dict, True)
        assert (result == abs_A*2).all()

        result_dict = tmul(expr, result, True)
        total = 0
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                total += abs_A[i, j]*result[i, j]
        assert (result_dict[t.id] == total)

        # Addition
        y = Variable(n, n)
        expr = (y + A*x).canonical_form[0]
        val_dict = {x.id: np.ones((n, n)),
                    y.id: np.ones((n, n))}

        result = mul(expr, val_dict)
        assert (result == A*ones + ones).all()

        result_dict = tmul(expr, result)
        assert (result_dict[y.id] == result).all()
        assert (result_dict[x.id] == A.T*result).all()

        val_dict = {x.id: A,
                    y.id: A}

        # Indexing
        expr = (x[:, 0] + y[:, 1]).canonical_form[0]
        result = mul(expr, val_dict)
        assert (result == A[:, 0] + A[:, 1]).all()

        result_dict = tmul(expr, result)
        mat = ones
        mat[:, 0] = result
        mat[:, 1] = 0
        assert (result_dict[x.id] == mat).all()

        # Negation
        val_dict = {x.id: A}
        expr = (-x).canonical_form[0]

        result = mul(expr, val_dict)
        assert (result == -A).all()

        result_dict = tmul(expr, result)
        assert (result_dict[x.id] == A).all()

        # Transpose
        expr = x.T.canonical_form[0]
        val_dict = {x.id: A}
        result = mul(expr, val_dict)
        assert (result == A.T).all()
        result_dict = tmul(expr, result)
        assert (result_dict[x.id] == A).all()

        # Convolution
        x = Variable(3)
        f = np.matrix(np.array([1, -2, -3])).T
        g = np.array([0, 1, 0.5])
        f_conv_g = np.array([ 0., 1., 2.5,  4., 1.5])
        expr = conv(f, x).canonical_form[0]
        val_dict = {x.id: g}
        result = mul(expr, val_dict, True)
        self.assertItemsAlmostEqual(result, f_conv_g)
        value = np.array(range(5))
        result_dict = tmul(expr, value, True)
        toep = LA.toeplitz(np.array([1,0,0]),
                           np.array([1, 2, 3, 0, 0]))
        x_val = toep.dot(value)
        self.assertItemsAlmostEqual(result_dict[x.id], x_val)

    def test_prune_constants(self):
        """Test pruning constants from constraints.
        """
        x = Variable(2)
        A = np.matrix("1 2; 3 4")
        constraints = (A*x <= 2).canonical_form[1]
        pruned = prune_constants(constraints)
        prod = mul(pruned[0].expr, {})
        self.assertItemsAlmostEqual(prod, np.zeros(A.shape[0]))

        # Test no-op
        constraints = (0*x <= 2).canonical_form[1]
        pruned = prune_constants(constraints)
        prod = mul(pruned[0].expr, {x.id: 1})
        self.assertItemsAlmostEqual(prod, np.zeros(A.shape[0]))

    def test_mul_funcs(self):
        """Test functions to multiply by A, A.T
        """
        n = 10
        x = Variable(n)
        obj = Minimize(norm(x, 1))
        constraints = [x >= 2]
        prob = Problem(obj, constraints)
        data = prob.get_problem_data(solver=SCS)
        A = data["A"]
        objective, constraints = prob.canonicalize()
        sym_data = SymData(objective, constraints, SOLVERS[SCS])
        sym_data.constraints = prune_constants(sym_data.constraints)
        Amul, ATmul = iterative.get_mul_funcs(sym_data)
        vec = np.array(range(sym_data.x_length))
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
