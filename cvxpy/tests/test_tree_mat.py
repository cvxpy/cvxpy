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
from cvxpy.lin_ops.tree_mat import mul, tmul
import numpy as np
import scipy.sparse as sp
import unittest
from base_test import BaseTest

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

        result = tmul(expr, result)
        assert (result[x.id] == A.T*A*ones).all()

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
