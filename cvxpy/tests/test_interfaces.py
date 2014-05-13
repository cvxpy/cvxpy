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

import cvxpy.interface as intf
from cvxpy.utilities import Sign
import numpy as np
import scipy.sparse as sp
import cvxopt
import scipy
import unittest

class TestInterfaces(unittest.TestCase):
    """ Unit tests for matrix interfaces. """
    def setUp(self):
        pass

    def sign_for_intf(self, interface):
        """Test sign for a given interface.
        """
        mat = interface.const_to_matrix([[1,2,3,4],[3,4,5,6]])
        self.assertEquals(intf.sign(mat), Sign.POSITIVE)
        self.assertEquals(intf.sign(-mat), Sign.NEGATIVE)
        self.assertEquals(intf.sign(0*mat), Sign.ZERO)
        mat = interface.const_to_matrix([[-1,2,3,4],[3,4,5,6]])
        self.assertEquals(intf.sign(mat), Sign.UNKNOWN)

    # Test cvxopt dense interface.
    def test_cvxopt_dense(self):
        interface = intf.get_matrix_interface(cvxopt.matrix)
        # const_to_matrix
        mat = interface.const_to_matrix([1,2,3])
        self.assertEquals(interface.size(mat), (3,1))
        sp_mat = sp.coo_matrix(([1,2], ([3,4], [2,1])), (5, 5))
        mat = interface.const_to_matrix(sp_mat)
        self.assertEquals(interface.size(mat), (5,5))
        # identity
        mat = interface.identity(4)
        cmp_mat = interface.const_to_matrix(np.eye(4))
        self.assertEquals(type(mat), type(cmp_mat))
        self.assertEquals(interface.size(mat), interface.size(cmp_mat))
        assert not mat - cmp_mat
        # scalar_matrix
        mat = interface.scalar_matrix(2,4,3)
        self.assertEquals(interface.size(mat), (4,3))
        self.assertEquals(interface.index(mat, (1,2)), 2)
        # reshape
        mat = interface.const_to_matrix([[1,2,3],[3,4,5]])
        mat = interface.reshape(mat, (6,1))
        self.assertEquals(interface.index(mat, (4,0)), 4)
        # index
        mat = interface.const_to_matrix([[1,2,3,4],[3,4,5,6]])
        self.assertEquals( interface.index(mat, (0,1)), 3)
        mat = interface.index(mat, (slice(1,4,2), slice(0,2,None)))
        self.assertEquals(list(mat), [2,4,4,6])
        # Sign
        self.sign_for_intf(interface)

    # Test cvxopt sparse interface.
    def test_cvxopt_sparse(self):
        interface = intf.get_matrix_interface(cvxopt.spmatrix)
        # const_to_matrix
        mat = interface.const_to_matrix([1,2,3])
        self.assertEquals(interface.size(mat), (3,1))
        # identity
        mat = interface.identity(4)
        cmp_mat = interface.const_to_matrix(np.eye(4))
        self.assertEquals(interface.size(mat), interface.size(cmp_mat))
        assert not mat - cmp_mat
        assert intf.is_sparse(mat)
        # scalar_matrix
        mat = interface.scalar_matrix(2,4,3)
        self.assertEquals(interface.size(mat), (4,3))
        self.assertEquals(interface.index(mat, (1,2)), 2)
        # reshape
        mat = interface.const_to_matrix([[1,2,3],[3,4,5]])
        mat = interface.reshape(mat, (6,1))
        self.assertEquals(interface.index(mat, (4,0)), 4)
        # Test scalars.
        scalar = interface.scalar_matrix(1, 1, 1)
        self.assertEquals(type(scalar), cvxopt.spmatrix)
        scalar = interface.scalar_matrix(1, 1, 3)
        self.assertEquals(scalar.size, (1,3))
        # index
        mat = interface.const_to_matrix([[1,2,3,4],[3,4,5,6]])
        self.assertEquals( interface.index(mat, (0,1)), 3)
        mat = interface.index(mat, (slice(1,4,2), slice(0,2,None)))
        self.assertEquals(list(mat), [2,4,4,6])
        # Sign
        self.sign_for_intf(interface)

    # Test numpy ndarray interface.
    def test_ndarray(self):
        interface = intf.get_matrix_interface(np.ndarray)
        # const_to_matrix
        mat = interface.const_to_matrix([1,2,3])
        self.assertEquals(interface.size(mat), (3,1))
        mat = interface.const_to_matrix([1,2])
        self.assertEquals(interface.size(mat), (2,1))
        # CVXOPT sparse conversion
        tmp = intf.get_matrix_interface(cvxopt.spmatrix).const_to_matrix([1,2,3])
        mat = interface.const_to_matrix(tmp)
        assert (mat == interface.const_to_matrix([1,2,3])).all()
        # identity
        mat = interface.identity(4)
        cvxopt_dense = intf.get_matrix_interface(cvxopt.matrix)
        cmp_mat = interface.const_to_matrix(cvxopt_dense.identity(4))
        self.assertEquals(interface.size(mat), interface.size(cmp_mat))
        assert (mat == cmp_mat).all()
        # scalar_matrix
        mat = interface.scalar_matrix(2,4,3)
        self.assertEquals(interface.size(mat), (4,3))
        self.assertEquals(interface.index(mat, (1,2)), 2)
        # reshape
        mat = interface.const_to_matrix([[1,2,3],[3,4,5]])
        mat = interface.reshape(mat, (6,1))
        self.assertEquals(interface.index(mat, (4,0)), 4)
        # index
        mat = interface.const_to_matrix([[1,2,3,4],[3,4,5,6]])
        self.assertEquals( interface.index(mat, (0,1)), 3)
        mat = interface.index(mat, (slice(1,4,2), slice(0,2,None)))
        self.assertEquals(list(mat.flatten('C')), [2,4,4,6])
        # Scalars and matrices.
        scalar = interface.const_to_matrix(2)
        mat = interface.const_to_matrix([1,2,3])
        assert (scalar*mat == interface.const_to_matrix([2,4,6])).all()
        assert (scalar - mat == interface.const_to_matrix([1,0,-1])).all()
        # Sign
        self.sign_for_intf(interface)

    # Test numpy matrix interface.
    def test_numpy_matrix(self):
        interface = intf.get_matrix_interface(np.matrix)
        # const_to_matrix
        mat = interface.const_to_matrix([1,2,3])
        self.assertEquals(interface.size(mat), (3,1))
        mat = interface.const_to_matrix([[1],[2],[3]])
        self.assertEquals(mat[0,0], 1)
        # identity
        mat = interface.identity(4)
        cvxopt_dense = intf.get_matrix_interface(cvxopt.matrix)
        cmp_mat = interface.const_to_matrix(cvxopt_dense.identity(4))
        self.assertEquals(interface.size(mat), interface.size(cmp_mat))
        assert not (mat - cmp_mat).any()
        # scalar_matrix
        mat = interface.scalar_matrix(2,4,3)
        self.assertEquals(interface.size(mat), (4,3))
        self.assertEquals(interface.index(mat, (1,2)), 2)
        # reshape
        mat = interface.const_to_matrix([[1,2,3],[3,4,5]])
        mat = interface.reshape(mat, (6,1))
        self.assertEquals(interface.index(mat, (4,0)), 4)
        # index
        mat = interface.const_to_matrix([[1,2,3,4],[3,4,5,6]])
        self.assertEquals( interface.index(mat, (0,1)), 3)
        mat = interface.index(mat, (slice(1,4,2), slice(0,2,None)))
        assert not (mat - np.matrix("2 4; 4 6")).any()
        # Sign
        self.sign_for_intf(interface)

    # Test cvxopt sparse interface.
    def test_scipy_sparse(self):
        interface = intf.get_matrix_interface(sp.csc_matrix)
        # const_to_matrix
        mat = interface.const_to_matrix([1,2,3])
        self.assertEquals(interface.size(mat), (3,1))
        C = cvxopt.spmatrix([1,1,1,1,1],[0,1,2,0,0,],[0,0,0,1,2])
        mat = interface.const_to_matrix(C)
        self.assertEquals(interface.size(mat), (3, 3))
        # identity
        mat = interface.identity(4)
        cmp_mat = interface.const_to_matrix(np.eye(4))
        self.assertEquals(interface.size(mat), interface.size(cmp_mat))
        assert (mat - cmp_mat).nnz == 0
        # scalar_matrix
        mat = interface.scalar_matrix(2,4,3)
        self.assertEquals(interface.size(mat), (4,3))
        self.assertEquals(interface.index(mat, (1,2)), 2)
        # reshape
        mat = interface.const_to_matrix([[1,2,3],[3,4,5]])
        mat = interface.reshape(mat, (6,1))
        self.assertEquals(interface.index(mat, (4,0)), 4)
        # Test scalars.
        scalar = interface.scalar_matrix(1, 1, 1)
        self.assertEquals(type(scalar), np.ndarray)
        scalar = interface.scalar_matrix(1, 1, 3)
        self.assertEquals(scalar.shape, (1,3))
        # index
        mat = interface.const_to_matrix([[1,2,3,4],[3,4,5,6]])
        self.assertEquals( interface.index(mat, (0,1)), 3)
        mat = interface.index(mat, (slice(1,4,2), slice(0,2,None)))
        assert not (mat - np.matrix("2 4; 4 6")).any()
        # scalar value
        mat = sp.eye(1)
        self.assertEqual(intf.scalar_value(mat), 1.0)
        # Sign
        self.sign_for_intf(interface)
