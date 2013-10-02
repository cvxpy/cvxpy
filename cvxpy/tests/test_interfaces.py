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

import cvxpy.interface.matrix_utilities as intf
import cvxpy.interface.numpy_wrapper as np
import cvxopt
import scipy
import unittest

class TestInterfaces(unittest.TestCase):
    """ Unit tests for matrix interfaces. """
    def setUp(self):
        pass

    # Test cvxopt dense interface.
    def test_cvxopt_dense(self):
        interface = intf.get_matrix_interface(cvxopt.matrix)
        # const_to_matrix
        mat = interface.const_to_matrix([1,2,3])
        self.assertEquals(interface.size(mat), (3,1))
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

    # Test numpy ndarray interface.
    def test_ndarray(self):
        interface = intf.get_matrix_interface(np.ndarray)
        # const_to_matrix
        mat = interface.const_to_matrix([1,2,3])
        self.assertEquals(interface.size(mat), (3,1))
        mat = interface.const_to_matrix([1,2])
        self.assertEquals(interface.size(mat), (2,1))
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
        self.assertEquals(list(mat.flatten('C')), [2,4,4,6])

    # Test numpy matrix interface.
    def test_numpy_matrix(self):
        interface = intf.get_matrix_interface(np.matrix)
        # const_to_matrix
        mat = interface.const_to_matrix([1,2,3])
        self.assertEquals(interface.size(mat), (3,1))
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

    # Test interface for lists.
    def test_lists(self):
        # index
        mat = intf.index([[1,2,3,4],[3,4,5,6]], (slice(1,4,2), slice(0,2,None)))
        self.assertItemsEqual(mat, [[2,4],[4,6]])
        mat = intf.index([[1,2,3,4],[3,4,5,6]], (slice(1,4,2), slice(0,1,None)))
        self.assertItemsEqual(mat, [2,4])
        mat = intf.index([[1,2,3,4],[3,4,5,6]], (slice(1,2,None), slice(1,2,None)))
        self.assertEquals(mat, 4)
        mat = intf.index([[2],[2]], (slice(0,1,None), slice(0,1,None)))
        self.assertEquals(mat, 2)