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

import cvxpy.interface as intf
import numpy as np
import scipy.sparse as sp
from cvxpy.tests.base_test import BaseTest


class TestInterfaces(BaseTest):
    """ Unit tests for matrix interfaces. """

    def setUp(self):
        pass

    def sign_for_intf(self, interface):
        """Test sign for a given interface.
        """
        mat = interface.const_to_matrix([[1, 2, 3, 4], [3, 4, 5, 6]])
        self.assertEqual(intf.sign(mat), (True, False))  # Positive.
        self.assertEqual(intf.sign(-mat), (False, True))  # Negative.
        self.assertEqual(intf.sign(0*mat), (True, True))  # Zero.
        mat = interface.const_to_matrix([[-1, 2, 3, 4], [3, 4, 5, 6]])
        self.assertEqual(intf.sign(mat), (False, False))  # Unknown.

    # # Test cvxopt dense interface.
    # def test_cvxopt_dense(self):
    #     interface = intf.get_matrix_interface(cvxopt.matrix)
    #     # const_to_matrix
    #     mat = interface.const_to_matrix([1, 2, 3])
    #     self.assertEqual(interface.shape(mat), (3, 1))
    #     sp_mat = sp.coo_matrix(([1, 2], ([3, 4], [2, 1])), (5, 5))
    #     mat = interface.const_to_matrix(sp_mat)
    #     self.assertEqual(interface.shape(mat), (5, 5))
    #     # identity
    #     mat = interface.identity(4)
    #     cmp_mat = interface.const_to_matrix(np.eye(4))
    #     self.assertEqual(type(mat), type(cmp_mat))
    #     self.assertEqual(interface.shape(mat), interface.shape(cmp_mat))
    #     assert not mat - cmp_mat
    #     # scalar_matrix
    #     mat = interface.scalar_matrix(2, 4, 3)
    #     self.assertEqual(interface.shape(mat), (4, 3))
    #     self.assertEqual(interface.index(mat, (1, 2)), 2)
    #     # reshape
    #     mat = interface.const_to_matrix([[1, 2, 3], [3, 4, 5]])
    #     mat = interface.reshape(mat, (6, 1))
    #     self.assertEqual(interface.index(mat, (4, 0)), 4)
    #     mat = interface.const_to_matrix(1, convert_scalars=True)
    #     self.assertEqual(type(interface.reshape(mat, (1, 1))), type(mat))
    #     # index
    #     mat = interface.const_to_matrix([[1, 2, 3, 4], [3, 4, 5, 6]])
    #     self.assertEqual(interface.index(mat, (0, 1)), 3)
    #     mat = interface.index(mat, (slice(1, 4, 2), slice(0, 2, None)))
    #     self.assertEqual(list(mat), [2, 4, 4, 6])
    #     # Sign
    #     self.sign_for_intf(interface)

    # # Test cvxopt sparse interface.
    # def test_cvxopt_sparse(self):
    #     interface = intf.get_matrix_interface(cvxopt.spmatrix)
    #     # const_to_matrix
    #     mat = interface.const_to_matrix([1, 2, 3])
    #     self.assertEqual(interface.shape(mat), (3, 1))
    #     # identity
    #     mat = interface.identity(4)
    #     cmp_mat = interface.const_to_matrix(np.eye(4))
    #     self.assertEqual(interface.shape(mat), interface.shape(cmp_mat))
    #     assert not mat - cmp_mat
    #     assert intf.is_sparse(mat)
    #     # scalar_matrix
    #     mat = interface.scalar_matrix(2, 4, 3)
    #     self.assertEqual(interface.shape(mat), (4, 3))
    #     self.assertEqual(interface.index(mat, (1, 2)), 2)
    #     # reshape
    #     mat = interface.const_to_matrix([[1, 2, 3], [3, 4, 5]])
    #     mat = interface.reshape(mat, (6, 1))
    #     self.assertEqual(interface.index(mat, (4, 0)), 4)
    #     mat = interface.const_to_matrix(1, convert_scalars=True)
    #     self.assertEqual(type(interface.reshape(mat, (1, 1))), type(mat))
    #     # Test scalars.
    #     scalar = interface.scalar_matrix(1, 1, 1)
    #     self.assertEqual(type(scalar), cvxopt.spmatrix)
    #     scalar = interface.scalar_matrix(1, 1, 3)
    #     self.assertEqual(scalar.shape, (1, 3))
    #     # index
    #     mat = interface.const_to_matrix([[1, 2, 3, 4], [3, 4, 5, 6]])
    #     self.assertEqual(interface.index(mat, (0, 1)), 3)
    #     mat = interface.index(mat, (slice(1, 4, 2), slice(0, 2, None)))
    #     self.assertEqual(list(mat), [2, 4, 4, 6])
    #     # Sign
    #     self.sign_for_intf(interface)

    # Test numpy ndarray interface.
    def test_ndarray(self):
        interface = intf.get_matrix_interface(np.ndarray)
        # const_to_matrix
        mat = interface.const_to_matrix([1, 2, 3])
        self.assertEqual(interface.shape(mat), (3,))
        mat = interface.const_to_matrix([1, 2])
        self.assertEqual(interface.shape(mat), (2,))
        # # CVXOPT sparse conversion
        # tmp = intf.get_matrix_interface(cvxopt.spmatrix).const_to_matrix([1, 2, 3])
        # mat = interface.const_to_matrix(tmp)
        # assert (mat == interface.const_to_matrix([1, 2, 3])).all()
        # # identity
        # mat = interface.identity(4)
        # cvxopt_dense = intf.get_matrix_interface(cvxopt.matrix)
        # cmp_mat = interface.const_to_matrix(cvxopt_dense.identity(4))
        # self.assertEqual(interface.shape(mat), interface.shape(cmp_mat))
        # assert (mat == cmp_mat).all()
        # scalar_matrix
        mat = interface.scalar_matrix(2, (4, 3))
        self.assertEqual(interface.shape(mat), (4, 3))
        self.assertEqual(interface.index(mat, (1, 2)), 2)
        # reshape
        mat = interface.const_to_matrix([[1, 2, 3], [3, 4, 5]])
        mat = interface.reshape(mat, (6, 1))
        self.assertEqual(interface.index(mat, (4, 0)), 4)
        mat = interface.const_to_matrix(1, convert_scalars=True)
        self.assertEqual(type(interface.reshape(mat, (1, 1))), type(mat))
        # index
        mat = interface.const_to_matrix([[1, 2, 3, 4], [3, 4, 5, 6]])
        self.assertEqual(interface.index(mat, (0, 1)), 3)
        mat = interface.index(mat, (slice(1, 4, 2), slice(0, 2, None)))
        self.assertEqual(list(mat.flatten('C')), [2, 4, 4, 6])
        # Scalars and matrices.
        scalar = interface.const_to_matrix(2)
        mat = interface.const_to_matrix([1, 2, 3])
        assert (scalar*mat == interface.const_to_matrix([2, 4, 6])).all()
        assert (scalar - mat == interface.const_to_matrix([1, 0, -1])).all()
        # Sign
        self.sign_for_intf(interface)
        # shape.
        assert interface.shape(np.array([1, 2, 3])) == (3,)

    # Test numpy matrix interface.
    def test_numpy_matrix(self):
        interface = intf.get_matrix_interface(np.matrix)
        # const_to_matrix
        mat = interface.const_to_matrix([1, 2, 3])
        self.assertEqual(interface.shape(mat), (3, 1))
        mat = interface.const_to_matrix([[1], [2], [3]])
        self.assertEqual(mat[0, 0], 1)
        # identity
        # mat = interface.identity(4)
        # cvxopt_dense = intf.get_matrix_interface(cvxopt.matrix)
        # cmp_mat = interface.const_to_matrix(cvxopt_dense.identity(4))
        # self.assertEqual(interface.shape(mat), interface.shape(cmp_mat))
        # assert not (mat - cmp_mat).any()
        # scalar_matrix
        mat = interface.scalar_matrix(2, (4, 3))
        self.assertEqual(interface.shape(mat), (4, 3))
        self.assertEqual(interface.index(mat, (1, 2)), 2)
        # reshape
        mat = interface.const_to_matrix([[1, 2, 3], [3, 4, 5]])
        mat = interface.reshape(mat, (6, 1))
        self.assertEqual(interface.index(mat, (4, 0)), 4)
        mat = interface.const_to_matrix(1, convert_scalars=True)
        self.assertEqual(type(interface.reshape(mat, (1, 1))), type(mat))
        # index
        mat = interface.const_to_matrix([[1, 2, 3, 4], [3, 4, 5, 6]])
        self.assertEqual(interface.index(mat, (0, 1)), 3)
        mat = interface.index(mat, (slice(1, 4, 2), slice(0, 2, None)))
        assert not (mat - np.array([[2, 4], [4, 6]])).any()
        # Sign
        self.sign_for_intf(interface)

    def test_scipy_sparse(self):
        """Test cvxopt sparse interface.
        """
        interface = intf.get_matrix_interface(sp.csc_matrix)
        # const_to_matrix
        mat = interface.const_to_matrix([1, 2, 3])
        self.assertEqual(interface.shape(mat), (3, 1))
        # C = cvxopt.spmatrix([1, 1, 1, 1, 1], [0, 1, 2, 0, 0, ], [0, 0, 0, 1, 2])
        # mat = interface.const_to_matrix(C)
        # self.assertEqual(interface.shape(mat), (3, 3))
        # identity
        mat = interface.identity(4)
        cmp_mat = interface.const_to_matrix(np.eye(4))
        self.assertEqual(interface.shape(mat), interface.shape(cmp_mat))
        assert (mat - cmp_mat).nnz == 0
        # scalar_matrix
        mat = interface.scalar_matrix(2, (4, 3))
        self.assertEqual(interface.shape(mat), (4, 3))
        self.assertEqual(interface.index(mat, (1, 2)), 2)
        # reshape
        mat = interface.const_to_matrix([[1, 2, 3], [3, 4, 5]])
        mat = interface.reshape(mat, (6, 1))
        self.assertEqual(interface.index(mat, (4, 0)), 4)
        # Test scalars.
        scalar = interface.scalar_matrix(1, (1, 1))
        self.assertEqual(type(scalar), np.ndarray)
        scalar = interface.scalar_matrix(1, (1, 3))
        self.assertEqual(scalar.shape, (1, 3))
        # index
        mat = interface.const_to_matrix([[1, 2, 3, 4], [3, 4, 5, 6]])
        self.assertEqual(interface.index(mat, (0, 1)), 3)
        mat = interface.index(mat, (slice(1, 4, 2), slice(0, 2, None)))
        assert not (mat - np.array([[2, 4], [4, 6]])).any()
        # scalar value
        mat = sp.eye(1)
        self.assertEqual(intf.scalar_value(mat), 1.0)
        # Sign
        self.sign_for_intf(interface)
        # Complex
        # define sparse matrix [[0, 1j],[-1j,0]]
        row = np.array([0, 1])
        col = np.array([1, 0])
        data = np.array([1j, -1j])
        A = sp.csr_matrix((data, (row, col)), shape=(2, 2))
        mat = interface.const_to_matrix(A)
        self.assertEquals(mat[0, 1], 1j)
        self.assertEquals(mat[1, 0], -1j)

    def test_conversion_between_intf(self):
        """Test conversion between every pair of interfaces.
        """
        interfaces = [intf.get_matrix_interface(np.ndarray),
                      intf.get_matrix_interface(np.matrix),
                      intf.get_matrix_interface(sp.csc_matrix)]
        cmp_mat = [[1, 2, 3, 4], [3, 4, 5, 6], [-1, 0, 2, 4]]
        for i in range(len(interfaces)):
            for j in range(i+1, len(interfaces)):
                intf1 = interfaces[i]
                mat1 = intf1.const_to_matrix(cmp_mat)
                intf2 = interfaces[j]
                mat2 = intf2.const_to_matrix(cmp_mat)
                for col in range(len(cmp_mat)):
                    for row in range(len(cmp_mat[0])):
                        key = (slice(row, row+1, None),
                               slice(col, col+1, None))
                        self.assertEqual(intf1.index(mat1, key),
                                         intf2.index(mat2, key))
                        # Convert between the interfaces.
                        self.assertEqual(cmp_mat[col][row],
                                         intf1.index(intf1.const_to_matrix(mat2), key))
                        self.assertEqual(intf2.index(intf2.const_to_matrix(mat1), key),
                                         cmp_mat[col][row])
