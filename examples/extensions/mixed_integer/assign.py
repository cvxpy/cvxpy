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
from munkres import Munkres

import cvxpy.lin_ops.lin_utils as lu

from .boolean import Boolean


class Assign(Boolean):
    """ An assignment matrix. """
    def __init__(self, rows, cols, *args, **kwargs) -> None:
        assert rows >= cols
        super(Assign, self).__init__(rows=rows, cols=cols, *args, **kwargs)

    def init_z(self):
        self.z.value = np.ones(self.size)/self.size[1]

    # Compute projection with maximal weighted matching.
    def _round(self, matrix):
        m = Munkres()
        lists = self.matrix_to_lists(matrix)
        indexes = m.compute(lists)
        matrix *= 0
        for row, column in indexes:
            matrix[row, column] = 1
        return matrix

    def matrix_to_lists(self, matrix):
        """Convert a matrix to a list of lists.
        """
        rows, cols = matrix.shape
        lists = []
        for i in range(rows):
            lists.append(matrix[i,:].tolist()[0])
        return lists

    # Constrain all entries to be zero that correspond to
    # zeros in the matrix.
    def _fix(self, matrix):
        return [self == matrix]

    # In the relaxation, we have 0 <= var <= 1.
    def canonicalize(self):
        obj, constraints = super(Assign, self).canonicalize()
        shape = (self.size[1], 1)
        one_row_vec = lu.create_const(np.ones(shape), shape)
        shape = (1, self.size[0])
        one_col_vec = lu.create_const(np.ones(shape), shape)
        # Row sum <= 1
        row_sum = lu.rmul_expr(obj, one_row_vec, (self.size[0], 1))
        constraints += [lu.create_leq(row_sum, lu.transpose(one_col_vec))]
        # Col sum == 1.
        col_sum = lu.mul_expr(one_col_vec, obj, (1, self.size[1]))
        constraints += [lu.create_eq(col_sum, lu.transpose(one_row_vec))]
        return (obj, constraints)
