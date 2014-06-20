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

from noncvx_variable import NonCvxVariable
import cvxpy.lin_ops.lin_utils as lu
from itertools import product

class permutation(NonCvxVariable):
    """ A permutation matrix. """
    def __init__(self, size, *args, **kwargs):
        super(permutation, self).__init__(rows=size, cols=size, *args, **kwargs)

    # Recursively set the largest value to 1 and zero out the
    # rest of that value's row and column.
    def _round(self, matrix):
        dims = range(self.size[0])
        ind_val = [(i, j, matrix[i, j]) for (i, j) in product(dims, dims)]
        chosen = self.get_largest(ind_val, [])
        matrix *= 0 # Zero out the matrix.
        for i,j,v in chosen:
            matrix[i,j] = 1
        return matrix

    # Get the index of the largest value by magnitude, filter out
    # all entries in the same row or column, and recurse.
    def get_largest(self, ind_val, chosen):
        # The final list will have 1 entry per row/col.
        if len(ind_val) == 0:
            return chosen
        largest = max(ind_val, key=lambda tup: abs(tup[2]))
        ind_val = [tup for tup in ind_val if \
                   tup[0] != largest[0] and tup[1] != largest[1]]
        return self.get_largest(ind_val, chosen + [largest])

    # Constrain all entries to be zero that correspond to
    # zeros in the matrix.
    def _fix(self, matrix):
        return [self == matrix]

    # In the relaxation, 0 <= var <= 1 and sum(var) == k.
    def constraints(self):
        obj, constraints = super(BoolVar, self).canonicalize()
        one = lu.create_const(1, (1, 1))
        constraints += [lu.create_geq(obj),
                        lu.create_leq(obj, one)]
        for i in range(self.size[0]):
            row_sum = lu.sum_expr([self[i, j] for j in range(self.size[0])])
            col_sum = lu.sum_expr([self[j, i] for j in range(self.size[0])])
            constraints += [lu.create_eq(row_sum, one),
                            lu.create_eq(col_sum, one)]
        return constraints
