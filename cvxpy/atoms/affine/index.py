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

from cvxpy.atoms.affine.affine_atom import AffAtom
import cvxpy.utilities as u
from cvxpy.utilities import key_utils as ku
import cvxpy.lin_ops.lin_utils as lu

class index(AffAtom):
    """ Indexing/slicing into a matrix. """
    # expr - the expression indexed/sliced into.
    # key - the index/slicing key (i.e. expr[key[0],key[1]]).
    def __init__(self, expr, key):
        # Format and validate key.
        self.key = ku.validate_key(key, expr._dcp_attr.shape)
        super(index, self).__init__(expr)

    # The string representation of the atom.
    def name(self):
        return self.args[0].name() + "[%s, %s]" % ku.to_str(self.key)

    # Returns the index/slice into the given value.
    @AffAtom.numpy_numeric
    def numeric(self, values):
        return values[0][self.key]

    def shape_from_args(self):
        """Returns the shape of the index expression.
        """
        return u.Shape(*ku.size(self.key, self.args[0]._dcp_attr.shape))

    def get_data(self):
        """Returns the (row slice, column slice).
        """
        return [self.key]

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        """Index/slice into the expression.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        size : tuple
            The size of the resulting expression.
        data : tuple
            A tuple of slices.

        Returns
        -------
        tuple
            (LinOp, [constraints])
        """
        obj = lu.index(arg_objs[0], size, data[0])
        return (obj, [])

    @staticmethod
    def get_index(matrix, constraints, row, col):
        """Returns a canonicalized index into a matrix.

        Parameters
        ----------
        matrix : LinOp
            The matrix to be indexed.
        constraints : list
            A list of constraints to append to.
        row : int
            The row index.
        col : int
            The column index.
        """
        key = (ku.index_to_slice(row),
               ku.index_to_slice(col))
        idx, idx_constr = index.graph_implementation([matrix],
                                                     (1, 1),
                                                     [key])
        constraints += idx_constr
        return idx

    @staticmethod
    def get_slice(matrix, constraints, row_start, row_end, col_start, col_end):
        """Gets a slice from a matrix

        Parameters
        ----------
        matrix : LinOp
            The matrix in the block equality.
        constraints : list
            A list of constraints to append to.
        row_start : int
            The first row of the matrix section.
        row_end : int
            The last row + 1 of the matrix section.
        col_start : int
            The first column of the matrix section.
        col_end : int
            The last column + 1 of the matrix section.
        """
        key = (slice(row_start, row_end, None),
               slice(col_start, col_end, None))
        rows = row_end - row_start
        cols = col_end - col_start
        slc, idx_constr = index.graph_implementation([matrix],
                                                     (rows, cols),
                                                     [key])
        constraints += idx_constr
        return slc

    @staticmethod
    def block_eq(matrix, block, constraints,
                 row_start, row_end, col_start, col_end):
        """Adds an equality setting a section of the matrix equal to block.

        Assumes block does not need to be promoted.

        Parameters
        ----------
        matrix : LinOp
            The matrix in the block equality.
        block : LinOp
            The block in the block equality.
        constraints : list
            A list of constraints to append to.
        row_start : int
            The first row of the matrix section.
        row_end : int
            The last row + 1 of the matrix section.
        col_start : int
            The first column of the matrix section.
        col_end : int
            The last column + 1 of the matrix section.
        """
        key = (slice(row_start, row_end, None),
               slice(col_start, col_end, None))
        rows = row_end - row_start
        cols = col_end - col_start
        assert block.size == (rows, cols)
        slc, idx_constr = index.graph_implementation([matrix],
                                                     (rows, cols),
                                                     [key])
        constraints += [lu.create_eq(slc, block)] + idx_constr
