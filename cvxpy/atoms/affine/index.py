"""
Copyright 2017 Steven Diamond

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

from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.affine.vec import vec
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.utilities import key_utils as ku
import cvxpy.lin_ops.lin_utils as lu
import scipy.sparse as sp
import numpy as np


class index(AffAtom):
    """ Indexing/slicing into a matrix. """
    # expr - the expression indexed/sliced into.
    # key - the index/slicing key (i.e. expr[key[0],key[1]]).

    def __init__(self, expr, key):
        # Format and validate key.
        self.key = ku.validate_key(key, expr.size)
        super(index, self).__init__(expr)

    # The string representation of the atom.
    def name(self):
        return self.args[0].name() + "[%s, %s]" % ku.to_str(self.key)

    # Returns the index/slice into the given value.
    @AffAtom.numpy_numeric
    def numeric(self, values):
        return values[0][self.key]

    def size_from_args(self):
        """Returns the shape of the index expression.
        """
        return ku.size(self.key, self.args[0].size)

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
    def get_special_slice(expr, key):
        """Indexing using logical indexing or a list of indices.

        Parameters
        ----------
        expr : Expression
            The expression being indexed/sliced into.
        key : tuple
            ndarrays or lists.
        Returns
        -------
        Expression
            An expression representing the index/slice.
        """
        expr = index.cast_to_const(expr)
        # Order the entries of expr and select them using key.
        idx_mat = np.arange(expr.size[0]*expr.size[1])
        idx_mat = np.reshape(idx_mat, expr.size, order='F')
        select_mat = idx_mat[key]
        if select_mat.ndim == 2:
            final_size = select_mat.shape
        else:  # Always cast 1d arrays as column vectors.
            final_size = (select_mat.size, 1)
        select_vec = np.reshape(select_mat, select_mat.size, order='F')
        # Select the chosen entries from expr.
        identity = sp.eye(expr.size[0]*expr.size[1]).tocsc()
        return reshape(identity[select_vec]*vec(expr), *final_size)

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
