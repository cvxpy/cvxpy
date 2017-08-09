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
from cvxpy.atoms.affine.vec import vec
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.utilities import key_utils as ku
import cvxpy.lin_ops.lin_utils as lu
import scipy.sparse as sp
import numpy as np


class index(AffAtom):
    """Indexing/slicing into an Expression.

    CVXPY supports NumPy-like indexing semantics via the Expression
    class' overloading of the ``[]`` operator. This is a low-level class
    constructed by that operator, and it should not be instantiated directly.

    Parameters
    ----------
    expr : Expression
        The expression indexed/sliced into.
    key :
        The index/slicing key (i.e. expr[key[0],key[1]]).
    """

    def __init__(self, expr, key, orig_key=None):
        # Format and validate key.
        if orig_key is None:
            self._orig_key = key
            self.key = ku.validate_key(key, expr.shape)
        else:
            self._orig_key = orig_key
            self.key = key
        super(index, self).__init__(expr)

    # The string representation of the atom.
    def name(self):
        # TODO string should be orig_key
        inner_str = "[%s" + ", %s"*(len(self.key)-1) + "]"
        return self.args[0].name() + inner_str % ku.to_str(self.key)

    def numeric(self, values):
        """ Returns the index/slice into the given value.
        """
        return values[0][self._orig_key]

    def shape_from_args(self):
        """Returns the shape of the index expression.
        """
        return ku.shape(self.key, self._orig_key, self.args[0].shape)

    def get_data(self):
        """Returns the (row slice, column slice).
        """
        return [self.key, self._orig_key]

    @staticmethod
    def graph_implementation(arg_objs, shape, data=None):
        """Index/slice into the expression.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        shape : tuple
            The shape of the resulting expression.
        data : tuple
            A tuple of slices.

        Returns
        -------
        tuple
            (LinOp, [constraints])
        """
        obj = lu.index(arg_objs[0], shape, data[0])
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
        idx_mat = np.arange(expr.size)
        idx_mat = np.reshape(idx_mat, expr.shape, order='F')
        select_mat = idx_mat[key]
        final_shape = select_mat.shape
        select_vec = np.reshape(select_mat, select_mat.size, order='F')
        # Select the chosen entries from expr.
        identity = sp.eye(expr.size).tocsc()
        return reshape(identity[select_vec]*vec(expr), final_shape)
