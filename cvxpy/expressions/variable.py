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

from cvxpy import settings as s
from cvxpy.expressions.leaf import Leaf
import cvxpy.lin_ops.lin_utils as lu
import scipy.sparse as sp


def upper_tri_to_full(n):
    """Returns a coefficient matrix to create a symmetric matrix.

    Parameters
    ----------
    n : int
        The width/height of the matrix.

    Returns
    -------
    SciPy CSC matrix
        The coefficient matrix.
    """
    entries = n*(n+1)//2

    val_arr = []
    row_arr = []
    col_arr = []
    count = 0
    for i in range(n):
        for j in range(i, n):
            # Index in the original matrix.
            col_arr.append(count)
            # Index in the filled matrix.
            row_arr.append(j*n + i)
            val_arr.append(1.0)
            if i != j:
                # Index in the original matrix.
                col_arr.append(count)
                # Index in the filled matrix.
                row_arr.append(i*n + j)
                val_arr.append(1.0)
            count += 1

    return sp.coo_matrix((val_arr, (row_arr, col_arr)),
                         (n*n, entries)).tocsc()


class Variable(Leaf):
    """The optimization variables in a problem.
    """

    def __init__(self, shape=(), name=None, var_id=None, **kwargs):
        if var_id is None:
            self.id = lu.get_id()
        else:
            self.id = var_id
        if name is None:
            self._name = "%s%d" % (s.VAR_PREFIX, self.id)
        else:
            self._name = name

        self._value = None
        super(Variable, self).__init__(shape, **kwargs)

    def name(self):
        """str : The name of the variable."""
        return self._name

    @property
    def grad(self):
        """Gives the (sub/super)gradient of the expression w.r.t. each variable.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Returns:
            A map of variable to SciPy CSC sparse matrix or None.
        """
        # TODO(akshayka): Do not assume shape is 2D.
        return {self: sp.eye(self.size).tocsc()}

    def variables(self):
        """Returns itself as a variable.
        """
        return [self]

    def canonicalize(self):
        """Returns the graph implementation of the object.

        Returns:
            A tuple of (affine expression, [constraints]).
        """
        obj = lu.create_var(self.shape, self.id)
        return (obj, [])

    def __repr__(self):
        """String to recreate the object.
        """
        attr_str = self._get_attr_str()
        if len(attr_str) > 0:
            return "Variable(%s%s)" % (self.shape, attr_str)
        else:
            return "Variable(%s)" % (self.shape,)
