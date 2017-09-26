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
from cvxpy.expressions.expression import Expression
import cvxpy.lin_ops.lin_utils as lu
import numpy as np


def promote(expr, shape):
    """ Promote a scalar expression to a vector/matrix.

    Parameters
    ----------
    expr : Expression
        The expression to promote.
    shape : tuple
        The shape to promote to.

    Raises
    ------
    ValueError
        If ``expr`` is not a scalar.
    """

    expr = Expression.cast_to_const(expr)
    if expr.shape != shape:
        if not expr.is_scalar():
            raise ValueError('Only scalars may be promoted.')
        return Promote(expr, shape)
    else:
        return expr


class Promote(AffAtom):
    """ Promote a scalar expression to a vector/matrix.

    Attributes
    ----------
    expr : Expression
        The expression to promote.
    shape : tuple
        The shape to promote to.
    """

    def __init__(self, expr, shape):
        self.promoted_shape = shape
        super(Promote, self).__init__(expr)

    @AffAtom.numpy_numeric
    def numeric(self, values):
        """Promotes the value.
        """
        return np.ones(self.promoted_shape) * values[0]

    def is_symmetric(self):
        """Is the expression symmetric?
        """
        return self.ndim == 2 and self.shape[0] == self.shape[1]

    def shape_from_args(self):
        """Returns the (row, col) shape of the expression.
        """
        return self.promoted_shape

    def get_data(self):
        """Returns info needed to reconstruct the expression besides the args.
        """
        return [self.promoted_shape]

    @staticmethod
    def graph_implementation(arg_objs, shape, data=None):
        """Promote scalar to vector/matrix

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        shape : tuple
            The shape of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        return (lu.promote(arg_objs[0], shape), [])
