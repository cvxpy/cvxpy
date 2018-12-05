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

    def is_atom_log_log_convex(self):
        """Is the atom log-log convex?"""
        return True

    def is_atom_log_log_concave(self):
        """Is the atom log-log concave?"""
        return True

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
