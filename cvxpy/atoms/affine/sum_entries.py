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
from cvxpy.atoms.axis_atom import AxisAtom
import cvxpy.lin_ops.lin_utils as lu
import numpy as np


class sum_entries(AxisAtom, AffAtom):
    """ Summing the entries of an expression.

    Attributes
    ----------
    expr : CVXPY Expression
        The expression to sum the entries of.
    """

    def __init__(self, expr, axis=None):
        super(sum_entries, self).__init__(expr, axis=axis)

    @AffAtom.numpy_numeric
    def numeric(self, values):
        """Sums the entries of value.
        """
        return np.sum(values[0], axis=self.axis)

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        """Sum the linear expression's entries.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        size : tuple
            The size of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        axis = data[0]
        if axis is None:
            obj = lu.sum_entries(arg_objs[0])
        elif axis == 1:
            const_size = (arg_objs[0].size[1], 1)
            ones = lu.create_const(np.ones(const_size), const_size)
            obj = lu.rmul_expr(arg_objs[0], ones, size)
        else:  # axis == 0
            const_size = (1, arg_objs[0].size[0])
            ones = lu.create_const(np.ones(const_size), const_size)
            obj = lu.mul_expr(ones, arg_objs[0], size)

        return (obj, [])
