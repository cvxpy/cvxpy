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
import cvxpy.lin_ops.lin_utils as lu
import numpy as np

class mul_elemwise(AffAtom):
    """ Multiplies two expressions elementwise.

    The first expression must be constant.
    """

    def __init__(self, lh_const, rh_expr):
        super(mul_elemwise, self).__init__(lh_const, rh_expr)

    @AffAtom.numpy_numeric
    def numeric(self, values):
        """Multiplies the values elementwise.
        """
        return np.multiply(values[0], values[1])

    def validate_arguments(self):
        """Checks that the arguments are valid.

           Left-hand argument must be constant.
        """
        if not self.args[0].is_constant():
            raise ValueError( ("The first argument to mul_elemwise must "
                               "be constant.") )

    def init_dcp_attr(self):
        """Sets the sign, curvature, and shape.
        """
        self._dcp_attr = u.DCPAttr.mul_elemwise(
            self.args[0]._dcp_attr,
            self.args[1]._dcp_attr,
        )

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        """Multiply the expressions elementwise.

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
        # One of the arguments is a scalar, so we can use normal multiplication.
        if arg_objs[0].size != arg_objs[1].size:
            return (lu.mul_expr(arg_objs[0], arg_objs[1], size), [])
        else:
            return (lu.mul_elemwise(arg_objs[0], arg_objs[1]), [])
