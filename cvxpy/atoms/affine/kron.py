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
import cvxpy.interface as intf
import cvxpy.lin_ops.lin_utils as lu
import numpy as np

class kron(AffAtom):
    """Kronecker product.
    """
    # TODO work with right hand constant.
    def __init__(self, lh_expr, rh_expr):
        super(kron, self).__init__(lh_expr, rh_expr)

    @AffAtom.numpy_numeric
    def numeric(self, values):
        """Kronecker product of the two values.
        """
        return np.kron(values[0], values[1])

    def validate_arguments(self):
        """Checks that both arguments are vectors, and the first is constant.
        """
        if not self.args[0].is_constant():
            raise ValueError("The first argument to kron must be constant.")

    def shape_from_args(self):
        """The sum of the argument dimensions - 1.
        """
        rows = self.args[0].size[0]*self.args[1].size[0]
        cols = self.args[0].size[1]*self.args[1].size[1]
        return u.Shape(rows, cols)

    def sign_from_args(self):
        """Same as times.
        """
        return self.args[0]._dcp_attr.sign*self.args[1]._dcp_attr.sign

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        """Kronecker product of two matrices.

        Parameters
        ----------
        arg_objs : list
            LinOp for each argument.
        size : tuple
            The size of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        return (lu.kron(arg_objs[0], arg_objs[1], size), [])
