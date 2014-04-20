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

import cvxpy.utilities as u
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.utilities import bool_mat_utils as bu
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.affine.index import index
import numpy as np

class vstack(AffAtom):
    """ Vertical concatenation """
    # Returns the vstack of the values.
    @AffAtom.numpy_numeric
    def numeric(self, values):
        return np.vstack(values)

    # The shape is the common width and the sum of the heights.
    def shape_from_args(self):
        cols = self.args[0].size[1]
        rows = sum(arg.size[0] for arg in self.args)
        return u.Shape(rows, cols)

    # All arguments must have the same width.
    def validate_arguments(self):
        arg_cols = [arg.size[1] for arg in self.args]
        if max(arg_cols) != min(arg_cols):
            raise TypeError( ("All arguments to vstack must have "
                              "the same number of columns.") )

    # Vertically concatenates sign and curvature as dense matrices.
    def sign_curv_from_args(self):
        signs = []
        curvatures = []
        # Promote the sign and curvature matrices to the declared size.
        for arg in self.args:
            signs.append( arg._dcp_attr.sign.promote(*arg.size) )
            curvatures.append( arg._dcp_attr.curvature.promote(*arg.size) )

        # Sign.
        neg_mat = bu.vstack([sign.neg_mat for sign in signs])
        pos_mat = bu.vstack([sign.pos_mat for sign in signs])
        # Curvature.
        cvx_mat = bu.vstack([c.cvx_mat for c in curvatures])
        conc_mat = bu.vstack([c.conc_mat for c in curvatures])
        constant = bu.vstack([c.nonconst_mat for c in curvatures])

        return (u.Sign(neg_mat, pos_mat),
                u.Curvature(cvx_mat, conc_mat, constant))

    # Sets the shape, sign, and curvature.
    def init_dcp_attr(self):
        shape = self.shape_from_args()
        sign,curvature = self.sign_curv_from_args()
        self._dcp_attr = u.DCPAttr(sign, curvature, shape)

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        """Stack the expressions vertically.

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
        X = lu.create_var(size)
        constraints = []
        # Create an equality constraint for each arg.
        offset = 0
        for arg in arg_objs:
            index.block_eq(X, arg, constraints,
                           offset, arg.size[0] + offset,
                           0, size[1])
            offset += arg.size[0]
        return (X, constraints)
