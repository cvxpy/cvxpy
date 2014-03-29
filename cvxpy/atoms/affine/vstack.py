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

from affine_atom import AffAtom
from ... import settings as s
from ... import utilities as u
from ...utilities import bool_mat_utils as bu
from ...utilities import coefficient_utils as cu
from ... import interface as intf
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

    def _id_to_var(self):
        """Returns a map of variable id to variable object.
        """
        id_to_var = {}
        for arg in self.args:
            for var in arg.variables():
                id_to_var[var.id] = var
        return id_to_var

    # Places all the coefficients as blocks in sparse matrices.
    def _tree_to_coeffs(self):
        id_to_var = self._id_to_var()
        # Use sparse matrices as coefficients.
        interface = intf.DEFAULT_SPARSE_INTERFACE
        new_coeffs = {}
        offset = 0
        for arg in self.args:
            rows = arg.size[0]
            arg_coeffs = arg.coefficients()
            for var_id, blocks in arg_coeffs.items():
                # Constant coefficients have one column.
                if var_id is s.CONSTANT:
                    cols = 1
                # Variable coefficients have a column for each entry.
                else:
                    var = id_to_var[var_id]
                    cols = var.size[0]*var.size[1]
                # Initialize blocks as zero matrices.
                if var_id not in new_coeffs:
                    new_blocks = []
                    for i in xrange(self.size[1]):
                        new_blocks.append( interface.zeros(self.size[0], cols) )
                    new_coeffs[var_id] = np.array(new_blocks, dtype="object", ndmin=1)
                # Add the coefficient blocks into the new blocks.
                for i, block in enumerate(blocks):
                    # Convert to lil before changing structure.
                    new_block = new_coeffs[var_id][i].tolil()
                    interface.block_add(new_block, block,
                                        offset, 0, rows, cols)
                    new_coeffs[var_id][i] = new_block.tocsc()
            offset += rows

        return cu.format_coeffs(new_coeffs)
