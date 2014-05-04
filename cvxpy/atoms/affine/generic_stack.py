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
from cvxpy.atoms.affine.affine_atom import AffAtom

class GenericStack(AffAtom):
    """ Concatenation along some dimension. """
    __metaclass__ = abc.ABCMeta
    # Concatenates sign and curvature as dense matrices.
    def sign_curv_from_args(self, stack_func):
        signs = []
        curvatures = []
        # Promote the sign and curvature matrices to the declared size.
        for arg in self.args:
            signs.append( arg._dcp_attr.sign.promote(*arg.size) )
            curvatures.append( arg._dcp_attr.curvature.promote(*arg.size) )

        # Sign.
        neg_mat = stack_func([sign.neg_mat for sign in signs])
        pos_mat = stack_func([sign.pos_mat for sign in signs])
        # Curvature.
        cvx_mat = stack_func([c.cvx_mat for c in curvatures])
        conc_mat = stack_func([c.conc_mat for c in curvatures])
        constant = stack_func([c.nonconst_mat for c in curvatures])

        return (u.Sign(neg_mat, pos_mat),
                u.Curvature(cvx_mat, conc_mat, constant))

    # Sets the shape, sign, and curvature.
    def init_dcp_attr(self):
        shape = self.shape_from_args()
        sign, curvature = self.sign_curv_from_args()
        self._dcp_attr = u.DCPAttr(sign, curvature, shape)
