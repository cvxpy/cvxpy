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

from ... import utilities as u
from ... import interface as intf
from ... import settings as s
from ..affine import AffExpression
from ..leaf import Leaf
import numpy as np

class Constant(Leaf, AffExpression):
    """
    A constant, either matrix or scalar.
    """
    def __init__(self, value):
        self._value = intf.DEFAULT_SPARSE_INTERFACE.const_to_matrix(value)
        # Set DCP attributes.
        dcp_attr = self.init_dcp_attr()
        # Set coefficients.
        coeffs = self.init_coefficients(*dcp_attr.shape.size)
        super(Constant, self).__init__(coeffs, dcp_attr)

    def name(self):
        return str(self.value)

    @property
    def value(self):
        return self._value

    # Return the DCP attributes of the constant.
    def init_dcp_attr(self):
        shape = u.Shape(*intf.size(self.value))
        sign = intf.sign(self.value)
        return u.DCPAttr(sign, u.Curvature.CONSTANT, shape)
        
    # Returns a coefficient dict with s.CONSTANT as the key
    # and the constant value split into columns as the value.
    def init_coefficients(self, rows, cols):
        blocks = []
        for i in range(cols):
            blocks.append( intf.index(self.value, (slice(None,None,None), i)) )
        return {s.CONSTANT: np.array(blocks, dtype="object", ndmin=1)}