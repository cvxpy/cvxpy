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
from ...utilities import coefficient_utils as cu
from ... import interface as intf
from ... import settings as s
from ..leaf import Leaf
import numpy as np
import scipy.sparse as sp

class Constant(Leaf):
    """
    A constant, either matrix or scalar.
    """
    def __init__(self, value):
        # TODO keep sparse matrices sparse.
        if sp.issparse(value):
            self._value = intf.DEFAULT_SPARSE_INTERFACE.const_to_matrix(value)
        else:
            self._value = intf.DEFAULT_INTERFACE.const_to_matrix(value)
        # Set DCP attributes.
        self.init_dcp_attr()

    def name(self):
        return str(self.value)

    @property
    def value(self):
        return self._value

    # Return the DCP attributes of the constant.
    def init_dcp_attr(self):
        shape = u.Shape(*intf.size(self.value))
        sign = intf.sign(self.value)
        self._dcp_attr = u.DCPAttr(sign, u.Curvature.CONSTANT, shape)

    # Returns a coefficient dict with s.CONSTANT as the key
    # and the constant value split into columns as the value.
    def _tree_to_coeffs(self):
        rows, cols = self.size
        blocks = []
        for i in range(cols):
            val = intf.index(self.value, (slice(None,None,None), i))
            blocks.append( intf.DEFAULT_SPARSE_INTERFACE.const_to_matrix(val) )
        coeffs = {s.CONSTANT: np.array(blocks, dtype="object", ndmin=1)}
        return cu.format_coeffs(coeffs)
