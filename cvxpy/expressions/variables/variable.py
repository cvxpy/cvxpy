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

from ... import settings as s
from ... import utilities as u
from ...utilities import coefficient_utils as cu
from ... import interface as intf
from ..constants import Constant
from ..leaf import Leaf
import numpy as np

class Variable(Leaf):
    """ The base variable class """
    VAR_COUNT = 0
    # name - unique identifier.
    # rows - variable height.
    # cols - variable width.
    def __init__(self, rows=1, cols=1, name=None):
        self._init_id()
        self._name = self.id if name is None else name
        self.primal_value = None
        self._dcp_attr = u.DCPAttr(u.Sign.UNKNOWN,
                                   u.Curvature.AFFINE,
                                   u.Shape(rows, cols))

    # Initialize the id.
    def _init_id(self):
        self.id = self.next_name(s.VAR_PREFIX)

    def name(self):
        return self._name

    # Save the value of the primal variable.
    def save_value(self, value):
        self.primal_value = value

    @property
    def value(self):
        return self.primal_value

    # Returns a coefficients dict with the variable as the key
    # and a list of offset identity matrices as the coefficients.
    def coefficients(self):
        rows, cols = self.size
        identity = intf.DEFAULT_SPARSE_INTERFACE.identity(rows*cols)
        blocks = [identity[i*rows:(i+1)*rows,:] for i in range(cols)]
        coeffs = {self: np.array(blocks, dtype="object", ndmin=1)}
        return cu.format_coeffs(coeffs)
