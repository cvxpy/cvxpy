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
import scipy.sparse as sp
import numpy as np

class Variable(Leaf):
    """ The base variable class """
    VAR_COUNT = 0
    # name - unique identifier.
    # rows - variable height.
    # cols - variable width.
    def __init__(self, rows=1, cols=1, name=None):
        self.set_id()
        if name is None:
            self._name = "%s%d" % (s.VAR_PREFIX, self.id)
        else:
            self._name = name
        self.primal_value = None
        self._dcp_attr = u.DCPAttr(u.Sign.UNKNOWN,
                                   u.Curvature.AFFINE,
                                   u.Shape(rows, cols))

    def name(self):
        return self._name

    # Save the value of the primal variable.
    def save_value(self, value):
        self.primal_value = value

    @property
    def value(self):
        return self.primal_value

    def variables(self):
        """Returns itself as a variable.
        """
        return [self]

    # Returns a coefficients dict with the variable as the key
    # and a list of offset identity matrices as the coefficients.
    def _tree_to_coeffs(self):
        rows, cols = self.size
        V = rows*[1.0]
        I = [i for i in range(rows)]
        # Create the blocks.
        blocks = []
        for col in range(cols):
            shape = (rows, rows*cols)
            selection = [i for i in range(col*rows, (col+1)*rows)]
            mat = sp.coo_matrix((V, (I, selection)), shape)
            blocks.append(mat.tocsc())
        coeffs = {self.id: np.array(blocks, dtype="object", ndmin=1)}
        return cu.format_coeffs(coeffs)
