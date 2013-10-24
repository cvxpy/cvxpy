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

from .. import utilities as u
from .. import interface as intf
from constants import Constant

class AffVstack(u.Affine):
    """ Vertical concatenation of Affine Objectives. """
    def __init__(self, *args):
        self.args = [self.cast_as_affine(arg) for arg in args]
        cols = self.args[0].size[1]
        rows = sum(arg.size[0] for arg in self.args)
        self._shape = u.Shape(rows, cols)
        self._vars = []
        map(self._vars.extend, (arg.variables() for arg in self.args))
        super(AffVstack, self).__init__()
    
    def variables(self):
        return self._vars

    # The dimensions of the vstack.
    @property
    def size(self):
        return self._shape.size

    # Places the coefficients of all the blocks
    # as blocks in zero matrices.
    def coefficients(self, interface):
        coeffs = {}
        offset = 0
        for arg in self.args:
            rows = arg.size[0]
            arg_coeffs = arg.coefficients(interface)
            for k,v in arg_coeffs.items():
                # Constant coefficients have the same dimension as the argument.
                if k is Constant:
                    cols = arg.size[1]
                # Variable coefficients have the dimensions arg rows by variable columns.
                else:
                    cols = k.size[0]
                if k in coeffs:
                    interface.block_add(coeffs[k], v, offset, 0, rows, cols)
                else:
                    zeros = interface.zeros(self.size[0], cols)
                    interface.block_add(zeros, v, offset, 0, rows, cols)
                    coeffs[k] = zeros
            offset += rows
        return coeffs