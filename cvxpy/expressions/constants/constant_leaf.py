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
from ... import interface as intf
from ..leaf import Leaf
import abc

class ConstantLeaf(Leaf):
    """
    ABC for Constant and Parameter.
    """
    __metaclass__ = abc.ABCMeta

    # Return the constant value split into column blocks as a coefficient.
    def coefficients(self):
        # TODO different versions for parameters and constants
        # For parameter multiply by vector to selected each column.
        # Scalars have scalar coefficients.
        if self.is_scalar():
            return {s.CONSTANT: [self.value]}
        else:
            blocks = [self.value[:,i] for i in range(self.size[1])]
            return {s.CONSTANT: blocks}

    # Casts the value into the appropriate matrix type.
    @staticmethod
    def cast_value(value):
        return intf.DEFAULT_SPARSE_INTERFACE.const_to_matrix(value)