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

import abc
import cvxpy.utilities as u
from cvxpy.atoms.atom import Atom

class AxisAtom(Atom):
    """
    An abstract base class for atoms that can be applied along an axis.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, expr, axis=None):
        self.axis = axis
        super(AxisAtom, self).__init__(expr)

    def shape_from_args(self):
        """Depends on axis.
        """
        if self.axis is None:
            return u.Shape(1, 1)
        elif self.axis == 0:
            return u.Shape(1, self.args[0].size[1])
        else: # axis == 1.
            return u.Shape(self.args[0].size[0], 1)

    def get_data(self):
        """Returns the axis being summed.
        """
        return [self.axis]

    def validate_arguments(self):
        """Checks that the new shape has the same number of entries as the old.
        """
        if self.axis is not None and not self.axis in [0, 1]:
            raise ValueError("Invalid argument for axis.")
