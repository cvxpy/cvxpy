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
from ..leaf import Leaf
import cvxpy.lin_ops.lin_utils as lu

class Variable(Leaf):
    """ The base variable class """
    # name - unique identifier.
    # rows - variable height.
    # cols - variable width.
    def __init__(self, rows=1, cols=1, name=None):
        self._rows = rows
        self._cols = cols
        self.id = lu.get_id()
        if name is None:
            self._name = "%s%d" % (s.VAR_PREFIX, self.id)
        else:
            self._name = name
        self.primal_value = None
        self.init_dcp_attr()
        super(Variable, self).__init__()

    def init_dcp_attr(self):
        """Determines the curvature, sign, and shape from the arguments.
        """
        self._dcp_attr = u.DCPAttr(u.Sign.UNKNOWN,
                                   u.Curvature.AFFINE,
                                   u.Shape(self._rows, self._cols))

    def get_data(self):
        """Returns info needed to reconstruct the expression besides the args.
        """
        return [self._rows, self._cols, self._name]

    def name(self):
        return self._name

    def save_value(self, value):
        """Save the value of the primal variable.
        """
        self.primal_value = value

    @property
    def value(self):
        return self.primal_value

    @value.setter
    def value(self, val):
        """Assign a value to the variable.
        """
        val = self._validate_value(val)
        self.save_value(val)

    def variables(self):
        """Returns itself as a variable.
        """
        return [self]

    def canonicalize(self):
        """Returns the graph implementation of the object.

        Returns:
            A tuple of (affine expression, [constraints]).
        """
        obj = lu.create_var(self.size, self.id)
        return (obj, [])

    def __repr__(self):
        """String to recreate the object.
        """
        return "Variable(%d, %d)" % self.size
