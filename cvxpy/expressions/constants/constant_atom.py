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

from constant import Constant
import cvxpy.lin_ops.lin_utils as lu

class ConstantAtom(Constant):
    """An atom with constant arguments.
    """

    def __init__(self, atom):
        self.atom = atom
        self._dcp_attr = self.atom._dcp_attr

    @property
    def value(self):
        """The value of the atom evaluated on its arguments.
        """
        return self.atom.value

    def parameters(self):
        """Return all the parameters in the atom.
        """
        return self.atom.parameters()

    def canonicalize(self):
        """Returns the graph implementation of the object.

        Returns:
            A tuple of (affine expression, [constraints]).
        """
        if len(self.parameters()) > 0:
            obj = lu.create_param(self, self.size)
        else:
            obj = lu.create_const(self.value, self.size)
        return (obj, [])
