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
import types
import expression
from .. import utilities as u
from .. import interface as intf

class Leaf(expression.Expression, u.Affine):
    """
    A leaf node, i.e. a Variable, Constant, or Parameter.
    """

    __metaclass__ = abc.ABCMeta

    def coefficients(self, cache=False):
        """Returns a dict of Variable to coefficient.

        Leaves cache their coefficients.

        Returns:
            A dict of Variable object to scipy sparse matrix
            of column coefficients.
            Also includes the key settings.CONSTANT for constant terms.
        """
        return super(Leaf, self).coefficients(cache=True)

    COUNT = 0
    def set_id(self):
        """Sets a new unique id based on a global counter.
        """
        Leaf.COUNT += 1
        self.id = Leaf.COUNT

    def canonicalize(self):
        """Default is the leaf as the objective and no constraints.
        """
        return (self, [])

    def variables(self):
        """Default is empty list of Variables.
        """
        return []

    def parameters(self):
        """Default is empty list of Parameters.
        """
        return []
