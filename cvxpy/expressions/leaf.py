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
from cvxpy.expressions import expression
import cvxpy.interface as intf
import numpy as np


class Leaf(expression.Expression):
    """
    A leaf node, i.e. a Variable, Constant, or Parameter.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.args = []

    def variables(self):
        """Default is empty list of Variables.
        """
        return []

    def parameters(self):
        """Default is empty list of Parameters.
        """
        return []

    def constants(self):
        """Default is empty list of Constants.
        """
        return []

    def is_convex(self):
        """Is the expression convex?
        """
        return True

    def is_concave(self):
        """Is the expression concave?
        """
        return True

    @property
    def domain(self):
        """A list of constraints describing the closure of the region
           where the expression is finite.
        """
        # Default is full domain.
        return []

    def _validate_value(self, val):
        """Check that the value satisfies the leaf's symbolic attributes.

        Parameters
        ----------
        val : numeric type
            The value assigned.

        Returns
        -------
        numeric type
            The value converted to the proper matrix type.
        """
        if val is not None:
            # Convert val to the proper matrix type.
            val = intf.DEFAULT_INTF.const_to_matrix(val)
            size = intf.size(val)
            if size != self.size:
                raise ValueError(
                    "Invalid dimensions (%s, %s) for %s value." %
                    (size[0], size[1], self.__class__.__name__)
                )
            # All signs are valid if sign is unknown.
            # Otherwise value sign must match declared sign.
            pos_val, neg_val = intf.sign(val)
            if self.is_positive() and not pos_val or \
               self.is_negative() and not neg_val:
                raise ValueError(
                    "Invalid sign for %s value." % self.__class__.__name__
                )
            # Round to correct sign.
            elif self.is_positive():
                val = np.maximum(val, 0)
            elif self.is_negative():
                val = np.minimum(val, 0)
        return val

    def is_quadratic(self):
        """Leaf nodes are always quadratic.
        """
        return True

    def is_pwl(self):
        """Leaf nodes are always piecewise linear.
        """
        return True
