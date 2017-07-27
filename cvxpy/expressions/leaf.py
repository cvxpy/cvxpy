"""
Copyright 2017 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
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
