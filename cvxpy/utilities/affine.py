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
from .. import settings as s

class Affine(object):
    """
    An interface for objects that can be or can contain affine expressions.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def coefficients(self):
        """Returns a dict representing the terms and their coefficients.

        Returns:
            A dict of Variable object to Numpy ndarray of cvxopt spmatrices.
            Also includes a string key for constant terms.
        """
        return NotImplemented

    def variables(self):
        """Returns the variables in the expression/constraint.

        Returns:
            A list of Variable objects.
        """
        variables = []
        for var in self.coefficients().keys():
            if var is not s.CONSTANT:
                variables.append(var)
        return variables
