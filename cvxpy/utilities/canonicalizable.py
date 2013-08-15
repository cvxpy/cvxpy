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

class Canonicalizable(object):
    """ Interface for objects that can be canonicalized. """
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        self._aff_obj,self._aff_constr = self.canonicalize()
        super(Canonicalizable, self).__init__()

    # Returns the objective and a shallow copy of the constraints list.
    def canonical_form(self):
        return (self._aff_obj,self._aff_constr[:])

    # Returns an affine expression and affine constraints
    # representing the expression's objective and constraints
    # as a partial optimization problem.
    # Creates new variables if necessary.
    @abc.abstractmethod
    def canonicalize(self):
        return NotImplemented