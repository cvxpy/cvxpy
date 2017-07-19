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


class Reduction(object):
    """Abstract base class for reductions."""
    __metaclass__ = abc.ABCMeta

    def accepts(self, problem):
        """Returns True (False) if the reduction can (not) be applied."""
        return NotImplemented

    @abc.abstractmethod
    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.
        """
        return NotImplemented

    @abc.abstractmethod
    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        return NotImplemented
