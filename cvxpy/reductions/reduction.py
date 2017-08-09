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
    """Abstract base class for reductions.

    A reduction is an actor that transforms a problem into an
    equivalent problem. By equivalent we mean that there exists
    a mapping between solutions of either problem: if we reduce a problem
    :math:`A` to another problem :math:`B` and then proceed to find a solution
    to :math:`B`, we can convert it to a solution of :math:`A` with at most a
    moderate amount of effort.

    Every reduction supports three methods: accepts, apply, and invert.
    The accepts method of a particular reduction codifies the types of problems
    that it is applicable to; the apply method takes a problem and reduces
    it to a (new) equivalent form, and the invert method maps solutions
    from reduced-to problems to their problems of provenance.
    """

    __metaclass__ = abc.ABCMeta

    def accepts(self, problem):
        """States whether the reduction accepts a problem.

        Parameters
        ----------
        problem : Problem
            The problem to check.

        Returns
        -------
        bool
            True if the reduction can be applied, False otherwise.
        """
        return NotImplemented

    @abc.abstractmethod
    def apply(self, problem):
        """Applies the reduction to a problem and returns an equivalent problem.

        Parameters
        ----------
        problem : Problem
            The problem to which the reduction will be applied.

        Returns
        -------
        Problem or dict
            An equivalent problem, encoded either as a Problem or a dict.

        InverseData, list or dict
            Data needed by the reduction in order to invert this particular
            application.
        """
        return NotImplemented

    @abc.abstractmethod
    def invert(self, solution, inverse_data):
        """Returns a solution to the original problem given the inverse_data.

        Parameters
        ----------
        solution : Solution
            A solution to a problem that generated the inverse_data.
        inverse_data
            The data encoding the original problem.

        Returns
        -------
        Solution
            A solution to the original problem.
        """
        return NotImplemented
