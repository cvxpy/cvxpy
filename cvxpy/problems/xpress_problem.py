"""
Copyright 2017 Fair Isaac Corp.

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

import cvxpy.settings as s

from collections import namedtuple

from cvxpy.problems.problem import Problem

# Used in self._cached_data to check if problem's objective or constraints have
# changed.
CachedProblem = namedtuple('CachedProblem', ['objective', 'constraints'])

# Used by pool.map to send solve result back.
SolveResult = namedtuple(
    'SolveResult', ['opt_value', 'status', 'primal_values', 'dual_values'])


class XpressProblem (Problem):

    """A convex optimization problem associated with the Xpress Optimizer

    Attributes
    ----------
    objective : Minimize or Maximize
        The expression to minimize or maximize.
    constraints : list
        The constraints on the problem variables.
    """

    # The solve methods available.
    REGISTERED_SOLVE_METHODS = {}

    def __init__(self, objective, constraints=None):

        super(XpressProblem, self).__init__(objective, constraints)
        self._iis = None
        self._xprob = None

    def _reset_iis(self):
        """Clears the iis information
        """

        self._iis = None
        self._transferRow = None

    def _update_problem_state(self, results_dict, sym_data, solver):
        """Updates the problem state given the solver results.

        Updates problem.status, problem.value and value of
        primal and dual variables.

        Parameters
        ----------
        results_dict : dict
            A dictionary containing the solver results.
        sym_data : SymData
            The symbolic data for the problem.
        solver : Solver
            The solver type used to obtain the results.
        """

        super(XpressProblem, self)._update_problem_state(results_dict, sym_data, solver)

        self._iis = results_dict[s.XPRESS_IIS]
        self._transferRow = results_dict[s.XPRESS_TROW]

    def __repr__(self):
        return "XpressProblem(%s, %s)" % (repr(self.objective),
                                          repr(self.constraints))

    def __neg__(self):
        return XpressProblem(-self.objective, self.constraints)

    def __add__(self, other):
        if other == 0:
            return self
        elif not isinstance(other, XpressProblem):
            return NotImplemented
        return XpressProblem(self.objective + other.objective,
                             list(set(self.constraints + other.constraints)))

    def __sub__(self, other):
        if not isinstance(other, XpressProblem):
            return NotImplemented
        return XpressProblem(self.objective - other.objective,
                             list(set(self.constraints + other.constraints)))

    def __mul__(self, other):
        if not isinstance(other, (int, float)):
            return NotImplemented
        return XpressProblem(self.objective * other, self.constraints)

    def __div__(self, other):
        if not isinstance(other, (int, float)):
            return NotImplemented
        return XpressProblem(self.objective * (1.0 / other), self.constraints)
