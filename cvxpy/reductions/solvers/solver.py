"""
Copyright 2017 Robin Verschueren

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
from collections import defaultdict

from cvxpy.reductions.reduction import Reduction


def group_constraints(constraints):
    """Organize the constraints into a dictionary keyed by constraint names.

    Paramters
    ---------
    constraints : list of constraints

    Returns
    -------
    dict
        A dict keyed by constraint types where dict[cone_type] maps to a list
        of exactly those constraints that are of type cone_type.
    """
    constr_map = defaultdict(list)
    for c in constraints:
        constr_map[type(c)].append(c)
    return constr_map


class Solver(Reduction):
    """Generic interface for a solver that uses reduction semantics
    """

    __metaclass__ = abc.ABCMeta

    # Solver capabilities.
    MIP_CAPABLE = False

    # Keys for inverse data.
    VAR_ID = 'var_id'
    EQ_CONSTR = 'eq_constr'
    NEQ_CONSTR = 'other_constr'

    @abc.abstractmethod
    def name(self):
        """The name of the solver.
        """
        return NotImplemented

    @abc.abstractmethod
    def import_solver(self):
        """Imports the solver.
        """
        return NotImplemented

    def is_installed(self):
        """Is the solver installed?
        """
        try:
            self.import_solver()
            return True
        except ImportError:
            return False

    @abc.abstractmethod
    def solve_via_data(self, data, warm_start, verbose, solver_opts, solver_cache=None):
        """Solve a problem represented by data returned from apply.
        """
        return NotImplemented

    def solve(self, problem, warm_start, verbose, solver_opts):
        """Solve the problem and return a Solution object.
        """
        data, inv_data = self.apply(problem)
        solution = self.solve_via_data(data, warm_start, verbose, solver_opts)
        return self.invert(solution, inv_data)
