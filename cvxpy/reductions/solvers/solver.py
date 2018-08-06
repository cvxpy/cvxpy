"""
Copyright 2017 Robin Verschueren

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
