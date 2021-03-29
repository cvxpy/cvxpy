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
from cvxpy.reductions.reduction import Reduction


class Solver(Reduction):
    """Generic interface for a solver that uses reduction semantics
    """

    DIMS = "dims"
    # ^ The key that maps to "ConeDims" in the data returned by apply().
    #
    #   There are separate ConeDims classes for cone programs vs QPs.
    #   See cone_matrix_stuffing.py and qp_matrix_stuffing.py for details.

    __metaclass__ = abc.ABCMeta

    # Solver capabilities.
    MIP_CAPABLE = False

    # Keys for inverse data.
    VAR_ID = 'var_id'
    DUAL_VAR_ID = 'dual_var_id'
    EQ_CONSTR = 'eq_constr'
    NEQ_CONSTR = 'other_constr'

    @abc.abstractmethod
    def name(self):
        """The name of the solver.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def import_solver(self):
        """Imports the solver.
        """
        raise NotImplementedError()

    def is_installed(self) -> bool:
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
        raise NotImplementedError()

    def solve(self, problem, warm_start, verbose, solver_opts):
        """Solve the problem and return a Solution object.
        """
        data, inv_data = self.apply(problem)
        solution = self.solve_via_data(data, warm_start, verbose, solver_opts)
        return self.invert(solution, inv_data)
