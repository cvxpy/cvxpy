"""
Copyright 2013 Steven Diamond, Copyright 2017 Robin Verschueren

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

import numpy as np
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.constraints import SOC, ExpCone, NonPos, Zero
from cvxpy.reductions.solution import Solution

from .conic_solver import ConicSolver
from .reduction_solver import ReductionSolver


class CVXOPT(ConicSolver):
    """An interface for the CVXOPT solver.
    """

    # Solver capabilities.
    LP_CAPABLE = True
    SOCP_CAPABLE = True
    SDP_CAPABLE = True
    EXP_CAPABLE = True
    MIP_CAPABLE = False

    # Map of CVXOPT status to CVXPY status.
    STATUS_MAP = {'optimal': s.OPTIMAL,
                  'primal infeasible': s.INFEASIBLE,
                  'dual infeasible': s.UNBOUNDED,
                  'unknown': s.SOLVER_ERROR}

    def name(self):
        """The name of the solver.
        """
        return s.CVXOPT

    def import_solver(self):
        """Imports the solver.
        """
        from cvxpy.problems.solvers.cvxopt_intf import CVXOPT as CVXOPT_old
        CVXOPT_old  # For flake8

    def accepts(self, problem):
        """Can CVXOPT solve the problem?
        """
        # TODO check if is matrix stuffed.
        if not problem.objective.args[0].is_affine():
            return False
        for constr in problem.constraints:
            if type(constr) not in [Zero, NonPos, SOC, PSD, ExpCone]:
                return False
            for arg in constr.args:
                if not arg.is_affine():
                    return False
        return True

    def apply(self, problem):
        pass
    
    def invert(self, solution, inverse_data):
        pass

    def solve(self, problem, warm_start, verbose, solver_opts):
        from cvxpy.problems.solvers.cvxopt_intf import CVXOPT as CVXOPT_old
        return CVXOPT_old.solve(problem.objective, problem.constraints, {}, \
             warm_start, verbose, solver_opts)

