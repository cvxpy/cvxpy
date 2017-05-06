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

import numpy as np
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.constraints import SOC, ExpCone, NonPos, Zero
from cvxpy.problems.solvers.elemental_intf import Elemental as ElementalOld
from cvxpy.reductions.solution import Solution
from cvxpy.solver_interface.reduction_solver import ReductionSolver

from .conic_solver import ConicSolver


class Elemental(ConicSolver):
    """An interface for the Elemental solver.
    """

    # Solver capabilities.
    LP_CAPABLE = True
    SOCP_CAPABLE = True
    SDP_CAPABLE = False
    EXP_CAPABLE = False
    MIP_CAPABLE = False

    # Map of Elemental status to CVXPY status.
    # TODO
    STATUS_MAP = {0: s.OPTIMAL}

    def import_solver(self):
        """Imports the solver.
        """
        import El
        El  # For flake8

    def name(self):
        """The name of the solver.
        """
        return s.ELEMENTAL

    def accepts(self, problem):
        """Can Elemental solve the problem?
        """
        # TODO check if is matrix stuffed.
        if not problem.objective.args[0].is_affine():
            return False
        for constr in problem.constraints:
            if type(constr) not in [Zero, NonPos, SOC, ExpCone]:
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
        old_solver = ElementalOld()
        return old_solver.solve(problem.objective, problem.constraints, {}, warm_start, verbose, solver_opts)
