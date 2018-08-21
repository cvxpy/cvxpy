"""
Copyright 2013 Steven Diamond

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

import cvxpy.settings as s
from cvxpy.reductions.solvers.conic_solvers import CVXOPT
from cvxpy.problems.problem_data.problem_data import ProblemData
from cvxpy.reductions.solution import Solution
import numpy as np

from .conic_solver import ConicSolver


class GLPK(CVXOPT):
    """An interface for the GLPK solver.
    """

    # Solver capabilities.
    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS

    def name(self):
        """The name of the solver.
        """
        return s.GLPK

    def import_solver(self):
        """Imports the solver.
        """
        from cvxopt import glpk
        glpk  # For flake8

    def accepts(self, problem):
        """Can CVXOPT solve the problem?
        """
        # TODO check if is matrix stuffed.
        if not problem.objective.args[0].is_affine():
            return False
        for constr in problem.constraints:
            if type(constr) not in GLPK.SUPPORTED_CONSTRAINTS:
                return False
            for arg in constr.args:
                if not arg.is_affine():
                    return False
        return True

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        status = solution['status']

        primal_vars = None
        dual_vars = None
        if status in s.SOLUTION_PRESENT:
            opt_val = solution['value']
            primal_vars = {inverse_data[self.VAR_ID]: solution['primal']}
        else:
            if status == s.INFEASIBLE:
                opt_val = np.inf
            elif status == s.UNBOUNDED:
                opt_val = -np.inf
            else:
                opt_val = None

        return Solution(status, opt_val, primal_vars, dual_vars, {})

    def solve_via_data(self, data, warm_start, verbose, solver_opts, solver_cache=None):
        from cvxpy.problems.solvers.glpk_intf import GLPK as GLPK_OLD
        solver = GLPK_OLD()
        return solver.solve(
            data["objective"],
            data["constraints"],
            {self.name(): ProblemData()},
            warm_start,
            verbose,
            solver_opts)
