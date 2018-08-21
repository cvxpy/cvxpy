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
from cvxpy.reductions.solvers.conic_solvers import GLPK
from cvxpy.problems.problem_data.problem_data import ProblemData
from .conic_solver import ConicSolver


class GLPK_MI(GLPK):
    """An interface for the GLPK MI solver.
    """

    # Solver capabilities.
    MIP_CAPABLE = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS

    def name(self):
        """The name of the solver.
        """
        return s.GLPK_MI

    def solve_via_data(self, data, warm_start, verbose, solver_opts, solver_cache=None):
        from cvxpy.problems.solvers.glpk_mi_intf import GLPK_MI as GLPK_OLD
        solver = GLPK_OLD()
        solver_opts[s.BOOL_IDX] = data[s.BOOL_IDX]
        solver_opts[s.INT_IDX] = data[s.INT_IDX]
        return solver.solve(
            data["objective"],
            data["constraints"],
            {self.name(): ProblemData()},
            warm_start,
            verbose,
            solver_opts)
