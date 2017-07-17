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

import cvxpy.settings as s
from cvxpy.constraints import NonPos, Zero
from cvxpy.solver_interface.conic_solvers.cvxopt_conif import CVXOPT
from cvxpy.problems.problem_data.problem_data import ProblemData


class GLPK(CVXOPT):
    """An interface for the GLPK solver.
    """

    # Solver capabilities.
    LP_CAPABLE = True
    SOCP_CAPABLE = False
    SDP_CAPABLE = False
    EXP_CAPABLE = False
    MIP_CAPABLE = False

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
            if type(constr) not in [Zero, NonPos]:
                return False
            for arg in constr.args:
                if not arg.is_affine():
                    return False
        return True

    def solve(self, problem, warm_start, verbose, solver_opts):
        from cvxpy.problems.solvers.glpk_intf import GLPK as GLPK_OLD
        solver = GLPK_OLD()
        _, inv_data = self.apply(problem)
        objective, _ = problem.objective.canonical_form
        constraints = [con for c in problem.constraints for con in c.canonical_form[1]]
        sol = solver.solve(
            objective,
            constraints,
            {self.name(): ProblemData()},
            warm_start,
            verbose,
            solver_opts)

        return self.invert(sol, inv_data)
