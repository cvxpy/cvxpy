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
from cvxpy.constraints import SOC, NonPos, Zero
from cvxpy.problems.problem_data.problem_data import ProblemData

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
            if type(constr) not in [Zero, NonPos, SOC]:
                return False
            for arg in constr.args:
                if not arg.is_affine():
                    return False
        return True

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """
        data = {}
        inv_data = {self.VAR_ID: problem.variables()[0].id}

        # Order and group constraints.
        eq_constr = [c for c in problem.constraints if type(c) == Zero]
        inv_data[self.EQ_CONSTR] = eq_constr
        leq_constr = [c for c in problem.constraints if type(c) == NonPos]
        inv_data[self.NEQ_CONSTR] = leq_constr
        return data, inv_data

    def solve(self, problem, warm_start, verbose, solver_opts):
        from cvxpy.problems.solvers.elemental_intf import Elemental as EL_OLD
        solver = EL_OLD()
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
