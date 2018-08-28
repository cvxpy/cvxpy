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
from cvxpy.constraints import SOC, NonPos, Zero
from cvxpy.problems.problem_data.problem_data import ProblemData

from .conic_solver import ConicSolver


class Elemental(ConicSolver):
    """An interface for the Elemental solver.
    """

    # Solver capabilities.
    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC]

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
            if type(constr) not in Elemental.SUPPORTED_CONSTRAINTS:
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
        objective, _ = problem.objective.canonical_form
        constraints = [con for c in problem.constraints for con in c.canonical_form[1]]
        data["objective"] = objective
        data["constraints"] = constraints
        inv_data = {self.VAR_ID: problem.variables()[0].id}

        # Order and group constraints.
        eq_constr = [c for c in problem.constraints if type(c) == Zero]
        inv_data[self.EQ_CONSTR] = eq_constr
        leq_constr = [c for c in problem.constraints if type(c) == NonPos]
        inv_data[self.NEQ_CONSTR] = leq_constr
        return data, inv_data

    def solve_via_data(self, data, warm_start, verbose, solver_opts, solver_cache=None):
        from cvxpy.problems.solvers.elemental_intf import Elemental as EL_OLD
        solver = EL_OLD()
        return solver.solve(
            data["objective"],
            data["constraints"],
            {self.name(): ProblemData()},
            warm_start,
            verbose,
            solver_opts)
